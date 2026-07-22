"""FastAPI deployment prototype for LC25000 histopathology classification."""

from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from config import (
    API_TITLE,
    API_VERSION,
    DEPLOYED_MODEL_NAME,
    DEPLOYED_MODEL_PATH,
    IMAGENET_NORMALIZE_MEAN,
    IMAGENET_NORMALIZE_STD,
    LC25000_CLASS_NAMES,
    RESEARCH_DISCLAIMER,
)
from model import TransferCNN


app = FastAPI(title=API_TITLE, version=API_VERSION)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_NORMALIZE_MEAN,
            std=IMAGENET_NORMALIZE_STD,
        ),
    ]
)

model = None


def load_lc25000_model():
    """Load the trained LC25000 ResNet-18 checkpoint."""

    if not DEPLOYED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {DEPLOYED_MODEL_PATH}"
        )

    loaded_model = TransferCNN(
        num_classes=len(LC25000_CLASS_NAMES),
        training_mode="staged_finetune",
    )
    loaded_model.model_name = DEPLOYED_MODEL_NAME

    loaded_model.load_state_dict(
        torch.load(
            DEPLOYED_MODEL_PATH,
            map_location=device,
        )
    )

    loaded_model.to(device)
    loaded_model.eval()

    return loaded_model


@app.on_event("startup")
def startup_event():
    """Load the trained model when the API starts."""

    global model
    model = load_lc25000_model()


@app.get("/health")
def health():
    """Return the current API and model status."""

    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": model is not None,
        "model_path": str(DEPLOYED_MODEL_PATH),
        "model": DEPLOYED_MODEL_NAME,
        "classes": LC25000_CLASS_NAMES,
        "disclaimer": RESEARCH_DISCLAIMER,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify an uploaded histopathology image."""

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        file_bytes = await file.read()
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {exc}",
        ) from exc

    try:
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        confidence, predicted_index_tensor = torch.max(
            probabilities,
            dim=0,
        )
        predicted_index = int(predicted_index_tensor.item())

        probability_map = {
            class_name: round(float(probability.item()), 6)
            for class_name, probability in zip(
                LC25000_CLASS_NAMES,
                probabilities,
            )
        }

        top_indices = torch.argsort(
            probabilities,
            descending=True,
        )[:3].tolist()

        top_predictions = [
            {
                "class_name": LC25000_CLASS_NAMES[index],
                "class_index": int(index),
                "probability": round(
                    float(probabilities[index].item()),
                    6,
                ),
            }
            for index in top_indices
        ]

        return {
            "model": DEPLOYED_MODEL_NAME,
            "predicted_class": LC25000_CLASS_NAMES[predicted_index],
            "predicted_index": predicted_index,
            "confidence": round(float(confidence.item()), 6),
            "top_predictions": top_predictions,
            "probabilities": probability_map,
            "disclaimer": RESEARCH_DISCLAIMER,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc
