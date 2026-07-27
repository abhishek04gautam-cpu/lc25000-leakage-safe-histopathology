"""FastAPI research prototype for LC25000 histopathology classification."""

import logging
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import torch
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision import transforms

from config import (
    API_TITLE,
    API_VERSION,
    DEPLOYED_MODEL_NAME,
    DEPLOYED_MODEL_PATH,
    IMAGE_SIZE,
    IMAGENET_NORMALIZE_MEAN,
    IMAGENET_NORMALIZE_STD,
    LC25000_CLASS_NAMES,
    RESEARCH_DISCLAIMER,
)
from model import TransferCNN

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class HealthResponse(BaseModel):
    """Response returned by the API health endpoint."""

    status: str
    device: str
    model_loaded: bool
    model: str
    classes: list[str]
    disclaimer: str


class TopPrediction(BaseModel):
    """A single ranked model prediction."""

    class_name: str
    class_index: int
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Structured prediction response."""

    model: str
    predicted_class: str
    predicted_index: int
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: list[TopPrediction]
    probabilities: dict[str, float]
    disclaimer: str


def select_device() -> torch.device:
    """Select the best available inference device."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


device = select_device()

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_NORMALIZE_MEAN,
            std=IMAGENET_NORMALIZE_STD,
        ),
    ]
)


def load_lc25000_model() -> TransferCNN:
    """Load the trained LC25000 ResNet-18 checkpoint."""

    if not DEPLOYED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint was not found at: {DEPLOYED_MODEL_PATH}"
        )

    loaded_model = TransferCNN(
        num_classes=len(LC25000_CLASS_NAMES),
        training_mode="staged_finetune",
        load_pretrained_weights=False,
    )
    loaded_model.model_name = DEPLOYED_MODEL_NAME

    checkpoint = torch.load(
        DEPLOYED_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    loaded_model.load_state_dict(checkpoint)
    loaded_model.to(device)
    loaded_model.eval()

    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model during startup and release it during shutdown."""

    logger.info(
        "Loading model '%s' on device '%s'.",
        DEPLOYED_MODEL_NAME,
        device,
    )

    app.state.model = load_lc25000_model()

    logger.info("Model loaded successfully.")

    try:
        yield
    finally:
        app.state.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model resources released.")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Research prototype for five-class lung and colon "
        "histopathology image classification. This API is not intended "
        "for clinical diagnosis or patient-care decisions."
    ),
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
def health(request: Request) -> HealthResponse:
    """Return the current API and model status."""

    loaded_model = getattr(request.app.state, "model", None)

    return HealthResponse(
        status="ok",
        device=str(device),
        model_loaded=loaded_model is not None,
        model=DEPLOYED_MODEL_NAME,
        classes=list(LC25000_CLASS_NAMES),
        disclaimer=RESEARCH_DISCLAIMER,
    )


async def read_uploaded_image(file: UploadFile) -> Image.Image:
    """Read, validate and decode an uploaded image."""

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The uploaded file must use an image media type.",
        )

    try:
        file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    except Exception as exc:
        logger.exception(
            "Failed to read uploaded file '%s'.",
            file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file could not be read.",
        ) from exc
    finally:
        await file.close()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The uploaded image exceeds the 10 MB size limit.",
        )

    try:
        with Image.open(BytesIO(file_bytes)) as candidate_image:
            candidate_image.verify()

        with Image.open(BytesIO(file_bytes)) as candidate_image:
            image = candidate_image.convert("RGB")

        return image

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        logger.warning(
            "Invalid image upload rejected: filename=%s",
            file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file could not be processed as a valid image.",
        ) from exc


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> PredictionResponse:
    """Classify an uploaded histopathology image."""

    loaded_model = getattr(request.app.state, "model", None)

    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction model is currently unavailable.",
        )

    image = await read_uploaded_image(file)

    try:
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = loaded_model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

        confidence_tensor, predicted_index_tensor = torch.max(
            probabilities,
            dim=0,
        )

        predicted_index = int(predicted_index_tensor.item())
        confidence = float(confidence_tensor.item())

        probability_values = probabilities.tolist()

        probability_map = {
            class_name: round(float(probability), 6)
            for class_name, probability in zip(
                LC25000_CLASS_NAMES,
                probability_values,
            )
        }

        number_of_top_predictions = min(
            3,
            len(LC25000_CLASS_NAMES),
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=number_of_top_predictions,
        )

        top_predictions = [
            TopPrediction(
                class_name=LC25000_CLASS_NAMES[int(class_index)],
                class_index=int(class_index),
                probability=round(float(probability), 6),
            )
            for probability, class_index in zip(
                top_probabilities.tolist(),
                top_indices.tolist(),
            )
        ]

        return PredictionResponse(
            model=DEPLOYED_MODEL_NAME,
            predicted_class=LC25000_CLASS_NAMES[predicted_index],
            predicted_index=predicted_index,
            confidence=round(confidence, 6),
            top_predictions=top_predictions,
            probabilities=probability_map,
            disclaimer=RESEARCH_DISCLAIMER,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Model inference failed for uploaded file '%s'.",
            file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be completed.",
        ) from exc
