"""Grad-CAM visual explanation generation for LC25000 ResNet18."""
import cv2, numpy as np, pandas as pd, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import DISSERTATION_FIGURES_DIR, LC25000_CLASS_NAMES, LC25000_LEAKAGE_SAFE_SPLIT_FILE, MODELS_DIR
from data_loader import load_datasets
from model import TransferCNN
from train import get_eval_transform
MODEL_NAME="TransferCNN_ResNet18_LeakageSafe"; MODEL_PATH=MODELS_DIR / f"LC25000_{MODEL_NAME}.pth"; PREDICTIONS_PATH=DISSERTATION_FIGURES_DIR.parent / "leakage_safe_evaluation" / f"LC25000_{MODEL_NAME}_predictions.csv"; OUTPUT_DIR=DISSERTATION_FIGURES_DIR / "gradcam_lc25000"; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model=model; self.target_layer=target_layer; self.gradients=None; self.activations=None
        self.forward_hook=target_layer.register_forward_hook(self.save_activation); self.backward_hook=target_layer.register_full_backward_hook(self.save_gradient)
    def save_activation(self, _module, _input_tensor, output): self.activations=output.detach()
    def save_gradient(self, _module, _grad_input, grad_output): self.gradients=grad_output[0].detach()
    def generate(self, input_tensor, class_index):
        self.model.zero_grad(); output=self.model(input_tensor); output[:,class_index].backward()
        if self.gradients is None or self.activations is None: raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")
        weights=self.gradients.mean(dim=(2,3), keepdim=True); cam=F.relu((weights*self.activations).sum(dim=1,keepdim=True)).squeeze().cpu().numpy(); cam=cv2.resize(cam,(224,224)); cam=cam-cam.min(); return cam/cam.max() if cam.max()>0 else cam
    def close(self): self.forward_hook.remove(); self.backward_hook.remove()
def get_device(): return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
def load_leakage_safe_test_data():
    if not LC25000_LEAKAGE_SAFE_SPLIT_FILE.exists(): raise FileNotFoundError(f"Leakage-safe split not found: {LC25000_LEAKAGE_SAFE_SPLIT_FILE}")
    dataset=[d for d in load_datasets() if d.name=="LC25000"][0]; test_idx=np.load(LC25000_LEAKAGE_SAFE_SPLIT_FILE)["test_idx"]; return dataset.X[test_idx], dataset.y[test_idx]
def load_model(device):
    if not MODEL_PATH.exists(): raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    model=TransferCNN(num_classes=len(LC25000_CLASS_NAMES), training_mode="staged_finetune"); model.model_name=MODEL_NAME; model.load_state_dict(torch.load(MODEL_PATH,map_location=device)); return model.to(device).eval()
def preprocess_image(image_array): return get_eval_transform()(image_array).unsqueeze(0)
def make_overlay(original_image, cam):
    original=cv2.resize(original_image,(224,224)); heatmap=np.uint8(255*cam); heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET); heatmap=cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB); overlay=cv2.addWeighted(original,0.55,heatmap,0.45,0); return original,heatmap,overlay
def save_gradcam_figure(original, heatmap, overlay, save_path, title):
    plt.figure(figsize=(12,4));
    for i,(img,t) in enumerate([(original,"Original"),(heatmap,"Grad-CAM heatmap"),(overlay,"Overlay")],1): plt.subplot(1,3,i); plt.imshow(img); plt.title(t); plt.axis("off")
    plt.suptitle(title,fontsize=10); plt.tight_layout(); plt.savefig(save_path,dpi=300); plt.close()
def select_cases(predictions_df):
    """
    Select publication-quality Grad-CAM examples.

    Strategy:
    - One representative correct example per class
      using median confidence (avoids cherry-picking).
    - Misclassified examples with moderate confidence
      and clinically meaningful confusion patterns.
    """

    correct = predictions_df[
        predictions_df["y_true"] == predictions_df["y_pred"]
    ].copy()

    incorrect = predictions_df[
        predictions_df["y_true"] != predictions_df["y_pred"]
    ].copy()

    selected = []

    # ------------------------------------------------------------------
    # Representative correct examples
    # ------------------------------------------------------------------
    for ci in range(len(LC25000_CLASS_NAMES)):

        c = correct[correct["y_true"] == ci].copy()

        if c.empty:
            continue

        # choose median-confidence example
        c = c.sort_values("confidence").reset_index()

        median_idx = len(c) // 2

        row = c.iloc[median_idx]

        selected.append(
            ("correct", int(row["index"]), row)
        )

    # ------------------------------------------------------------------
    # Representative misclassifications
    # ------------------------------------------------------------------

    # Avoid extreme low/high confidence failures
    incorrect = incorrect[
        (incorrect["confidence"] >= 0.40) &
        (incorrect["confidence"] <= 0.85)
    ].copy()

    # prioritize clinically meaningful confusion
    preferred_pairs = [
        (4, 2),  # lung SCC -> lung adenocarcinoma
        (2, 4),  # reverse
        (3, 1),  # colon adenocarcinoma -> benign colon
        (1, 3),  # reverse
    ]

    used = set()

    for true_cls, pred_cls in preferred_pairs:

        pair_df = incorrect[
            (incorrect["y_true"] == true_cls) &
            (incorrect["y_pred"] == pred_cls)
        ]

        if pair_df.empty:
            continue

        row = pair_df.iloc[
            (pair_df["confidence"] - 0.65).abs().argsort().iloc[0]
        ]

        key = int(row.name)

        if key not in used:
            selected.append(("misclassified", key, row))
            used.add(key)

    # fallback if too few examples
    if len([x for x in selected if x[0] == "misclassified"]) < 4:

        remaining = incorrect.sort_values(
            by="confidence",
            ascending=False
        )

        for _, row in remaining.iterrows():

            key = int(row.name)

            if key in used:
                continue

            selected.append(("misclassified", key, row))
            used.add(key)

            if len([x for x in selected if x[0] == "misclassified"]) >= 4:
                break

    return selected
def main():
    device=get_device(); X_test,_=load_leakage_safe_test_data()
    if not PREDICTIONS_PATH.exists(): raise FileNotFoundError(f"Prediction file not found: {PREDICTIONS_PATH}")
    preds=pd.read_csv(PREDICTIONS_PATH)
    if len(preds)!=len(X_test): raise ValueError(f"Prediction rows ({len(preds)}) do not match test samples ({len(X_test)}).")
    model=load_model(device); gradcam=GradCAM(model, model.model.layer4[-1]); rows=[]
    for case_type,test_position,row in select_cases(preds):
        image=X_test[test_position]; true_i=int(row["y_true"]); pred_i=int(row["y_pred"]); conf=float(row["confidence"]); cam=gradcam.generate(preprocess_image(image).to(device), pred_i); original,heatmap,overlay=make_overlay(image,cam); filename=f"{case_type}_testpos_{test_position}_true_{true_i}_pred_{pred_i}.png"; path=OUTPUT_DIR / filename; title=f"{case_type.upper()} | True: {LC25000_CLASS_NAMES[true_i]} | Predicted: {LC25000_CLASS_NAMES[pred_i]} | Confidence: {conf:.3f}"; save_gradcam_figure(original,heatmap,overlay,path,title); rows.append({"case_type":case_type,"test_position":test_position,"true_index":true_i,"true_label":LC25000_CLASS_NAMES[true_i],"predicted_index":pred_i,"predicted_label":LC25000_CLASS_NAMES[pred_i],"confidence":round(conf,6),"gradcam_file":str(path)}); print(f"Saved: {path}")
    gradcam.close(); pd.DataFrame(rows).to_csv(OUTPUT_DIR / "gradcam_lc25000_summary.csv", index=False)
if __name__ == "__main__": main()
