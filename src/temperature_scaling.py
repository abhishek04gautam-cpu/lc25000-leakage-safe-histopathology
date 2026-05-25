"""Post-hoc temperature scaling for LC25000 ResNet18 calibration."""
import random
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from config import CALIBRATION_ANALYSIS_DIR, CALIBRATION_BINS, LC25000_BATCH_SIZE, LC25000_LEAKAGE_SAFE_SPLIT_FILE, LC25000_CLASS_NAMES, MODELS_DIR, RANDOM_STATE
from data_loader import load_datasets
from model import TransferCNN
from train import ImageDataset, get_eval_transform
OUTPUT_DIR = CALIBRATION_ANALYSIS_DIR / "temperature_scaling"; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "TransferCNN_ResNet18_LeakageSafe"; MODEL_PATH = MODELS_DIR / f"LC25000_{MODEL_NAME}.pth"

def set_seed(seed=RANDOM_STATE):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available(): torch.manual_seed(seed)

def get_device(): return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
def load_leakage_safe_split():
    if not LC25000_LEAKAGE_SAFE_SPLIT_FILE.exists(): raise FileNotFoundError(f"Leakage-safe split not found: {LC25000_LEAKAGE_SAFE_SPLIT_FILE}")
    s=np.load(LC25000_LEAKAGE_SAFE_SPLIT_FILE); return s["train_idx"],s["val_idx"],s["test_idx"]
def make_loader(X,y): return DataLoader(ImageDataset(X,y,transform=get_eval_transform()), batch_size=LC25000_BATCH_SIZE, shuffle=False)

def collect_logits_and_labels(model, loader, device):
    model.eval(); logits=[]; labels=[]
    with torch.no_grad():
        for images,y in loader:
            images=images.to(device); y=y.to(device); logits.append(model(images).cpu()); labels.append(y.cpu())
    return torch.cat(logits,0), torch.cat(labels,0)
class TemperatureScaler(nn.Module):
    def __init__(self): super().__init__(); self.log_temperature=nn.Parameter(torch.zeros(1))
    def temperature(self): return torch.exp(self.log_temperature)
    def forward(self, logits): return logits / self.temperature()

def fit_temperature(validation_logits, validation_labels, max_iter=300):
    scaler=TemperatureScaler(); criterion=nn.CrossEntropyLoss(); validation_logits=validation_logits.detach(); validation_labels=validation_labels.detach(); opt=torch.optim.LBFGS([scaler.log_temperature], lr=0.01, max_iter=max_iter, line_search_fn="strong_wolfe")
    def closure(): opt.zero_grad(); loss=criterion(scaler(validation_logits), validation_labels); loss.backward(); return loss
    opt.step(closure); return float(scaler.temperature().item())
def softmax_with_temperature(logits, temperature): return F.softmax(logits / temperature, dim=1).numpy()
def multiclass_brier_score(y_true, probabilities): return np.mean(np.sum((probabilities - np.eye(probabilities.shape[1])[y_true]) ** 2, axis=1))

def expected_calibration_error(y_true, probabilities, n_bins=CALIBRATION_BINS):
    conf=np.max(probabilities,axis=1); pred=np.argmax(probabilities,axis=1); correct=pred==y_true; edges=np.linspace(0,1,n_bins+1); ece=0.0; mce=0.0; rows=[]
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]; mask=((conf>=lo)&(conf<=hi)) if i==n_bins-1 else ((conf>=lo)&(conf<hi)); count=int(mask.sum())
        if count==0: rows.append({"bin":i+1,"lower":round(float(lo),2),"upper":round(float(hi),2),"count":0,"accuracy":np.nan,"confidence":np.nan,"gap":np.nan}); continue
        a=float(correct[mask].mean()); c=float(conf[mask].mean()); gap=abs(a-c); ece+=(count/len(y_true))*gap; mce=max(mce,gap); rows.append({"bin":i+1,"lower":round(float(lo),2),"upper":round(float(hi),2),"count":count,"accuracy":round(a,6),"confidence":round(c,6),"gap":round(gap,6)})
    return ece,mce,pd.DataFrame(rows)

def compute_metrics(y_true, probabilities):
    y_pred=np.argmax(probabilities,axis=1); precision,recall,weighted_f1,_=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0); macro_f1=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)[2]
    try: roc_auc=roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
    except ValueError: roc_auc=np.nan
    try: nll=log_loss(y_true, probabilities, labels=list(range(probabilities.shape[1])))
    except ValueError: nll=np.nan
    ece,mce,bins=expected_calibration_error(y_true, probabilities); return {"accuracy":accuracy_score(y_true,y_pred),"precision":precision,"recall":recall,"weighted_f1":weighted_f1,"macro_f1":macro_f1,"balanced_accuracy":balanced_accuracy_score(y_true,y_pred),"roc_auc":roc_auc,"nll":nll,"ece":ece,"mce":mce,"brier":multiclass_brier_score(y_true, probabilities)}, bins

def save_reliability_plot(before_bins, after_bins):
    b=before_bins.dropna(subset=["accuracy","confidence"]); a=after_bins.dropna(subset=["accuracy","confidence"]); plt.figure(figsize=(7,6)); plt.plot(b["confidence"],b["accuracy"],marker="o",label="Before temperature scaling"); plt.plot(a["confidence"],a["accuracy"],marker="o",label="After temperature scaling"); plt.plot([0,1],[0,1],linestyle="--",label="Perfect calibration"); plt.xlabel("Mean confidence"); plt.ylabel("Observed accuracy"); plt.title("LC25000 ResNet18 Reliability Before and After Temperature Scaling"); plt.xlim(0,1); plt.ylim(0,1); plt.legend(); plt.tight_layout(); out=OUTPUT_DIR / "LC25000_temperature_scaling_reliability.png"; plt.savefig(out,dpi=300); plt.close(); print(f"Reliability plot saved to: {out}")

def save_prediction_file(y_true, probs_before, probs_after):
    ypb=np.argmax(probs_before,axis=1); ypa=np.argmax(probs_after,axis=1); df=pd.DataFrame({"y_true":y_true,"y_pred_before":ypb,"y_pred_after":ypa,"confidence_before":np.round(np.max(probs_before,axis=1),6),"confidence_after":np.round(np.max(probs_after,axis=1),6),"true_label":[LC25000_CLASS_NAMES[int(i)] for i in y_true],"pred_label_before":[LC25000_CLASS_NAMES[int(i)] for i in ypb],"pred_label_after":[LC25000_CLASS_NAMES[int(i)] for i in ypa]})
    for i,name in enumerate(LC25000_CLASS_NAMES): df[f"prob_before_{name}"]=np.round(probs_before[:,i],6); df[f"prob_after_{name}"]=np.round(probs_after[:,i],6)
    out=OUTPUT_DIR / "LC25000_temperature_scaled_predictions.csv"; df.to_csv(out,index=False); print(f"Temperature-scaled predictions saved to: {out}")

def load_lc25000_dataset_and_split():
    dataset=[d for d in load_datasets() if d.name=="LC25000"][0]; _train,val,test=load_leakage_safe_split(); return dataset, dataset.X[val], dataset.y[val], dataset.X[test], dataset.y[test]
def load_resnet18_model(device, num_classes):
    if not MODEL_PATH.exists(): raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    model=TransferCNN(num_classes=num_classes, training_mode="staged_finetune"); model.model_name=MODEL_NAME; model.load_state_dict(torch.load(MODEL_PATH,map_location=device)); return model.to(device).eval()
def run_temperature_scaling():
    set_seed(); device=get_device(); dataset,X_val,y_val,X_test,y_test=load_lc25000_dataset_and_split(); model=load_resnet18_model(device,dataset.num_classes); val_logits,val_labels=collect_logits_and_labels(model,make_loader(X_val,y_val),device); test_logits,test_labels=collect_logits_and_labels(model,make_loader(X_test,y_test),device); temperature=fit_temperature(val_logits,val_labels); print(f"Learned temperature: {temperature:.6f}"); probs_before=softmax_with_temperature(test_logits,1.0); probs_after=softmax_with_temperature(test_logits,temperature); y=test_labels.numpy().astype(int); before,b_bins=compute_metrics(y,probs_before); after,a_bins=compute_metrics(y,probs_after); b_bins.to_csv(OUTPUT_DIR / "LC25000_calibration_bins_before_temperature_scaling.csv",index=False); a_bins.to_csv(OUTPUT_DIR / "LC25000_calibration_bins_after_temperature_scaling.csv",index=False); save_reliability_plot(b_bins,a_bins); save_prediction_file(y,probs_before,probs_after); summary={"dataset":"LC25000","model":MODEL_NAME,"temperature":round(float(temperature),6)}; summary.update({f"{k}_before":round(float(v),6) if not np.isnan(v) else np.nan for k,v in before.items()}); summary.update({f"{k}_after":round(float(v),6) if not np.isnan(v) else np.nan for k,v in after.items()}); pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "temperature_scaling_summary.csv",index=False); print(pd.DataFrame([summary]).to_string(index=False))
if __name__ == "__main__": run_temperature_scaling()
