"""Model efficiency analysis for leakage-safe LC25000 models."""
import random, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from config import LC25000_LEAKAGE_SAFE_SPLIT_FILE, MODEL_EFFICIENCY_DIR, MODELS_DIR, RANDOM_STATE
from data_loader import load_datasets
from model import SimpleCNN, TransferCNN, TransferDenseNet121, TransferEfficientNetB0
from train import ImageDataset, get_eval_transform
OUTPUT_DIR=MODEL_EFFICIENCY_DIR; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CONFIGS=[{"name":"SimpleCNN_LeakageSafe","class":SimpleCNN,"kwargs":{},"checkpoint":MODELS_DIR/"LC25000_SimpleCNN_LeakageSafe.pth"},{"name":"TransferCNN_ResNet18_LeakageSafe","class":TransferCNN,"kwargs":{"training_mode":"staged_finetune"},"checkpoint":MODELS_DIR/"LC25000_TransferCNN_ResNet18_LeakageSafe.pth"},{"name":"TransferCNN_DenseNet121_LeakageSafe","class":TransferDenseNet121,"kwargs":{"training_mode":"staged_finetune"},"checkpoint":MODELS_DIR/"LC25000_TransferCNN_DenseNet121_LeakageSafe.pth"},{"name":"TransferCNN_EfficientNetB0_LeakageSafe","class":TransferEfficientNetB0,"kwargs":{"training_mode":"staged_finetune"},"checkpoint":MODELS_DIR/"LC25000_TransferCNN_EfficientNetB0_LeakageSafe.pth"}]
def set_seed(seed=RANDOM_STATE): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def get_device(): return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
def count_parameters(model): return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
def checkpoint_size_mb(path): return np.nan if not path.exists() else path.stat().st_size/(1024*1024)
def load_lc25000_dataset(): return [d for d in load_datasets() if d.name=="LC25000"][0]
def get_eval_loader(dataset, max_samples=512, batch_size=32):
    if not LC25000_LEAKAGE_SAFE_SPLIT_FILE.exists(): raise FileNotFoundError(f"Leakage-safe split not found: {LC25000_LEAKAGE_SAFE_SPLIT_FILE}")
    idx=np.load(LC25000_LEAKAGE_SAFE_SPLIT_FILE)["test_idx"][:max_samples]; ds=ImageDataset(dataset.X[idx], dataset.y[idx], transform=get_eval_transform()); return DataLoader(ds,batch_size=batch_size,shuffle=False), len(idx)
def benchmark_inference(model, loader, device, warmup_batches=2, repeats=3):
    model.eval()
    with torch.no_grad():
        for i,(images,_) in enumerate(loader):
            _=model(images.to(device))
            if i+1>=warmup_batches: break
    times=[]; counts=[]
    for _ in range(repeats):
        start=time.perf_counter(); count=0
        with torch.no_grad():
            for images,_ in loader: _=model(images.to(device)); count+=images.shape[0]
        times.append(time.perf_counter()-start); counts.append(count)
    mean=float(np.mean(times)); samples=int(counts[0]); return mean,(mean/samples)*1000,samples/mean
def load_model_from_config(config, num_classes, device):
    model=config["class"](num_classes=num_classes, **config["kwargs"]); model.model_name=config["name"]
    if config["checkpoint"].exists(): model.load_state_dict(torch.load(config["checkpoint"], map_location=device))
    else: print(f"Warning: checkpoint not found: {config['checkpoint']}")
    return model.to(device).eval()
def main():
    set_seed(); device=get_device(); print(f"Using device: {device}"); dataset=load_lc25000_dataset(); loader,sample_count=get_eval_loader(dataset); rows=[]
    for cfg in MODEL_CONFIGS:
        print("="*80); print(cfg["name"]); model=load_model_from_config(cfg,dataset.num_classes,device); total,trainable=count_parameters(model); total_time,ms,ips=benchmark_inference(model,loader,device); size=checkpoint_size_mb(cfg["checkpoint"]); row={"model":cfg["name"],"device":str(device),"benchmark_samples":sample_count,"total_parameters":int(total),"trainable_parameters_after_load":int(trainable),"checkpoint_size_mb":round(float(size),6) if not np.isnan(size) else np.nan,"mean_inference_time_seconds":round(float(total_time),6),"mean_ms_per_image":round(float(ms),6),"images_per_second":round(float(ips),6)}; rows.append(row); print(pd.DataFrame([row]).to_string(index=False))
    pd.DataFrame(rows).to_csv(OUTPUT_DIR/"lc25000_model_efficiency_summary.csv",index=False); print(f"\nSaved model efficiency summary to: {OUTPUT_DIR/'lc25000_model_efficiency_summary.csv'}")
if __name__ == "__main__": main()
