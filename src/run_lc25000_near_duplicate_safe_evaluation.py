"""Final near-duplicate-safe LC25000 model evaluation."""
import argparse, random, time
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import RANDOM_STATE, RESULTS_DIR
from data_loader import load_datasets
from model import SimpleCNN, TransferCNN, TransferDenseNet121, TransferEfficientNetB0
from train import train_cnn
from visualization import save_confusion_matrix, save_training_curves

def set_seed(seed=RANDOM_STATE):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available(): torch.manual_seed(seed)

NEAR_DUPLICATE_SAFE_SPLIT_FILE = RESULTS_DIR / "near_duplicate_safe_split" / "LC25000_near_duplicate_safe_split.npz"
NEAR_DUPLICATE_SAFE_EVAL_DIR = RESULTS_DIR / "near_duplicate_safe_evaluation"


def load_near_duplicate_safe_split():
    if not NEAR_DUPLICATE_SAFE_SPLIT_FILE.exists():
        raise FileNotFoundError(f"Near-duplicate-safe split not found: {NEAR_DUPLICATE_SAFE_SPLIT_FILE}\nRun: python src/lc25000_near_duplicate_safe_split.py")
    s = np.load(NEAR_DUPLICATE_SAFE_SPLIT_FILE)
    return s["train_idx"], s["val_idx"], s["test_idx"]

def make_downsampled_grayscale_features(images, size=32):
    return np.asarray([(np.asarray(Image.fromarray(a).convert("L").resize((size, size)), dtype=np.float32) / 255.0).flatten() for a in images], dtype=np.float32)

def save_logistic_outputs(dataset, model_name, y_test, y_pred, y_prob):
    NEAR_DUPLICATE_SAFE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_test, y_pred, labels=list(range(dataset.num_classes)), target_names=dataset.class_names, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(NEAR_DUPLICATE_SAFE_EVAL_DIR / f"LC25000_{model_name}_classification_report.csv")
    df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "confidence": np.round(np.max(y_prob, axis=1), 6), "true_label": [dataset.class_names[int(i)] for i in y_test], "pred_label": [dataset.class_names[int(i)] for i in y_pred]})
    for class_index, class_name in enumerate(dataset.class_names): df[f"prob_{class_name}"] = np.round(y_prob[:, class_index], 6)
    df.to_csv(NEAR_DUPLICATE_SAFE_EVAL_DIR / f"LC25000_{model_name}_predictions.csv", index=False)

def run_logistic(dataset, X_train, y_train, X_val, y_val, X_test, y_test):
    model_name = "LogisticRegression_Downsampled_NearDuplicateSafe"
    X_train_full = np.concatenate([X_train, X_val], axis=0); y_train_full = np.concatenate([y_train, y_val], axis=0)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE))])
    model.fit(make_downsampled_grayscale_features(X_train_full), y_train_full)
    y_pred = model.predict(make_downsampled_grayscale_features(X_test)); y_prob = model.predict_proba(make_downsampled_grayscale_features(X_test))
    accuracy = accuracy_score(y_test, y_pred); precision, recall, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0); macro_f1 = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)[2]; balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    try: roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except ValueError: roc_auc = np.nan
    cm = confusion_matrix(y_test, y_pred, labels=list(range(dataset.num_classes)))
    save_logistic_outputs(dataset, model_name, y_test, y_pred, y_prob)
    return {"dataset": dataset.name, "split_type": "near_duplicate_safe_average_hash_grouped_split", "model": model_name, "accuracy": round(float(accuracy), 6), "precision": round(float(precision), 6), "recall": round(float(recall), 6), "weighted_f1": round(float(weighted_f1), 6), "macro_f1": round(float(macro_f1), 6), "balanced_accuracy": round(float(balanced_accuracy), 6), "roc_auc": round(float(roc_auc), 6) if not np.isnan(roc_auc) else np.nan, "epochs": "n/a", "train_samples": len(y_train_full), "validation_samples": len(y_val), "test_samples": len(y_test), "notes": "Downsampled 32x32 grayscale Logistic Regression baseline on near-duplicate-safe split"}, cm

def build_model(model_key, num_classes):
    if model_key == "simplecnn": model = SimpleCNN(num_classes); model.model_name = "SimpleCNN_NearDuplicateSafe"; return model
    if model_key == "resnet18": model = TransferCNN(num_classes, "staged_finetune"); model.model_name = "TransferCNN_ResNet18_NearDuplicateSafe"; return model
    if model_key == "headonly": model = TransferCNN(num_classes, "head_only"); model.model_name = "TransferCNN_ResNet18_HeadOnly_NearDuplicateSafe"; return model
    if model_key == "densenet121": model = TransferDenseNet121(num_classes, "staged_finetune"); model.model_name = "TransferCNN_DenseNet121_NearDuplicateSafe"; return model
    if model_key == "efficientnetb0": model = TransferEfficientNetB0(num_classes, "staged_finetune"); model.model_name = "TransferCNN_EfficientNetB0_NearDuplicateSafe"; return model
    raise ValueError(f"Unknown model key: {model_key}")

def copy_deep_model_outputs_to_near_duplicate_safe_dir(dataset, model):
    NEAR_DUPLICATE_SAFE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["predictions", "classification_report"]:
        src = RESULTS_DIR / f"{dataset.name}_{model.model_name}_{suffix}.csv"
        dst = NEAR_DUPLICATE_SAFE_EVAL_DIR / f"LC25000_{model.model_name}_{suffix}.csv"
        if src.exists(): pd.read_csv(src).to_csv(dst, index=False); print(f"Copied {suffix} to: {dst}")
        else: print(f"Warning: file not found: {src}")

def run_deep_model(model_key, dataset, X_train, y_train, X_val, y_val, X_test, y_test):
    model = build_model(model_key, dataset.num_classes)
    accuracy, precision, recall, weighted_f1, macro_f1, balanced_accuracy, roc_auc, cm, history = train_cnn(model, X_train, y_train, X_val, y_val, X_test, y_test, class_names=dataset.class_names, dataset_name=dataset.name, use_augmentation=True, use_class_weights=True, augmentation_policy="standard")
    save_confusion_matrix(cm, f"{dataset.name}_{model.model_name}"); save_training_curves(history, f"{dataset.name}_near_duplicate_safe", model.model_name); copy_deep_model_outputs_to_near_duplicate_safe_dir(dataset, model)
    return {"dataset": dataset.name, "split_type": "near_duplicate_safe_average_hash_grouped_split", "model": model.model_name, "accuracy": round(float(accuracy), 6), "precision": round(float(precision), 6), "recall": round(float(recall), 6), "weighted_f1": round(float(weighted_f1), 6), "macro_f1": round(float(macro_f1), 6), "balanced_accuracy": round(float(balanced_accuracy), 6), "roc_auc": round(float(roc_auc), 6) if not np.isnan(roc_auc) else np.nan, "epochs": history["epochs_ran"], "train_samples": len(y_train), "validation_samples": len(y_val), "test_samples": len(y_test), "notes": "Deep model evaluated on near-duplicate-safe split"}, cm

def update_all_model_summary(result_row):
    path = NEAR_DUPLICATE_SAFE_EVAL_DIR / "lc25000_near_duplicate_safe_all_model_summary.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not df.empty: df = df[df["model"] != result_row["model"]]
    df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True).sort_values(["macro_f1", "balanced_accuracy", "roc_auc", "weighted_f1"], ascending=[False, False, False, False])
    df.to_csv(path, index=False); print(f"Updated combined near-duplicate-safe summary: {path}")

def parse_args():
    p = argparse.ArgumentParser(description="Run near-duplicate-safe LC25000 model evaluation."); p.add_argument("--model", required=True, choices=["logistic", "simplecnn", "resnet18", "headonly", "densenet121", "efficientnetb0"]); return p.parse_args()

def main():
    args = parse_args(); start = time.perf_counter(); set_seed(); NEAR_DUPLICATE_SAFE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    dataset = [d for d in load_datasets() if d.name == "LC25000"][0]
    train_idx, val_idx, test_idx = load_near_duplicate_safe_split()
    X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]; X_val, y_val = dataset.X[val_idx], dataset.y[val_idx]; X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]
    result_row, cm = run_logistic(dataset, X_train, y_train, X_val, y_val, X_test, y_test) if args.model == "logistic" else run_deep_model(args.model, dataset, X_train, y_train, X_val, y_val, X_test, y_test)
    result_path = NEAR_DUPLICATE_SAFE_EVAL_DIR / f"lc25000_{args.model}_near_duplicate_safe_metrics.csv"; pd.DataFrame([result_row]).to_csv(result_path, index=False); update_all_model_summary(result_row)
    print("\nNear-duplicate-safe result:"); [print(f"{k}: {v}") for k, v in result_row.items()]; print("Confusion Matrix:\n", cm); print(f"\nSaved metrics to: {result_path}"); print(f"Total time: {(time.perf_counter() - start) / 60:.4f} minutes")

if __name__ == "__main__":
    main()
