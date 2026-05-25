"""Main LC25000 experiment runner."""
import random
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from config import ACTIVE_DATASETS, LC25000_SPLIT_FILE, RANDOM_STATE, SPLIT_DIR
from data_loader import load_datasets
from logger import log_result, reset_results, save_summary
from model import get_models
from train import train_cnn
from visualization import save_confusion_matrix, save_training_curves

def set_seed(seed=RANDOM_STATE):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available(): torch.manual_seed(seed)

def get_split_file(dataset_name):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    return LC25000_SPLIT_FILE if dataset_name == "LC25000" else SPLIT_DIR / f"{dataset_name}_split.npz"

def save_split_indices(split_file, train_idx, val_idx, test_idx):
    split_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(split_file, train_idx=np.asarray(train_idx), val_idx=np.asarray(val_idx), test_idx=np.asarray(test_idx))

def load_split_indices(split_file):
    split_data = np.load(split_file)
    return split_data["train_idx"], split_data["val_idx"], split_data["test_idx"]

def create_or_load_split(dataset):
    split_file = get_split_file(dataset.name)
    if split_file.exists():
        print(f"Loaded saved split: {split_file}")
        return load_split_indices(split_file)
    all_indices = np.arange(len(dataset.X))
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.30, random_state=RANDOM_STATE, stratify=dataset.y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_STATE, stratify=dataset.y[temp_idx])
    save_split_indices(split_file, train_idx, val_idx, test_idx)
    print(f"Saved new split: {split_file}")
    return train_idx, val_idx, test_idx

def print_split_summary(dataset, train_idx, val_idx, test_idx):
    print("\nSplit summary")
    print("Train samples:", len(train_idx)); print("Validation samples:", len(val_idx)); print("Test samples:", len(test_idx))
    for split_name, indices in [("Train", train_idx), ("Validation", val_idx), ("Test", test_idx)]:
        unique, counts = np.unique(dataset.y[indices], return_counts=True)
        print(f"\n{split_name} class distribution:")
        for class_index, count in zip(unique, counts):
            class_name = dataset.class_names[int(class_index)] if dataset.class_names else str(class_index)
            print(f"  {int(class_index)} - {class_name}: {int(count)}")

def run_dataset_experiment(dataset):
    print("=" * 80); print(f"Dataset: {dataset.name}"); print("=" * 80)
    set_seed(RANDOM_STATE)
    train_idx, val_idx, test_idx = create_or_load_split(dataset)
    print_split_summary(dataset, train_idx, val_idx, test_idx)
    X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
    X_val, y_val = dataset.X[val_idx], dataset.y[val_idx]
    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]
    for model_label, model in get_models(dataset).items():
        print("\n" + "-" * 80); print(f"Model: {model_label}"); print("-" * 80)
        accuracy, precision, recall, weighted_f1, macro_f1, balanced_accuracy, roc_auc, cm, history = train_cnn(model, X_train, y_train, X_val, y_val, X_test, y_test, class_names=dataset.class_names, dataset_name=dataset.name, use_augmentation=True, use_class_weights=True, augmentation_policy="standard")
        print("\nTest metrics")
        print("Accuracy:", round(accuracy, 6)); print("Precision:", round(precision, 6)); print("Recall:", round(recall, 6)); print("Weighted F1:", round(weighted_f1, 6)); print("Macro F1:", round(macro_f1, 6)); print("Balanced accuracy:", round(balanced_accuracy, 6)); print("ROC-AUC:", round(roc_auc, 6) if not np.isnan(roc_auc) else "nan")
        print("Confusion Matrix:\n", cm)
        save_confusion_matrix(cm, f"{dataset.name}_{model.model_name}")
        save_training_curves(history, f"{dataset.name}_main", model.model_name)
        backbone = "CustomCNN" if model.model_name == "SimpleCNN" else "ResNet18" if model.model_name == "TransferCNN_ResNet18" else "PretrainedCNN"
        log_result(dataset, model.model_name, accuracy, precision, recall, weighted_f1, macro_f1, balanced_accuracy, roc_auc, backbone, "horizontal_flip+rotation10", "stratified_saved_split", "class_weighted_cross_entropy", history["epochs_ran"], "macro_f1")

def main():
    start_time = time.perf_counter(); set_seed(RANDOM_STATE)
    datasets = [d for d in load_datasets() if d.name in ACTIVE_DATASETS]
    if not datasets: raise ValueError("No active datasets were loaded.")
    reset_results(); print("Training main LC25000 models...\n")
    for dataset in datasets: run_dataset_experiment(dataset)
    save_summary()
    print(f"\nTotal time taken: {(time.perf_counter() - start_time) / 60:.4f} minutes")

if __name__ == "__main__":
    main()
