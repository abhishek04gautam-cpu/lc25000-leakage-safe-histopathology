"""Plotting utilities for LC25000 experiments."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from config import CONFUSION_MATRIX_DIR, RESULTS_DIR

def save_confusion_matrix(cm, dataset_name):
    CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    threshold = cm.max() / 2.0 if cm.size > 0 else 0
    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            value = int(cm[row_index, col_index])
            plt.text(col_index, row_index, value, ha="center", va="center", color="white" if value > threshold else "black")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    file_path = CONFUSION_MATRIX_DIR / f"{dataset_name}_cm.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {file_path}")

def save_training_curves(history, dataset_name, model_name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{dataset_name} - {model_name} Loss Curve")
    plt.legend(); plt.tight_layout()
    loss_path = RESULTS_DIR / f"{dataset_name}_{model_name}_loss_curve.png"
    plt.savefig(loss_path, dpi=300); plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(history["val_f1"], label="Validation macro-F1")
    plt.xlabel("Epoch"); plt.ylabel("Macro-F1")
    plt.title(f"{dataset_name} - {model_name} Validation Macro-F1 Curve")
    plt.legend(); plt.tight_layout()
    f1_path = RESULTS_DIR / f"{dataset_name}_{model_name}_f1_curve.png"
    plt.savefig(f1_path, dpi=300); plt.close()
    print(f"Training loss curve saved to: {loss_path}")
    print(f"Validation F1 curve saved to: {f1_path}")

def save_summary_plot(summary_df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df.copy().sort_values("dataset")
    plt.figure(figsize=(10, 6))
    x = np.arange(len(plot_df)); width = 0.25
    plt.bar(x - width, plot_df["f1"], width, label="Weighted F1")
    plt.bar(x, plot_df["macro_f1"], width, label="Macro F1")
    plt.bar(x + width, plot_df["balanced_accuracy"], width, label="Balanced accuracy")
    plt.xticks(x, plot_df["dataset"], rotation=20, ha="right")
    plt.ylabel("Score"); plt.ylim(0, 1.05); plt.title("Best Model Performance by Dataset")
    plt.legend(); plt.tight_layout()
    file_path = RESULTS_DIR / "best_models_comparison.png"
    plt.savefig(file_path, dpi=300); plt.close()
    print(f"Summary plot saved to: {file_path}")

def save_roc_curve(y_true, y_prob, class_names, dataset_name, model_name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    num_classes = y_prob.shape[1]
    plt.figure(figsize=(8, 6))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    else:
        y_binary = label_binarize(y_true, classes=np.arange(num_classes))
        for class_index in range(num_classes):
            fpr, tpr, _ = roc_curve(y_binary[:, class_index], y_prob[:, class_index])
            roc_auc = auc(fpr, tpr)
            label = class_names[class_index] if class_names is not None else f"class_{class_index}"
            plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"{dataset_name} - {model_name} ROC Curve")
    plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
    file_path = RESULTS_DIR / f"{dataset_name}_{model_name}_roc_curve.png"
    plt.savefig(file_path, dpi=300); plt.close()
    print(f"ROC curve saved to: {file_path}")
