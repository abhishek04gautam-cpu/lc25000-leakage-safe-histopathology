"""Classical Logistic Regression baseline for LC25000."""
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import RESULTS_DIR
from data_loader import load_datasets
from logger import log_result, save_summary
from main import get_split_file, load_split_indices
from visualization import save_confusion_matrix, save_roc_curve

def make_downsampled_grayscale_features(images, size=(32, 32)):
    features = []
    for image_array in images:
        image = Image.fromarray(image_array).convert("L").resize(size)
        features.append((np.asarray(image, dtype=np.float32) / 255.0).flatten())
    return np.asarray(features, dtype=np.float32)

def save_prediction_outputs(dataset, model_name, y_test, y_pred, y_prob, cm):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_test, y_pred, labels=list(range(dataset.num_classes)), target_names=dataset.class_names, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(RESULTS_DIR / f"{dataset.name}_{model_name}_classification_report.csv")
    confidence = np.max(y_prob, axis=1)
    df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "confidence": np.round(confidence, 6), "true_label": [dataset.class_names[int(i)] for i in y_test], "pred_label": [dataset.class_names[int(i)] for i in y_pred]})
    for class_index, class_name in enumerate(dataset.class_names):
        df[f"prob_{class_name}"] = np.round(y_prob[:, class_index], 6)
    df.to_csv(RESULTS_DIR / f"{dataset.name}_{model_name}_predictions.csv", index=False)
    wrong = df[df["y_true"] != df["y_pred"]].copy()
    if not wrong.empty:
        wrong.sort_values("confidence", ascending=False).to_csv(RESULTS_DIR / f"{dataset.name}_{model_name}_misclassifications.csv", index=False)

def main():
    start = time.perf_counter()
    dataset = [d for d in load_datasets() if d.name == "LC25000"][0]
    split_file = get_split_file(dataset.name)
    if not split_file.exists(): raise FileNotFoundError(f"Saved split not found: {split_file}. Run python src/main.py first.")
    train_idx, val_idx, test_idx = load_split_indices(split_file)
    trainval_idx = np.concatenate([train_idx, val_idx])
    X_trainval, y_trainval = dataset.X[trainval_idx], dataset.y[trainval_idx]
    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]
    print("Creating 32x32 grayscale features...")
    X_trainval_features = make_downsampled_grayscale_features(X_trainval)
    X_test_features = make_downsampled_grayscale_features(X_test)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, solver="lbfgs"))])
    model.fit(X_trainval_features, y_trainval)
    y_pred = model.predict(X_test_features); y_prob = model.predict_proba(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1 = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)[2]
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    try: roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except ValueError: roc_auc = np.nan
    cm = confusion_matrix(y_test, y_pred, labels=list(range(dataset.num_classes)))
    print(classification_report(y_test, y_pred, labels=list(range(dataset.num_classes)), target_names=dataset.class_names, zero_division=0))
    model_name = "LogisticRegression_Downsampled"
    save_confusion_matrix(cm, f"{dataset.name}_{model_name}")
    save_roc_curve(y_test, y_prob, dataset.class_names, dataset.name, model_name)
    save_prediction_outputs(dataset, model_name, y_test, y_pred, y_prob, cm)
    log_result(dataset, model_name, accuracy, precision, recall, weighted_f1, macro_f1, balanced_accuracy, roc_auc, "n/a", "downsampled_32x32_grayscale", "saved_original_split_trainval_fit", "logistic_regression", "n/a", "macro_f1")
    save_summary()
    print(f"\nTotal time taken: {(time.perf_counter() - start) / 60:.4f} minutes")

if __name__ == "__main__":
    main()
