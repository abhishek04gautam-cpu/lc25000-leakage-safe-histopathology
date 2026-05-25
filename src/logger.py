"""Experiment logging utilities."""
import csv
import math
import pandas as pd
from config import RESULTS_DIR
from visualization import save_summary_plot

RESULT_FILE = RESULTS_DIR / "experiments.csv"

def reset_results():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RESULT_FILE.is_file():
        RESULT_FILE.unlink()

def safe_round(value, digits=6):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(value) else round(value, digits)

def log_result(dataset, model_name, accuracy, precision, recall, f1, macro_f1,
               balanced_accuracy, roc_auc, backbone, augmentation, sampling,
               loss_name, epochs, selection_metric):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = RESULT_FILE.is_file()
    with open(RESULT_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["dataset", "type", "model", "accuracy", "precision", "recall", "f1", "macro_f1", "balanced_accuracy", "roc_auc", "backbone", "augmentation", "sampling", "loss", "epochs", "selection_metric"])
        writer.writerow([dataset.name, dataset.data_type, model_name, safe_round(accuracy), safe_round(precision), safe_round(recall), safe_round(f1), safe_round(macro_f1), safe_round(balanced_accuracy), safe_round(roc_auc), backbone, augmentation, sampling, loss_name, epochs, selection_metric])

def save_summary():
    if not RESULT_FILE.is_file():
        print("No results file found for summary.")
        return
    df = pd.read_csv(RESULT_FILE)
    metric_columns = ["accuracy", "precision", "recall", "f1", "macro_f1", "balanced_accuracy", "roc_auc"]
    for column in metric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    ranked_df = df.sort_values(["dataset", "macro_f1", "balanced_accuracy", "roc_auc", "f1"], ascending=[True, False, False, False, False])
    summary = ranked_df.groupby("dataset", as_index=False).first()
    summary_path = RESULTS_DIR / "best_models_summary.csv"
    rankings_path = RESULTS_DIR / "all_models_ranked.csv"
    summary.to_csv(summary_path, index=False)
    ranked_df.to_csv(rankings_path, index=False)
    print(f"Summary saved to: {summary_path}")
    print(f"Ranked results saved to: {rankings_path}")
    print(summary)
    save_summary_plot(summary)
