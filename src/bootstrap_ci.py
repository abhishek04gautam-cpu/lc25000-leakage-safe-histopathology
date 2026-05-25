"""Bootstrap confidence intervals for LC25000 model metrics."""
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score
from config import BOOTSTRAP_CI_DIR, LEAKAGE_SAFE_EVAL_DIR, N_BOOTSTRAPS, RANDOM_STATE
OUTPUT_DIR = BOOTSTRAP_CI_DIR; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_FILES = {m: LEAKAGE_SAFE_EVAL_DIR / f"{m}_predictions.csv" for m in ["LC25000_TransferCNN_ResNet18_LeakageSafe", "LC25000_SimpleCNN_LeakageSafe", "LC25000_TransferCNN_DenseNet121_LeakageSafe", "LC25000_TransferCNN_EfficientNetB0_LeakageSafe"]}

def calculate_metrics(y_true, y_pred, y_prob):
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0); macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {"accuracy": accuracy_score(y_true, y_pred), "weighted_precision": weighted[0], "weighted_recall": weighted[1], "weighted_f1": weighted[2], "macro_f1": macro[2], "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}
    try: metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:,1]) if y_prob.shape[1] == 2 else roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError: metrics["roc_auc"] = np.nan
    return metrics

def get_probability_columns(df):
    cols = [c for c in df.columns if c.startswith("prob_")]
    if not cols: raise ValueError("No probability columns found in prediction file.")
    return cols

def bootstrap_dataset(model_name, prediction_path):
    if not prediction_path.exists(): print(f"Skipping missing prediction file: {prediction_path}"); return None
    df = pd.read_csv(prediction_path)
    missing = {"y_true", "y_pred"} - set(df.columns)
    if missing: raise ValueError(f"{prediction_path} is missing required columns: {missing}")
    y_true = df["y_true"].to_numpy().astype(int); y_pred = df["y_pred"].to_numpy().astype(int); y_prob = df[get_probability_columns(df)].to_numpy(float)
    rng = np.random.default_rng(RANDOM_STATE); point = calculate_metrics(y_true, y_pred, y_prob); rows=[]; idx=np.arange(len(df))
    for _ in range(N_BOOTSTRAPS):
        b = rng.choice(idx, size=len(df), replace=True); rows.append(calculate_metrics(y_true[b], y_pred[b], y_prob[b]))
    boot = pd.DataFrame(rows); summary=[]
    for metric, point_value in point.items():
        vals = boot[metric].dropna(); lo, hi = (np.nan, np.nan) if vals.empty else np.percentile(vals, [2.5, 97.5])
        summary.append({"model": model_name, "metric": metric, "point_estimate": round(float(point_value),6) if not np.isnan(point_value) else np.nan, "ci_lower_95": round(float(lo),6) if not np.isnan(lo) else np.nan, "ci_upper_95": round(float(hi),6) if not np.isnan(hi) else np.nan, "n_bootstraps": N_BOOTSTRAPS, "test_samples": len(df)})
    boot.to_csv(OUTPUT_DIR / f"{model_name}_bootstrap_samples.csv", index=False); summary_df = pd.DataFrame(summary); summary_df.to_csv(OUTPUT_DIR / f"{model_name}_bootstrap_ci_summary.csv", index=False); print(summary_df); return summary_df

def main():
    summaries=[s for m,p in PREDICTION_FILES.items() if (s:=bootstrap_dataset(m,p)) is not None]
    if not summaries: raise FileNotFoundError("No prediction files were available for bootstrap analysis.")
    pd.concat(summaries, ignore_index=True).to_csv(OUTPUT_DIR / "all_leakage_safe_bootstrap_ci_summary.csv", index=False); print("Done.")
if __name__ == "__main__": main()
