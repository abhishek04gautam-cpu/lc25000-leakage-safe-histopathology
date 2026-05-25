"""Paired statistical comparison of leakage-safe LC25000 model predictions."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from config import LEAKAGE_SAFE_EVAL_DIR, N_PAIRED_BOOTSTRAPS, RANDOM_STATE, STATISTICAL_TESTS_LEAKAGE_SAFE_DIR
try:
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
OUTPUT_DIR = STATISTICAL_TESTS_LEAKAGE_SAFE_DIR; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def pred_file(model): return LEAKAGE_SAFE_EVAL_DIR / f"LC25000_{model}_predictions.csv"
COMPARISONS = [
    ("ResNet18 vs Logistic Regression", "TransferCNN_ResNet18_LeakageSafe", "LogisticRegression_Downsampled_LeakageSafe"),
    ("SimpleCNN vs Logistic Regression", "SimpleCNN_LeakageSafe", "LogisticRegression_Downsampled_LeakageSafe"),
    ("DenseNet121 vs Logistic Regression", "TransferCNN_DenseNet121_LeakageSafe", "LogisticRegression_Downsampled_LeakageSafe"),
    ("EfficientNet-B0 vs Logistic Regression", "TransferCNN_EfficientNetB0_LeakageSafe", "LogisticRegression_Downsampled_LeakageSafe"),
    ("ResNet18 vs SimpleCNN", "TransferCNN_ResNet18_LeakageSafe", "SimpleCNN_LeakageSafe"),
    ("DenseNet121 vs SimpleCNN", "TransferCNN_DenseNet121_LeakageSafe", "SimpleCNN_LeakageSafe"),
    ("EfficientNet-B0 vs SimpleCNN", "TransferCNN_EfficientNetB0_LeakageSafe", "SimpleCNN_LeakageSafe"),
    ("DenseNet121 vs ResNet18", "TransferCNN_DenseNet121_LeakageSafe", "TransferCNN_ResNet18_LeakageSafe"),
    ("EfficientNet-B0 vs ResNet18", "TransferCNN_EfficientNetB0_LeakageSafe", "TransferCNN_ResNet18_LeakageSafe"),
    ("DenseNet121 vs EfficientNet-B0", "TransferCNN_DenseNet121_LeakageSafe", "TransferCNN_EfficientNetB0_LeakageSafe"),
    ("ResNet18 vs ResNet18 Head Only", "TransferCNN_ResNet18_LeakageSafe", "TransferCNN_ResNet18_HeadOnly_LeakageSafe"),
]
METRICS = {"accuracy": accuracy_score, "weighted_f1": lambda y,p: f1_score(y,p,average="weighted",zero_division=0), "macro_f1": lambda y,p: f1_score(y,p,average="macro",zero_division=0), "balanced_accuracy": balanced_accuracy_score}

def read_predictions(path):
    if not path.exists(): raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    for c in ["y_true", "y_pred"]:
        if c not in df.columns: raise ValueError(f"{path} is missing required column: {c}")
    return df

def align_prediction_files(df_a, df_b, path_a, path_b):
    if len(df_a) != len(df_b): raise ValueError(f"Prediction length mismatch: {path_a} vs {path_b}")
    y_true_a = df_a["y_true"].to_numpy(); y_true_b = df_b["y_true"].to_numpy()
    if not np.array_equal(y_true_a, y_true_b): raise ValueError(f"y_true mismatch between files: {path_a} and {path_b}")
    return y_true_a, df_a["y_pred"].to_numpy(), df_b["y_pred"].to_numpy()

def run_mcnemar(y_true, pred_a, pred_b):
    ca, cb = pred_a == y_true, pred_b == y_true
    table = [[int(np.sum(ca & cb)), int(np.sum(ca & ~cb))], [int(np.sum(~ca & cb)), int(np.sum(~ca & ~cb))]]
    if STATSMODELS_AVAILABLE:
        result = mcnemar(table, exact=True); statistic, p_value, method = result.statistic, result.pvalue, "McNemar exact"
    else:
        statistic, p_value, method = np.nan, np.nan, "statsmodels not installed"
    return {"both_correct": table[0][0], "a_correct_b_wrong": table[0][1], "a_wrong_b_correct": table[1][0], "both_wrong": table[1][1], "mcnemar_statistic": statistic, "p_value": p_value, "method": method}

def paired_bootstrap_metric_difference(y_true, pred_a, pred_b, metric_fn, n_bootstrap=N_PAIRED_BOOTSTRAPS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed); n = len(y_true); idx = np.arange(n)
    obs_a, obs_b = metric_fn(y_true, pred_a), metric_fn(y_true, pred_b); obs_diff = obs_a - obs_b
    diffs = []
    for _ in range(n_bootstrap):
        b = rng.choice(idx, size=n, replace=True); diffs.append(metric_fn(y_true[b], pred_a[b]) - metric_fn(y_true[b], pred_b[b]))
    diffs = np.asarray(diffs); lower, upper = np.percentile(diffs, [2.5, 97.5]); p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"metric_a": obs_a, "metric_b": obs_b, "difference_a_minus_b": obs_diff, "ci_lower": lower, "ci_upper": upper, "bootstrap_p_approx": min(float(p), 1.0), "n_bootstrap": n_bootstrap}

def interpret_ci(lo, hi):
    return "A higher; CI excludes zero" if lo > 0 else "B higher; CI excludes zero" if hi < 0 else "Difference uncertain; CI includes zero"

def main():
    mcnemar_rows, bootstrap_rows = [], []
    for comparison, model_a, model_b in COMPARISONS:
        print("=" * 80); print(f"LC25000 leakage-safe comparison: {comparison}"); print("=" * 80)
        path_a, path_b = pred_file(model_a), pred_file(model_b)
        y_true, pred_a, pred_b = align_prediction_files(read_predictions(path_a), read_predictions(path_b), path_a, path_b)
        mrow = {"dataset": "LC25000", "comparison": comparison, "model_a": model_a, "model_b": model_b, **run_mcnemar(y_true, pred_a, pred_b)}; mcnemar_rows.append(mrow)
        for metric_name, metric_fn in METRICS.items():
            r = paired_bootstrap_metric_difference(y_true, pred_a, pred_b, metric_fn)
            brow = {"dataset": "LC25000", "comparison": comparison, "model_a": model_a, "model_b": model_b, "metric": metric_name, "model_a_score": round(r["metric_a"],6), "model_b_score": round(r["metric_b"],6), "difference_a_minus_b": round(r["difference_a_minus_b"],6), "ci_lower": round(r["ci_lower"],6), "ci_upper": round(r["ci_upper"],6), "bootstrap_p_approx": round(r["bootstrap_p_approx"],6), "n_bootstrap": r["n_bootstrap"], "interpretation": interpret_ci(r["ci_lower"], r["ci_upper"])}
            bootstrap_rows.append(brow); print(f"{metric_name}: diff={brow['difference_a_minus_b']}, 95% CI=({brow['ci_lower']}, {brow['ci_upper']}), {brow['interpretation']}")
        pd.DataFrame(mcnemar_rows).to_csv(OUTPUT_DIR / "mcnemar_results_leakage_safe.csv", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(OUTPUT_DIR / "paired_bootstrap_metric_differences_leakage_safe.csv", index=False)
    print(f"McNemar leakage-safe results saved to: {OUTPUT_DIR / 'mcnemar_results_leakage_safe.csv'}")

if __name__ == "__main__": main()
