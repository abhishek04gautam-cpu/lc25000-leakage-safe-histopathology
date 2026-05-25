"""Confidence-bin analysis for leakage-safe LC25000 model predictions."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CONFIDENCE_ANALYSIS_DIR, LEAKAGE_SAFE_EVAL_DIR
OUTPUT_DIR = CONFIDENCE_ANALYSIS_DIR; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_FILES = {m: LEAKAGE_SAFE_EVAL_DIR / f"{m}_predictions.csv" for m in ["LC25000_TransferCNN_ResNet18_LeakageSafe", "LC25000_SimpleCNN_LeakageSafe", "LC25000_TransferCNN_DenseNet121_LeakageSafe", "LC25000_TransferCNN_EfficientNetB0_LeakageSafe"]}

def validate_prediction_file(df, prediction_path):
    missing = {"y_true", "y_pred", "confidence"} - set(df.columns)
    if missing: raise ValueError(f"{prediction_path} is missing required columns: {missing}")

def build_confidence_summary(df, model_name):
    df = df.copy(); df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)
    df["confidence_bin"] = pd.cut(df["confidence"], bins=np.linspace(0,1,6), labels=["0.0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1.0"], include_lowest=True)
    summary = df.groupby("confidence_bin", observed=False).agg(samples=("correct","count"), accuracy=("correct","mean"), mean_confidence=("confidence","mean")).reset_index()
    summary["accuracy"] = summary["accuracy"].round(6); summary["mean_confidence"] = summary["mean_confidence"].round(6); summary.insert(0, "model", model_name); return summary

def save_confidence_plot(summary, model_name):
    plot_df = summary[summary["samples"] > 0].copy()
    if plot_df.empty: print(f"No non-empty confidence bins for {model_name}."); return
    plt.figure(figsize=(7,5)); plt.plot(plot_df["confidence_bin"].astype(str), plot_df["accuracy"], marker="o", label="Observed accuracy"); plt.plot(plot_df["confidence_bin"].astype(str), plot_df["mean_confidence"], marker="o", label="Mean confidence")
    plt.xlabel("Confidence bin"); plt.ylabel("Score"); plt.title(f"{model_name} Confidence Reliability"); plt.ylim(0,1.05); plt.legend(); plt.tight_layout(); path=OUTPUT_DIR / f"{model_name}_confidence_reliability.png"; plt.savefig(path,dpi=300); plt.close(); print(f"Saved confidence plot to: {path}")

def analyse_confidence(model_name, prediction_path):
    if not prediction_path.exists(): print(f"Skipping missing prediction file: {prediction_path}"); return None
    df = pd.read_csv(prediction_path); validate_prediction_file(df, prediction_path); summary=build_confidence_summary(df, model_name); summary.to_csv(OUTPUT_DIR / f"{model_name}_confidence_bins.csv", index=False); print(summary); save_confidence_plot(summary, model_name); return summary

def main():
    summaries=[s for m,p in PREDICTION_FILES.items() if (s:=analyse_confidence(m,p)) is not None]
    if not summaries: raise FileNotFoundError("No prediction files were available for confidence analysis.")
    pd.concat(summaries, ignore_index=True).to_csv(OUTPUT_DIR / "all_leakage_safe_confidence_bins.csv", index=False); print("Done.")
if __name__ == "__main__": main()
