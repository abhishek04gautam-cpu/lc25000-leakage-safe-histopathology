"""Calibration analysis for leakage-safe LC25000 model predictions."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CALIBRATION_ANALYSIS_DIR, CALIBRATION_BINS, LEAKAGE_SAFE_EVAL_DIR
OUTPUT_DIR = CALIBRATION_ANALYSIS_DIR; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_FILES = {m: LEAKAGE_SAFE_EVAL_DIR / f"{m}_predictions.csv" for m in ["LC25000_TransferCNN_ResNet18_LeakageSafe", "LC25000_SimpleCNN_LeakageSafe", "LC25000_TransferCNN_DenseNet121_LeakageSafe", "LC25000_TransferCNN_EfficientNetB0_LeakageSafe"]}

def get_probability_columns(df):
    cols=[c for c in df.columns if c.startswith("prob_")]
    if not cols: raise ValueError("No probability columns found in prediction file.")
    return cols

def expected_calibration_error(confidences, correct, n_bins=CALIBRATION_BINS):
    edges=np.linspace(0,1,n_bins+1); ece=0.0; mce=0.0; rows=[]; n=len(confidences)
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]; mask=((confidences>=lo)&(confidences<=hi)) if i==n_bins-1 else ((confidences>=lo)&(confidences<hi)); count=int(mask.sum())
        if count==0: rows.append({"bin_lower":round(float(lo),2),"bin_upper":round(float(hi),2),"samples":0,"mean_confidence":np.nan,"accuracy":np.nan,"absolute_gap":np.nan}); continue
        c=float(confidences[mask].mean()); a=float(correct[mask].mean()); gap=abs(a-c); ece+=(count/n)*gap; mce=max(mce,gap); rows.append({"bin_lower":round(float(lo),2),"bin_upper":round(float(hi),2),"samples":count,"mean_confidence":round(c,6),"accuracy":round(a,6),"absolute_gap":round(gap,6)})
    return ece,mce,pd.DataFrame(rows)

def multiclass_brier_score(y_true, probability_array):
    return np.mean(np.sum((probability_array - np.eye(probability_array.shape[1])[y_true]) ** 2, axis=1))

def save_reliability_diagram(bin_df, model_name):
    plot_df=bin_df[bin_df["samples"]>0].copy()
    if plot_df.empty: print(f"No non-empty calibration bins for {model_name}."); return
    plt.figure(figsize=(6,6)); plt.plot([0,1],[0,1],linestyle="--",label="Perfect calibration"); plt.plot(plot_df["mean_confidence"],plot_df["accuracy"],marker="o",label="Observed"); plt.xlabel("Mean confidence"); plt.ylabel("Observed accuracy"); plt.title(f"{model_name}\nReliability Diagram"); plt.xlim(0,1); plt.ylim(0,1); plt.legend(); plt.tight_layout(); path=OUTPUT_DIR / f"{model_name}_reliability_diagram.png"; plt.savefig(path,dpi=300); plt.close(); print(f"Saved reliability diagram to: {path}")

def analyse_model(model_name, prediction_path):
    if not prediction_path.exists(): print(f"Skipping missing prediction file: {prediction_path}"); return None
    df=pd.read_csv(prediction_path); missing={"y_true","y_pred"}-set(df.columns)
    if missing: raise ValueError(f"{prediction_path} is missing required columns: {missing}")
    probs=df[get_probability_columns(df)].to_numpy(float); y_true=df["y_true"].to_numpy(int); y_pred=df["y_pred"].to_numpy(int); conf=np.max(probs,axis=1); correct=(y_true==y_pred).astype(int)
    ece,mce,bin_df=expected_calibration_error(conf,correct,CALIBRATION_BINS); brier=multiclass_brier_score(y_true,probs); bin_df.insert(0,"model",model_name); bin_df.to_csv(OUTPUT_DIR / f"{model_name}_calibration_bins.csv", index=False); save_reliability_diagram(bin_df, model_name)
    summary={"model":model_name,"samples":len(df),"accuracy":round(float(correct.mean()),6),"mean_confidence":round(float(conf.mean()),6),"expected_calibration_error":round(float(ece),6),"maximum_calibration_error":round(float(mce),6),"multiclass_brier_score":round(float(brier),6),"n_bins":CALIBRATION_BINS}; print(pd.DataFrame([summary])); return summary

def main():
    summaries=[s for m,p in PREDICTION_FILES.items() if (s:=analyse_model(m,p)) is not None]
    if not summaries: raise FileNotFoundError("No prediction files were available for calibration analysis.")
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "calibration_summary.csv", index=False); print("Done.")
if __name__ == "__main__": main()
