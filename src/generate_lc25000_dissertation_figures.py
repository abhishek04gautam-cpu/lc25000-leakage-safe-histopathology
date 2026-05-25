"""Generate dissertation figures for final LC25000 leakage-safe evaluation."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from config import BOOTSTRAP_CI_DIR, CALIBRATION_ANALYSIS_DIR, CONFIDENCE_ANALYSIS_DIR, DISSERTATION_FIGURES_DIR, LC25000_CLASS_NAMES, LEAKAGE_SAFE_EVAL_DIR, STATISTICAL_TESTS_LEAKAGE_SAFE_DIR
OUTPUT_DIR=DISSERTATION_FIGURES_DIR; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILES={"Logistic Regression":LEAKAGE_SAFE_EVAL_DIR/"LC25000_LogisticRegression_Downsampled_LeakageSafe_predictions.csv","SimpleCNN":LEAKAGE_SAFE_EVAL_DIR/"LC25000_SimpleCNN_LeakageSafe_predictions.csv","ResNet18":LEAKAGE_SAFE_EVAL_DIR/"LC25000_TransferCNN_ResNet18_LeakageSafe_predictions.csv","EfficientNet-B0":LEAKAGE_SAFE_EVAL_DIR/"LC25000_TransferCNN_EfficientNetB0_LeakageSafe_predictions.csv","DenseNet121":LEAKAGE_SAFE_EVAL_DIR/"LC25000_TransferCNN_DenseNet121_LeakageSafe_predictions.csv"}
SHORT_CLASS_LABELS=["colon aca","colon benign","lung aca","lung benign","lung scc"]
def load_predictions(path):
    if not path.exists(): raise FileNotFoundError(f"Missing prediction file: {path}")
    return pd.read_csv(path)
def get_probability_columns(df): return [c for c in df.columns if c.startswith("prob_")]
def safe_model_filename(model_name): return model_name.replace(" ","_").replace("-","").replace("/","_")
def save_model_performance_chart():
    rows=[]
    for name,path in MODEL_FILES.items():
        df=load_predictions(path); y=df["y_true"].to_numpy(); p=df["y_pred"].to_numpy(); rows.append({"model":name,"accuracy":float((y==p).mean()),"errors":int((y!=p).sum())})
    res=pd.DataFrame(rows); res.to_csv(OUTPUT_DIR/"figure_model_summary_data.csv",index=False)
    plt.figure(figsize=(9,5)); plt.bar(res["model"],res["accuracy"]); plt.ylim(0.5,1.01); plt.ylabel("Accuracy"); plt.xlabel("Model"); plt.title("LC25000 Leakage-Safe Model Accuracy Comparison"); plt.xticks(rotation=25,ha="right"); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"figure_model_accuracy_comparison.png",dpi=300); plt.close()
    plt.figure(figsize=(9,5)); plt.bar(res["model"],res["errors"]); plt.ylabel("Number of test errors"); plt.xlabel("Model"); plt.title("LC25000 Leakage-Safe Test Errors by Model"); plt.xticks(rotation=25,ha="right"); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"figure_model_error_counts.png",dpi=300); plt.close()
def save_confusion_matrices():
    for name,path in MODEL_FILES.items():
        df=load_predictions(path); cm=confusion_matrix(df["y_true"],df["y_pred"],labels=list(range(len(LC25000_CLASS_NAMES)))); fig,ax=plt.subplots(figsize=(8,7)); ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=SHORT_CLASS_LABELS).plot(ax=ax,values_format="d",xticks_rotation=35); ax.set_title(f"LC25000 Leakage-Safe {name} Confusion Matrix"); plt.tight_layout(); plt.savefig(OUTPUT_DIR/f"figure_confusion_matrix_{safe_model_filename(name)}.png",dpi=300); plt.close()
def save_roc_curves():
    for name,path in MODEL_FILES.items():
        df=load_predictions(path); prob_cols=get_probability_columns(df)
        if not prob_cols: print(f"Skipping ROC for {name}: no probability columns."); continue
        y=df["y_true"].to_numpy(); probs=df[prob_cols].to_numpy(float); ybin=label_binarize(y,classes=list(range(len(LC25000_CLASS_NAMES)))); plt.figure(figsize=(8,6))
        for i,cls in enumerate(LC25000_CLASS_NAMES):
            fpr,tpr,_=roc_curve(ybin[:,i],probs[:,i]); short=cls.replace("colon ","c. ").replace("lung ","l. ").replace("adenocarcinoma","aca").replace("benign tissue","benign").replace("squamous cell carcinoma","scc"); plt.plot(fpr,tpr,label=f"{short} AUC={auc(fpr,tpr):.3f}")
        plt.plot([0,1],[0,1],linestyle="--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(f"LC25000 Leakage-Safe {name} One-vs-Rest ROC Curves"); plt.legend(fontsize=8,loc="lower right"); plt.tight_layout(); plt.savefig(OUTPUT_DIR/f"figure_roc_curve_{safe_model_filename(name)}.png",dpi=300); plt.close()
def save_bootstrap_ci_chart():
    path=BOOTSTRAP_CI_DIR/"LC25000_TransferCNN_ResNet18_LeakageSafe_bootstrap_ci_summary.csv"
    if not path.exists(): print(f"Bootstrap CI file not found: {path}"); return
    df=pd.read_csv(path); keep=df[df["metric"].isin(["accuracy","weighted_f1","macro_f1","balanced_accuracy","roc_auc"])].copy().sort_values("metric"); x=keep["point_estimate"].to_numpy(); lo=keep["ci_lower_95"].to_numpy(); hi=keep["ci_upper_95"].to_numpy(); y=np.arange(len(keep)); plt.figure(figsize=(8,5)); plt.errorbar(x,y,xerr=[x-lo,hi-x],fmt="o",capsize=4); plt.yticks(y,keep["metric"]); plt.xlim(0.99,1.001); plt.xlabel("Metric value with 95% confidence interval"); plt.title("LC25000 Leakage-Safe ResNet18 Bootstrap Confidence Intervals"); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"figure_resnet18_bootstrap_ci.png",dpi=300); plt.close()
def save_statistical_difference_chart():
    path=STATISTICAL_TESTS_LEAKAGE_SAFE_DIR/"paired_bootstrap_metric_differences_leakage_safe.csv"
    if not path.exists(): print(f"Statistical comparison file not found: {path}"); return
    df=pd.read_csv(path); m=df[df["metric"]=="accuracy"].copy().sort_values("difference_a_minus_b"); x=m["difference_a_minus_b"].to_numpy(); lo=m["ci_lower"].to_numpy(); hi=m["ci_upper"].to_numpy(); y=np.arange(len(m)); plt.figure(figsize=(9,7)); plt.axvline(0,linestyle="--"); plt.errorbar(x,y,xerr=[x-lo,hi-x],fmt="o",capsize=4); plt.yticks(y,m["comparison"]); plt.xlabel("Accuracy difference: model A minus model B"); plt.title("LC25000 Leakage-Safe Paired Bootstrap Accuracy Differences"); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"figure_paired_bootstrap_accuracy_differences.png",dpi=300); plt.close()
def save_confidence_and_calibration_charts():
    path=CONFIDENCE_ANALYSIS_DIR/"LC25000_TransferCNN_ResNet18_LeakageSafe_confidence_bins.csv"
    if path.exists():
        df=pd.read_csv(path); p=df[df["samples"]>0].copy(); plt.figure(figsize=(7,5)); plt.plot(p["confidence_bin"],p["accuracy"],marker="o",label="Observed accuracy"); plt.plot(p["confidence_bin"],p["mean_confidence"],marker="o",label="Mean confidence"); plt.ylim(0,1.05); plt.xlabel("Confidence bin"); plt.ylabel("Score"); plt.title("LC25000 Leakage-Safe ResNet18 Confidence Reliability"); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"figure_confidence_reliability.png",dpi=300); plt.close()
    path=CALIBRATION_ANALYSIS_DIR/"LC25000_TransferCNN_ResNet18_LeakageSafe_calibration_bins.csv"
    if path.exists():
        df=pd.read_csv(path); p=df[df["samples"]>0].copy(); plt.figure(figsize=(6,6)); plt.plot([0,1],[0,1],linestyle="--",label="Perfect calibration"); plt.plot(p["mean_confidence"],p["accuracy"],marker="o",label="Observed"); plt.xlabel("Mean confidence"); plt.ylabel("Observed accuracy"); plt.title("LC25000 Leakage-Safe ResNet18 Reliability Diagram"); plt.xlim(0,1); plt.ylim(0,1); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"figure_reliability_diagram.png",dpi=300); plt.close()
def main():
    save_model_performance_chart(); save_confusion_matrices(); save_roc_curves(); save_bootstrap_ci_chart(); save_statistical_difference_chart(); save_confidence_and_calibration_charts(); print(f"Figures saved to: {OUTPUT_DIR}")
if __name__ == "__main__": main()
