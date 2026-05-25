"""Class-wise responsible-AI audit for LC25000."""
import numpy as np, pandas as pd
from config import LEAKAGE_SAFE_EVAL_DIR, RESPONSIBLE_AI_AUDIT_DIR
OUTPUT_DIR=RESPONSIBLE_AI_AUDIT_DIR; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLASSIFICATION_REPORTS={"LC25000_TransferCNN_ResNet18_LeakageSafe": LEAKAGE_SAFE_EVAL_DIR / "LC25000_TransferCNN_ResNet18_LeakageSafe_classification_report.csv"}
SUMMARY_FILES={"LC25000_TransferCNN_ResNet18_LeakageSafe": LEAKAGE_SAFE_EVAL_DIR / "lc25000_resnet18_leakage_safe_metrics.csv"}
def extract_dataset_name(model_key): return "LC25000" if model_key.startswith("LC25000") else model_key
def extract_model_name(model_key): return model_key[len("LC25000_"):] if model_key.startswith("LC25000_") else model_key
def load_model_level_metrics(model_key):
    path=SUMMARY_FILES.get(model_key)
    if path is None or not path.exists(): return {"macro_f1":np.nan,"weighted_f1":np.nan,"balanced_accuracy":np.nan,"roc_auc":np.nan}
    df=pd.read_csv(path); row=df.iloc[0] if not df.empty else {}
    return {"macro_f1":row.get("macro_f1",np.nan),"weighted_f1":row.get("weighted_f1",np.nan),"balanced_accuracy":row.get("balanced_accuracy",np.nan),"roc_auc":row.get("roc_auc",np.nan)}
def load_class_rows(report_path):
    df=pd.read_csv(report_path,index_col=0); df=df[~df.index.isin({"accuracy","macro avg","weighted avg"})].copy(); return df[pd.to_numeric(df["support"], errors="coerce").notna()].assign(support=lambda x: x["support"].astype(float))
def build_classwise_audit():
    rows=[]
    for key,path in CLASSIFICATION_REPORTS.items():
        if not path.exists(): print(f"Missing classification report: {path}"); continue
        metrics=load_model_level_metrics(key); cls=load_class_rows(path); total=cls["support"].sum(); mins,maxs=cls["support"].min(),cls["support"].max()
        for cname,row in cls.iterrows():
            rows.append({"dataset":extract_dataset_name(key),"model":extract_model_name(key),"class_name":cname,"support":int(row["support"]),"support_ratio":round(float(row["support"]/total),6),"minority_class":bool(row["support"]==mins),"largest_class":bool(row["support"]==maxs),"precision":round(float(row["precision"]),6),"recall":round(float(row["recall"]),6),"f1_score":round(float(row["f1-score"]),6),"dataset_macro_f1":round(float(metrics["macro_f1"]),6),"dataset_weighted_f1":round(float(metrics["weighted_f1"]),6),"dataset_balanced_accuracy":round(float(metrics["balanced_accuracy"]),6),"dataset_roc_auc":metrics["roc_auc"]})
    audit=pd.DataFrame(rows)
    if not audit.empty: audit["dataset_roc_auc"]=pd.to_numeric(audit["dataset_roc_auc"], errors="coerce").round(6)
    return audit
def build_dataset_level_summary(audit_df):
    rows=[]
    for dataset_name,group in audit_df.groupby("dataset"):
        smallest=group.sort_values("support").iloc[0]; largest=group.sort_values("support",ascending=False).iloc[0]; worst_recall=group.sort_values("recall").iloc[0]; worst_f1=group.sort_values("f1_score").iloc[0]
        rows.append({"dataset":dataset_name,"model":group["model"].iloc[0],"total_test_samples":int(group["support"].sum()),"smallest_class":smallest["class_name"],"smallest_class_support":int(smallest["support"]),"smallest_class_recall":smallest["recall"],"largest_class":largest["class_name"],"largest_class_support":int(largest["support"]),"largest_class_recall":largest["recall"],"worst_recall_class":worst_recall["class_name"],"worst_recall":worst_recall["recall"],"worst_f1_class":worst_f1["class_name"],"worst_f1":worst_f1["f1_score"],"macro_f1":group["dataset_macro_f1"].iloc[0],"weighted_f1":group["dataset_weighted_f1"].iloc[0],"balanced_accuracy":group["dataset_balanced_accuracy"].iloc[0],"roc_auc":group["dataset_roc_auc"].iloc[0]})
    return pd.DataFrame(rows)
def main():
    audit=build_classwise_audit()
    if audit.empty: raise FileNotFoundError("No class-wise audit rows were generated. Run leakage-safe ResNet18 evaluation first.")
    audit.to_csv(OUTPUT_DIR / "classwise_responsible_ai_audit.csv", index=False); summary=build_dataset_level_summary(audit); summary.to_csv(OUTPUT_DIR / "dataset_level_responsible_ai_summary.csv", index=False); print(audit); print(summary); print("Done.")
if __name__ == "__main__": main()
