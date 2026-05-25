"""Central configuration for the LC25000 cancer image classification framework."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
LC25000_PATH = DATA_DIR / "lc25000"

RANDOM_STATE = 42
IMAGE_SIZE = 224

ACTIVE_DATASETS = ["LC25000"]
LC25000_CLASS_FOLDER_MAP = {
    "colon_aca": "colon adenocarcinoma",
    "colon_n": "colon benign tissue",
    "lung_aca": "lung adenocarcinoma",
    "lung_n": "lung benign tissue",
    "lung_scc": "lung squamous cell carcinoma",
}
LC25000_CLASS_NAMES = list(LC25000_CLASS_FOLDER_MAP.values())

DEFAULT_BATCH_SIZE = 16
LC25000_BATCH_SIZE = 24
LC25000_UNFREEZE_EPOCH = 2
LC25000_ROTATION_DEGREES = 10
LC25000_HORIZONTAL_FLIP_P = 0.5
IMAGENET_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGENET_NORMALIZE_STD = [0.229, 0.224, 0.225]

SPLIT_DIR = BASE_DIR / "saved_splits"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrices"
ABLATION_OUTPUT_DIR = RESULTS_DIR / "ablations"
LEAKAGE_AUDIT_DIR = RESULTS_DIR / "leakage_audit"
LEAKAGE_SAFE_EVAL_DIR = RESULTS_DIR / "leakage_safe_evaluation"
BOOTSTRAP_CI_DIR = RESULTS_DIR / "bootstrap_confidence_intervals"
CONFIDENCE_ANALYSIS_DIR = RESULTS_DIR / "confidence_analysis"
CALIBRATION_ANALYSIS_DIR = RESULTS_DIR / "calibration_analysis"
RESPONSIBLE_AI_AUDIT_DIR = RESULTS_DIR / "responsible_ai_audit"
STATISTICAL_TESTS_LEAKAGE_SAFE_DIR = RESULTS_DIR / "statistical_tests_leakage_safe"
MODEL_EFFICIENCY_DIR = RESULTS_DIR / "model_efficiency"
DISSERTATION_FIGURES_DIR = RESULTS_DIR / "dissertation_figures"

LC25000_SPLIT_FILE = SPLIT_DIR / "LC25000_split.npz"
LC25000_LEAKAGE_SAFE_SPLIT_FILE = SPLIT_DIR / "LC25000_leakage_safe_split.npz"
LC25000_LEAKAGE_SAFE_SPLIT_SUMMARY = SPLIT_DIR / "LC25000_leakage_safe_split_summary.csv"

CORE_ABLATION_DATASETS = ["LC25000"]
CORE_CALIBRATION_DATASETS = ["LC25000"]
N_BOOTSTRAPS = 1000
N_PAIRED_BOOTSTRAPS = 2000
CALIBRATION_BINS = 10

API_TITLE = "LC25000 Cancer Histopathology Classification API"
API_VERSION = "1.0.0"
DEPLOYED_MODEL_NAME = "LC25000_TransferCNN_ResNet18_LeakageSafe"
DEPLOYED_MODEL_PATH = MODELS_DIR / f"{DEPLOYED_MODEL_NAME}.pth"
RESEARCH_DISCLAIMER = (
    "Research prototype only. This output is not a clinical diagnosis and must "
    "not be used without expert pathological review."
)
