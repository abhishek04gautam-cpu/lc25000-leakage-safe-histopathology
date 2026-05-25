"""Central dataset-loading entry point."""
from dataset import Dataset
from data.lc25000_loader import load_lc25000_dataset
from config import ACTIVE_DATASETS, LC25000_CLASS_NAMES, LC25000_PATH

def load_datasets():
    datasets = []
    if "LC25000" in ACTIVE_DATASETS:
        X_lc25000, y_lc25000 = load_lc25000_dataset(LC25000_PATH)
        datasets.append(Dataset("LC25000", X_lc25000, y_lc25000, "image", class_names=LC25000_CLASS_NAMES))
    if not datasets:
        raise ValueError("No datasets were loaded. Check ACTIVE_DATASETS in config.py.")
    return datasets
