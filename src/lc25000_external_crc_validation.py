"""
External colon-only validation on CRC-VAL-HE-7K.

This script evaluates the existing LC25000 leakage-safe ResNet18 checkpoint
on an independent colorectal histology validation set using only:

- TUM  -> tumour / LC25000 colon adenocarcinoma
- NORM -> normal / LC25000 colon benign tissue

The model remains the original 5-class LC25000 model. Two outputs are reported:

1. Strict 5-class external accuracy:
   A sample is correct only if TUM is predicted as LC25000 class 0 or
   NORM is predicted as LC25000 class 1. Predictions into lung classes are
   counted as non-colon predictions.

2. Colon-restricted binary accuracy:
   Only the LC25000 colon probabilities are compared:
   class 0 = colon adenocarcinoma, class 1 = colon benign tissue.
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


LC25000_CLASS_NAMES = [
    "colon adenocarcinoma",
    "colon benign tissue",
    "lung adenocarcinoma",
    "lung benign tissue",
    "lung squamous cell carcinoma",
]

CRC_TO_BINARY = {
    "NORM": 0,  # normal
    "TUM": 1,   # tumour
}

BINARY_LABEL_NAMES = ["normal", "tumour"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResNet18LC25000(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ExternalCRCDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        return image, row["binary_label"], row["crc_class"], str(row["path"])


def find_checkpoint(repo_root: Path) -> Path:
    candidates = [
        repo_root / "models" / "LC25000_TransferCNN_ResNet18_LeakageSafe.pth",
        repo_root.parent / "models" / "LC25000_TransferCNN_ResNet18_LeakageSafe.pth",
        Path.cwd() / "models" / "LC25000_TransferCNN_ResNet18_LeakageSafe.pth",
        Path.cwd().parent / "models" / "LC25000_TransferCNN_ResNet18_LeakageSafe.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find LC25000_TransferCNN_ResNet18_LeakageSafe.pth. "
        "Expected it in publication_repo 3/models or ../models."
    )


def find_crc_root() -> Path:
    import os

    candidates = []

    env_path = os.environ.get("CRC_VAL_HE_7K_ROOT")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path.cwd() / "datasets" / "crc-val-he-7k" / "CRC-VAL-HE-7K",
            Path.cwd().parent / "datasets" / "crc-val-he-7k" / "CRC-VAL-HE-7K",
            Path.home() / "Datasets" / "kaggle" / "crc-val-he-7k" / "CRC-VAL-HE-7K",
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find CRC-VAL-HE-7K root directory. "
        "Set CRC_VAL_HE_7K_ROOT to the folder containing ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR and TUM."
    )


def collect_rows(crc_root: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    rows = []

    for crc_class, binary_label in CRC_TO_BINARY.items():
        class_dir = crc_root / crc_class
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                rows.append(
                    {
                        "path": path,
                        "crc_class": crc_class,
                        "binary_label": binary_label,
                    }
                )

    return rows


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results" / "external_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = find_checkpoint(repo_root)
    crc_root = find_crc_root()

    print("Repository root:", repo_root)
    print("Checkpoint:", checkpoint_path)
    print("CRC-VAL root:", crc_root)

    rows = collect_rows(crc_root)
    print("External validation rows:", len(rows))
    print("Class counts:", Counter(row["crc_class"] for row in rows))

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    loader = DataLoader(
        ExternalCRCDataset(rows, transform),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    device = get_device()
    print("Device:", device)

    model = ResNet18LC25000(num_classes=5)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    all_true_binary = []
    all_crc_class = []
    all_paths = []
    all_probs = []
    all_full_pred = []

    with torch.no_grad():
        for images, binary_labels, crc_classes, paths in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            full_pred = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_full_pred.extend(full_pred.tolist())
            all_true_binary.extend(binary_labels.numpy().astype(int).tolist())
            all_crc_class.extend(list(crc_classes))
            all_paths.extend(list(paths))

    y_true_binary = np.asarray(all_true_binary, dtype=int)
    y_prob = np.vstack(all_probs)
    full_pred = np.asarray(all_full_pred, dtype=int)

    # Binary positive class = tumour.
    # TUM maps to LC25000 class 0, NORM maps to LC25000 class 1.
    p_tumour = y_prob[:, 0]
    p_normal = y_prob[:, 1]

    colon_prob_sum = p_tumour + p_normal
    lung_or_other_prob_sum = y_prob[:, 2:].sum(axis=1)

    colon_restricted_pred_binary = (p_tumour >= p_normal).astype(int)

    # Strict 5-class correctness:
    # tumour is correct only if full 5-class pred is LC class 0.
    # normal is correct only if full 5-class pred is LC class 1.
    strict_correct = (
        ((y_true_binary == 1) & (full_pred == 0))
        | ((y_true_binary == 0) & (full_pred == 1))
    )
    strict_accuracy = float(strict_correct.mean())

    non_colon_pred = ~np.isin(full_pred, [0, 1])
    non_colon_prediction_rate = float(non_colon_pred.mean())

    precision, recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true_binary,
        colon_restricted_pred_binary,
        average="weighted",
        zero_division=0,
    )
    macro_f1 = precision_recall_fscore_support(
        y_true_binary,
        colon_restricted_pred_binary,
        average="macro",
        zero_division=0,
    )[2]

    try:
        roc_auc = roc_auc_score(y_true_binary, p_tumour)
    except ValueError:
        roc_auc = float("nan")

    colon_restricted_accuracy = accuracy_score(
        y_true_binary, colon_restricted_pred_binary
    )
    balanced_accuracy = balanced_accuracy_score(
        y_true_binary, colon_restricted_pred_binary
    )

    cm_binary = confusion_matrix(
        y_true_binary,
        colon_restricted_pred_binary,
        labels=[0, 1],
    )

    # 2x6 strict table: true binary label vs full LC25000 predicted class.
    strict_cm = pd.crosstab(
        pd.Series([BINARY_LABEL_NAMES[i] for i in y_true_binary], name="true_external_label"),
        pd.Series([LC25000_CLASS_NAMES[i] for i in full_pred], name="full_lc25000_prediction"),
        dropna=False,
    )

    pred_rows = []
    for i, path in enumerate(all_paths):
        row = {
            "path": path,
            "crc_class": all_crc_class[i],
            "true_binary_label": int(y_true_binary[i]),
            "true_binary_name": BINARY_LABEL_NAMES[int(y_true_binary[i])],
            "full_pred_index": int(full_pred[i]),
            "full_pred_name": LC25000_CLASS_NAMES[int(full_pred[i])],
            "colon_restricted_pred_binary": int(colon_restricted_pred_binary[i]),
            "colon_restricted_pred_name": BINARY_LABEL_NAMES[int(colon_restricted_pred_binary[i])],
            "strict_5class_correct": bool(strict_correct[i]),
            "colon_prob_sum": float(colon_prob_sum[i]),
            "lung_or_other_prob_sum": float(lung_or_other_prob_sum[i]),
        }
        for class_index, class_name in enumerate(LC25000_CLASS_NAMES):
            safe_name = class_name.replace(" ", "_")
            row[f"prob_{safe_name}"] = float(y_prob[i, class_index])
        pred_rows.append(row)

    predictions_df = pd.DataFrame(pred_rows)

    summary = {
        "external_dataset": "CRC-VAL-HE-7K",
        "external_classes_used": "TUM,NORM",
        "model": "LC25000_TransferCNN_ResNet18_LeakageSafe",
        "checkpoint": str(checkpoint_path),
        "tumour_samples_TUM": int((y_true_binary == 1).sum()),
        "normal_samples_NORM": int((y_true_binary == 0).sum()),
        "total_samples": int(len(y_true_binary)),
        "strict_5class_external_accuracy": round(strict_accuracy, 6),
        "non_colon_prediction_rate": round(non_colon_prediction_rate, 6),
        "colon_restricted_accuracy": round(float(colon_restricted_accuracy), 6),
        "colon_restricted_precision_weighted": round(float(precision), 6),
        "colon_restricted_recall_weighted": round(float(recall), 6),
        "colon_restricted_weighted_f1": round(float(weighted_f1), 6),
        "colon_restricted_macro_f1": round(float(macro_f1), 6),
        "colon_restricted_balanced_accuracy": round(float(balanced_accuracy), 6),
        "colon_restricted_roc_auc": round(float(roc_auc), 6) if not np.isnan(roc_auc) else np.nan,
        "mean_colon_prob_sum": round(float(np.mean(colon_prob_sum)), 6),
        "mean_lung_or_other_prob_sum": round(float(np.mean(lung_or_other_prob_sum)), 6),
        "notes": (
            "External colon-only validation using CRC-VAL-HE-7K TUM/NORM. "
            "TUM mapped to LC25000 colon adenocarcinoma; NORM mapped to LC25000 colon benign tissue."
        ),
    }

    summary_df = pd.DataFrame([summary])

    summary_path = out_dir / "lc25000_resnet18_crc_val_external_colon_validation_summary.csv"
    predictions_path = out_dir / "lc25000_resnet18_crc_val_external_colon_predictions.csv"
    binary_cm_path = out_dir / "lc25000_resnet18_crc_val_external_colon_binary_confusion_matrix.csv"
    strict_cm_path = out_dir / "lc25000_resnet18_crc_val_external_colon_strict_full_prediction_table.csv"

    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    pd.DataFrame(
        cm_binary,
        index=["true_normal", "true_tumour"],
        columns=["pred_normal", "pred_tumour"],
    ).to_csv(binary_cm_path)
    strict_cm.to_csv(strict_cm_path)

    print()
    print("External validation summary:")
    print(summary_df.to_string(index=False))

    print()
    print("Colon-restricted binary confusion matrix:")
    print(pd.DataFrame(
        cm_binary,
        index=["true_normal", "true_tumour"],
        columns=["pred_normal", "pred_tumour"],
    ).to_string())

    print()
    print("Strict full 5-class prediction table:")
    print(strict_cm.to_string())

    print()
    print("Saved:")
    print(summary_path)
    print(predictions_path)
    print(binary_cm_path)
    print(strict_cm_path)


if __name__ == "__main__":
    main()
