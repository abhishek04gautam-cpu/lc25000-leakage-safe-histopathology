"""Create leakage-safe LC25000 split by grouping duplicate resized images."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import IMAGE_SIZE, LC25000_CLASS_FOLDER_MAP, LC25000_LEAKAGE_SAFE_SPLIT_FILE, LC25000_LEAKAGE_SAFE_SPLIT_SUMMARY, LC25000_PATH, RANDOM_STATE
from data.lc25000_loader import VALID_IMAGE_EXTENSIONS, find_class_folders

def collect_lc25000_paths_in_loader_order(root_dir: Path):
    class_folders = find_class_folders(root_dir); rows = []
    for class_index, folder_key in enumerate(LC25000_CLASS_FOLDER_MAP.keys()):
        class_dir = class_folders[folder_key]
        for image_path in sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]):
            rows.append({"path": image_path, "class_index": class_index, "class_name": LC25000_CLASS_FOLDER_MAP[folder_key]})
    return rows

def resized_image_hash(path: Path):
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()

def build_group_dataframe(rows):
    print("Computing resized image hashes...")
    image_hashes = [resized_image_hash(row["path"]) for row in tqdm(rows)]
    df = pd.DataFrame(rows); df["original_index"] = np.arange(len(df)); df["image_hash"] = image_hashes
    group_df = df.groupby("image_hash").agg(group_label=("class_index", "first"), group_class_name=("class_name", "first"), group_size=("original_index", "count")).reset_index()
    mixed_label_hashes = df.groupby("image_hash")["class_index"].nunique()
    mixed_label_hashes = mixed_label_hashes[mixed_label_hashes > 1]
    if len(mixed_label_hashes) > 0:
        raise ValueError(f"Found {len(mixed_label_hashes)} identical image hashes with multiple class labels.")
    return df, group_df

def create_group_level_split(group_df):
    group_indices = np.arange(len(group_df)); group_labels = group_df["group_label"].to_numpy()
    train_groups, temp_groups = train_test_split(group_indices, test_size=0.30, random_state=RANDOM_STATE, stratify=group_labels)
    val_groups, test_groups = train_test_split(temp_groups, test_size=0.50, random_state=RANDOM_STATE, stratify=group_labels[temp_groups])
    return train_groups, val_groups, test_groups

def convert_groups_to_original_indices(df, group_df, train_groups, val_groups, test_groups):
    train_hashes = set(group_df.iloc[train_groups]["image_hash"]); val_hashes = set(group_df.iloc[val_groups]["image_hash"]); test_hashes = set(group_df.iloc[test_groups]["image_hash"])
    if not train_hashes.isdisjoint(val_hashes): raise AssertionError("Train and validation hash groups overlap.")
    if not train_hashes.isdisjoint(test_hashes): raise AssertionError("Train and test hash groups overlap.")
    if not val_hashes.isdisjoint(test_hashes): raise AssertionError("Validation and test hash groups overlap.")
    return (df[df["image_hash"].isin(train_hashes)]["original_index"].to_numpy(), df[df["image_hash"].isin(val_hashes)]["original_index"].to_numpy(), df[df["image_hash"].isin(test_hashes)]["original_index"].to_numpy(), train_hashes, val_hashes, test_hashes)

def save_split_summary(df, train_idx, val_idx, test_idx):
    rows = []
    for split_name, indices in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        counts = df.iloc[indices]["class_name"].value_counts().sort_index()
        for class_name, count in counts.items(): rows.append({"split": split_name, "class_name": class_name, "samples": int(count)})
    summary_df = pd.DataFrame(rows); summary_df.to_csv(LC25000_LEAKAGE_SAFE_SPLIT_SUMMARY, index=False); return summary_df

def main():
    rows = collect_lc25000_paths_in_loader_order(Path(LC25000_PATH)); print(f"Images found: {len(rows)}")
    df, group_df = build_group_dataframe(rows)
    print(f"Unique resized-image groups: {len(group_df)}"); print(f"Duplicate groups: {int((group_df['group_size'] > 1).sum())}"); print(f"Maximum duplicate-group size: {int(group_df['group_size'].max())}")
    train_groups, val_groups, test_groups = create_group_level_split(group_df)
    train_idx, val_idx, test_idx, train_hashes, val_hashes, test_hashes = convert_groups_to_original_indices(df, group_df, train_groups, val_groups, test_groups)
    LC25000_LEAKAGE_SAFE_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(LC25000_LEAKAGE_SAFE_SPLIT_FILE, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    summary_df = save_split_summary(df, train_idx, val_idx, test_idx)
    print("\nLeakage-safe split saved:", LC25000_LEAKAGE_SAFE_SPLIT_FILE)
    print(summary_df); print("\nHash overlap checks:"); print("Train/validation overlap:", len(train_hashes & val_hashes)); print("Train/test overlap:", len(train_hashes & test_hashes)); print("Validation/test overlap:", len(val_hashes & test_hashes))

if __name__ == "__main__":
    main()
