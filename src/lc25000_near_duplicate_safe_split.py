"""Create a near-duplicate-safe LC25000 split.

This script creates a stricter grouped split than the exact resized-image-hash
leakage-safe split. It groups images that are exact resized-image duplicates
and also connects visually near-duplicate images using average-hash Hamming
distance <= 2 within each class.

The goal is to prevent exact and near-duplicate images from being split across
train, validation and test partitions.
"""

import hashlib
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    IMAGE_SIZE,
    LC25000_CLASS_FOLDER_MAP,
    LC25000_PATH,
    RESULTS_DIR,
    RANDOM_STATE,
)
from data.lc25000_loader import VALID_IMAGE_EXTENSIONS, find_class_folders


OUTPUT_DIR = RESULTS_DIR / "near_duplicate_safe_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_PATH = OUTPUT_DIR / "LC25000_near_duplicate_safe_split.npz"
SUMMARY_PATH = OUTPUT_DIR / "LC25000_near_duplicate_safe_split_summary.csv"
GROUPS_PATH = OUTPUT_DIR / "LC25000_near_duplicate_groups.csv"

HAMMING_THRESHOLD = 2


def collect_lc25000_paths_in_loader_order(root_dir: Path):
    class_folders = find_class_folders(root_dir)
    rows = []

    for class_index, folder_key in enumerate(LC25000_CLASS_FOLDER_MAP.keys()):
        class_dir = class_folders[folder_key]
        image_paths = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ]
        )

        for image_path in image_paths:
            rows.append(
                {
                    "path": image_path,
                    "class_index": class_index,
                    "class_name": LC25000_CLASS_FOLDER_MAP[folder_key],
                }
            )

    return rows


def resized_image_hash(path: Path):
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()


def average_hash(path: Path, hash_size=8):
    image = Image.open(path).convert("L").resize(
        (hash_size, hash_size),
        Image.Resampling.LANCZOS,
    )
    pixels = np.asarray(image, dtype=np.float32)
    return (pixels > pixels.mean()).astype(np.uint8).flatten()


def union_find_make(n):
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return parent, find, union


def build_groups(rows):
    print("Computing resized-image hashes...")
    resized_hashes = np.asarray(
        [resized_image_hash(row["path"]) for row in tqdm(rows)],
        dtype=object,
    )

    print("Computing average hashes...")
    ahashes = np.asarray(
        [average_hash(row["path"]) for row in tqdm(rows)],
        dtype=np.uint8,
    )

    n = len(rows)
    parent, find, union = union_find_make(n)

    print("Unioning exact resized-image duplicates...")
    hash_to_indices = defaultdict(list)
    for idx, h in enumerate(resized_hashes):
        hash_to_indices[h].append(idx)

    for indices in hash_to_indices.values():
        if len(indices) > 1:
            first = indices[0]
            for other in indices[1:]:
                union(first, other)

    print("Unioning near-duplicates within each class...")
    class_to_indices = defaultdict(list)
    for idx, row in enumerate(rows):
        class_to_indices[int(row["class_index"])].append(idx)

    # Brute force within each class: 5,000 images per class is manageable.
    for class_index, indices in class_to_indices.items():
        print(f"Class {class_index}: checking {len(indices)} images")
        class_hashes = ahashes[indices]

        for i in tqdm(range(len(indices))):
            distances = np.sum(class_hashes[i + 1 :] != class_hashes[i], axis=1)
            matches = np.where(distances <= HAMMING_THRESHOLD)[0]

            if len(matches) == 0:
                continue

            original_i = indices[i]
            for m in matches:
                original_j = indices[i + 1 + int(m)]
                union(original_i, original_j)

    group_roots = [find(i) for i in range(n)]
    root_to_group_id = {}
    group_ids = []

    for root in group_roots:
        if root not in root_to_group_id:
            root_to_group_id[root] = len(root_to_group_id)
        group_ids.append(root_to_group_id[root])

    group_df = pd.DataFrame(
        {
            "original_index": list(range(n)),
            "path": [str(row["path"]) for row in rows],
            "class_index": [int(row["class_index"]) for row in rows],
            "class_name": [row["class_name"] for row in rows],
            "near_duplicate_group_id": group_ids,
            "resized_image_hash": resized_hashes,
        }
    )

    return group_df


def split_groups(group_df):
    group_summary = (
        group_df.groupby("near_duplicate_group_id")
        .agg(
            class_index=("class_index", "first"),
            class_name=("class_name", "first"),
            group_size=("original_index", "count"),
        )
        .reset_index()
    )

    group_ids = group_summary["near_duplicate_group_id"].values
    group_labels = group_summary["class_index"].values

    train_groups, temp_groups = train_test_split(
        group_ids,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=group_labels,
    )

    temp_summary = group_summary[
        group_summary["near_duplicate_group_id"].isin(temp_groups)
    ]
    temp_group_ids = temp_summary["near_duplicate_group_id"].values
    temp_labels = temp_summary["class_index"].values

    val_groups, test_groups = train_test_split(
        temp_group_ids,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    train_idx = group_df[
        group_df["near_duplicate_group_id"].isin(train_groups)
    ]["original_index"].values

    val_idx = group_df[
        group_df["near_duplicate_group_id"].isin(val_groups)
    ]["original_index"].values

    test_idx = group_df[
        group_df["near_duplicate_group_id"].isin(test_groups)
    ]["original_index"].values

    return train_idx, val_idx, test_idx, group_summary


def check_group_overlap(group_df, train_idx, val_idx, test_idx):
    train_groups = set(group_df.loc[train_idx, "near_duplicate_group_id"])
    val_groups = set(group_df.loc[val_idx, "near_duplicate_group_id"])
    test_groups = set(group_df.loc[test_idx, "near_duplicate_group_id"])

    return {
        "train_val_group_overlap": len(train_groups & val_groups),
        "train_test_group_overlap": len(train_groups & test_groups),
        "val_test_group_overlap": len(val_groups & test_groups),
    }


def class_counts(group_df, indices, split_name):
    subset = group_df.loc[indices]
    rows = []

    for class_index, count in subset["class_index"].value_counts().sort_index().items():
        class_name = subset[subset["class_index"] == class_index]["class_name"].iloc[0]
        rows.append(
            {
                "split": split_name,
                "class_index": int(class_index),
                "class_name": class_name,
                "samples": int(count),
            }
        )

    return rows


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("Collecting LC25000 paths...")
    rows = collect_lc25000_paths_in_loader_order(Path(LC25000_PATH))
    print(f"Images found: {len(rows)}")

    group_df = build_groups(rows)
    group_df.to_csv(GROUPS_PATH, index=False)

    train_idx, val_idx, test_idx, group_summary = split_groups(group_df)

    overlap = check_group_overlap(group_df, train_idx, val_idx, test_idx)

    np.savez(
        SPLIT_PATH,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    summary_rows = []
    summary_rows.extend(class_counts(group_df, train_idx, "train"))
    summary_rows.extend(class_counts(group_df, val_idx, "validation"))
    summary_rows.extend(class_counts(group_df, test_idx, "test"))

    summary_df = pd.DataFrame(summary_rows)

    metadata = {
        "total_images": len(group_df),
        "total_near_duplicate_groups": group_df["near_duplicate_group_id"].nunique(),
        "largest_group_size": int(group_summary["group_size"].max()),
        "groups_with_more_than_one_image": int((group_summary["group_size"] > 1).sum()),
        "hamming_threshold": HAMMING_THRESHOLD,
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
        **overlap,
    }

    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(
        OUTPUT_DIR / "LC25000_near_duplicate_safe_split_metadata.csv",
        index=False,
    )

    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\nNear-duplicate-safe split metadata:")
    print(metadata_df.to_string(index=False))
    print("\nClass counts:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved split to: {SPLIT_PATH}")
    print(f"Saved group mapping to: {GROUPS_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()