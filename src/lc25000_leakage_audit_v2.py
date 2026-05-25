"""Leakage audit for original LC25000 split."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from config import IMAGE_SIZE, LC25000_CLASS_FOLDER_MAP, LC25000_PATH, LEAKAGE_AUDIT_DIR
from data.lc25000_loader import VALID_IMAGE_EXTENSIONS, find_class_folders
from main import get_split_file, load_split_indices

OUTPUT_DIR = LEAKAGE_AUDIT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def collect_lc25000_paths_in_loader_order(root_dir: Path):
    class_folders = find_class_folders(root_dir)
    rows = []
    for class_index, folder_key in enumerate(LC25000_CLASS_FOLDER_MAP.keys()):
        class_dir = class_folders[folder_key]
        for image_path in sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]):
            rows.append({"path": image_path, "class_index": class_index, "class_name": LC25000_CLASS_FOLDER_MAP[folder_key]})
    return rows

def sha256_file(path: Path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()

def resized_image_hash(path: Path):
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()

def average_hash(path: Path, hash_size=8):
    image = Image.open(path).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    return (pixels > pixels.mean()).astype(np.uint8).flatten()

def make_pair_contact_sheet(rows_df, all_rows, max_pairs=30):
    if rows_df.empty:
        print("No rows available for visual contact sheet."); return
    selected = rows_df.sort_values("min_hamming_distance").head(max_pairs)
    tw, th, lh, gap = 160, 160, 50, 20
    sheet = Image.new("RGB", (2 * tw + gap * 3, len(selected) * (th + lh) + gap * (len(selected) + 1)), "white")
    draw = ImageDraw.Draw(sheet); y = gap
    for _, row in selected.iterrows():
        train_path = all_rows[int(row["nearest_train_original_index"])]["path"]
        test_path = all_rows[int(row["test_original_index"])]["path"]
        train_img = Image.open(train_path).convert("RGB").resize((tw, th))
        test_img = Image.open(test_path).convert("RGB").resize((tw, th))
        sheet.paste(train_img, (gap, y)); sheet.paste(test_img, (gap * 2 + tw, y))
        label = f"Hamming={int(row['min_hamming_distance'])} | Train: {row['nearest_train_label_name']} | Test: {row['test_label_name']}"
        draw.text((gap, y + th + 5), "Nearest train", fill="black")
        draw.text((gap * 2 + tw, y + th + 5), "Test image", fill="black")
        draw.text((gap, y + th + 22), label, fill="black")
        y += th + lh + gap
    out = OUTPUT_DIR / "lc25000_near_duplicate_visual_pairs_top30.png"
    sheet.save(out); print(f"Saved visual pair contact sheet: {out}")

def build_nearest_neighbour_rows(all_rows, train_idx, test_idx, file_hashes, image_hashes, average_hashes):
    train_hashes = average_hashes[train_idx]
    rows = []
    print("\nChecking nearest train image for each test image...")
    for test_position, test_hash in tqdm(enumerate(average_hashes[test_idx]), total=len(test_idx)):
        distances = np.sum(train_hashes != test_hash, axis=1)
        nearest_train_position = int(np.argmin(distances))
        test_original_index = int(test_idx[test_position])
        train_original_index = int(train_idx[nearest_train_position])
        ti = int(all_rows[test_original_index]["class_index"]); tr = int(all_rows[train_original_index]["class_index"])
        rows.append({"test_position": test_position, "test_original_index": test_original_index, "test_label": ti, "test_label_name": all_rows[test_original_index]["class_name"], "test_path": str(all_rows[test_original_index]["path"]), "nearest_train_position": nearest_train_position, "nearest_train_original_index": train_original_index, "nearest_train_label": tr, "nearest_train_label_name": all_rows[train_original_index]["class_name"], "nearest_train_path": str(all_rows[train_original_index]["path"]), "min_hamming_distance": int(distances[nearest_train_position]), "same_class": ti == tr, "possible_duplicate_hamming_le_2": int(distances[nearest_train_position]) <= 2, "high_similarity_hamming_le_5": int(distances[nearest_train_position]) <= 5, "exact_same_file_hash": file_hashes[test_original_index] == file_hashes[train_original_index], "exact_same_resized_image_hash": image_hashes[test_original_index] == image_hashes[train_original_index]})
    return pd.DataFrame(rows)

def main():
    root_dir = Path(LC25000_PATH)
    print("Collecting image paths in loader order...")
    all_rows = collect_lc25000_paths_in_loader_order(root_dir)
    print(f"Images found: {len(all_rows)}")
    split_file = get_split_file("LC25000")
    if not split_file.exists(): raise FileNotFoundError(f"Saved split not found: {split_file}. Run main.py first.")
    train_idx, _val_idx, test_idx = load_split_indices(split_file)
    file_hashes = np.asarray([sha256_file(r["path"]) for r in tqdm(all_rows)])
    image_hashes = np.asarray([resized_image_hash(r["path"]) for r in tqdm(all_rows)])
    average_hashes = np.asarray([average_hash(r["path"]) for r in tqdm(all_rows)], dtype=np.uint8)
    results_df = build_nearest_neighbour_rows(all_rows, train_idx, test_idx, file_hashes, image_hashes, average_hashes)
    detailed_path = OUTPUT_DIR / "lc25000_train_test_leakage_audit_v2_detailed.csv"
    results_df.to_csv(detailed_path, index=False)
    exact_file_overlap = set(file_hashes[train_idx]) & set(file_hashes[test_idx])
    exact_resized_overlap = set(image_hashes[train_idx]) & set(image_hashes[test_idx])
    summary = {"dataset":"LC25000","train_samples":len(train_idx),"test_samples":len(test_idx),"exact_file_hash_overlaps_train_test":len(exact_file_overlap),"exact_resized_image_hash_overlaps_train_test":len(exact_resized_overlap),"nearest_pair_exact_file_hash_matches":int(results_df["exact_same_file_hash"].sum()),"nearest_pair_exact_resized_image_hash_matches":int(results_df["exact_same_resized_image_hash"].sum()),"average_hash_hamming_le_2":int(results_df["possible_duplicate_hamming_le_2"].sum()),"average_hash_hamming_le_5":int(results_df["high_similarity_hamming_le_5"].sum()),"hamming_le_2_same_class_count":int(results_df[results_df["possible_duplicate_hamming_le_2"] & results_df["same_class"]].shape[0]),"hamming_le_2_cross_class_count":int(results_df[results_df["possible_duplicate_hamming_le_2"] & (~results_df["same_class"])].shape[0]),"minimum_hamming_distance_observed":int(results_df["min_hamming_distance"].min()),"median_min_hamming_distance":float(results_df["min_hamming_distance"].median()),"mean_min_hamming_distance":float(results_df["min_hamming_distance"].mean())}
    summary_df = pd.DataFrame([summary])
    summary_path = OUTPUT_DIR / "lc25000_leakage_audit_v2_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    make_pair_contact_sheet(results_df, all_rows)
    print(summary_df.to_string(index=False)); print(f"Saved detailed audit to: {detailed_path}"); print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()
