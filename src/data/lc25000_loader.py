"""LC25000 dataset loader."""
from pathlib import Path
import numpy as np
from PIL import Image
from config import IMAGE_SIZE, LC25000_CLASS_FOLDER_MAP, LC25000_CLASS_NAMES

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_class_folders(root_dir: Path):
    root_dir = Path(root_dir)
    class_folders = {}
    for folder_key in LC25000_CLASS_FOLDER_MAP.keys():
        matches = [path for path in root_dir.rglob(folder_key) if path.is_dir()]
        if not matches:
            raise FileNotFoundError(f"Could not find LC25000 class folder '{folder_key}' under {root_dir}")
        class_folders[folder_key] = matches[0]
    return class_folders

def load_lc25000_dataset(root_dir):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"LC25000 path not found: {root_dir}")
    class_folders = find_class_folders(root_dir)
    images, labels = [], []
    print("\nLoading LC25000 dataset...")
    print(f"Root: {root_dir}")
    for class_index, folder_key in enumerate(LC25000_CLASS_FOLDER_MAP.keys()):
        class_dir = class_folders[folder_key]
        class_name = LC25000_CLASS_FOLDER_MAP[folder_key]
        image_paths = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS])
        print(f"{class_index} - {class_name}: {len(image_paths)} images")
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                images.append(np.asarray(image, dtype=np.uint8))
                labels.append(class_index)
            except Exception as exc:
                print(f"Skipping unreadable image: {image_path} ({exc})")
    X = np.asarray(images, dtype=np.uint8)
    y = np.asarray(labels).astype(int).flatten()
    if len(y) == 0:
        raise ValueError(f"No LC25000 images were loaded from {root_dir}")
    print("LC25000 loaded.")
    print("Samples:", len(y))
    print("Shape:", X.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for class_index, count in zip(unique, counts):
        print(f"  {class_index} - {LC25000_CLASS_NAMES[int(class_index)]}: {int(count)}")
    return X, y
