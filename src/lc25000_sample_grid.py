"""Create a representative LC25000 sample-image grid."""
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from config import DISSERTATION_FIGURES_DIR, LC25000_CLASS_FOLDER_MAP, LC25000_PATH, RANDOM_STATE
from data.lc25000_loader import VALID_IMAGE_EXTENSIONS, find_class_folders
OUTPUT_DIR=DISSERTATION_FIGURES_DIR; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
def collect_class_images(class_folder): return sorted([p for p in class_folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS])
def main():
    random.seed(RANDOM_STATE); root=Path(LC25000_PATH)
    if not root.exists(): raise FileNotFoundError(f"LC25000 path not found: {root}")
    folders=find_class_folders(root); n_rows=len(LC25000_CLASS_FOLDER_MAP); n_cols=5; fig,axes=plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(12,10))
    for r,(folder_key,class_name) in enumerate(LC25000_CLASS_FOLDER_MAP.items()):
        paths=collect_class_images(folders[folder_key])
        if len(paths)<n_cols: raise ValueError(f"Class folder {folders[folder_key]} contains only {len(paths)} valid images.")
        for c,image_path in enumerate(random.sample(paths,n_cols)):
            ax=axes[r,c]; ax.imshow(Image.open(image_path).convert("RGB")); ax.axis("off")
            if c==0: ax.set_ylabel(class_name, fontsize=10, rotation=0, labelpad=55, va="center")
    plt.suptitle("Representative LC25000 Histopathology Images by Class", fontsize=14); plt.tight_layout(); out=OUTPUT_DIR / "figure_lc25000_sample_grid.png"; plt.savefig(out,dpi=300,bbox_inches="tight"); plt.close(); print(f"Saved sample grid to: {out}")
if __name__ == "__main__": main()
