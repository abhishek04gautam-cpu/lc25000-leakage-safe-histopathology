"""
download_lc25000_dataset.py

Downloads the LC25000 Lung and Colon Histopathological Image Dataset using
kagglehub and copies it into:

    project_root/datasets/lc25000

Expected final structure includes folders such as:
    colon_aca
    colon_n
    lung_aca
    lung_n
    lung_scc
"""

from pathlib import Path
import shutil
import sys


try:
    import kagglehub
except ImportError:
    print("Missing package: kagglehub")
    print("Install it with:")
    print("    pip install kagglehub")
    sys.exit(1)


# Project root = parent of src/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
LC25000_DIR = DATA_DIR / "lc25000"

# KaggleHub dataset slug
LC25000_SLUG = "andrewmvd/lung-and-colon-cancer-histopathological-images"

# Set to True if you want to delete the old lc25000 folder before downloading
CLEAR_BEFORE_DOWNLOAD = True


def clear_dir(path: Path) -> None:
    """Delete and recreate a directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def copy_downloaded_files(downloaded_path: Path, target_dir: Path) -> None:
    """
    Copy files/folders from kagglehub cache into the project dataset folder.
    """
    for item in downloaded_path.iterdir():
        target = target_dir / item.name

        if target.exists():
            print(f"Skipping existing: {target}")
            continue

        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def verify_lc25000_structure(root_dir: Path) -> None:
    """
    Check whether the five expected LC25000 class folders are present.
    """
    expected_folders = [
        "colon_aca",
        "colon_n",
        "lung_aca",
        "lung_n",
        "lung_scc",
    ]

    found = {}
    for folder_name in expected_folders:
        matches = [path for path in root_dir.rglob(folder_name) if path.is_dir()]
        found[folder_name] = matches[0] if matches else None

    print("\nLC25000 folder check:")
    all_found = True

    for folder_name, path in found.items():
        if path is None:
            print(f"  Missing: {folder_name}")
            all_found = False
        else:
            image_count = len(
                [
                    file
                    for file in path.iterdir()
                    if file.is_file()
                    and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                ]
            )
            print(f"  Found: {folder_name} -> {path} ({image_count} images)")

    if not all_found:
        raise FileNotFoundError(
            "\nLC25000 downloaded, but one or more expected class folders were not found."
        )

    print("\nDataset structure looks correct.")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAR_BEFORE_DOWNLOAD:
        print(f"Clearing existing dataset folder: {LC25000_DIR}")
        clear_dir(LC25000_DIR)
    else:
        ensure_dir(LC25000_DIR)

    print(f"\nDownloading LC25000 from KaggleHub:")
    print(f"  {LC25000_SLUG}")

    downloaded_path = Path(kagglehub.dataset_download(LC25000_SLUG))

    print(f"\nDownloaded to KaggleHub cache:")
    print(f"  {downloaded_path}")

    print(f"\nCopying dataset into:")
    print(f"  {LC25000_DIR}")

    copy_downloaded_files(downloaded_path, LC25000_DIR)

    verify_lc25000_structure(LC25000_DIR)

    print("\nLC25000 download finished successfully.")
    print(f"Dataset location: {LC25000_DIR}")


if __name__ == "__main__":
    main()