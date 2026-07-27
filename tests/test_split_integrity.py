"""Integrity checks for persisted train, validation and test splits."""

from pathlib import Path

import numpy as np
import pytest


SAVED_SPLITS_DIR = Path("saved_splits")
SPLIT_FILES = sorted(SAVED_SPLITS_DIR.rglob("*.npz"))


@pytest.mark.parametrize("split_file", SPLIT_FILES)
def test_saved_split_indices_do_not_overlap(split_file: Path) -> None:
    """Train, validation and test indices must be mutually exclusive."""

    split = np.load(split_file)

    required_keys = {"train_idx", "val_idx", "test_idx"}
    assert required_keys.issubset(split.files), (
        f"{split_file} does not contain all required index arrays. "
        f"Available keys: {split.files}"
    )

    train_indices = set(split["train_idx"].tolist())
    validation_indices = set(split["val_idx"].tolist())
    test_indices = set(split["test_idx"].tolist())

    assert train_indices
    assert validation_indices
    assert test_indices

    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)


def test_at_least_one_saved_split_is_available() -> None:
    """The repository should contain at least one persisted split file."""

    assert SPLIT_FILES, f"No .npz split files found under {SAVED_SPLITS_DIR}"
