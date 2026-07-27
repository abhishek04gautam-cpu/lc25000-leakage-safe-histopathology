# Dataset Card: LC25000 and Limited CRC-VAL-HE-7K Validation

## Overview

This repository uses two public histopathology datasets:

1. **LC25000** as the primary five-class training and internal-evaluation dataset.
2. **CRC-VAL-HE-7K** for a limited, colon-only external-validation experiment.

The datasets are not redistributed in this repository. Users must obtain them
from their original sources and comply with the applicable data terms.

## Primary dataset: LC25000

| Field | Value |
|---|---|
| Dataset | Lung and Colon Cancer Histopathological Image Dataset (LC25000) |
| Primary task in this project | Five-class image-patch classification |
| Distributed image count | 25,000 |
| Distributed image format | JPEG |
| Distributed image size | `768 × 768` pixels |
| Classes | 5 |
| Images per class | 5,000 |
| Project input size | `224 × 224` RGB |
| Local source used by setup script | KaggleHub dataset `andrewmvd/lung-and-colon-cancer-histopathological-images` |
| Redistribution in this repository | No |

### Classes

| Project index | Folder | Class |
|---:|---|---|
| 0 | `colon_aca` | Colon adenocarcinoma |
| 1 | `colon_n` | Colon benign tissue |
| 2 | `lung_aca` | Lung adenocarcinoma |
| 3 | `lung_n` | Lung benign tissue |
| 4 | `lung_scc` | Lung squamous-cell carcinoma |

### Provenance

The LC25000 publication describes 25,000 de-identified, HIPAA-compliant,
validated colour images across five balanced histological classes.

The Kaggle distribution used by this repository states that the 25,000 images
were generated from 1,250 source images:

- 750 lung-tissue images
- 500 colon-tissue images

The source images were augmented to 25,000 using the `Augmentor` package.

This augmentation history is important when interpreting model performance.
Randomly separating augmented relatives can produce overly optimistic estimates
when closely related image patches cross dataset partitions.

### Source references

- Borkowski, A. A., Bui, M. M., Thomas, L. B., Wilson, C. P., DeLand, L. A.,
  and Mastorides, S. (2019). *Lung and Colon Cancer Histopathological Image
  Dataset (LC25000)*.
  https://arxiv.org/abs/1912.12142
- Kaggle distribution used by the setup script:
  https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

## Dataset acquisition

The repository provides:

```bash
python src/download_lc25000_dataset.py
```

The script downloads:

```text
andrewmvd/lung-and-colon-cancer-histopathological-images
```

through KaggleHub and copies the downloaded contents to:

```text
datasets/lc25000/
```

It then recursively verifies that all five expected class folders are present.

### Destructive-download warning

The current setup script defines:

```python
CLEAR_BEFORE_DOWNLOAD = True
```

Running the script therefore deletes an existing local
`datasets/lc25000/` directory before downloading. Users should preserve any
local modifications or derived files elsewhere before running it.

## Loading and preprocessing

The LC25000 loader:

- Searches recursively for the five expected class folders
- Accepts `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif` and `.tiff`
- Sorts image paths deterministically within each class
- Converts every image to RGB
- Resizes every image to `224 × 224`
- Stores loaded pixels as unsigned 8-bit arrays
- Skips unreadable files and reports them

Training-time preprocessing then applies:

- Random horizontal flip with probability `0.5`
- Random rotation up to `10` degrees
- Tensor conversion
- ImageNet normalisation

Validation, testing and API inference use deterministic resizing, tensor
conversion and ImageNet normalisation without random augmentation.

## Initial leakage audit

The repository preserves an audit of an earlier LC25000 train-test allocation:

| Audit field | Result |
|---|---:|
| Training samples audited | 17,500 |
| Test samples audited | 3,750 |
| Exact file-hash overlaps between train and test | 277 |
| Exact resized-image-hash overlaps between train and test | 277 |
| Nearest pairs with average-hash Hamming distance ≤ 2 | 543 |
| Nearest pairs with average-hash Hamming distance ≤ 5 | 987 |
| Hamming-distance ≤ 2 pairs from the same class | 543 |
| Hamming-distance ≤ 2 cross-class pairs | 0 |
| Minimum observed Hamming distance | 0 |
| Median minimum Hamming distance | 11.0 |
| Mean minimum Hamming distance | 9.8256 |

These findings motivated the grouped splitting procedures used for the reported
experiments.

## Exact-duplicate-aware leakage-safe split

The primary leakage-safe split groups images using SHA-256 hashes of their
resized `224 × 224` RGB pixel arrays.

All images with the same resized-image hash remain in one partition. The script
also raises an error if an identical resized-image hash appears with multiple
class labels.

Group-level splitting is:

- Stratified by class
- Deterministic with random seed `42`
- Approximately `70%` training, `15%` validation and `15%` test
- Checked explicitly for zero hash-group overlap between all partitions

### Saved split counts

| Partition | Colon adenocarcinoma | Colon benign | Lung adenocarcinoma | Lung benign | Lung SCC | Total |
|---|---:|---:|---:|---:|---:|---:|
| Training | 3,513 | 3,499 | 3,492 | 3,501 | 3,505 | 17,510 |
| Validation | 750 | 753 | 755 | 748 | 755 | 3,761 |
| Test | 737 | 748 | 753 | 751 | 740 | 3,729 |

Protocol label:

```text
leakage_safe_resized_image_hash_grouped_split
```

Resized-image hashing prevents exact equality after project preprocessing from
crossing partitions. It does not establish patient-level independence or remove
all visually related images.

## Near-duplicate-aware split

The stricter companion split combines two grouping rules:

1. Exact equality of resized `224 × 224` RGB arrays, identified by SHA-256.
2. Average-hash similarity within the same class.

The average hash is produced by:

- Converting the image to greyscale
- Resizing it to `8 × 8`
- Comparing each pixel with the image mean
- Flattening the result into a 64-bit binary representation

Images within the same class are connected when their average-hash Hamming
distance is at most `2`. Exact and near-duplicate relationships are merged into
connected components using union-find, and every component remains in one
partition.

The grouped split is stratified, uses seed `42`, follows an approximate
`70/15/15` allocation and checks for zero group overlap.

### Saved near-duplicate-safe split counts

| Partition | Colon adenocarcinoma | Colon benign | Lung adenocarcinoma | Lung benign | Lung SCC | Total |
|---|---:|---:|---:|---:|---:|---:|
| Training | 3,509 | 3,496 | 3,500 | 3,480 | 3,495 | 17,480 |
| Validation | 746 | 747 | 753 | 755 | 744 | 3,745 |
| Test | 745 | 757 | 747 | 765 | 761 | 3,775 |

Protocol label:

```text
near_duplicate_safe_average_hash_grouped_split
```

Average-hash grouping is a heuristic. A threshold of `2` reduces one measurable
form of visual redundancy but cannot guarantee biological, slide-level or
patient-level independence.

## External validation dataset: CRC-VAL-HE-7K

| Field | Value |
|---|---|
| Dataset | CRC-VAL-HE-7K |
| Full dataset size | 7,180 image patches |
| Reported patients | 50 |
| Patch size | `224 × 224` pixels |
| Resolution | `0.5` microns per pixel |
| Use in this project | Limited colon-only external validation |
| Images used in this project | 1,974 |
| Redistribution in this repository | No |

The official Zenodo record describes CRC-VAL-HE-7K as a validation set from
50 patients with colorectal adenocarcinoma, with no patient overlap with the
associated NCT-CRC-HE-100K training dataset.

Source:

- Kather, J. N., Halama, N., and Marx, A. (2018).
  *100,000 Histological Images of Human Colorectal Cancer and Healthy Tissue*.
  https://doi.org/10.5281/zenodo.1214456

### Classes used by this project

Only two CRC-VAL-HE-7K classes were used:

| External class | Samples | Project mapping |
|---|---:|---|
| `TUM` | 1,233 | LC25000 colon adenocarcinoma |
| `NORM` | 741 | LC25000 colon benign tissue |
| **Total** | **1,974** | — |

This experiment is not a full five-class external validation. It does not
validate the three lung-related LC25000 outputs.

## Intended use

The documented datasets and saved splits are intended for:

- Machine-learning research and education
- Histopathology image-patch classification experiments
- Data-leakage and near-duplicate analysis
- Reproducibility studies
- Evaluation of calibration, robustness and domain shift
- Non-clinical software-engineering demonstrations

## Out-of-scope use

They must not be treated as sufficient evidence for:

- Clinical diagnosis or screening
- Patient-level performance claims
- Whole-slide-image performance claims
- Deployment in patient-care workflows
- Demographic or clinical subgroup performance
- Complete cross-institutional validation
- Full external validation of the five LC25000 classes

## Known limitations and risks

- LC25000 is composed of image patches rather than patient-level cases.
- Patient identifiers are unavailable in the project copy.
- The distributed 25,000-image collection was created through augmentation from
  a much smaller set of source images.
- Exact and perceptual hashing cannot prove biological independence.
- Connected or morphologically related patches may remain after grouping.
- Images can contain staining, scanner, preparation or compression shortcuts.
- Balanced image counts do not imply balanced patient representation.
- Demographic and clinical metadata are limited or unavailable.
- The project cannot perform patient-level, demographic or subgroup fairness
  analysis on LC25000.
- Training images are resized from `768 × 768` to `224 × 224`, which discards
  spatial detail.
- Horizontal flips and rotations may not reproduce the full range of real
  laboratory variation.
- CRC-VAL-HE-7K differs from LC25000 in source, staining and task structure.
- Only `TUM` and `NORM` were used from CRC-VAL-HE-7K.
- External results therefore assess limited colorectal transfer, not full model
  generalisation.
- No prospective clinical evaluation or pathologist-reader study was performed.

## Privacy and ethics

The LC25000 publication describes the released images as de-identified and
HIPAA compliant. The CRC-VAL-HE-7K Zenodo record describes retrospective,
anonymised archival material and provides its ethics information.

Public availability does not remove the need for responsible use. Users should:

- Avoid attempting re-identification
- Follow the original dataset terms and institutional requirements
- Keep predictions within research and educational contexts
- Retain qualified pathological review for any medically relevant analysis
- Avoid presenting benchmark accuracy as clinical validity

## Licensing and attribution

The repository source code is licensed under the MIT License.

Dataset rights are separate:

- The Kaggle LC25000 distribution used by this project's download script is
  marked **CC BY-SA 4.0** on Kaggle.
- CRC-VAL-HE-7K users must follow the terms and citation guidance on its Zenodo
  record.
- Pretrained model weights and other third-party artefacts may have additional
  terms.

The repository's MIT licence does not relicense any dataset, image, pretrained
weight or external artefact. Users are responsible for verifying the current
source terms before redistribution or commercial use.

## Reproducibility artefacts

The repository includes:

- `saved_splits/LC25000_leakage_safe_split.npz`
- `saved_splits/LC25000_leakage_safe_split_summary.csv`
- `saved_splits/near_duplicate_safe/LC25000_near_duplicate_safe_split.npz`
- `saved_splits/near_duplicate_safe/LC25000_near_duplicate_safe_split_summary.csv`
- Leakage-audit summaries and detailed outputs
- Per-sample prediction tables
- Internal and external evaluation summaries
- Split-integrity tests

These artefacts preserve the project partitioning and evaluation evidence but
do not redistribute the underlying image datasets.

## Related documentation

- [README](README.md)
- [Model card](MODEL_CARD.md)
- [Results overview](RESULTS_OVERVIEW.md)
- [Citation metadata](CITATION.cff)
