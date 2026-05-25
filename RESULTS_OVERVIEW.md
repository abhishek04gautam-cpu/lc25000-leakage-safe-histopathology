# Results Overview

This repository contains supplementary code, saved split files, figures, and result summaries for a leakage-safe evaluation study on LC25000 histopathology classification.

## Main Evaluation Protocols

The repository includes results for:

1. Initial stratified saved split
2. Leakage-safe split using resized-image hash grouping
3. Near-duplicate-safe split using perceptual average-hash grouping
4. Test-time robustness evaluation
5. Calibration and confidence analysis
6. Grad-CAM explainability
7. Limited external colon-only validation on CRC-VAL-HE-7K

## Key Findings

- The initial LC25000 split contained exact train-test image overlaps.
- The leakage audit detected 277 exact train-test resized-image overlaps.
- The near-duplicate audit identified 543 highly similar train-test relationships at average-hash Hamming distance <= 2.
- Leakage-safe ResNet18 evaluation achieved approximately 0.9992 macro-F1 on LC25000.
- Near-duplicate-safe ResNet18 evaluation achieved approximately 0.9992 macro-F1.
- Brightness reduction produced the largest observed robustness degradation.
- External CRC-VAL-HE-7K colon-only validation showed a substantial generalization drop, with colon-restricted binary accuracy of approximately 0.858 and ROC-AUC of approximately 0.903.

## External Validation

The external validation experiment uses CRC-VAL-HE-7K classes:

- TUM mapped to tumour / LC25000 colon adenocarcinoma
- NORM mapped to normal / LC25000 colon benign tissue

This is a limited colon-only external validation, not a full five-class external validation of LC25000.

## Important Files

- `results_summary/leakage_safe/`
- `results_summary/near_duplicate_safe/`
- `results_summary/robustness/`
- `results_summary/external_validation/`
- `saved_splits/`
- `leakage_audit/`
- `figures/`

## Dataset Availability

Datasets are not redistributed in this repository. Users should download LC25000 and CRC-VAL-HE-7K from their original public sources.
