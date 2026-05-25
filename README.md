# LC25000 Leakage-Safe Histopathology Benchmark

This repository contains the dissertation code, experimental pipelines, saved dataset splits, and evaluation outputs for leakage-aware deep learning experiments on the LC25000 histopathology dataset.

The project investigates how duplicate and near-duplicate image patches can inflate evaluation performance in histopathology classification benchmarks.

---

# Repository Contents

The repository includes:

- Deep learning baselines
- Classical machine learning baselines
- Leakage auditing
- Duplicate and near-duplicate detection
- Leakage-safe dataset splitting
- Near-duplicate-safe grouped splitting
- Calibration analysis
- Robustness evaluation
- Statistical significance testing
- Bootstrap confidence intervals
- Grad-CAM explainability analysis

All source files are organized under:

```text
src/
```

---

# Expected Dataset Layout

Put the LC25000 Kaggle dataset under:

```text
lc25000_project/datasets/lc25000/
```

The dataset loader searches recursively for:

```text
colon_aca
colon_n
lung_aca
lung_n
lung_scc
```

---

# Data Leakage and Near-Duplicate Mitigation

Histopathology patch datasets may contain duplicated or highly similar image regions across train and test splits.

This can artificially inflate evaluation metrics.

To reduce optimistic bias, this repository implements:

1. Exact duplicate detection using SHA256 hashes
2. Resized-image duplicate detection
3. Perceptual average-hash near-duplicate grouping
4. Group-aware train/validation/test splitting

The repository therefore reports:

- Original split performance
- Leakage-safe performance
- Near-duplicate-safe performance

---

# Main Findings

Key observations from the leakage audit:

- 277 exact train–test overlaps were detected in the original LC25000 split
- 543 highly similar near-duplicate relationships were identified
- Leakage-safe evaluation slightly reduced performance
- Near-duplicate-safe evaluation still achieved approximately 99.92% macro-F1 using ResNet18
- Brightness perturbations produced the largest robustness degradation

These findings suggest that LC25000 remains highly separable even after duplicate mitigation, although patch-level redundancy may still exist.

---

# Suggested Run Order

From the project root:

```bash
cd lc25000_project
```

Install dependencies:

```bash
python -m pip install numpy pandas pillow scikit-learn matplotlib torch torchvision tqdm opencv-python fastapi uvicorn statsmodels
```

Compile-check all source files:

```bash
python -m py_compile $(find src -name "*.py")
```

Run experiments:

```bash
# Original split experiments
python src/main.py

# Classical baseline
python src/lc25000_classical_baseline.py

# Leakage audit
python src/lc25000_leakage_audit_v2.py

# Leakage-safe split
python src/lc25000_leakage_safe_split.py

# Near-duplicate-safe split
python src/lc25000_near_duplicate_safe_split.py

# Leakage-safe evaluations
python src/run_lc25000_leakage_safe_evaluation.py --model logistic
python src/run_lc25000_leakage_safe_evaluation.py --model simplecnn
python src/run_lc25000_leakage_safe_evaluation.py --model resnet18
python src/run_lc25000_leakage_safe_evaluation.py --model headonly
python src/run_lc25000_leakage_safe_evaluation.py --model densenet121
python src/run_lc25000_leakage_safe_evaluation.py --model efficientnetb0

# Near-duplicate-safe evaluation
python src/run_lc25000_near_duplicate_safe_evaluation.py --model resnet18

# Statistical analysis
python src/statistical_comparison_leakage_safe.py
python src/bootstrap_ci.py

# Confidence and calibration analysis
python src/confidence_analysis.py
python src/calibration_analysis.py
python src/temperature_scaling.py

# Additional analysis
python src/imbalance_audit.py
python src/gradcam_lc25000.py
python src/lc25000_sample_grid.py
python src/generate_lc25000_dissertation_figures.py
python src/model_efficiency_analysis.py
```

---

# Reproducibility

All experiments were executed using fixed random seeds.

Saved split index files are included to support deterministic experiment reproduction.

---

# Notes

Two syntax issues from the dissertation appendix text were corrected during repository preparation:

1. `run_lc25000_leakage_safe_evaluation.py`
   - The final `main()` call required indentation under:
   ```python
   if __name__ == "__main__":
   ```

2. `temperature_scaling.py`
   - Missing closing bracket:
   ```python
   y_test = dataset.y[test_idx]
   ```

A syntax compile check was successfully executed across all Python source files.

---

# Clinical Disclaimer

This repository is intended for research purposes only.

The models and evaluation results presented here are not approved for clinical deployment and should not be interpreted as diagnostic systems for real-world pathology workflows.

---

# Ethical Considerations

Important limitations include:

- Potential dataset bias
- Patch-level redundancy
- Limited demographic diversity
- Absence of patient-level evaluation
- Unknown external generalization capability
- Risk of shortcut learning from duplicated tissue regions

Even near-duplicate-safe evaluation may not fully eliminate morphological redundancy within LC25000.

---

# Limitations

Although duplicate mitigation was applied, the dataset remains patch-based rather than patient-based.

Therefore:

- Morphological overlap may still exist
- External cohort generalization remains unverified
- Very high accuracy values should be interpreted cautiously

Future work should include:

- External dataset validation
- Patient-level splitting
- Cross-dataset transfer evaluation
- Stain normalization experiments

---

# License

This repository is provided for academic and research purposes.