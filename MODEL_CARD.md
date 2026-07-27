# Model Card: LC25000 Leakage-Safe ResNet-18

## Model overview

| Field | Value |
|---|---|
| Model identifier | `LC25000_TransferCNN_ResNet18_LeakageSafe` |
| Task | Five-class histopathology patch classification |
| Framework | PyTorch |
| Backbone | ResNet-18 |
| Input | One RGB histopathology image patch |
| Input size | `224 × 224` pixels |
| Output | Softmax probabilities across five classes |
| Status | Research prototype |
| Clinical status | Not clinically approved and not a medical device |

This model was developed to investigate leakage-aware evaluation of deep-learning
models on the LC25000 histopathology dataset. The repository emphasises saved
splits, duplicate-aware grouping, statistical evaluation, calibration,
explainability and reproducible inference.

The primary checkpoint described by this card is the leakage-safe ResNet-18 used
by the FastAPI inference service and the limited external colon validation.

## Intended use

The model is intended for:

- Machine-learning research and education
- Studying data leakage and duplicate-aware evaluation
- Reproducibility and model-validation demonstrations
- Software-engineering portfolio demonstration
- Non-clinical experimentation with histopathology image classification

## Out-of-scope use

The model must not be used for:

- Clinical diagnosis
- Patient screening or triage
- Treatment or patient-care decisions
- Autonomous pathology reporting
- Use without qualified pathological review
- Claims of validated performance on whole-slide images
- Claims of complete external validation across all five classes

## Predicted classes

The model predicts one of the following LC25000 classes:

| Index | Class |
|---:|---|
| 0 | Colon adenocarcinoma |
| 1 | Colon benign tissue |
| 2 | Lung adenocarcinoma |
| 3 | Lung benign tissue |
| 4 | Lung squamous-cell carcinoma |

## Architecture

The model uses an ImageNet-pretrained ResNet-18 backbone. Its original
classification layer is replaced with:

```text
ResNet-18 features
→ Linear(512, 256)
→ ReLU
→ Dropout(p=0.5)
→ Linear(256, 5)
```

During API startup, the architecture is created without downloading pretrained
weights because the complete trained state is restored from the project
checkpoint.

## Input preprocessing

Inference preprocessing consists of:

1. Converting the uploaded image to RGB
2. Resizing it to `224 × 224`
3. Converting it to a PyTorch tensor
4. Applying ImageNet normalisation

Normalisation values:

```text
Mean: [0.485, 0.456, 0.406]
Std:  [0.229, 0.224, 0.225]
```

The API validates both the uploaded media type and the image contents. Uploads
are limited to 10 MB.

## Training configuration

The recorded leakage-safe ResNet-18 run used:

| Setting | Value |
|---|---|
| Random seed | `42` |
| Batch size | `24` |
| Recorded epochs | `24` |
| Selection metric | Validation macro F1 |
| Loss | Cross-entropy |
| Optimiser | AdamW |
| Weight decay | `1e-4` |
| Head learning rate | `1e-3` |
| Final-block learning rate | `1e-4` |
| Early-stopping patience | `8` epochs |
| LR scheduler | ReduceLROnPlateau |
| LR reduction factor | `0.5` |
| Scheduler patience | `2` epochs |

The training implementation supports optional class weighting.

### Data augmentation

Training augmentation includes:

- Random horizontal flip with probability `0.5`
- Random rotation of up to `10` degrees
- ImageNet normalisation

Evaluation and inference do not apply random augmentation.

### Staged fine-tuning

Training proceeds through staged transfer learning:

1. Train the replacement classification head.
2. Train the final ResNet block and classification head.
3. Unfreeze the full network for discriminative fine-tuning.

The discriminative learning rates range from `1e-6` for early layers to
`1e-4` for the classification head.

## Training and evaluation data

### Primary dataset

LC25000 contains 25,000 histopathology image patches distributed across five
balanced lung and colon tissue classes.

The dataset is not redistributed in this repository. Its original source and
usage conditions remain authoritative.

### Leakage-safe split

The primary model was evaluated using an exact-duplicate-aware grouped split
labelled:

```text
leakage_safe_resized_image_hash_grouped_split
```

| Partition | Samples |
|---|---:|
| Training | 17,510 |
| Validation | 3,761 |
| Test | 3,729 |

Grouping was used to prevent matching resized-image hashes from crossing split
boundaries.

### Near-duplicate-safe companion experiment

A separate ResNet-18 was trained and evaluated using average-hash grouping:

```text
near_duplicate_safe_average_hash_grouped_split
```

| Partition | Samples |
|---|---:|
| Training | 17,480 |
| Validation | 3,745 |
| Test | 3,775 |

This result belongs to the separately trained
`TransferCNN_ResNet18_NearDuplicateSafe` model. It is not a post-hoc evaluation
of the primary leakage-safe checkpoint.

Average-hash grouping reduces one form of morphological redundancy but cannot
guarantee that all visually or biologically related patches are separated.

## Internal evaluation

### Primary leakage-safe checkpoint

| Metric | Result |
|---|---:|
| Accuracy | 0.999195 |
| Weighted precision | 0.999199 |
| Weighted recall | 0.999195 |
| Weighted F1 | 0.999195 |
| Macro F1 | 0.999196 |
| Balanced accuracy | 0.999189 |
| Macro one-vs-rest ROC-AUC | 0.999999 |
| Test samples | 3,729 |

### Near-duplicate-safe companion model

| Metric | Result |
|---|---:|
| Accuracy | 0.999205 |
| Weighted precision | 0.999206 |
| Weighted recall | 0.999205 |
| Weighted F1 | 0.999205 |
| Macro F1 | 0.999204 |
| Balanced accuracy | 0.999202 |
| Macro one-vs-rest ROC-AUC | 0.999999 |
| Test samples | 3,775 |

The very high internal results should be interpreted in the context of
LC25000's patch construction, augmentation history and visual redundancy. They
must not be treated as evidence of equivalent clinical or cross-institutional
performance.

## Calibration

Calibration analysis for the primary leakage-safe checkpoint used the 3,729
internal test images:

| Metric | Result |
|---|---:|
| Accuracy | 0.999195 |
| Mean confidence | 0.998814 |
| Expected calibration error | 0.001106 |
| Maximum calibration error | 0.459248 |
| Multiclass Brier score | 0.001502 |
| Calibration bins | 10 |

Low aggregate calibration error on the internal test set does not guarantee
reliable confidence under domain shift. Individual predictions may still be
incorrect or overconfident.

## Limited external validation

The primary leakage-safe checkpoint was evaluated on 1,974 images from the
CRC-VAL-HE-7K dataset:

| External class | Samples | LC25000 mapping |
|---|---:|---|
| `TUM` | 1,233 | Colon adenocarcinoma |
| `NORM` | 741 | Colon benign tissue |

### Strict five-class interpretation

| Metric | Result |
|---|---:|
| Accuracy | 0.8308 |
| Non-colon prediction rate | 0.0329 |

### Colon-restricted binary interpretation

| Metric | Result |
|---|---:|
| Accuracy | 0.8582 |
| Weighted precision | 0.8609 |
| Weighted recall | 0.8582 |
| Weighted F1 | 0.8545 |
| Macro F1 | 0.8418 |
| Balanced accuracy | 0.8296 |
| ROC-AUC | 0.9032 |

This experiment covers only tumour and normal colorectal tissue. It is not
external validation of the three lung-related classes or of the complete
five-class task.

The reduction from internal to external performance indicates meaningful domain
shift and illustrates why internal benchmark results alone are insufficient for
clinical claims.

## Explainability

The repository includes Grad-CAM examples for correctly classified and
misclassified LC25000 images.

These visualisations show regions that influenced model activations, but they:

- Do not prove causal reasoning
- Do not establish pathological correctness
- May highlight artefacts or dataset-specific shortcuts
- Must not be interpreted as clinical explanations without expert review

## Known limitations

- LC25000 is a patch-based rather than patient-based dataset.
- Patient identifiers are unavailable, so patient-level splitting cannot be
  verified.
- Morphologically related images may remain across partitions despite hashing.
- Perceptual hashes are imperfect proxies for biological independence.
- Dataset-specific colour, texture or acquisition shortcuts may influence the
  predictions.
- Demographic and clinical metadata are limited.
- Stain and scanner variation may reduce generalisation.
- External evaluation is restricted to two colorectal classes.
- Lung classes have not been independently externally validated.
- No whole-slide-image evaluation was performed.
- No prospective clinical evaluation was performed.
- No pathologist-reader study was conducted.
- No subgroup or demographic fairness analysis was possible with the available
  metadata.
- Internal confidence estimates may not remain calibrated under domain shift.
- The API and Docker image are research prototypes, not regulated deployment
  systems.

## Ethical and safety considerations

Histopathology predictions can affect serious medical decisions when misused.
False-negative and false-positive outputs may both cause harm.

Users must:

- Treat every prediction as an experimental model output
- Retain qualified human pathological review
- Avoid presenting probabilities as diagnostic certainty
- Evaluate performance on representative local data
- Consider staining, scanner, laboratory and population differences
- Follow applicable medical-device, privacy and data-governance requirements

The API returns the following disclaimer with every prediction:

> Research prototype only. This output is not a clinical diagnosis and must not
> be used without expert pathological review.

## Checkpoint integrity

Expected local path:

```text
models/LC25000_TransferCNN_ResNet18_LeakageSafe.pth
```

SHA-256:

```text
7e55baa06af27d2e6188933e539e0d9a759ffe6a9f0939a326055fab87adecbf
```

Verify the checkpoint with:

```bash
shasum -a 256 models/LC25000_TransferCNN_ResNet18_LeakageSafe.pth
```

The trained checkpoint is intentionally excluded from Git. The checksum
identifies the artefact used for the documented API and limited external
validation.

## Reproducibility resources

The repository provides:

- Saved leakage-safe and near-duplicate-safe split files
- Split summaries
- Per-sample prediction tables
- Classification reports
- Confusion matrices and ROC curves
- Bootstrap confidence intervals
- McNemar comparisons
- Calibration outputs
- Grad-CAM examples
- External-validation summaries
- FastAPI inference code
- A CPU-oriented Docker image
- Automated Ruff, pytest and Docker-build checks

## Licence and data rights

The repository source code is released under the MIT License.

This licence does not replace or modify the terms governing LC25000,
CRC-VAL-HE-7K, pretrained weights or other third-party artefacts. Users are
responsible for obtaining datasets and model artefacts lawfully and complying
with their original terms.

## Citation

Repository citation metadata is available in [`CITATION.cff`](CITATION.cff).
