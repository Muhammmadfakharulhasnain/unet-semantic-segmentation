# Month 5 — U-Net Semantic Segmentation

A research-oriented PyTorch project for three-class semantic segmentation on the Oxford-IIIT Pet trimap masks. The project implements a compact U-Net, validates multiclass losses and metrics, runs four controlled experiments, and documents the resulting error patterns.

The final selected model is **E03: augmented U-Net with multiclass Dice Loss** because it achieved the highest Mean IoU, Mean Dice, and Boundary IoU.

## Research motivation and objectives

The project investigates how augmentation and loss-function choice affect semantic segmentation when the boundary class is thin and difficult to recover.

Objectives:

- Implement and validate a U-Net from basic PyTorch layers.
- Segment pet, background, and boundary regions.
- Compare Cross-Entropy, Dice, and combined CrossEntropy + Dice losses.
- Investigate whether horizontal-flip augmentation improves generalization.
- Evaluate boundary precision, recall, IoU, and Dice rather than relying only on pixel accuracy.
- Conduct a reproducible, single-seed experimental comparison.

This is an educational and research-portfolio project. The results are not statistically conclusive and make no novelty or clinical claims.

## Project highlights

- Compact U-Net with 486,571 trainable parameters.
- Three-class Oxford-IIIT Pet trimap segmentation.
- Deterministic 85%/15% internal train/validation split.
- Official test set held out for final evaluation.
- Synchronized image/mask horizontal flips.
- NumPy and PyTorch metric implementations with cross-implementation tests.
- Four completed experiments, E01–E04.
- Primary model: [`checkpoints/augmented_dice.pt`](checkpoints/augmented_dice.pt) when checkpoints are available locally.
- Full technical report: [`docs/FINAL_PROJECT_REPORT.md`](docs/FINAL_PROJECT_REPORT.md).
- Completion checklist: [`docs/PROJECT_COMPLETION_CHECKLIST.md`](docs/PROJECT_COMPLETION_CHECKLIST.md).

## Dataset: Oxford-IIIT Pet

The project uses the Oxford-IIIT Pet dataset with pixel-level trimap segmentation masks. The raw labels are mapped to contiguous model class IDs:

| Model class | Meaning | Raw mask label |
|---:|---|---:|
| 0 | Pet | 1 |
| 1 | Background | 2 |
| 2 | Boundary | 3 |

This is semantic segmentation, not binary foreground-only segmentation.

The dataset is downloaded or reused through `torchvision.datasets.OxfordIIITPet`. The dataset itself is excluded by `.gitignore` and is not required to be committed to GitHub.

## Dataset preparation and preprocessing

The official `trainval` partition contains 3,680 samples and is split using seed 42:

- 3,128 internal training samples — 85%.
- 552 internal validation samples — 15%.
- 0 internal test samples.
- 3,669 separate official test samples.

The official test partition remains untouched during model selection.

Preprocessing:

- Images resized to `128 × 128` with bilinear interpolation.
- Masks resized to `128 × 128` with nearest-neighbor interpolation.
- Images converted to tensors.
- Images normalized with ImageNet mean and standard deviation.
- Training augmentation, when enabled, is a synchronized horizontal flip with probability `0.5`.
- Validation and test data are not augmented.

## U-Net architecture

The implementation in [`src/model.py`](src/model.py) is a compact U-Net:

- Input channels: 3.
- Output classes: 3.
- Base channels: 8.
- Four encoder stages with max-pooling.
- Bottleneck after the fourth downsampling stage.
- Four transposed-convolution upsampling stages.
- Skip connections concatenate encoder features with decoder features.
- Each block contains two 3×3 convolutions, batch normalization, and ReLU.
- A 1×1 convolution produces three raw logits per pixel.
- Softmax is not applied by the model.

All four experiments used the same architecture and had 486,571 trainable parameters.

## Loss functions

### Cross-Entropy

Cross-Entropy receives raw logits and integer target masks. It is effective for per-pixel classification and provides well-established gradients, but it can under-emphasize small or thin classes when pixel frequencies are imbalanced.

### Multiclass Dice Loss

Dice Loss applies softmax internally, converts targets to one-hot form, computes class-wise soft overlap, and averages the class Dice scores. It directly emphasizes region overlap and can help smaller or thin regions such as boundaries.

### Combined Loss

E04 uses the existing implementation:

```text
L_combined = 0.5 × L_CrossEntropy + 0.5 × L_Dice
```

Cross-Entropy receives raw logits, while the Dice component applies softmax internally. Raw loss values from different objectives should not be compared directly.

## Evaluation metrics

The project reports:

- **Pixel Accuracy:** fraction of correctly classified pixels.
- **Mean IoU:** macro-average of per-class Intersection over Union.
- **Mean Dice:** macro-average of per-class Dice coefficients.
- **Per-class IoU:** overlap quality for pet, background, and boundary.
- **Per-class Dice:** overlap-sensitive F1-style score for each class.
- **Precision:** proportion of predicted class pixels that are correct.
- **Recall:** proportion of ground-truth class pixels recovered.
- **Boundary precision and recall:** specifically track the precision/recall trade-off for the thin boundary class.

The metric implementations are in [`src/metrics.py`](src/metrics.py), with tests in [`tests/test_metrics.py`](tests/test_metrics.py).

## Experimental methodology

All experiments used:

- Python 3.12.10.
- PyTorch 2.2.2.
- torchvision 0.17.2.
- Intel Mac CPU execution.
- AdamW, learning rate `0.001`, weight decay `0.0001`.
- Batch size `2`.
- Maximum `5` epochs.
- Early-stopping patience `2`.
- Random seed `42`.
- Checkpoint selected using the lowest validation loss.

The official test set was evaluated once after model selection for each completed experiment.

## Experiment comparison

Percentages below are the recorded values converted to percentages for readability. The full-precision results are in [`results/tables/experiment_comparison.csv`](results/tables/experiment_comparison.csv) and [`results/tables/experiment_comparison.json`](results/tables/experiment_comparison.json) when generated artifacts are available locally.

| Metric | E01 Baseline CE | E02 Augmented CE | E03 Augmented Dice | E04 Augmented Combined |
|---|---:|---:|---:|---:|
| Pixel Accuracy | 84.90% | 84.77% | 85.28% | **85.63%** |
| Mean IoU | 63.27% | 63.78% | **65.97%** | 65.31% |
| Mean Dice | 75.33% | 75.89% | **78.03%** | 77.19% |
| Pet IoU | 71.59% | 71.45% | 71.96% | **72.88%** |
| Background IoU | 83.48% | 83.60% | 84.03% | **84.50%** |
| Boundary IoU | 34.74% | 36.28% | **41.91%** | 38.56% |
| Boundary recall | 40.78% | 44.77% | **55.71%** | 47.17% |
| Boundary precision | 70.08% | 65.68% | 62.84% | **67.87%** |

Metric leaders:

- Mean IoU: E03 — `0.659680`.
- Mean Dice: E03 — `0.780276`.
- Boundary IoU: E03 — `0.419066`.
- Pixel accuracy: E04 — `0.856337`.
- Pet IoU: E04 — `0.728767`.
- Background IoU: E04 — `0.845016`.

## Results and model selection

The selected primary model is E03:

[`checkpoints/augmented_dice.pt`](checkpoints/augmented_dice.pt)

E03 is selected because the project prioritizes overlap quality and the difficult boundary class. It achieved the highest Mean IoU, Mean Dice, Boundary IoU, and Boundary recall.

E04 is retained as an alternative. It achieved the best pixel accuracy, pet IoU, background IoU, and boundary precision. However, E04's Boundary IoU and Boundary recall were lower than E03, showing that improved boundary precision came with reduced boundary recovery.

## Visual predictions and training curves

### Dataset sample

![Dataset sample visualization](results/figures/dataset_sample_visualization.png)

### E01 — Baseline Cross-Entropy

![E01 training curves](results/figures/baseline_ce_curves.png)

![E01 prediction comparisons](results/figures/baseline_ce_predictions.png)

### E02 — Augmented Cross-Entropy

![E02 training curves](results/figures/augmented_ce_curves.png)

![E02 prediction comparisons](results/figures/augmented_ce_predictions.png)

### E03 — Augmented Dice Loss

![E03 training curves](results/figures/augmented_dice_curves.png)

![E03 prediction comparisons](results/figures/augmented_dice_predictions.png)

### E04 — Augmented Combined Loss

![E04 training curves](results/figures/augmented_combined_curves.png)

![E04 prediction comparisons](results/figures/augmented_combined_predictions.png)

## Error analysis

The saved prediction grids show that all models recover the main pet regions but continue to make errors around thin contours, legs, ears, and lower silhouettes.

E03 provides the strongest boundary overlap and recall, but its boundary precision is lower than E01 and E04. This indicates more aggressive boundary recovery with additional false positives.

E04 increases boundary precision relative to E03 but loses boundary recall and Boundary IoU. The result is a measurable precision/recall trade-off, not a universally better model.

No official-test confusion matrices or raw per-image prediction arrays were saved by the original runs. A validation-only diagnostic matrix for E03 is documented in [`docs/FINAL_PROJECT_REPORT.md`](docs/FINAL_PROJECT_REPORT.md); it must not be interpreted as an official test result.

Detailed experiment analyses:

- [`docs/E01_error_analysis.md`](docs/E01_error_analysis.md)
- [`docs/E02_error_analysis.md`](docs/E02_error_analysis.md)
- [`docs/E03_error_analysis.md`](docs/E03_error_analysis.md)
- [`docs/E04_error_analysis.md`](docs/E04_error_analysis.md)

## Repository structure

```text
month5_unet_segmentation/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/default.yaml
├── src/
│   ├── data.py
│   ├── engine.py
│   ├── evaluate.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── run_experiments.py
│   └── visualize.py
├── tests/
│   ├── test_core.py
│   ├── test_losses.py
│   └── test_metrics.py
├── notebooks/
│   ├── 00_dataset_and_segmentation_fundamentals.ipynb
│   ├── 01_unet_from_scratch.ipynb
│   ├── 02_training_and_experiments.ipynb
│   └── 03_evaluation_and_error_analysis.ipynb
├── experiments/records/
│   ├── E01_baseline_ce.md
│   ├── E02_augmented_ce.md
│   ├── E03_augmented_dice.md
│   └── E04_augmented_combined.md
├── checkpoints/
├── results/tables/
├── results/figures/
└── docs/
```

The dataset, virtual environments, checkpoints, caches, generated histories, comparison tables, and PNG figures are excluded by [`.gitignore`](.gitignore) where appropriate. Large datasets and model files may be distributed separately rather than committed to GitHub.

## Installation and setup

The completed project was validated on an Intel Mac using Python 3.12.10 and a dedicated `.venv-pytorch` environment. The pinned PyTorch installation used for the reported experiments was:

```bash
cd /path/to/month5_unet_segmentation
python3.12 -m venv .venv-pytorch
source .venv-pytorch/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2
python -m pip install -r requirements.txt
```

The `requirements.txt` file contains the project dependencies. If installing it after the pinned PyTorch packages, keep the PyTorch and torchvision versions pinned for compatibility with the reported Intel-macOS experiments. Do not commit `.venv/` or `.venv-pytorch/`.

The alternative [`environment.yml`](environment.yml) is also provided for Conda-based setup, but the reported experiments used the `.venv-pytorch` environment above.

## Running tests

From the repository root with the environment activated:

```bash
pytest -q
```

The test suite covers core model/loss/metric behavior, loss gradients, metric edge cases, and NumPy/PyTorch metric consistency.

## Running experiments

The default experiment matrix is defined in [`configs/default.yaml`](configs/default.yaml). To run the configured matrix:

```bash
PYTHONPATH=. python -m src.run_experiments --config configs/default.yaml
```

The runner writes checkpoints to `checkpoints/`, histories and comparisons to `results/tables/`, and curves/prediction grids to `results/figures/`. Running an experiment can overwrite shared generated comparison files, so preserve existing experiment-specific artifacts and use isolated configurations when extending the study.

For a saved checkpoint evaluation, the existing entry point is:

```bash
PYTHONPATH=. python -m src.evaluate \
  --config configs/default.yaml \
  --checkpoint checkpoints/augmented_dice.pt \
  --loss dice
```

This evaluates the official test loader; use it only when a new evaluation is intended. The final reported results in this README are the recorded results from the closed E01–E04 study.

## Limitations

- One random seed was used per experiment.
- Each experiment ran for at most five epochs.
- The input resolution was limited to 128×128.
- The U-Net is intentionally compact with base channels 8.
- The boundary class is sensitive to small localization errors.
- Qualitative figures show only two examples per experiment.
- No statistical significance or multi-seed variance estimates are available.
- No official-test confusion matrices or raw per-image outputs were saved.

## Future research directions

Potential future work includes:

- Repeating the experiments across multiple seeds and reporting mean ± standard deviation.
- Training for longer with a larger early-stopping budget.
- Testing higher input resolution.
- Saving official-test confusion matrices and per-image metrics.
- Investigating class weighting or explicit boundary-aware objectives.
- Tuning the CrossEntropy/Dice weighting in the combined objective.
- Evaluating whether the observed boundary precision/recall trade-off is stable across pet poses, scales, and backgrounds.

The four-experiment phase is closed; these are future research questions, not completed experiments in this repository.

## Acknowledgements and dataset citation

This project uses the Oxford-IIIT Pet dataset and its trimap segmentation annotations through torchvision. Please follow the dataset's attribution and usage requirements when redistributing data or derived assets. The U-Net implementation is informed by:

> Ronneberger, O., Fischer, P., and Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.

Paper: [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

## Additional documentation

- [`docs/FINAL_PROJECT_REPORT.md`](docs/FINAL_PROJECT_REPORT.md)
- [`docs/PROJECT_COMPLETION_CHECKLIST.md`](docs/PROJECT_COMPLETION_CHECKLIST.md)
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
- [`docs/segmentation_metrics.md`](docs/segmentation_metrics.md)
- [`docs/segmentation_losses.md`](docs/segmentation_losses.md)
