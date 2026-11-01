# E03 Augmented Dice-Loss Error Analysis

Experiment: `E03` (`augmented_dice`)  
E01 baseline: [`../experiments/records/E01_baseline_ce.md`](../experiments/records/E01_baseline_ce.md)  
E02 baseline: [`../experiments/records/E02_augmented_ce.md`](../experiments/records/E02_augmented_ce.md)  
E03 record: [`../experiments/records/E03_augmented_dice.md`](../experiments/records/E03_augmented_dice.md)  
Three-experiment table: [`../results/tables/experiment_comparison.csv`](../results/tables/experiment_comparison.csv)

E03 kept E02's horizontal-flip augmentation and changed only the loss from CrossEntropy to multiclass Dice Loss. The dataset, model, split, seed, and training hyperparameters were unchanged. The checkpoint and dataset were not modified, and E04 was not started.

## Learning curves

![E03 learning curves](../results/figures/augmented_dice_curves.png)

Saved history: [`augmented_dice_history.csv`](../results/tables/augmented_dice_history.csv).

- Training Dice loss decreased from `0.378124` to `0.248084`.
- Validation Dice loss decreased from `0.300341` to `0.238523`.
- Validation mIoU increased every epoch: `0.569972`, `0.601009`, `0.621040`, `0.641263`, `0.647594`.
- Validation mIoU continued improving through epoch 5; it did not show a plateau in this five-epoch run.
- The best validation-loss checkpoint was epoch 5.

Raw E03 Dice-loss values must not be compared numerically with E01/E02 CrossEntropy-loss values because they represent different objectives. The comparable quantities are the segmentation metrics.

## Official test metrics

| Class | Precision | Recall | IoU | Dice |
|---|---:|---:|---:|---:|
| Pet | 0.829996 | 0.844058 | 0.719643 | 0.836968 |
| Background | 0.906334 | 0.920250 | 0.840331 | 0.913239 |
| Boundary | 0.628430 | 0.557105 | 0.419066 | 0.590622 |

Overall test results were pixel accuracy `0.852783`, mean IoU `0.659680`, mean Dice `0.780276`, and Dice loss `0.227806`.

## Comparison with E01 and E02

| Metric | E01 baseline CE | E02 augmented CE | E03 augmented Dice | E03 − E02 |
|---|---:|---:|---:|---:|
| Pixel accuracy | 0.849045 | 0.847688 | 0.852783 | +0.005096 |
| Mean IoU | 0.632692 | 0.637779 | 0.659680 | +0.021901 |
| Mean Dice | 0.753340 | 0.758877 | 0.780276 | +0.021399 |
| Pet IoU | 0.715888 | 0.714524 | 0.719643 | +0.005119 |
| Background IoU | 0.834827 | 0.835969 | 0.840331 | +0.004362 |
| Boundary IoU | 0.347363 | 0.362843 | 0.419066 | +0.056223 |
| Boundary recall | 0.407846 | 0.447739 | 0.557105 | +0.109366 |

Relative to E01, E03 improved mean IoU by `0.026988`, mean Dice by `0.026936`, boundary IoU by `0.071703`, and boundary recall by `0.149259`. E03 also had the highest pet IoU, background IoU, and pixel accuracy of the three runs.

Measured results therefore support the hypothesis for this controlled single-seed comparison: Dice Loss substantially improved the boundary metrics while also improving the overall overlap metrics.

## Prediction inspection

![E03 saved predictions](../results/figures/augmented_dice_predictions.png)

The two saved examples show that the main pet regions are detected, but errors remain:

- The first prediction contains broad contour disagreement, irregular boundary regions, and isolated predicted fragments outside the main object.
- The second prediction recovers the main body but differs around the head, limbs, lower contour, and thin boundary regions.
- Pixel-error panels still show substantial edge and protrusion errors, so Dice Loss did not eliminate contour displacement or missed boundary pixels.

These are observations from the two saved examples only. They do not establish that every test image improved.

## Boundary analysis

Boundary IoU increased from `0.347363` in E01 and `0.362843` in E02 to `0.419066` in E03. Boundary recall increased from `0.407846` and `0.447739` to `0.557105`. Boundary precision decreased to `0.628430`, so the recall improvement came with more boundary false positives, but the net IoU and Dice results improved substantially.

This pattern is consistent with Dice Loss encouraging overlap recovery for the thin boundary class. That mechanism is a hypothesis about why the result occurred; the saved outputs do not include a confusion matrix to identify whether boundary false negatives were assigned mainly to pet or background.

## Limitations and unresolved questions

- E03 uses one random seed and five epochs; repeated seeds are needed to assess robustness.
- The qualitative figure contains only two examples.
- The three runs are controlled but not a statistical study.
- Dice and CrossEntropy loss values are not directly comparable.
- No per-class confusion matrix or class-frequency summary was saved.
- Boundary errors remain visible despite the metric improvement.
- The 128×128 resolution and small U-Net capacity may still limit thin-contour recovery.

## Questions for E04

1. Does a combined CrossEntropy + Dice objective preserve E03's boundary gains while improving boundary precision?
2. Are the E03 improvements stable across multiple seeds?
3. Which boundary confusions dominate after Dice training?
4. Would higher input resolution improve the remaining contour errors?
