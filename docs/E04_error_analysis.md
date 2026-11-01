# E04 Augmented Combined-Loss Error Analysis

Experiment: `E04` (`augmented_combined`)  
E03 baseline: [`../experiments/records/E03_augmented_dice.md`](../experiments/records/E03_augmented_dice.md)  
Four-experiment table: [`../results/tables/experiment_comparison.csv`](../results/tables/experiment_comparison.csv)

E04 kept the E03 model, augmentation, data split, seed, resolution, optimizer settings, and training schedule. It changed only the loss to `0.5 × CrossEntropy + 0.5 × Dice`. The dataset and prior experiment-specific artifacts were preserved, and no further experiment was started.

## Learning curves

![E04 learning curves](../results/figures/augmented_combined_curves.png)

Saved history: [`augmented_combined_history.csv`](../results/tables/augmented_combined_history.csv).

- Combined training loss decreased from `0.579593` to `0.353855`.
- Combined validation loss decreased from `0.459349` to `0.349321`.
- Validation mIoU increased from `0.554791` to `0.641466`.
- Validation mIoU improved through epoch 4 and essentially stabilized between epochs 4 (`0.641246`) and 5 (`0.641466`).
- E04's final validation mIoU was below E03's final validation mIoU (`0.647594`) but above E01 (`0.625063`) and E02 (`0.626708`).

Raw E04 combined-loss values are not directly comparable with E01/E02 CrossEntropy or E03 Dice-loss values.

## Official test metrics

| Class | Precision | Recall | IoU | Dice |
|---|---:|---:|---:|---:|
| Pet | 0.829103 | 0.857590 | 0.728767 | 0.843106 |
| Background | 0.895418 | 0.937547 | 0.845016 | 0.915998 |
| Boundary | 0.678677 | 0.471733 | 0.385609 | 0.556592 |

Overall test results were pixel accuracy `0.856337`, mean IoU `0.653131`, mean Dice `0.771899`, and combined loss `0.339481`.

## Four-experiment comparison

| Metric | E01 baseline CE | E02 augmented CE | E03 augmented Dice | E04 combined | E04 − E03 |
|---|---:|---:|---:|---:|---:|
| Pixel accuracy | 0.849045 | 0.847688 | 0.852783 | 0.856337 | +0.003554 |
| Mean IoU | 0.632692 | 0.637779 | 0.659680 | 0.653131 | -0.006549 |
| Mean Dice | 0.753340 | 0.758877 | 0.780276 | 0.771899 | -0.008378 |
| Pet IoU | 0.715888 | 0.714524 | 0.719643 | 0.728767 | +0.009124 |
| Background IoU | 0.834827 | 0.835969 | 0.840331 | 0.845016 | +0.004685 |
| Boundary IoU | 0.347363 | 0.362843 | 0.419066 | 0.385609 | -0.033457 |
| Boundary recall | 0.407846 | 0.447739 | 0.557105 | 0.471733 | -0.085373 |
| Boundary precision | 0.700804 | 0.656786 | 0.628430 | 0.678677 | +0.050247 |

E04 achieved the best pixel accuracy, pet IoU, and background IoU. E03 remained best on mean IoU, mean Dice, boundary IoU, and boundary recall. The combined loss partially restored boundary precision relative to E03, but it did not preserve E03's boundary recall or boundary IoU.

## Boundary precision/recall trade-off

Compared with E03, E04 boundary precision increased from `0.628430` to `0.678677`, while boundary recall decreased from `0.557105` to `0.471733`. Boundary IoU consequently fell from `0.419066` to `0.385609`.

This is a measured precision/recall rebalancing, not evidence that E04 reduced all boundary false positives: the aggregate metrics do not include a confusion matrix. The result is consistent with the combined objective making boundary predictions more selective, but the exact error destinations cannot be identified from the saved outputs.

## Prediction inspection

![E04 saved predictions](../results/figures/augmented_combined_predictions.png)

The two saved examples show:

- The main pet regions are detected in both examples.
- Contour displacement and missed thin regions remain visible around legs, head/ears, and lower silhouettes.
- The first prediction contains broad object-edge disagreement and isolated small predictions away from the main region.
- The second prediction captures the dominant body but differs around the head and lower contour; the error panel still contains substantial edge errors.

These observations are limited to the two saved visualization examples and should not be generalized to every test image without per-image analysis.

## Conclusion and limitations

For this single-seed, five-epoch comparison, the combined loss did not outperform Dice Loss on overall overlap or boundary segmentation. It improved boundary precision and produced the highest pixel accuracy, pet IoU, and background IoU, but E03's Dice-only objective produced higher mean IoU, mean Dice, boundary IoU, and boundary recall.

The result does not establish that the combined loss is inferior in general. The study uses one seed, a short schedule, 128×128 inputs, and a small U-Net. A confusion matrix, class-frequency summary, repeated seeds, and longer training would be needed to determine whether the precision/recall trade-off is robust.

## Future research questions

1. Would another CE/Dice weighting improve the boundary precision/recall balance?
2. Would class weighting or an explicit boundary-aware term preserve E03 recall while improving E04 precision?
3. Are the E03/E04 differences stable across multiple random seeds?
4. Which boundary confusions dominate after combined-loss training?
