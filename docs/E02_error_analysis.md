# E02 Augmented Cross-Entropy Error Analysis

Experiment: `E02` (`augmented_ce`)  
Baseline: [`../experiments/records/E01_baseline_ce.md`](../experiments/records/E01_baseline_ce.md)  
E02 record: [`../experiments/records/E02_augmented_ce.md`](../experiments/records/E02_augmented_ce.md)  
Comparison table: [`../results/tables/experiment_comparison.csv`](../results/tables/experiment_comparison.csv)

E02 changed only the existing training-time random horizontal flip relative to E01. The model, CrossEntropy loss, image size, batch size, learning rate, epoch limit, patience, seed, data split, and metrics were unchanged. The dataset and checkpoints were not modified, and E03 was not started.

## Learning curves

![E02 learning curves](../results/figures/augmented_ce_curves.png)

Saved history: [`augmented_ce_history.csv`](../results/tables/augmented_ce_history.csv).

- Training loss decreased from `0.677206` to `0.405464`.
- Validation loss decreased from `0.519933` to `0.403686`.
- Validation mIoU increased from `0.529811` to `0.626708`.
- Validation mIoU improved strongly through epoch 3, changed only slightly at epoch 4, and increased again slightly at epoch 5. It did not show a clear decline during this five-epoch run.
- E02's final validation mIoU (`0.626708`) was slightly above E01's final validation mIoU (`0.625063`), while E02's final validation loss (`0.403686`) was higher than E01's (`0.395379`). These are different signals and should not be treated as identical selection criteria.

The run completed all five epochs; the best validation-loss checkpoint was epoch 5.

## Official test metrics

| Class | Precision | Recall | IoU | Dice |
|---|---:|---:|---:|---:|
| Pet | 0.807826 | 0.860850 | 0.714524 | 0.833496 |
| Background | 0.895838 | 0.925975 | 0.835969 | 0.910657 |
| Boundary | 0.656786 | 0.447739 | 0.362843 | 0.532480 |

Overall test metrics were pixel accuracy `0.847688`, mean IoU `0.637779`, mean Dice `0.758877`, and loss `0.393926`.

## E01 versus E02

| Metric | E01 | E02 | Difference |
|---|---:|---:|---:|
| Test loss | 0.391949 | 0.393926 | +0.001977 |
| Pixel accuracy | 0.849045 | 0.847688 | -0.001357 |
| Mean IoU | 0.632692 | 0.637779 | +0.005086 |
| Mean Dice | 0.753340 | 0.758877 | +0.005537 |
| Pet IoU | 0.715888 | 0.714524 | -0.001364 |
| Background IoU | 0.834827 | 0.835969 | +0.001142 |
| Boundary IoU | 0.347363 | 0.362843 | +0.015481 |
| Boundary recall | 0.407846 | 0.447739 | +0.039893 |

Measured changes indicate that augmentation improved mean IoU, mean Dice, background IoU, boundary IoU, and boundary recall. Pixel accuracy, test loss, and pet IoU were slightly worse. The largest improvement was boundary recall, followed by boundary IoU.

The per-class precision/recall tradeoff is also important:

- Pet recall increased (`+0.028268`), but pet precision decreased (`-0.028446`), producing a small pet-IoU decrease.
- Background precision increased (`+0.023912`), while background recall decreased (`-0.025530`); background IoU and Dice nevertheless increased slightly.
- Boundary recall increased (`+0.039893`), while boundary precision decreased (`-0.044019`). The net boundary IoU and Dice improvements show that the recall gain outweighed the additional boundary false positives for this run.

## Prediction inspection

![E02 saved predictions](../results/figures/augmented_ce_predictions.png)

The saved figure contains two input/ground-truth/prediction/pixel-error examples. In both examples, the main pet region is detected, but contour differences remain visible:

- The first prediction has visible disagreement around the outer silhouette, lower body/legs, and thin contour regions. The error panel contains broad object-edge error regions and some isolated errors.
- The second prediction recovers the dominant body but differs around the head/legs and lower contour. Thin boundary regions remain visibly imperfect.

These examples support the conclusion that augmentation did not eliminate contour displacement or missed boundary regions. They do not by themselves prove that every test image improved, because only the saved examples are shown.

## Interpretation and limitations

The measured results support a modest positive answer to the research question for boundary performance: horizontal flipping improved boundary IoU by `0.015481` and boundary recall by `0.039893` on the official test set under this single seed. It did not improve every metric, and the test loss and pixel accuracy decreased slightly.

The results do not establish that flipping alone caused a statistically robust improvement. E01 and E02 each use one seed and five epochs. The qualitative figure contains only two examples. No confusion matrix was saved, so the exact destinations of boundary errors remain unknown. The result table also does not quantify class frequencies or per-image variance.

## Research questions for the next experiment

1. Does the combined CrossEntropy + Dice objective improve boundary IoU without reducing pet IoU?
2. Are the E02 boundary gains stable across multiple random seeds?
3. Does a saved confusion matrix show that boundary false negatives are primarily assigned to pet or background?
4. Would higher input resolution improve the remaining thin-contour errors?
5. Are the apparent pet recall/precision tradeoffs consistent across the test set or driven by a small number of images?
