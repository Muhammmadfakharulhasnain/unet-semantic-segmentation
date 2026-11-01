# E03 Experiment Record — Augmented Multiclass Dice Loss

Status: Completed successfully.

## Research question

Does multiclass Dice Loss improve segmentation performance compared with CrossEntropy Loss, particularly for the underperforming boundary class?

## Hypothesis

Using the existing synchronized horizontal-flip augmentation with Multiclass Dice Loss may improve overlap-oriented segmentation metrics, especially boundary IoU and boundary Dice, because Dice directly optimizes region overlap. The expected tradeoff is that pixelwise calibration or background/pet precision may differ from the CrossEntropy baseline.

## Motivation from E01 and E02

E01 CrossEntropy without augmentation achieved boundary IoU `0.347363` and boundary recall `0.407846`. E02 showed that horizontal flipping improved boundary IoU to `0.362843` and boundary recall to `0.447739`, while pet IoU decreased slightly. E03 keeps E02's augmentation and changes only the loss to investigate whether an overlap-focused objective improves the remaining boundary weakness.

## Controlled experimental configuration

| Parameter | E03 value |
|---|---|
| Experiment ID | E03 |
| Experiment name | `augmented_dice` |
| Architecture | U-Net |
| Input size | 128 × 128 |
| Input channels | 3 |
| Output classes | 3 |
| Base channels | 8 |
| Loss | Multiclass Dice Loss |
| Augmentation | Existing synchronized horizontal flip, p=0.5 |
| Batch size | 2 |
| Learning rate | 0.001 |
| Maximum epochs | 5 |
| Early-stopping patience | 2 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Device | CPU selected by `auto` |
| Random seed | 42 |
| Dataset split | Official trainval → 85% internal train / 15% internal validation; official test held out |

Temporary configuration: `/tmp/month5_e03.yaml`  
Comparison baseline: `/tmp/month5_e02.yaml`

Only these experiment-level fields differ from E02:

- `experiments[0].name`: `augmented_ce` → `augmented_dice`
- `experiments[0].loss`: `cross_entropy` → `dice`

All other configuration values are intended to remain identical.

## Implementation verification

The existing [`src/losses.py`](../../src/losses.py) implementation defines `MulticlassDiceLoss` with:

- Input logits shaped `[B, C, H, W]`.
- Target labels shaped `[B, H, W]` with integer class IDs.
- Softmax over `dim=1`, the class dimension.
- One-hot target conversion to `[B, C, H, W]`.
- Class/batch/spatial Dice aggregation over `(0, 2, 3)`.
- Scalar loss `1 - mean(Dice)` with smoothing `1e-6` and background included.

A focused runtime check passed for synthetic `[2, 3, 8, 8]` logits and `[2, 8, 8]` integer targets: the loss was finite and scalar, gradients were present and finite, and the configured class count was 3.

## Evaluation metrics

Report the same metrics used by E01 and E02:

- Test loss and pixel accuracy.
- Mean IoU and mean Dice.
- Per-class precision, recall, IoU, and Dice for pet, background, and boundary.
- Training/validation loss and mean IoU by epoch.

The primary outcome is boundary IoU, with boundary recall and boundary Dice as supporting outcomes. E03 must use the official test set only once after model selection by validation loss.

## Expected observations and limitations

- Dice Loss may improve overlap metrics for smaller or thinner regions, but this is a hypothesis rather than an expected guaranteed result.
- Dice Loss can behave differently from CrossEntropy on pixel calibration and class-frequency weighting; lower loss values across the two objectives are not directly comparable.
- The boundary class remains sensitive to small localization errors at 128×128 resolution.
- Five epochs and one seed provide a controlled comparison but not statistical evidence of robustness.
- The existing result pipeline does not save a confusion matrix or class-frequency summary, limiting diagnosis of boundary-to-pet versus boundary-to-background errors.

## Training results

Training completed for all 5 configured epochs; early stopping did not trigger.

| Epoch | Train loss | Train mIoU | Validation loss | Validation mIoU |
|---:|---:|---:|---:|---:|
| 1 | 0.3781242237 | 0.4972469786 | 0.3003406082 | 0.5699723246 |
| 2 | 0.2886024202 | 0.5803214364 | 0.2764906162 | 0.6010094012 |
| 3 | 0.2690538475 | 0.6036266805 | 0.2572513201 | 0.6210399578 |
| 4 | 0.2541874661 | 0.6222295148 | 0.2420647440 | 0.6412625542 |
| 5 | 0.2480844026 | 0.6304736114 | 0.2385231144 | 0.6475938588 |

- Best checkpoint epoch: 5
- Best validation loss: `0.2385231144`
- Training duration: `665.14` seconds
- Trainable parameters: `486571`

## Official test results

| Metric | Value |
|---|---:|
| Test loss | 0.2278057328 |
| Pixel accuracy | 0.8527833362 |
| Mean IoU | 0.6596799843 |
| Mean Dice | 0.7802763853 |
| Macro precision | 0.7882533671 |
| Macro recall | 0.7738044758 |
| Pet IoU | 0.7196429978 |
| Pet Dice | 0.8369679041 |
| Pet precision | 0.8299960647 |
| Pet recall | 0.8440578604 |
| Background IoU | 0.8403309085 |
| Background Dice | 0.9132389231 |
| Background precision | 0.9063337847 |
| Background recall | 0.9202500864 |
| Boundary IoU | 0.4190660464 |
| Boundary Dice | 0.5906223287 |
| Boundary precision | 0.6284302520 |
| Boundary recall | 0.5571054805 |

## Three-experiment comparison

| Metric | E01 | E02 | E03 |
|---|---:|---:|---:|
| Test loss | 0.391949 | 0.393926 | 0.227806 |
| Pixel accuracy | 0.849045 | 0.847688 | 0.852783 |
| Mean IoU | 0.632692 | 0.637779 | 0.659680 |
| Mean Dice | 0.753340 | 0.758877 | 0.780276 |
| Pet IoU | 0.715888 | 0.714524 | 0.719643 |
| Background IoU | 0.834827 | 0.835969 | 0.840331 |
| Boundary IoU | 0.347363 | 0.362843 | 0.419066 |
| Boundary recall | 0.407846 | 0.447739 | 0.557105 |

Relative to E02, E03 changed by `+0.021901` mean IoU, `+0.021399` mean Dice, `+0.056223` boundary IoU, `+0.109366` boundary recall, `+0.005119` pet IoU, and `+0.004362` background IoU. Pixel accuracy increased by `+0.005096`. The raw loss values are not comparable across E02 CrossEntropy and E03 Dice objectives.

## Error analysis

See [`docs/E03_error_analysis.md`](../../docs/E03_error_analysis.md) for curve and prediction observations, limitations, and research questions.

## Saved outputs

- Checkpoint: `checkpoints/augmented_dice.pt`
- Epoch history: `results/tables/augmented_dice_history.csv`
- Comparison CSV: `results/tables/experiment_comparison.csv`
- Comparison JSON: `results/tables/experiment_comparison.json`
- Training curves: `results/figures/augmented_dice_curves.png`
- Prediction examples: `results/figures/augmented_dice_predictions.png`
