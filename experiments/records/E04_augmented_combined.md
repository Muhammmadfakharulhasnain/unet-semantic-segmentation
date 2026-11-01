# E04 Experiment Record — Augmented Combined CrossEntropy + Dice Loss

Status: Completed successfully.

## Research question

Does combining CrossEntropy and multiclass Dice Loss improve segmentation performance compared with Dice Loss alone, particularly for the boundary class?

## Hypothesis

The combined objective may preserve E03's overlap and boundary-recall gains while restoring some of the pixelwise precision and calibration encouraged by CrossEntropy. The primary hypothesis is that a 0.5/0.5 combination will improve boundary IoU and boundary precision relative to E03 without sacrificing mean IoU, mean Dice, pet IoU, or background IoU.

## Motivation from E03

E03 is currently the strongest completed experiment: mean IoU `0.659680`, mean Dice `0.780276`, boundary IoU `0.419066`, and boundary recall `0.557105`. Its boundary precision is lower at `0.628430`, indicating a measured recall/precision trade-off. E04 tests whether adding CrossEntropy to Dice can reduce boundary false positives while retaining Dice's overlap-oriented behavior.

## Combined loss formulation

The existing implementation defines:

`L_combined = 0.5 × L_CE + 0.5 × L_Dice`

`CrossEntropyLoss` receives raw model logits and integer targets directly. The Dice component is `MulticlassDiceLoss`, which applies softmax internally over the class dimension before calculating the soft multiclass Dice loss. The combined result is a scalar suitable for backpropagation.

## Controlled experimental configuration

| Parameter | E04 value |
|---|---|
| Experiment ID | E04 |
| Experiment name | `augmented_combined` |
| Architecture | U-Net |
| Input size | 128 × 128 |
| Input channels | 3 |
| Output classes | 3 |
| Base channels | 8 |
| Loss | `0.5 × CrossEntropy + 0.5 × Dice` |
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

Temporary configuration: `/tmp/month5_e04.yaml`  
Comparison baseline: `/tmp/month5_e03.yaml`

Only these experiment-level fields differ from E03:

- `experiments[0].name`: `augmented_dice` → `augmented_combined`
- `experiments[0].loss`: `dice` → `combined`

All other configuration values are intended to remain identical.

## Implementation and runtime verification

The inspected [`src/losses.py`](../../src/losses.py) implementation contains `CombinedLoss` with:

- `CrossEntropyLoss` component receiving raw logits.
- `MulticlassDiceLoss` component applying softmax internally.
- `ce_weight=0.5`.
- `dice_weight=0.5`.
- Three-class Dice configuration.

A focused runtime check passed on synthetic logits shaped `[2, 3, 8, 8]` and integer targets shaped `[2, 8, 8]`: the loss was scalar and finite, gradients were present and finite, and the selected components were `CrossEntropyLoss` and `MulticlassDiceLoss`.

## Evaluation metrics

Use the same metrics as E01–E03:

- Test loss and pixel accuracy.
- Mean IoU and mean Dice.
- Per-class precision, recall, IoU, and Dice for pet, background, and boundary.
- Training/validation loss and mean IoU by epoch.

The primary comparisons are boundary IoU and boundary precision, with boundary recall as a supporting metric. E04 should evaluate the official test set once after validation-based checkpoint selection.

## Expected observations and limitations

- The combined objective may improve the boundary precision/recall trade-off, but this is a hypothesis rather than a guaranteed outcome.
- Combined-loss values should be compared only within E04; raw loss values are not directly comparable with E01/E02 CrossEntropy or E03 Dice loss.
- Five epochs and one seed are adequate for this controlled portfolio experiment but not for statistical claims.
- The current pipeline does not save a confusion matrix or class-frequency summary, limiting diagnosis of boundary-to-pet versus boundary-to-background errors.
- Existing 128×128 resolution and the small U-Net may continue to limit thin-contour accuracy.

## Training results

Training completed for all 5 configured epochs; early stopping did not trigger.

| Epoch | Train loss | Train mIoU | Validation loss | Validation mIoU |
|---:|---:|---:|---:|---:|
| 1 | 0.5795934297 | 0.4445624283 | 0.4593490304 | 0.5547912207 |
| 2 | 0.4442813264 | 0.5721768964 | 0.4167542054 | 0.5929948190 |
| 3 | 0.4016786011 | 0.6065796324 | 0.3691405204 | 0.6293530513 |
| 4 | 0.3738699935 | 0.6272263150 | 0.3554495010 | 0.6412459398 |
| 5 | 0.3538546845 | 0.6436919248 | 0.3493212386 | 0.6414660282 |

- Best checkpoint epoch: 5
- Best validation loss: `0.3493212386`
- Training duration: `920.59` seconds
- Trainable parameters: `486571`

## Official test results

| Metric | Value |
|---|---:|
| Test loss | 0.3394808428 |
| Pixel accuracy | 0.8563372658 |
| Mean IoU | 0.6531307556 |
| Mean Dice | 0.7718987549 |
| Macro precision | 0.8010662003 |
| Macro recall | 0.7556233344 |
| Pet IoU | 0.7287672859 |
| Pet Dice | 0.8431062895 |
| Pet precision | 0.8291033721 |
| Pet recall | 0.8575903296 |
| Background IoU | 0.8450158395 |
| Background Dice | 0.9159984661 |
| Background precision | 0.8954182601 |
| Background recall | 0.9375469540 |
| Boundary IoU | 0.3856091416 |
| Boundary Dice | 0.5565915091 |
| Boundary precision | 0.6786769688 |
| Boundary recall | 0.4717327198 |

## Four-experiment comparison

| Metric | E01 | E02 | E03 | E04 |
|---|---:|---:|---:|---:|
| Pixel accuracy | 0.849045 | 0.847688 | 0.852783 | 0.856337 |
| Mean IoU | 0.632692 | 0.637779 | 0.659680 | 0.653131 |
| Mean Dice | 0.753340 | 0.758877 | 0.780276 | 0.771899 |
| Pet IoU | 0.715888 | 0.714524 | 0.719643 | 0.728767 |
| Background IoU | 0.834827 | 0.835969 | 0.840331 | 0.845016 |
| Boundary IoU | 0.347363 | 0.362843 | 0.419066 | 0.385609 |
| Boundary recall | 0.407846 | 0.447739 | 0.557105 | 0.471733 |
| Boundary precision | 0.700804 | 0.656786 | 0.628430 | 0.678677 |

Relative to E03, E04 improved pixel accuracy by `+0.003554`, pet IoU by `+0.009124`, background IoU by `+0.004685`, and boundary precision by `+0.050247`. It reduced mean IoU by `0.006549`, mean Dice by `0.008378`, boundary IoU by `0.033457`, and boundary recall by `0.085373`.

## Error analysis

See [`docs/E04_error_analysis.md`](../../docs/E04_error_analysis.md) for learning-curve and prediction observations, the boundary precision/recall trade-off, and limitations.

## Saved outputs

- Checkpoint: `checkpoints/augmented_combined.pt`
- Epoch history: `results/tables/augmented_combined_history.csv`
- Comparison CSV: `results/tables/experiment_comparison.csv`
- Comparison JSON: `results/tables/experiment_comparison.json`
- Training curves: `results/figures/augmented_combined_curves.png`
- Prediction examples: `results/figures/augmented_combined_predictions.png`
