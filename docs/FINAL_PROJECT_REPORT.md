# Month 5 U-Net Semantic Segmentation — Final Project Report

## Abstract

This project implemented and evaluated a compact U-Net for three-class semantic segmentation on the Oxford-IIIT Pet trimap masks. Four controlled experiments compared CrossEntropy, horizontal-flip augmentation, multiclass Dice Loss, and a 0.5/0.5 CrossEntropy-plus-Dice objective. All experiments used the same U-Net configuration, data split, seed, optimizer settings, and five-epoch training budget. E03, the augmented Dice-loss model, was selected as the primary model because it achieved the highest mean IoU, mean Dice, and boundary IoU. E04 achieved higher pixel accuracy, pet IoU, background IoU, and boundary precision, but its boundary recall and boundary IoU were lower than E03.

## 1. Introduction and motivation

The project investigates practical choices for multiclass semantic segmentation when the boundary class is thin and difficult to recover. The Oxford-IIIT Pet trimap contains pet, background, and boundary labels. The experimental question progressed from a CrossEntropy baseline to augmentation, Dice optimization, and a combined objective. The purpose is educational and research-oriented comparison; no statistical significance or novelty claim is made from the single-seed runs.

## 2. Dataset

The dataset is Oxford-IIIT Pet with trimap segmentation masks. Raw mask labels are mapped to contiguous model classes:

| Model class | Meaning | Raw label |
|---:|---|---:|
| 0 | Pet | 1 |
| 1 | Background | 2 |
| 2 | Boundary | 3 |

The existing extracted dataset was reused. No dataset files were modified during the experiments.

## 3. Splitting and preprocessing

The official `trainval` partition contains 3,680 samples. It was split deterministically with seed 42 into:

- 3,128 internal training samples (85%).
- 552 internal validation samples (15%).
- 0 internal test samples.
- 3,669 samples in the separate official test partition.

The official test partition was held out during model selection. Images were resized to 128×128 with bilinear interpolation, while masks used nearest-neighbor interpolation. Images were converted to tensors and normalized with ImageNet mean and standard deviation. Training augmentation, when enabled, was a synchronized horizontal flip with probability 0.5; validation and test data were not augmented.

## 4. U-Net architecture

The implementation is a compact U-Net with three input channels, three output classes, and base channel count 8. Each encoder/decoder block uses two 3×3 convolutions, batch normalization, and ReLU. Four max-pooling operations downsample the encoder. A bottleneck follows the fourth pooling stage. Four transposed-convolution upsampling stages restore resolution and concatenate encoder skip connections. A 1×1 convolution produces three raw class logits per pixel. Softmax is not applied in the model output.

All experiments had 486,571 trainable parameters.

## 5. Loss functions

- E01 and E02 used CrossEntropy on raw logits and integer target masks.
- E03 used soft multiclass Dice Loss, applying softmax internally and averaging class Dice scores.
- E04 used `0.5 × CrossEntropy + 0.5 × Dice`.

The raw loss values are not directly comparable across these different objectives. Segmentation metrics are used for cross-experiment comparison.

## 6. Training methodology

All experiments used:

- Python 3.12.10.
- PyTorch 2.2.2 and torchvision 0.17.2.
- Intel Mac CPU execution.
- AdamW with learning rate 0.001 and weight decay 0.0001.
- Batch size 2.
- Maximum five epochs and early-stopping patience 2.
- Random seed 42.
- Checkpoint selection by lowest validation loss.

The official test set was evaluated once per completed experiment after loading the validation-selected checkpoint.

## 7. Experimental design and results

| Experiment | Augmentation | Loss | Test pixel accuracy | Mean IoU | Mean Dice | Pet IoU | Background IoU | Boundary IoU | Boundary recall | Boundary precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E01 baseline_ce | No | CrossEntropy | 0.849045 | 0.632692 | 0.753340 | 0.715888 | 0.834827 | 0.347363 | 0.407846 | 0.700804 |
| E02 augmented_ce | Yes | CrossEntropy | 0.847688 | 0.637779 | 0.758877 | 0.714524 | 0.835969 | 0.362843 | 0.447739 | 0.656786 |
| E03 augmented_dice | Yes | Dice | 0.852783 | 0.659680 | 0.780276 | 0.719643 | 0.840331 | 0.419066 | 0.557105 | 0.628430 |
| E04 augmented_combined | Yes | 0.5 CE + 0.5 Dice | 0.856337 | 0.653131 | 0.771899 | 0.728767 | 0.845016 | 0.385609 | 0.471733 | 0.678677 |

Metric leaders:

- Mean IoU: E03, `0.659680`.
- Mean Dice: E03, `0.780276`.
- Boundary IoU: E03, `0.419066`.
- Pixel accuracy: E04, `0.856337`.
- Pet IoU: E04, `0.728767`.
- Background IoU: E04, `0.845016`.

## 8. Final model selection

The selected primary model is [`checkpoints/augmented_dice.pt`](../checkpoints/augmented_dice.pt). E03 is preferred because the project emphasizes overlap quality and the underperforming boundary class: it leads on mean IoU, mean Dice, and boundary IoU, and has the highest boundary recall.

E04 is retained as an important alternative. It has the best pixel accuracy, pet IoU, background IoU, and boundary precision, but lower mean IoU, mean Dice, boundary IoU, and boundary recall than E03. This demonstrates a measurable boundary precision/recall trade-off rather than a universally superior loss.

## 9. Error analysis

The saved curves show stable loss reduction and improving validation metrics for all four runs. E03's validation mIoU increased through epoch 5 to `0.647594`. E04 stabilized around `0.641466` at epochs 4–5. The saved prediction grids continue to show contour displacement, missed thin boundary regions, and errors around limbs and silhouettes.

For the selected E03 checkpoint, no test confusion matrix or raw prediction arrays were saved by the original evaluation pipeline. A diagnostic confusion matrix was therefore computed on the internal validation split only:

Rows are ground-truth classes and columns are predicted classes. Class order is pet, background, boundary.

| Ground truth \\ Prediction | Pet | Background | Boundary |
|---|---:|---:|---:|
| Pet | 2,264,488 | 228,174 | 164,641 |
| Background | 258,947 | 4,842,300 | 258,831 |
| Boundary | 251,650 | 206,124 | 568,813 |

This validation-only diagnostic gives boundary precision `0.573236`, boundary recall `0.554082`, and boundary IoU `0.392269` under the same metric implementation. It must not be confused with the official E03 test metrics. The matrix helps show that boundary errors are distributed into both neighboring classes, but it does not replace a held-out test confusion matrix.

## 10. Limitations

- Each experiment used one seed and only five epochs.
- The input resolution was limited to 128×128.
- The model is intentionally small with base channels 8.
- The boundary class remains sensitive to one- or two-pixel localization errors.
- The qualitative prediction figures contain only two examples per experiment.
- The original test evaluation did not save a test confusion matrix, class-frequency table, or per-image metrics.
- The comparison is controlled but not a statistical study; differences should not be described as statistically significant.
- The internal validation split is fixed for this project but is still a single random split.

## 11. Conclusion and future work

The project is complete as a controlled four-experiment study. E03 is the primary model for the research report because it provides the best mean IoU, mean Dice, and boundary IoU. E04 provides a useful alternative when pixel accuracy, pet IoU, background IoU, or boundary precision is prioritized.

Future work, outside this closed experimentation phase, could include repeated seeds, longer training, higher resolution, confusion-matrix logging during official evaluation, class-frequency analysis, boundary-aware objectives, or tuning the CE/Dice weighting. These are research questions, not completed experiments in this project.

## References to project artifacts

- Four-experiment metrics: [`../results/tables/experiment_comparison.csv`](../results/tables/experiment_comparison.csv)
- Four-experiment JSON: [`../results/tables/experiment_comparison.json`](../results/tables/experiment_comparison.json)
- E03 record: [`../experiments/records/E03_augmented_dice.md`](../experiments/records/E03_augmented_dice.md)
- E03 error analysis: [`E03_error_analysis.md`](E03_error_analysis.md)
- E04 error analysis: [`E04_error_analysis.md`](E04_error_analysis.md)
