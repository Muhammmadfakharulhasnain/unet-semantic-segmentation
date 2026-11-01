# Experimental Protocol

## Research question
How do data augmentation and loss-function choice affect U-Net semantic segmentation on Oxford-IIIT Pet trimap masks?

## Dataset protocol
- Use Oxford-IIIT Pet.
- Use the official `trainval` partition for a fixed 85/15 internal train/validation split.
- Use the official `test` partition as the held-out test set; it is not included in the internal split.
- With the current Oxford-IIIT Pet metadata, the internal split contains 3,128 training samples and 552 validation samples from 3,680 `trainval` samples; the official test partition contains 3,669 samples.
- Apply identical resizing and normalization in all runs.
- Use nearest-neighbor interpolation for segmentation masks.

## Controlled variables
- Same model architecture and base channel count.
- Same image size and batch size.
- Same fixed train/validation/test data.
- Same optimizer and learning rate.
- Same maximum epochs and early-stopping patience.
- Same random seed for the default comparison.

## Independent variables
1. Training augmentation: off/on.
2. Loss function: Cross-Entropy, Dice, Combined.

## Dependent variables
- Mean IoU over all three classes.
- Mean IoU over pet and boundary classes (reported separately from the all-class mean).
- Mean Dice.
- Pixel accuracy.
- Macro precision and recall.
- Per-class IoU, Dice, precision and recall.
- Training time and best epoch.

## Experiment matrix
1. baseline_ce: no augmentation + Cross-Entropy.
2. augmented_ce: horizontal flip + Cross-Entropy.
3. augmented_dice: horizontal flip + Dice.
4. augmented_combined: horizontal flip + 0.5 CE + 0.5 Dice.

## Model selection
Choose the checkpoint with the lowest validation loss. Do not select checkpoints using test metrics.

## Reporting
- Report actual test metrics from `experiment_comparison.csv`.
- Include training/validation curves.
- Include prediction grids and failure cases.
- Record runtime, device, package versions, seed and configuration.
- Do not claim that differences are statistically significant from a single run.
- For stronger evidence, repeat each experiment with at least three seeds and report mean ± standard deviation.

## Threats to validity
- Small dataset and limited training budget.
- One seed in the default run.
- Reduced input resolution.
- Class imbalance, especially for the narrow boundary class.
- The random internal split may vary if the seed changes.
