# E02 Experiment Record — Augmented Cross-Entropy

Status: Completed successfully.

## Research hypothesis

Adding the existing synchronized random horizontal-flip augmentation to the E01 baseline may improve generalization to pet pose and left-right variation. The primary hypothesis is that augmentation will improve official test mean IoU, with particular interest in boundary IoU and boundary recall. Because the augmentation changes only the training images/masks and leaves the objective and model unchanged, any measured difference can be attributed more directly to the augmentation policy, subject to random variation.

## Controlled comparison

E02 should differ from E01 only in `augmentation`:

| Field | E01 baseline | Proposed E02 |
|---|---|---|
| Experiment ID | E01 | E02 |
| Experiment name | `baseline_ce` | `augmented_ce` |
| Architecture | U-Net | U-Net |
| Input size | 128 × 128 | 128 × 128 |
| Input channels | 3 | 3 |
| Output classes | 3 | 3 |
| Base channels | 8 | 8 |
| Loss | CrossEntropy | CrossEntropy |
| Augmentation | Disabled | Enabled: random horizontal flip, p=0.5 |
| Batch size | 2 | 2 |
| Learning rate | 0.001 | 0.001 |
| Maximum epochs | 5 | 5 |
| Early-stopping patience | 2 | 2 |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.0001 | 0.0001 |
| Device | CPU | CPU |
| Random seed | 42 | 42 |
| Dataset split | Official trainval → 85% train / 15% validation; official test held out | Same |
| Evaluation metrics | Pixel accuracy, macro IoU/Dice, per-class metrics | Same |

The proposed experiment should use a temporary configuration selecting only `augmented_ce`; `configs/default.yaml` should remain unchanged.

## Current augmentation implementation

In [`src/data.py`](../../src/data.py), every sample is first resized to the configured square size:

- Image: `BILINEAR` interpolation.
- Mask: `NEAREST` interpolation.

When `augment=True`, one call to `random.random()` is made. If it is less than `0.5`, `TF.hflip` is applied to both the image and mask. The training dataset receives `augment_train=exp["augmentation"]`; validation and official test datasets are constructed with `augment=False`.

## Synchronization and label safety

The image and mask share the same flip decision and both are flipped before conversion of the mask into contiguous class IDs. This preserves image-mask alignment for the supported transformation. Nearest-neighbor mask resizing preserves discrete raw labels; bilinear interpolation is used only for RGB images. The existing defensive mask-label check rejects unexpected raw labels.

The split logic remains unchanged: the official trainval partition is split into 85% internal training and 15% internal validation using seed 42, while the official test partition remains separate and unaugmented.

## Augmentation risks

- **Horizontal flip:** Usually reasonable for pet segmentation because the semantic labels are left-right invariant. It can nevertheless alter natural asymmetry, text or directional context in backgrounds, and pose-specific appearance statistics.
- **Randomness:** The flip is sampled at each `__getitem__` call, so the same training image can be flipped on one access and not flipped on another. The experiment runner seeds Python's `random` module, but exact reproducibility still depends on the loader configuration and random-access order.
- **Limited policy:** The current policy provides only left-right variation. It does not address scale, crop, illumination, blur, rotation, or background variation.
- **No geometric interpolation risk for masks:** The supported flip itself is label-preserving, and mask resizing is nearest-neighbor. More geometric transforms must not use bilinear or bicubic interpolation on masks.
- **Distribution mismatch risk:** If the dataset contains meaningful directional cues, flipping may create less realistic samples and could reduce performance.
- **Boundary sensitivity:** Flipping should preserve boundary geometry, but it cannot by itself correct annotation ambiguity or recover boundary detail lost during 128x128 resizing.

## Acceptance and comparison criteria

Compare E02 with E01 using the same reported metrics and seed. The primary comparison is boundary IoU, with boundary recall as a diagnostic. Also report mean IoU, mean Dice, pixel accuracy, pet IoU, and background IoU. A single-seed improvement should be treated as preliminary; repeated seeds would be needed to establish robustness.

## Saved outputs

- `checkpoints/augmented_ce.pt`
- `results/tables/augmented_ce_history.csv`
- `results/figures/augmented_ce_curves.png`
- `results/figures/augmented_ce_predictions.png`
- Updated `results/tables/experiment_comparison.csv`

## Training results

Training completed for all 5 configured epochs; early stopping did not trigger.

| Epoch | Train loss | Train mIoU | Validation loss | Validation mIoU |
|---:|---:|---:|---:|---:|
| 1 | 0.6772056781 | 0.4085573615 | 0.5199327895 | 0.5298110389 |
| 2 | 0.5209376223 | 0.5458379736 | 0.4652263833 | 0.5839944282 |
| 3 | 0.4616646943 | 0.5889306191 | 0.4150262843 | 0.6193919921 |
| 4 | 0.4261128135 | 0.6123440808 | 0.4047571817 | 0.6206499326 |
| 5 | 0.4054642522 | 0.6265512297 | 0.4036858295 | 0.6267084025 |

- Best checkpoint epoch: 5
- Best validation loss: `0.4036858295`
- Training duration: `726.46` seconds
- Trainable parameters: `486571`

## Official test results

| Metric | Value |
|---|---:|
| Test loss | 0.3939262371 |
| Pixel accuracy | 0.8476875079 |
| Mean IoU | 0.6377788711 |
| Mean Dice | 0.7588775000 |
| Macro precision | 0.7868163499 |
| Macro recall | 0.7448547821 |
| Pet IoU | 0.7145240594 |
| Pet Dice | 0.8334955179 |
| Pet precision | 0.8078258465 |
| Pet recall | 0.8608501009 |
| Background IoU | 0.8359692740 |
| Background Dice | 0.9106571508 |
| Background precision | 0.8958376567 |
| Background recall | 0.9259751985 |
| Boundary IoU | 0.3628432798 |
| Boundary Dice | 0.5324798312 |
| Boundary precision | 0.6567855464 |
| Boundary recall | 0.4477390468 |

## E01 comparison

| Metric | E01 | E02 | Difference (E02 − E01) |
|---|---:|---:|---:|
| Test loss | 0.3919490123 | 0.3939262371 | +0.0019772248 |
| Pixel accuracy | 0.8490446875 | 0.8476875079 | -0.0013571797 |
| Mean IoU | 0.6326923763 | 0.6377788711 | +0.0050864948 |
| Mean Dice | 0.7533400132 | 0.7588775000 | +0.0055374868 |
| Pet IoU | 0.7158876896 | 0.7145240594 | -0.0013636302 |
| Background IoU | 0.8348267988 | 0.8359692740 | +0.0011424752 |
| Boundary IoU | 0.3473626404 | 0.3628432798 | +0.0154806394 |
| Boundary recall | 0.4078462685 | 0.4477390468 | +0.0398927783 |

The detailed interpretation is in [`docs/E02_error_analysis.md`](../../docs/E02_error_analysis.md).
