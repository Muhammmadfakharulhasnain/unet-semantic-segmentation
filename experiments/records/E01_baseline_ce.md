# E01 Experiment Record — Baseline Cross-Entropy

Status: Completed successfully.

## Experiment definition

| Field | Value |
|---|---|
| Experiment ID | E01 |
| Experiment name | baseline_ce |
| Architecture | U-Net |
| Input size | 128 × 128 |
| Input channels | 3 |
| Output classes | 3 |
| Base channels | 8 |
| Loss | CrossEntropy |
| Augmentation | Disabled |
| Batch size | 2 |
| Learning rate | 0.001 |
| Maximum epochs | 5 |
| Early-stopping patience | 2 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Selected device | CPU |
| Configured device mode | `auto` |
| Random seed | 42 |

## Dataset split

- Dataset: Oxford-IIIT Pet
- Source partition: official `trainval`
- Internal training fraction: 85%
- Internal validation fraction: 15%
- Internal test fraction: 0%
- Official test partition: separate and held out
- Verified counts: 3,128 train, 552 validation, 3,669 official test
- Local extracted dataset reused; no download was triggered during verification

## Environment and hardware

| Field | Observed value |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.2.2 |
| torchvision | 0.17.2 |
| NumPy | 1.26.4 |
| Architecture | x86_64 |
| Hardware model identifier | MacBookPro15,1 |
| Logical CPU count | 16 |
| CUDA available | False |
| MPS available | True (reported by this PyTorch build) |
| Device selected by `auto` | CPU |

## Reproducibility notes

- E01 uses the temporary configuration `/tmp/month5_e01.yaml`.
- The tracked `configs/default.yaml` was not modified.
- The temporary configuration contains only `baseline_ce`.
- Training completed for all 5 configured epochs; early stopping did not trigger.
- Best epoch: 5 (selected by validation loss).
- Training time: 985.0 seconds.
- Final validation loss: 0.39537858898225037.
- Official test loss: 0.3919490123150881.
- Official test pixel accuracy: 0.84904468751597.
- Official test mean IoU: 0.6326923762829235.
- Official test mean Dice: 0.753340013217597.
- The implementation selects the best checkpoint by validation loss.

## Saved outputs

- Checkpoint: `checkpoints/baseline_ce.pt`
- Epoch history: `results/tables/baseline_ce_history.csv`
- Experiment comparison: `results/tables/experiment_comparison.csv`
- JSON comparison: `results/tables/experiment_comparison.json`
- Training curves: `results/figures/baseline_ce_curves.png`
- Prediction examples: `results/figures/baseline_ce_predictions.png`
- Error analysis: [`docs/E01_error_analysis.md`](../../docs/E01_error_analysis.md)
