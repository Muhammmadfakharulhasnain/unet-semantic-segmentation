# Project Completion Checklist

## Completed components

- [x] Dedicated Python 3.12.10 environment configured in `.venv-pytorch`.
- [x] PyTorch 2.2.2 and torchvision 0.17.2 verified.
- [x] Oxford-IIIT Pet loader validated with local extracted data.
- [x] Official trainval → 85% internal train / 15% validation split documented.
- [x] Official test partition kept separate during model selection.
- [x] U-Net architecture validated with three-class raw-logit output.
- [x] CrossEntropy, multiclass Dice, and combined losses implemented and gradient-tested.
- [x] NumPy and PyTorch segmentation metrics implemented and cross-tested.
- [x] E01 baseline completed and documented.
- [x] E02 augmentation experiment completed and documented.
- [x] E03 Dice-loss experiment completed and documented.
- [x] E04 combined-loss experiment completed and documented.
- [x] Primary model selected: `checkpoints/augmented_dice.pt`.
- [x] Validation-only E03 diagnostic confusion matrix computed.
- [x] Final comparison CSV and JSON contain E01, E02, E03, and E04.
- [x] Final project report created.

## Verified artifacts

- [x] Four checkpoints:
  - `checkpoints/baseline_ce.pt`
  - `checkpoints/augmented_ce.pt`
  - `checkpoints/augmented_dice.pt`
  - `checkpoints/augmented_combined.pt`
- [x] Four training-history CSV files.
- [x] Four experiment records in `experiments/records/`.
- [x] Four error-analysis documents in `docs/`.
- [x] `results/tables/experiment_comparison.csv` with four rows.
- [x] `results/tables/experiment_comparison.json` with four records.
- [x] Eight experiment figures: one curve figure and one prediction figure per experiment.
- [x] E01–E04 experiment-specific checkpoints and histories preserved.

## Final model decision

- Primary model: `checkpoints/augmented_dice.pt`.
- Selection basis: highest mean IoU, mean Dice, boundary IoU, and boundary recall.
- Alternative: `checkpoints/augmented_combined.pt`, which has higher pixel accuracy, pet IoU, background IoU, and boundary precision but lower boundary IoU and recall.

## Known limitations

- Single random seed per experiment.
- Five-epoch training budget.
- 128×128 input resolution and compact U-Net capacity.
- No original official-test confusion matrices or raw per-image prediction arrays were saved.
- Qualitative figures show only two examples per experiment.
- No statistical significance testing or multi-seed variance estimates.

## Remaining packaging tasks

- [ ] Optionally add a reproducible environment lock file or verified requirements export.
- [ ] Optionally add a README section linking the final report and primary checkpoint.
- [ ] Optionally add a small inference command/example for loading the selected checkpoint.
- [ ] Optionally add a license and dataset attribution notice appropriate for the portfolio repository.
- [ ] Optionally archive the validation-only diagnostic confusion matrix as a dedicated table.

No additional experiment is required for project closure. E05 and further training remain outside the closed experimentation phase.
