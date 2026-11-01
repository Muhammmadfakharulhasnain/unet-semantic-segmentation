# Month 5 — Four-Week Execution Plan

## Week 1 — Segmentation foundations and dataset

Learn:
- Classification vs detection vs semantic segmentation vs instance segmentation.
- Pixel masks, class IDs and dense prediction.
- FCN overview.
- Oxford-IIIT Pet annotations and trimap labels.

Deliverables:
- Finish Notebook 00.
- Inspect at least 12 image/mask pairs.
- Write a one-page explanation of the three mask classes.
- Verify mask labels are exactly the expected categories.

Checkpoint questions:
1. What does one output pixel represent?
2. Why must masks use nearest-neighbor interpolation?
3. Why can pixel accuracy be misleading?

## Week 2 — U-Net architecture from scratch

Learn:
- Convolution blocks.
- Max pooling and spatial resolution.
- Bottleneck.
- Transposed convolution.
- Skip connections and concatenation.
- Output logits.

Deliverables:
- Finish Notebook 01.
- Run a forward pass and record every tensor shape.
- Calculate trainable parameter count.
- Explain the U-Net diagram in your own words.

Checkpoint:
Input `[B,3,128,128]` must produce logits `[B,3,128,128]`.

## Week 3 — Training and controlled experiments

Learn:
- Cross-Entropy.
- Dice coefficient and Dice loss.
- Combined loss.
- Optimizer, learning rate, epochs, early stopping.
- Train/validation/test separation.

Deliverables:
- Finish Notebook 02.
- Run all four configured experiments.
- Save histories and checkpoints.
- Confirm that CSVs contain actual measured values.

## Week 4 — Evaluation, error analysis and reporting

Learn:
- Confusion matrix.
- Per-class IoU and Dice.
- Precision and recall.
- Qualitative evaluation and failure analysis.

Deliverables:
- Finish Notebook 03.
- Complete the comparison table.
- Save prediction grids and training curves.
- Analyze at least 10 failed or difficult examples.
- Complete the final report and README.
- Prepare a 3–5 minute project explanation.

## End-of-month acceptance criteria

- [ ] Can explain semantic segmentation without notes.
- [ ] Can draw and explain U-Net.
- [ ] Can explain why skip connections matter.
- [ ] Can run the model independently of the notebooks.
- [ ] Can explain each evaluation metric.
- [ ] Has actual experiment tables and plots.
- [ ] Has documented failure cases and limitations.
- [ ] Can reproduce the results from configuration and seed.
