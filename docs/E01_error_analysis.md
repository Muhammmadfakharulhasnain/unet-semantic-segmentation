# E01 Baseline Results and Error Analysis

Experiment: `E01` (`baseline_ce`)  
Checkpoint: [`../checkpoints/baseline_ce.pt`](../checkpoints/baseline_ce.pt)  
Test results: [`../results/tables/experiment_comparison.csv`](../results/tables/experiment_comparison.csv)  
Epoch history: [`../results/tables/baseline_ce_history.csv`](../results/tables/baseline_ce_history.csv)

This analysis uses the saved E01 outputs only. The checkpoint and dataset were not modified, and E02 was not started.

## 1. Official test metrics

| Class | Precision | Recall | IoU | Dice |
|---|---:|---:|---:|---:|
| Pet | 0.836272 | 0.832582 | 0.715888 | 0.834423 |
| Background | 0.871926 | 0.951505 | 0.834827 | 0.909979 |
| Boundary | 0.700804 | 0.407846 | 0.347363 | 0.515619 |
| Macro average | 0.803001 | 0.730644 | 0.632692 mean IoU | 0.753340 mean Dice |

Overall pixel accuracy was `0.849045`; test loss was `0.391949`.

The boundary class has the weakest results by a clear margin. Its recall (`0.407846`) is much lower than its precision (`0.700804`), indicating that many ground-truth boundary pixels were not predicted as boundary, while boundary predictions that were made were more selective. This is consistent with the low boundary IoU and Dice.

## 2. Training curves

![E01 training curves](../results/figures/baseline_ce_curves.png)

Observed from the saved curve figure and history:

- Training loss decreased at every epoch, from `0.687398` to `0.400591`.
- Validation loss also decreased at every epoch, from `0.539883` to `0.395379`.
- Training mean IoU increased from `0.412764` to `0.631121`.
- Validation mean IoU increased from `0.516806` to `0.625063`.
- Validation mean IoU improved only slightly from epoch 4 (`0.624725`) to epoch 5 (`0.625063`), so the short run was approaching a plateau.
- There is no strong divergence between training and validation curves in these five epochs. This is evidence against obvious short-run overfitting, but five epochs is too short to establish a generalization trend.

## 3. Saved prediction examples

![E01 saved predictions](../results/figures/baseline_ce_predictions.png)

The figure presents input, ground truth, prediction, and a pixel-error view for two saved test examples. The displayed colors encode background, pet, and boundary consistently across the ground-truth and prediction panels.

### Example 1

- The predicted pet occupies a somewhat different extent from the ground truth, especially around the outer contour and lower body/leg regions.
- Several contour regions disagree while the central pet body is broadly recovered.
- The pixel-error panel shows a substantial error outline around the object, which is consistent with boundary displacement rather than a completely missed pet.

### Example 2

- The main pet region is recovered, but the predicted silhouette is narrower/different in places than the ground-truth silhouette.
- Thin contour and leg regions show visible disagreement in the error panel.
- Small isolated error marks are also visible away from the dominant predicted region, indicating localized pixel mistakes rather than a uniformly shifted mask.

These are observations about the two examples rendered by the saved figure; they are not a claim that these two images represent the complete test distribution.

## 4. Likely failure patterns

The saved examples and class metrics support the following interpretations:

### Incorrect object boundaries

The prediction contours do not always align with the ground-truth contours. A one- or two-pixel displacement is especially consequential for a thin boundary class and can produce many false negatives even when the pet interior is correct.

### Missing pet regions

The examples show missed or differently shaped thin regions around legs and other protrusions. The pet recall of `0.832582` indicates that the model does recover most pet pixels overall, but not all of the object extent.

### Background classified as pet

The pet precision of `0.836272` is below perfect, so some predicted pet pixels are false positives from background or boundary areas. The first example visibly contains pet-extent differences around the silhouette.

### Boundary pixels classified incorrectly

This is the dominant measured weakness. Boundary recall is only `0.407846`, compared with background recall `0.951505` and pet recall `0.832582`. The model therefore often assigns boundary pixels to another class, most plausibly neighboring pet or background classes. The saved aggregate table does not contain a confusion matrix, so the exact destination of each boundary false negative cannot be quantified from the saved E01 results alone.

## 5. Why boundary IoU is substantially lower

The results support several compatible explanations:

1. Boundary pixels form a thin, relatively small target region. Small localization errors create a large fraction of false negatives for this class.
2. Cross-entropy optimizes per-pixel classification and does not explicitly reward contour alignment or connected shape structure.
3. The boundary class is adjacent to both pet and background, making its appearance and label assignment more ambiguous than the broad background region.
4. A boundary false negative is costly for boundary IoU because it contributes directly to the union while not contributing to the intersection.

These are research hypotheses informed by the class metrics and visual examples. A class-frequency table and a saved test confusion matrix would be needed to quantify class imbalance and the exact boundary-to-pet/background error split.

## 6. Baseline limitations

- Only five epochs were run, so convergence and longer-run overfitting were not evaluated.
- The baseline uses CrossEntropy alone; it has no explicit contour, Dice, class-weighting, or boundary-aware objective.
- The model uses 128x128 inputs, which limits the spatial detail available for thin boundaries and small protrusions.
- The experiment uses a small U-Net configuration with base channels 8 and CPU-oriented settings; capacity may limit fine-grained segmentation.
- The saved prediction figure contains two examples, which is useful for qualitative inspection but is not a representative error sample.
- Aggregate per-class metrics do not identify the exact confusion destinations for boundary errors. Future runs should save a confusion matrix and class-frequency summary.
- E01 is a single seed (`42`) and does not measure variance across random seeds.

## 7. Research questions for future experiments

1. Does the validated combined CrossEntropy + Dice loss improve boundary recall and boundary IoU without reducing pet IoU?
2. Does a boundary-aware or class-weighted objective reduce boundary false negatives?
3. Does increasing input resolution improve contour and thin-leg recovery enough to justify CPU cost?
4. Does augmentation improve robustness to pose, lighting, and background variation?
5. How do results vary across random seeds, and are the observed class gaps stable?
6. Which boundary errors dominate: boundary-to-pet or boundary-to-background? A confusion matrix can answer this directly.
7. Are the qualitative failures concentrated in particular pet poses, scales, or background types?

## References to saved artifacts

- Curves: [`baseline_ce_curves.png`](../results/figures/baseline_ce_curves.png)
- Predictions: [`baseline_ce_predictions.png`](../results/figures/baseline_ce_predictions.png)
- History: [`baseline_ce_history.csv`](../results/tables/baseline_ce_history.csv)
- Test comparison: [`experiment_comparison.csv`](../results/tables/experiment_comparison.csv)
