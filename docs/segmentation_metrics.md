# Semantic Segmentation Metrics

This project evaluates the three Oxford-IIIT Pet trimap classes after mapping
the raw labels to contiguous model labels:

| Class ID | Class |
|---:|---|
| 0 | pet |
| 1 | background |
| 2 | boundary |

## Confusion matrix

The NumPy implementation in `src.metrics.confusion_matrix` uses rows for the
ground-truth class and columns for the predicted class. Therefore, for class
`c`:

- **TP** is `CM[c, c]`: pixels truly belonging to `c` and predicted as `c`.
- **FP** is column `c` excluding `CM[c, c]`: pixels predicted as `c` but belonging to another class.
- **FN** is row `c` excluding `CM[c, c]`: pixels belonging to `c` but predicted as another class.

Every pixel contributes exactly once to the matrix. This makes the matrix a
compact summary of pixel-level classification errors.

## Metrics

For each class, the implementation calculates:

```text
IoU  = TP / (TP + FP + FN)
Dice = 2TP / (2TP + FP + FN)
```

Pixel accuracy is the fraction of all pixels classified correctly:

```text
Pixel Accuracy = sum(TP) / total pixels
```

IoU measures overlap relative to the union of the predicted and true pixels.
Dice weights the overlap twice and is numerically more forgiving than IoU.
Pixel accuracy is easy to interpret, but can hide poor performance on a small
class when background pixels dominate.

## Macro averages and absent classes

`mean_iou` and `mean_dice` are macro averages: they calculate each class's
metric first, then average the class values. A class whose ground truth and
prediction are both absent has a zero denominator, so its per-class value is
reported as `NaN` and excluded from the macro average. If every class is
absent, the macro function returns `0.0`.

If a class is absent from the ground truth but appears in the prediction, its
denominator is nonzero and its score is `0.0`, correctly penalizing the false
positive prediction. An empty confusion matrix has pixel accuracy `0.0`.

This differs from calculating one metric from a globally aggregated confusion
matrix and calling that a class average. Global aggregation weights classes by
their pixel counts; macro averaging gives each present class equal weight.
For this reason, this project reports per-class values and macro means rather
than relying on pixel accuracy alone.

## NumPy and PyTorch implementations

The NumPy functions are intentionally separate from the PyTorch functions used
during model evaluation, but both implementations now use the same formulas
and edge-case policy. They are cross-tested with identical confusion matrices.

This policy means a class absent from both ground truth and predictions does
not contribute a zero to a macro score. It is represented as `NaN` per class
and ignored by the macro mean. A class that is absent from ground truth but
incorrectly predicted has false positives, so its denominator is nonzero and
its IoU/Dice is exactly `0.0` in both implementations.
