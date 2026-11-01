# Mathematics Notes

## 1. Pixel-wise classification

For each pixel `(i, j)`, the model produces class logits `z_0, ..., z_(C-1)`. Softmax converts them to probabilities:

`p_c = exp(z_c) / sum_k exp(z_k)`

The predicted class is the index with the largest probability.

## 2. Cross-Entropy Loss

For a target class `y` and predicted probability `p_y`:

`L_CE = -log(p_y)`

Cross-entropy strongly penalizes assigning low probability to the correct class. In dense prediction, the pixel-wise losses are aggregated over the image/batch.

## 3. Dice coefficient

For binary sets A and B:

`Dice = 2|A intersection B| / (|A| + |B|)`

For a class in a confusion matrix:

`Dice = 2TP / (2TP + FP + FN)`

Dice ranges from 0 to 1. Larger is better.

## 4. Intersection over Union

`IoU = |A intersection B| / |A union B|`

Using confusion-matrix counts:

`IoU = TP / (TP + FP + FN)`

IoU ranges from 0 to 1. Larger is better.

## 5. Precision and recall

`Precision = TP / (TP + FP)`

Of the pixels predicted as a class, how many were correct?

`Recall = TP / (TP + FN)`

Of the true pixels belonging to a class, how many were recovered?

## 6. Pixel accuracy

`Pixel accuracy = total correct pixels / total pixels`

Pixel accuracy can be high even when a small class is poorly segmented. Report per-class IoU/Dice and macro averages alongside it.

## 7. Confusion matrix

Rows are ground-truth classes; columns are predicted classes. The diagonal contains correct predictions. Off-diagonal cells reveal which classes are confused.

## 8. Combined loss

This project uses:

`L_combined = 0.5 * L_CE + 0.5 * L_Dice`

These weights are an experimental choice, not a universal optimum. Keep them fixed when comparing runs and report them in the experiment protocol.
