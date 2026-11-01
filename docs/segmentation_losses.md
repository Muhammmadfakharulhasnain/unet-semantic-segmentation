# Segmentation Loss Functions

The model produces raw segmentation logits with shape `[B, C, H, W]` and the
Oxford-IIIT Pet target masks have shape `[B, H, W]`, where `C=3`:

| Class ID | Class |
|---:|---|
| 0 | pet |
| 1 | background |
| 2 | boundary |

The loss functions are implemented in `src/losses.py` and are selected by
`build_loss` using the names `cross_entropy`, `dice`, and `combined`.

## 1. Multiclass Cross-Entropy

For pixel `p`, target class `y_p`, and model logit `z_{p,c}`:

```text
p_{p,c} = softmax(z_{p,:})_c
CE = - mean_p log(p_{p,y_p})
```

The implementation uses `torch.nn.CrossEntropyLoss`, which expects raw logits
and internally applies the numerically stable log-softmax operation. The code
must therefore not apply softmax before calling CrossEntropyLoss.

Input and target shapes:

```text
logits:  [B, C, H, W]  floating point raw logits
targets: [B, H, W]     integer class IDs in [0, C)
output:  scalar loss
```

Cross-Entropy provides a strong pixel-wise learning signal and is usually easy
to optimize. Its limitation here is that every pixel contributes to the mean,
so a large background region can dominate the objective when classes are
imbalanced.

## 2. Soft Multiclass Dice Loss

The implementation first converts logits to class probabilities with
`softmax(logits, dim=1)`. The target mask is converted to one-hot format:

```text
targets: [B, H, W]
one_hot: [B, C, H, W]
```

For each class `c`, probabilities and one-hot targets are summed over batch
and spatial dimensions `(0, 2, 3)`:

```text
Dice_c = (2 * intersection_c + smooth)
         / (prediction_mass_c + target_mass_c + smooth)
Dice Loss = 1 - mean_c Dice_c
```

The default includes all three classes, including background. Passing
`include_background=False` averages only the pet and boundary classes. The
small `smooth` value prevents zero-denominator numerical failures.

Dice directly rewards overlap and can counter class imbalance better than raw
pixel averaging. It can be less stable when a class is extremely sparse or
absent, and its batch-level aggregation means the loss depends on the class
composition of the batch.

## 3. Combined CE + Dice Loss

The configured combined loss is:

```text
L_combined = 0.5 * L_CE + 0.5 * L_Dice
```

The implementation computes Cross-Entropy from raw logits and Dice from its
own softmax probabilities. There is no manual softmax before Cross-Entropy.
The result is a scalar and gradients flow through both terms.

Cross-Entropy supplies local, class-discriminative gradients for each pixel;
Dice supplies an overlap-oriented signal that is more sensitive to region
coverage and minority classes. The weights are an experimental choice, not a
universal optimum. Loss values from different loss functions should not be
ranked directly: CE, Dice loss, and their weighted sum have different scales
and meanings.

## Gradient behavior and validation

All three losses are differentiable with respect to the logits. The tests in
`tests/test_losses.py` verify:

- expected segmentation tensor shapes;
- scalar, finite loss values;
- raw-logit Cross-Entropy behavior;
- softmax along the class dimension (`dim=1`);
- correct one-hot target layout;
- optional background exclusion for Dice;
- finite, nonzero gradients; and
- the exact half-CE/half-Dice combined-loss identity.

No optimizer step is part of loss validation. A successful backward pass shows
that the loss can provide gradients; it does not measure model quality or
replace a training and validation experiment.
