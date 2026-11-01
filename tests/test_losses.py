import torch
import torch.nn.functional as F

from src.losses import CombinedLoss, MulticlassDiceLoss, build_loss


def make_inputs():
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    targets = torch.randint(0, 3, (2, 4, 5))
    return logits, targets


def test_all_losses_return_finite_scalars_for_segmentation_shapes():
    logits, targets = make_inputs()

    losses = [
        build_loss("cross_entropy"),
        MulticlassDiceLoss(num_classes=3),
        CombinedLoss(num_classes=3),
    ]
    for loss_fn in losses:
        value = loss_fn(logits, targets)
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_cross_entropy_receives_raw_logits_without_manual_softmax():
    logits, targets = make_inputs()
    implementation = build_loss("cross_entropy")

    assert torch.allclose(implementation(logits, targets), F.cross_entropy(logits, targets))


def test_dice_uses_class_dimension_and_one_hot_targets():
    logits, targets = make_inputs()
    loss_fn = MulticlassDiceLoss(num_classes=3, smooth=1e-6)

    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=3).permute(0, 3, 1, 2).to(probabilities.dtype)
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    denominator = (probabilities + one_hot).sum((0, 2, 3))
    expected = 1.0 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()

    assert torch.allclose(loss_fn(logits, targets), expected)


def test_dice_can_exclude_background_class():
    logits, targets = make_inputs()
    all_classes = MulticlassDiceLoss(num_classes=3, include_background=True)
    foreground_only = MulticlassDiceLoss(num_classes=3, include_background=False)

    assert torch.isfinite(all_classes(logits, targets))
    assert torch.isfinite(foreground_only(logits, targets))
    assert not torch.allclose(all_classes(logits, targets), foreground_only(logits, targets))


def test_all_losses_produce_finite_nonzero_gradients():
    _, targets = make_inputs()
    for loss_fn in [
        build_loss("cross_entropy"),
        MulticlassDiceLoss(num_classes=3),
        CombinedLoss(num_classes=3),
    ]:
        logits, _ = make_inputs()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0


def test_combined_loss_is_the_configured_half_half_sum():
    logits, targets = make_inputs()
    ce = build_loss("cross_entropy")(logits, targets)
    dice = MulticlassDiceLoss(num_classes=3)(logits, targets)
    combined = CombinedLoss(num_classes=3)(logits, targets)

    assert torch.allclose(combined, 0.5 * ce + 0.5 * dice)
