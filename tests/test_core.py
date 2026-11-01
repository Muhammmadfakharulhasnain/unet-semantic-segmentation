import torch
from src.model import UNet, count_parameters
from src.losses import MulticlassDiceLoss, CombinedLoss
from src.metrics import metrics_from_confusion_matrix

def test_unet_output_shape():
    model = UNet(in_channels=3, num_classes=3, base_channels=8)
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 3, 128, 128)
    assert count_parameters(model) > 0

def test_losses_are_finite():
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randint(0, 3, (2, 32, 32))
    for loss_fn in [MulticlassDiceLoss(3), CombinedLoss(3)]:
        loss = loss_fn(logits, target)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)

def test_metrics_are_bounded():
    cm = torch.tensor([[8, 1, 1], [2, 7, 1], [1, 1, 8]])
    result = metrics_from_confusion_matrix(cm, ["pet", "background", "boundary"])
    assert 0 <= result["mean_iou"] <= 1
    assert 0 <= result["mean_dice"] <= 1
    assert 0 <= result["pixel_accuracy"] <= 1
