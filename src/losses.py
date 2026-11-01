import torch
from torch import nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    """Soft multiclass Dice loss. Expects logits [B,C,H,W], targets [B,H,W]."""
    def __init__(self, num_classes=3, smooth=1e-6, include_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=probs.dtype)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        denominator = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=3, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MulticlassDiceLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)

def build_loss(name, num_classes=3):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return MulticlassDiceLoss(num_classes=num_classes)
    if name == "combined":
        return CombinedLoss(num_classes=num_classes)
    raise ValueError(f"Unknown loss: {name}")
