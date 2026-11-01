import torch
import numpy as np


NUM_CLASSES = 3
CLASS_NAMES = ("pet", "background", "boundary")


def confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    """Return a NumPy confusion matrix with rows=true and columns=predicted.

    The inputs may have any matching shape. Labels must be integer class IDs in
    ``[0, num_classes)``. This implementation is intentionally independent of
    the PyTorch metric path below.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not isinstance(num_classes, (int, np.integer)) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")

    true = true.reshape(-1)
    pred = pred.reshape(-1)
    if not np.issubdtype(true.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("y_true and y_pred must contain integer class IDs")
    if np.any((true < 0) | (true >= num_classes)):
        raise ValueError("y_true contains an out-of-range class ID")
    if np.any((pred < 0) | (pred >= num_classes)):
        raise ValueError("y_pred contains an out-of-range class ID")

    encoded = num_classes * true + pred
    return np.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def _validate_numpy_confusion_matrix(confmat):
    cm = np.asarray(confmat)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confmat must be a square 2D array")
    if np.any(cm < 0) or not np.issubdtype(cm.dtype, np.number):
        raise ValueError("confmat must contain non-negative numbers")
    return cm.astype(np.float64, copy=False)


def _per_class_counts(confmat):
    cm = _validate_numpy_confusion_matrix(confmat)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return tp, fp, fn


def per_class_iou(confmat):
    """Return per-class IoU, using NaN when a class has zero union."""
    tp, fp, fn = _per_class_counts(confmat)
    denominator = tp + fp + fn
    return np.divide(tp, denominator, out=np.full_like(tp, np.nan), where=denominator != 0)


def mean_iou(confmat):
    """Return macro IoU, ignoring classes absent from truth and prediction."""
    values = per_class_iou(confmat)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else 0.0


def per_class_dice(confmat):
    """Return per-class Dice, using NaN when a class has zero denominator."""
    tp, fp, fn = _per_class_counts(confmat)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.full_like(tp, np.nan), where=denominator != 0)


def mean_dice(confmat):
    """Return macro Dice, ignoring classes absent from truth and prediction."""
    values = per_class_dice(confmat)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else 0.0


def pixel_accuracy(confmat):
    """Return correct-pixel fraction, or 0.0 for an empty confusion matrix."""
    cm = _validate_numpy_confusion_matrix(confmat)
    total = cm.sum()
    return float(np.trace(cm) / total) if total != 0 else 0.0

@torch.no_grad()
def update_confusion_matrix(confmat, logits, targets, num_classes):
    predictions = logits.argmax(dim=1).view(-1)
    targets = targets.view(-1)
    valid = (targets >= 0) & (targets < num_classes)
    encoded = num_classes * targets[valid] + predictions[valid]
    confmat += torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)
    return confmat

def metrics_from_confusion_matrix(confmat, class_names=None, eps=1e-8):
    """Return PyTorch metrics using the same convention as the NumPy API.

    ``eps`` is retained for public API compatibility. Zero denominators are
    represented as NaN and excluded from macro averages instead of being
    converted to zero.
    """
    cm = confmat.to(torch.float64)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    iou = torch.where(union != 0, tp / union, torch.full_like(tp, torch.nan))
    dice = torch.where(
        dice_denominator != 0,
        (2 * tp) / dice_denominator,
        torch.full_like(tp, torch.nan),
    )
    precision = torch.where(
        precision_denominator != 0,
        tp / precision_denominator,
        torch.full_like(tp, torch.nan),
    )
    recall = torch.where(
        recall_denominator != 0,
        tp / recall_denominator,
        torch.full_like(tp, torch.nan),
    )
    total = cm.sum()
    pixel_accuracy = torch.where(total != 0, tp.sum() / total, torch.zeros_like(total))

    result = {
        "pixel_accuracy": float(pixel_accuracy),
        "mean_iou": float(torch.nanmean(iou)) if not torch.isnan(iou).all() else 0.0,
        "mean_dice": float(torch.nanmean(dice)) if not torch.isnan(dice).all() else 0.0,
        "macro_precision": float(torch.nanmean(precision)) if not torch.isnan(precision).all() else 0.0,
        "macro_recall": float(torch.nanmean(recall)) if not torch.isnan(recall).all() else 0.0,
        "mean_iou_pet_and_boundary": float(torch.nanmean(iou[[0, 2]]))
        if not torch.isnan(iou[[0, 2]]).all()
        else 0.0,
    }
    names = class_names or [f"class_{i}" for i in range(len(tp))]
    for i, name in enumerate(names):
        result[f"iou_{name}"] = float(iou[i])
        result[f"dice_{name}"] = float(dice[i])
        result[f"precision_{name}"] = float(precision[i])
        result[f"recall_{name}"] = float(recall[i])
    return result
