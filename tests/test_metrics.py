import numpy as np
import pytest
import torch

from src.metrics import (
    confusion_matrix,
    mean_dice,
    mean_iou,
    metrics_from_confusion_matrix,
    per_class_dice,
    per_class_iou,
    pixel_accuracy,
)


def test_perfect_predictions():
    truth = np.array([[0, 1], [2, 0]])
    prediction = truth.copy()
    cm = confusion_matrix(truth, prediction, num_classes=3)

    np.testing.assert_array_equal(cm, np.diag([2, 1, 1]))
    np.testing.assert_allclose(per_class_iou(cm), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(per_class_dice(cm), [1.0, 1.0, 1.0])
    assert mean_iou(cm) == pytest.approx(1.0)
    assert mean_dice(cm) == pytest.approx(1.0)
    assert pixel_accuracy(cm) == pytest.approx(1.0)


def test_completely_incorrect_predictions():
    truth = np.array([0, 1, 2])
    prediction = np.array([1, 2, 0])
    cm = confusion_matrix(truth, prediction, num_classes=3)

    np.testing.assert_allclose(per_class_iou(cm), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(per_class_dice(cm), [0.0, 0.0, 0.0])
    assert mean_iou(cm) == pytest.approx(0.0)
    assert mean_dice(cm) == pytest.approx(0.0)
    assert pixel_accuracy(cm) == pytest.approx(0.0)


def test_partially_correct_predictions():
    truth = np.array([0, 0, 1, 1, 2, 2])
    prediction = np.array([0, 1, 1, 2, 2, 0])
    cm = confusion_matrix(truth, prediction, num_classes=3)

    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    np.testing.assert_allclose(per_class_iou(cm), [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(per_class_dice(cm), [0.5, 0.5, 0.5])
    assert mean_iou(cm) == pytest.approx(1 / 3)
    assert mean_dice(cm) == pytest.approx(0.5)
    assert pixel_accuracy(cm) == pytest.approx(0.5)


def test_absent_classes_are_excluded_from_macro_means():
    truth = np.zeros(4, dtype=np.int64)
    prediction = np.zeros(4, dtype=np.int64)
    cm = confusion_matrix(truth, prediction, num_classes=3)

    assert per_class_iou(cm)[0] == pytest.approx(1.0)
    assert per_class_dice(cm)[0] == pytest.approx(1.0)
    assert np.isnan(per_class_iou(cm)[1:]).all()
    assert np.isnan(per_class_dice(cm)[1:]).all()
    assert mean_iou(cm) == pytest.approx(1.0)
    assert mean_dice(cm) == pytest.approx(1.0)


def test_zero_denominators_and_empty_input_are_explicit():
    cm = np.zeros((3, 3), dtype=np.int64)
    assert np.isnan(per_class_iou(cm)).all()
    assert np.isnan(per_class_dice(cm)).all()
    assert mean_iou(cm) == 0.0
    assert mean_dice(cm) == 0.0
    assert pixel_accuracy(cm) == 0.0


def test_class_absent_from_truth_but_predicted_has_zero_score():
    cm = confusion_matrix(np.array([0, 0]), np.array([0, 1]), num_classes=3)
    assert per_class_iou(cm)[1] == pytest.approx(0.0)
    assert per_class_dice(cm)[1] == pytest.approx(0.0)
    assert np.isnan(per_class_iou(cm)[2])


@pytest.mark.parametrize(
    "cm",
    [
        np.diag([2, 1, 1]),
        np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        np.array([[4, 0, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]]),
        np.zeros((3, 3), dtype=np.int64),
    ],
)
def test_numpy_and_pytorch_metrics_match_for_identical_confusion_matrices(cm):
    numpy_iou = per_class_iou(cm)
    numpy_dice = per_class_dice(cm)
    numpy_metrics = {
        "mean_iou": mean_iou(cm),
        "mean_dice": mean_dice(cm),
        "pixel_accuracy": pixel_accuracy(cm),
    }

    torch_metrics = metrics_from_confusion_matrix(
        torch.as_tensor(cm),
        class_names=["pet", "background", "boundary"],
    )
    torch_iou = np.array(
        [torch_metrics[f"iou_{name}"] for name in ["pet", "background", "boundary"]]
    )
    torch_dice = np.array(
        [torch_metrics[f"dice_{name}"] for name in ["pet", "background", "boundary"]]
    )

    np.testing.assert_allclose(torch_iou, numpy_iou, equal_nan=True)
    np.testing.assert_allclose(torch_dice, numpy_dice, equal_nan=True)
    for name, expected in numpy_metrics.items():
        assert torch_metrics[name] == pytest.approx(expected)


def test_confusion_matrix_rejects_mismatched_or_invalid_labels():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0], num_classes=3)
    with pytest.raises(ValueError):
        confusion_matrix([0, 3], [0, 1], num_classes=3)
    with pytest.raises(ValueError):
        confusion_matrix([0.0], [0], num_classes=3)
