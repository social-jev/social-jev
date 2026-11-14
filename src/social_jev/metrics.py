"""Metrics for batch evaluation."""

from .evaluation import Label


def accuracy(predictions: list[list[float]], labels: list[Label]) -> float:
    """Return the fraction of top choices that match integer reference labels."""
    total = 0.0
    for probabilities, label in zip(predictions, labels, strict=True):
        choice = max(range(len(probabilities)), key=probabilities.__getitem__)
        if not isinstance(label, int):
            msg = "Accuracy requires integer option indices as labels."
            raise TypeError(msg)
        total += float(choice == label)
    return total / len(labels)


def f1(predictions: list[list[float]], labels: list[Label]) -> float:
    """Return binary F1 with option index 0 as the positive class."""
    true_positives = false_positives = false_negatives = 0
    for (positive, negative), label in zip(predictions, labels, strict=True):
        if not isinstance(label, int) or label not in (0, 1):
            msg = "Binary F1 requires integer labels 0 and 1."
            raise ValueError(msg)
        predicts_positive = positive >= negative
        is_positive = label == 0
        true_positives += predicts_positive and is_positive
        false_positives += predicts_positive and not is_positive
        false_negatives += not predicts_positive and is_positive

    denominator = 2 * true_positives + false_positives + false_negatives
    return 2 * true_positives / denominator if denominator else 0.0
