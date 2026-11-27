"""Evaluate datasets through the ordinary query API."""

from ..metrics import accuracy
from .core import (
    create_eval_requests,
    get_named_metrics,
    score_evaluation,
)


def evaluate(model, *, data, queries=None, metrics=accuracy, calibration_bins=10):
    """Submit the dataset and score its ordered predictions."""
    if type(calibration_bins) is not int or calibration_bins < 1:
        message = "calibration_bins must be a positive integer."
        raise ValueError(message)

    requests, labels, selected_metrics = create_eval_requests(
        data, queries=queries, metric_names=get_named_metrics(metrics)
    )
    predictions = [prediction for results in model.query(requests) for prediction in results]
    submitted_queries = [query for request in requests for query in request.queries]
    probabilities = [
        [prediction.probabilities[option] for option in query.options]
        for prediction, query in zip(predictions, submitted_queries, strict=True)
    ]
    return score_evaluation(
        probabilities,
        requests=requests,
        labels=labels,
        selected_metrics=selected_metrics,
        metrics=metrics,
        calibration_bins=calibration_bins,
    )
