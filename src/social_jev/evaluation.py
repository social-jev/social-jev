"""Evaluation inputs, named metrics, and batch results."""

from collections.abc import Sequence
from statistics import fmean

from .types import CalibrationBin, CalibrationMetrics, Metric, Metrics, Query, QueryLabel, Request


def get_named_metrics(metrics: Metrics) -> dict[str, Metric]:
    """Use explicit metric keys or callable names."""
    if isinstance(metrics, dict):
        return metrics
    functions = metrics if isinstance(metrics, list) else [metrics]
    # Callable also covers objects without names; those require explicit dictionary keys.
    named_metrics = {getattr(metric, "__name__"): metric for metric in functions}  # noqa: B009
    if len(named_metrics) != len(functions):
        msg = "Metrics share a function name. Pass a dict with distinct names."
        raise ValueError(msg)
    return named_metrics


# Keep setup checks together so request construction stays readable.
def validate_eval_setup(rows, queries, metric_names) -> None:  # noqa: C901
    """Reject ambiguous inputs, invalid labels, and unknown metric selections."""
    has_metrics = ["metrics" in row for row in rows]
    # metric names must appear in every row or no rows
    if any(has_metrics) and not all(has_metrics):
        msg = "Specify metrics in every row or omit them everywhere."
        raise ValueError(msg)

    for row in rows:
        if queries is None:
            options = [row["options"]]
        else:
            # queries must come from one place only
            if "query" in row or "options" in row:
                msg = "Supply queries either in data or through queries=, not both."
                raise ValueError(msg)
            options = [query.options for query in queries]

        labels = [row["label"]] if len(options) == 1 else row["label"]
        # each query needs one label
        if len(labels) != len(options):
            msg = "Provide one label per query, in query order."
            raise ValueError(msg)
        for label, query_options in zip(labels, options, strict=True):
            if isinstance(label, int):
                # the label must point to an available option
                if not 0 <= label < len(query_options):
                    msg = "Integer labels must be valid zero-based option indices."
                    raise ValueError(msg)
            # a list label needs one value for each option
            elif len(label) != len(query_options):
                msg = "Vector labels need one value per option."
                raise ValueError(msg)

        unknown = set(row.get("metrics", metric_names)) - set(metric_names)
        # each named metric must have a supplied function
        if unknown:
            msg = f"Unknown metrics: {sorted(unknown)}"
            raise ValueError(msg)


def create_eval_requests(data, *, queries, metric_names):
    """Expand shared or per-row queries and align labels and metric selections."""
    rows = list(data)
    if isinstance(queries, Query):
        queries = [queries]
    elif queries is not None:
        queries = list(queries)

    validate_eval_setup(rows, queries, metric_names)
    requests = []
    labels = []
    selected_metrics = []
    for row in rows:
        row_queries = (
            queries if queries is not None else [Query(text=row["query"], options=row["options"])]
        )
        row_labels = [row["label"]] if len(row_queries) == 1 else row["label"]
        names = list(row.get("metrics", metric_names))
        requests.append(Request(state=row["state"], queries=row_queries))
        labels.extend(row_labels)
        selected_metrics.extend([names] * len(row_queries))
    return requests, labels, selected_metrics


def compute_calibration(
    probabilities: Sequence[Sequence[float]], labels: Sequence[QueryLabel], *, bins: int
) -> CalibrationMetrics:
    """Compute top-option ECE with an already validated positive bin count.

    Vector labels are interpreted as target probabilities, not normalized here.
    Integer labels and their one-hot equivalents give identical measurements.
    No user-supplied metrics are needed for this calculation.
    """
    groups: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for prediction, label in zip(probabilities, labels, strict=True):
        choice = max(range(len(prediction)), key=prediction.__getitem__)
        confidence = prediction[choice]
        target = float(choice == label) if isinstance(label, int) else label[choice]
        bucket = min(int(confidence * bins), bins - 1)
        groups[bucket].append((confidence, target))

    bin_measurements: list[CalibrationBin] = []
    weighted_error = 0.0
    observations = 0
    for bucket, rows in enumerate(groups):
        confidence = accuracy = None
        if rows:
            confidences, correctness = zip(*rows, strict=True)
            confidence = fmean(confidences)
            accuracy = fmean(correctness)
            weighted_error += len(rows) * abs(confidence - accuracy)
            observations += len(rows)
        bin_measurements.append(
            {
                "lower": bucket / bins,
                "upper": (bucket + 1) / bins,
                "observations": len(rows),
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )

    return CalibrationMetrics(
        ece=round(weighted_error / observations, 2) if observations else None,
        bins=bin_measurements,
    )
