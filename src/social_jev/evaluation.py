"""Evaluation inputs, named metrics, and batch results."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .types import Query, Request, Result

Label = int | Sequence[float]
Metric = Callable[[list[list[float]], list[Label]], float | int]
Metrics = Metric | list[Metric] | dict[str, Metric]


@dataclass
class EvaluationResult:
    """Named scores and flat predictions and labels in document/query order."""

    scores: dict[str, float | int]
    predictions: list[Result]
    labels: list[Label]


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
    if any(has_metrics) and not all(has_metrics):
        msg = "Specify metrics in every row or omit them everywhere."
        raise ValueError(msg)

    for row in rows:
        if queries is None:
            options = [row["options"]]
        else:
            if "query" in row or "options" in row:
                msg = "Supply queries either in data or through query=, not both."
                raise ValueError(msg)
            options = [query.options for query in queries]

        labels = [row["label"]] if len(options) == 1 else row["label"]
        if len(labels) != len(options):
            msg = "Provide one label per query, in query order."
            raise ValueError(msg)
        for label, query_options in zip(labels, options, strict=True):
            if isinstance(label, int):
                if not 0 <= label < len(query_options):
                    msg = "Integer labels must be valid zero-based option indices."
                    raise ValueError(msg)
            elif len(label) != len(query_options):
                msg = "Vector labels need one value per option."
                raise ValueError(msg)

        unknown = set(row.get("metrics", metric_names)) - set(metric_names)
        if unknown:
            msg = f"Unknown metrics: {sorted(unknown)}"
            raise ValueError(msg)


def create_eval_requests(data, *, query, metric_names):
    """Expand shared or per-row queries and align labels and metric selections."""
    rows = list(data)
    if query is None:
        queries = None
    elif isinstance(query, Query):
        queries = [query]
    else:
        queries = list(query)

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
