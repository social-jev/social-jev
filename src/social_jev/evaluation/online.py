"""Datasets and scoring functions to evaluate during training."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..metrics import accuracy
from ..types import Metrics, Query
from .common import create_eval_requests, get_named_metrics, validate_calibration_bins


@dataclass(kw_only=True)
class OnlineEvaluation:
    """Evaluate a reusable dataset or factory with the same inputs as evaluate()."""

    data: Iterable[dict] | Callable[[], Iterable[dict]]
    queries: Query | list[Query] | None = None
    metrics: Metrics = accuracy
    calibration_bins: int = 10

    def __post_init__(self):
        validate_calibration_bins(self.calibration_bins)

    def prepare(self):
        """Build ordered requests, labels, and metric selections for this evaluation."""
        data = self.data() if callable(self.data) else self.data
        return create_eval_requests(
            data, queries=self.queries, metric_names=get_named_metrics(self.metrics)
        )
