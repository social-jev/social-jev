"""Datasets and scoring functions to evaluate during training."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from ..metrics import accuracy
from ..types import Metrics, Query


@dataclass(kw_only=True)
class OnlineEvaluation:
    """Evaluate a reusable dataset or factory with the same inputs as evaluate()."""

    data: Iterable[dict] | Callable[[], Iterable[dict]]
    queries: Query | list[Query] | None = None
    metrics: Metrics = accuracy
    calibration_bins: int = 10

    def __post_init__(self):
        if type(self.calibration_bins) is not int or self.calibration_bins < 1:
            message = "calibration_bins must be a positive integer."
            raise ValueError(message)
        if isinstance(self.data, Iterator):
            message = (
                "Online evaluation needs a reusable dataset "
                "or a callable returning a fresh iterator."
            )
            raise TypeError(message)
