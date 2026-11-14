"""Batch evaluation results."""

from collections.abc import Sequence
from dataclasses import dataclass

from .types import Result

Label = int | Sequence[float]


@dataclass
class EvaluationResult:
    """The batch score, predictions, and labels in input order."""

    score: float
    predictions: list[Result]
    labels: list[Label]
