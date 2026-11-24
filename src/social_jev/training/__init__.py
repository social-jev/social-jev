"""Configure post-training and choose the per-query training loss."""

from typing import TYPE_CHECKING

from ..data import Sample
from .config import TrainConfig
from .loss import BrierCrossEntropy, BrierLoss, CrossEntropy, FocalLoss, Loss

if TYPE_CHECKING:
    from .trainer import Trainer


__all__ = [
    "BrierCrossEntropy",
    "BrierLoss",
    "CrossEntropy",
    "FocalLoss",
    "Loss",
    "Sample",
    "TrainConfig",
    "Trainer",
]


def __getattr__(name):
    """Import the optional training runtime only when Trainer is requested."""
    if name == "Trainer":
        from .trainer import Trainer  # noqa: PLC0415 -- optional runtime.

        return Trainer
    raise AttributeError(name)
