"""Per-query losses over the supplied answer options."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from ..types import OptionLabels, OptionLogits, OptionMask, QueryLosses


class Loss(ABC):
    """Return one differentiable loss per query from option logits and labels."""

    @abstractmethod
    def __call__(
        self, logits: OptionLogits, labels: OptionLabels, option_mask: OptionMask
    ) -> QueryLosses: ...


@dataclass
class FocalLoss(Loss):
    """Weight each option's cross-entropy by (1 - its model probability) ** gamma."""

    gamma: float = 2.0

    def __call__(
        self, logits: OptionLogits, labels: OptionLabels, option_mask: OptionMask
    ) -> QueryLosses:
        log_probs = (
            logits.float()
            .masked_fill(~option_mask, -torch.inf)
            .log_softmax(dim=-1)
            .masked_fill(~option_mask, 0.0)
        )
        weights = (1.0 - log_probs.exp()).pow(self.gamma)
        return -(labels * weights * log_probs).sum(dim=-1)


class CrossEntropy(FocalLoss):
    """Compute cross-entropy over options, retaining distribution labels."""

    def __init__(self):
        super().__init__(gamma=0.0)


@dataclass
class BrierLoss(Loss):
    """Sum squared differences between predicted and reference option probabilities."""

    def __call__(
        self, logits: OptionLogits, labels: OptionLabels, option_mask: OptionMask
    ) -> QueryLosses:
        probabilities = logits.float().masked_fill(~option_mask, -torch.inf).softmax(dim=-1)
        errors = (probabilities - labels).square()
        return errors.masked_fill(~option_mask, 0.0).sum(dim=-1)


@dataclass
class BrierCrossEntropy(Loss):
    """Combine Brier and cross-entropy as weight * Brier + (1 - weight) * CE."""

    brier_weight: float = 0.5
    brier: BrierLoss = field(default_factory=BrierLoss, init=False)
    cross_entropy: CrossEntropy = field(default_factory=CrossEntropy, init=False)

    def __post_init__(self):
        if not 0.0 <= self.brier_weight <= 1.0:
            message = "brier_weight must be between 0 and 1."
            raise ValueError(message)

    def __call__(
        self, logits: OptionLogits, labels: OptionLabels, option_mask: OptionMask
    ) -> QueryLosses:
        brier = self.brier(logits, labels, option_mask)
        cross_entropy = self.cross_entropy(logits, labels, option_mask)
        return self.brier_weight * brier + (1.0 - self.brier_weight) * cross_entropy
