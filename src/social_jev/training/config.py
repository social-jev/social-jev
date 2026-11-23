"""Settings for post-training a SocialJev checkpoint."""

from dataclasses import dataclass, field
from math import isfinite

import torch

from .loss import FocalLoss, Loss


@dataclass(kw_only=True, frozen=True)
class TrainConfig:
    """Configure supervised post-training."""

    steps: int
    seq_len: int = 4096
    local_batch_size: int = 1
    global_batch_size: int = -1
    data_parallel_replicate_degree: int | None = None
    data_parallel_shard_degree: int = 1
    reduce_dtype: torch.dtype = torch.float32
    head_lr: float = 1e-3
    backbone_lr: float | None = None
    val_steps: int | None = None
    checkpoint_steps: int | None = None
    shuffle_buffer_size: int = 256
    loss_fn: Loss = field(default_factory=FocalLoss)

    def __post_init__(self):  # noqa: C901 -- check independent public settings together.
        """Validate settings that do not depend on the available devices."""
        for name in (
            "steps",
            "seq_len",
            "local_batch_size",
            "data_parallel_shard_degree",
            "shuffle_buffer_size",
        ):
            if getattr(self, name) < 1:
                message = f"{name} must be positive."
                raise ValueError(message)
        for name in ("global_batch_size",):
            value = getattr(self, name)
            if value != -1 and value < 1:
                message = f"{name} must be positive or -1."
                raise ValueError(message)
        if (
            self.data_parallel_replicate_degree is not None
            and self.data_parallel_replicate_degree < 1
        ):
            message = "data_parallel_replicate_degree must be positive or None."
            raise ValueError(message)
        for name in ("head_lr", "backbone_lr"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                message = f"{name} must be finite and positive."
                raise ValueError(message)
        for name in ("val_steps", "checkpoint_steps"):
            value = getattr(self, name)
            if value is not None and value < 1:
                message = f"{name} must be positive."
                raise ValueError(message)
