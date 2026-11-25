"""Settings for post-training a SocialJev checkpoint."""

from dataclasses import dataclass, field

import torch

from .loss import FocalLoss, Loss
from .lr_schedule import LRSchedule, WarmupStableDecay


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
    log_steps: int = 1
    """Log training metrics every this many optimizer updates."""
    val_steps: int | None = 10
    """Validate every this many updates, plus before and after training."""
    checkpoint_steps: int | None = None
    shuffle_buffer_size: int = 256
    loss_fn: Loss = field(default_factory=FocalLoss)
    lr_schedule: LRSchedule = field(default_factory=WarmupStableDecay)
    """Multiply each parameter group's learning rate for the current update."""

    def __post_init__(self):
        """Require positive sizes and intervals used by our pipeline."""
        for name in (
            "steps",
            "seq_len",
            "local_batch_size",
            "data_parallel_shard_degree",
            "shuffle_buffer_size",
            "log_steps",
            "val_steps",
            "checkpoint_steps",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                message = f"{name} must be positive."
                raise ValueError(message)
