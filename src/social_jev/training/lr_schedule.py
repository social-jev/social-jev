"""Learning-rate multipliers for optimizer updates."""

from collections.abc import Callable
from dataclasses import dataclass

type LRSchedule = Callable[[int, int], float]
"""Return a multiplier from the zero-based current update and total updates."""


@dataclass(frozen=True)
class WarmupStableDecay:
    """Linear warmup, a stable phase, and optional linear cooldown."""

    warmup_steps: int = 10
    cooldown_steps: int | None = None
    min_lr_factor: float = 0.1

    def __call__(self, current_step: int, total_steps: int) -> float:
        warmup = min(self.warmup_steps, total_steps)
        if current_step < warmup:
            return (current_step + 1) / warmup

        cooldown = min(self.cooldown_steps or 0, total_steps - warmup)
        if cooldown == 0:
            return 1.0

        # match torchtitan's update indexing, including its final virtual step
        decay_start = total_steps + 1 - cooldown
        progress = min(1.0, max(0.0, (current_step + 1 - decay_start) / cooldown))
        return self.min_lr_factor + (1.0 - self.min_lr_factor) * (1.0 - progress)
