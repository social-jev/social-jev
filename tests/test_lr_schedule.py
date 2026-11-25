"""Check update learning rates and continuation through TorchTitan checkpoints."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("torchtitan")
pytest.importorskip("triton")

from torchtitan.components.optimizer import LRSchedulersContainer

from social_jev.training import TrainConfig
from social_jev.training.lr_schedule import WarmupStableDecay
from social_jev.training.trainer import ScheduleConfig


@pytest.mark.parametrize("cooldown_steps", [None, 0])
def test_default_warmup_then_constant(cooldown_steps):
    config = TrainConfig(steps=20)
    assert config.lr_schedule == WarmupStableDecay()
    schedule = WarmupStableDecay(cooldown_steps=cooldown_steps)
    assert [schedule(step, 20) for step in range(20)] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [1.0] * 10
    )


def test_linear_cooldown_matches_titan():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    reference = LRSchedulersContainer.Config(
        warmup_steps=10, decay_ratio=0.25, decay_type="linear", min_lr_factor=0.1
    ).build(optimizers=[optimizer], training_steps=20)
    schedule = WarmupStableDecay(cooldown_steps=5)
    for current_step in range(21):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(schedule(current_step, 20))
        optimizer.step()
        reference.step()


@pytest.mark.parametrize(
    "lr_schedule",
    [
        WarmupStableDecay(cooldown_steps=5),
        lambda current_step, total_steps: 1 - current_step / total_steps,
    ],
)
def test_schedule_restores_update_position_and_group_rates(lr_schedule):
    def build():
        optimizer = torch.optim.SGD(
            [
                {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3},
                {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-4},
            ]
        )
        schedulers = ScheduleConfig(lr_schedule=lr_schedule).build(
            optimizers=[optimizer], training_steps=20
        )
        return optimizer, schedulers

    optimizer, schedulers = build()
    for _ in range(7):
        optimizer.step()
        schedulers.step()
    resumed_optimizer, resumed_schedulers = build()
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    with pytest.warns(UserWarning, match="get_last_lr"):
        resumed_schedulers.load_state_dict(schedulers.state_dict())

    for current_step in range(7, 20):
        expected = [base * lr_schedule(current_step, 20) for base in (1e-3, 1e-4)]
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(expected)
        assert [group["lr"] for group in resumed_optimizer.param_groups] == pytest.approx(expected)
        optimizer.step()
        schedulers.step()
        resumed_optimizer.step()
        resumed_schedulers.step()
