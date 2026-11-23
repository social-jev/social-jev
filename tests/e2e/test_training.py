"""Exercise training orchestration on CPU when the TorchTitan runtime is installed."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from social_jev import SocialJev
from social_jev.data import Sample, SampleStep, packed_batches
from social_jev.training import BrierLoss, CrossEntropy, FocalLoss, TrainConfig

pytest.importorskip("torchtitan")
pytest.importorskip("triton")

from social_jev.training.trainer import (
    TitanModel,
    Trainer,
    Validation,
    run,
)


def source():
    yield Sample(
        [
            {"role": "state", "content": "S"},
            {"role": "query", "content": "Q"},
            {"role": "option", "content": "A"},
            {"role": "option", "content": "B"},
        ],
        targets=[[0.25, 0.75]],
        metadata={"language": "en"},
    )


@pytest.mark.parametrize("interval", [None, 2])
def test_validation_schedule_and_custom_loss(checkpoint, interval):
    model = SocialJev.from_pretrained(checkpoint)
    wrapped = TitanModel(TitanModel.Config(checkpoint=str(checkpoint), train_backbone=True))
    wrapped.load_state_dict(model.engine.model.state_dict())
    loss_fn = Mock(wraps=BrierLoss())
    metrics = Mock()
    validator = Validation(
        Validation.Config(
            enable=True,
            source=source,
            loss_fn=loss_fn,
            interval=interval,
            seq_len=64,
            batch_size=2,
            steps=5,
        ),
        dp_world_size=1,
        dp_rank=0,
        tokenizer=SimpleNamespace(tokenizer=model.processor.tokenizer),
        parallel_dims=None,
        metrics_processor=metrics,
    )
    scheduled = [step for step in range(6) if validator.should_validate(step)]
    assert scheduled == ([0, 5] if interval is None else [0, 2, 4, 5])
    validator.validate([wrapped], 0)
    assert loss_fn.call_count == 1
    assert wrapped.training
    logged, step, extras = metrics.log_validation.call_args.args
    assert step == 0
    assert extras["eval/language/en/loss"] == pytest.approx(logged)
    assert 0 <= logged <= 2
    assert all(parameter.grad is None for parameter in wrapped.parameters())
    model.stop()


@pytest.mark.parametrize("loss_fn", [CrossEntropy(), FocalLoss(), BrierLoss()])
def test_accumulated_update_matches_mean_query_loss(checkpoint, loss_fn):
    loaded = SocialJev.from_pretrained(checkpoint)
    model = TitanModel(TitanModel.Config(checkpoint=str(checkpoint), train_backbone=True))
    model.load_state_dict(loaded.engine.model.state_dict())
    reference = deepcopy(model)
    batches = packed_batches(
        SampleStep(source, 0, 1, infinite=False),
        loaded.processor.tokenizer,
        seq_len=64,
        batch_size=1,
    )
    batch = batches.next()
    assert batch is not None
    assert batch.targets is not None
    repeated = deepcopy(batch)
    repeated.targets = 1 - batch.targets
    inputs = [({"batch": item}, item.targets) for item in (batch, repeated, repeated)]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    # exercise the actual update without initializing CUDA or distributed startup
    scheduler = Mock()
    trainer = cast(
        "Trainer",
        SimpleNamespace(
            config=SimpleNamespace(loss_fn=loss_fn, training=SimpleNamespace(max_norm=100.0)),
            model_parts=[model],
            device=torch.device("cpu"),
            gradient_accumulation_steps=len(inputs),
            parallel_dims=SimpleNamespace(dp_enabled=False),
            optimizers=optimizer,
            lr_schedulers=scheduler,
            metrics_processor=SimpleNamespace(
                should_log=lambda _step: False, ntokens_since_last_log=0
            ),
            train_context=nullcontext,
            step=1,
            ntokens_seen=0,
        ),
    )
    Trainer.train_step(trainer, iter(inputs))

    expected = torch.cat(
        [
            loss_fn(reference(item), item.targets, item.option_indices >= 0)
            for item in (batch, repeated, repeated)
        ]
    ).mean()
    expected.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 100.0)
    expected_optimizer.step()
    for actual, expected_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected_parameter)
    scheduler.step.assert_called_once()
    loaded.stop()


def test_run_validates_before_training_and_once_at_end(checkpoint, monkeypatch):
    events = []
    model = torch.nn.Linear(1, 1)
    validator = SimpleNamespace(last_step=None)

    def validate(model_parts, step):
        assert model_parts == [model]
        events.append(("validate", step))
        validator.last_step = step

    def train():
        events.append(("train", trainer.step))
        trainer.step = 3
        validator.validate([model], 3)

    validator.validate = validate
    trainer = SimpleNamespace(
        step=0,
        validator=validator,
        train=train,
        model_parts=[model],
        tokenizer=SimpleNamespace(tokenizer=None),
        close=lambda: events.append(("close", 3)),
    )
    monkeypatch.setattr(Trainer, "from_checkpoint", lambda *_args: trainer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    trained = run(
        checkpoint,
        revision=None,
        trainset=source,
        valset=source,
        config=TrainConfig(steps=3),
        run_dir=None,
        save=None,
        stop_inference=None,
    )
    assert events == [("validate", 0), ("train", 0), ("validate", 3), ("close", 3)]
    assert trained is None
