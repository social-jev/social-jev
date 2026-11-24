"""Exercise training orchestration on CPU when the TorchTitan runtime is installed."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from torch.distributed import checkpoint as dcp

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


@pytest.mark.parametrize(("shards", "replicas"), [(1, 4), (2, 2), (4, 1)])
def test_training_configures_replication_and_checkpoints(tmp_path, monkeypatch, shards, replicas):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(Trainer.Config, "build", lambda self: self)
    config = Trainer.from_checkpoint(
        tmp_path,
        TrainConfig(steps=100, data_parallel_shard_degree=shards, checkpoint_steps=25),
        source,
        None,
        tmp_path / "run",
    )
    assert config.parallelism.data_parallel_replicate_degree == replicas
    assert config.parallelism.data_parallel_shard_degree == shards
    assert config.checkpoint.enable
    assert config.checkpoint.interval == 25
    assert not config.checkpoint.last_save_model_only
    assert config.model_spec.parallelize_fn.keywords["reduce_dtype"] == torch.float32


def test_run_without_directory_rejects_checkpoint_interval_before_loading(monkeypatch):
    download = Mock()
    monkeypatch.setattr("social_jev.training.trainer.snapshot_download", download)
    with pytest.raises(ValueError, match="checkpoint_steps requires run_dir"):
        run(
            "org/model",
            revision=None,
            trainset=source,
            valset=None,
            config=TrainConfig(steps=3, checkpoint_steps=1),
            run_dir=None,
            save=None,
            stop_inference=None,
        )
    download.assert_not_called()


def test_training_export_roundtrip_preserves_weights_and_tokenizer(
    checkpoint, tmp_path, monkeypatch
):
    loaded = SocialJev.from_pretrained(checkpoint)
    model = TitanModel(TitanModel.Config(checkpoint=str(checkpoint), train_backbone=True))
    model.load_state_dict(loaded.engine.model.state_dict())
    # represent an update before exporting
    with torch.no_grad():
        model.decision_head.network[0].weight.add_(0.25)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    exported = model.to_hf(loaded.processor.tokenizer)
    assert exported is not None
    assert not exported.training
    assert all(parameter.device.type == "cpu" for parameter in exported.parameters())
    destination = tmp_path / "export"
    exported.save_pretrained(destination, max_shard_size="5GB")
    restored = SocialJev.from_pretrained(destination)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(restored.engine.model.state_dict()[name], parameter)
    assert restored.processor.tokenizer.chat_template == loaded.processor.tokenizer.chat_template
    assert restored.processor.tokenizer.get_vocab() == loaded.processor.tokenizer.get_vocab()
    batch = packed_batches(
        SampleStep(source, 0, 1, infinite=False),
        loaded.processor.tokenizer,
        seq_len=64,
        batch_size=1,
    ).next()
    assert batch is not None
    with torch.no_grad():
        torch.testing.assert_close(exported(batch), model(batch))
    loaded.stop()
    restored.stop()


@pytest.mark.parametrize("destination", ["local", "hub", "none", "other-rank", "failure"])
def test_training_saves_requested_destination_and_cleans_up(tmp_path, monkeypatch, destination):
    exported = Mock()
    model = Mock()
    model.to_hf.return_value = None if destination == "other-rank" else exported
    trainer = SimpleNamespace(
        step=0,
        train=Mock(),
        model_parts=[model],
        tokenizer=SimpleNamespace(tokenizer=object()),
        close=Mock(),
    )
    monkeypatch.setattr(Trainer, "from_checkpoint", lambda *_args: trainer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    save = tmp_path / "export" if destination == "local" else "org/model"
    if destination == "none":
        save = None
    if destination == "failure":
        exported.push_to_hub.side_effect = RuntimeError("upload failed")
    context = (
        pytest.raises(RuntimeError, match="upload failed")
        if destination == "failure"
        else nullcontext()
    )
    with context:
        result = run(
            tmp_path,
            revision=None,
            trainset=source,
            valset=None,
            config=TrainConfig(steps=1),
            run_dir=None,
            save=save,
            stop_inference=None,
        )
        assert result is None
    trainer.close.assert_called_once()
    if destination == "local":
        exported.save_pretrained.assert_called_once_with(save, max_shard_size="5GB")
        exported.push_to_hub.assert_not_called()
    elif destination in {"hub", "failure"}:
        exported.push_to_hub.assert_called_once_with(save, max_shard_size="5GB")
        exported.save_pretrained.assert_not_called()
    else:
        exported.save_pretrained.assert_not_called()
        exported.push_to_hub.assert_not_called()
    if destination == "none":
        model.to_hf.assert_not_called()


def test_final_checkpoint_retains_optimizer_state(tmp_path, monkeypatch):
    from torchtitan.components.optimizer import (  # noqa: PLC0415 -- optional runtime.
        LRSchedulersContainer,
        OptimizersContainer,
        ParamGroupConfig,
    )

    monkeypatch.setattr(Trainer.Config, "build", lambda self: self)
    config = Trainer.from_checkpoint(
        tmp_path, TrainConfig(steps=2, checkpoint_steps=2), source, None, tmp_path / "run"
    )
    torch.distributed.init_process_group(
        "gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1
    )
    manager = None
    try:
        model = torch.nn.Linear(2, 1)
        optimizers = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=".*", optimizer_name="AdamW", optimizer_kwargs={"lr": 0.01}
                )
            ],
            implementation="for-loop",
        ).build(model_parts=[model])
        schedulers = LRSchedulersContainer(optimizers, lambda step: 1 - step / 4)
        model(torch.ones(1, 2)).sum().backward()
        optimizers.step()
        schedulers.step()
        expected = model.weight.detach().clone()
        optimizer = optimizers.optimizers[0]
        expected_state = deepcopy(optimizer.state[model.weight])
        expected_lr = optimizer.param_groups[0]["lr"]
        expected_scheduler = schedulers.state_dict()
        manager = config.checkpoint.build(
            dataloader=None,
            model_parts=[model],
            optimizers=optimizers,
            lr_schedulers=schedulers,
            states={"train_state": {"step": 2}},
            sd_adapter=None,
            base_folder=str(tmp_path / "run"),
        )
        save = Mock(wraps=manager.dcp_save)
        monkeypatch.setattr(manager, "dcp_save", save)
        assert not manager.save(1)
        assert manager.save(2, last_step=True)
        save.assert_called_once()
        saved = tmp_path / "run" / "step-2"
        metadata = dcp.FileSystemReader(saved).read_metadata()
        assert any("exp_avg" in key for key in metadata.state_dict_metadata)
        assert any("lr_scheduler" in key for key in metadata.state_dict_metadata)
        assert "train_state.step" in metadata.state_dict_metadata
        schedulers.step()
        optimizer.param_groups[0]["lr"] = 0.123
        for value in optimizer.state[model.weight].values():
            value.zero_()
        with torch.no_grad():
            model.weight.zero_()
        # torch titan recomputes the learning rate through get_lr() on restore
        with pytest.warns(UserWarning, match="To get the last learning rate"):
            assert manager.load(step=2)
        torch.testing.assert_close(model.weight, expected)
        torch.testing.assert_close(optimizer.state[model.weight], expected_state)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
        assert schedulers.state_dict() == expected_scheduler
        assert schedulers.schedulers[0].get_last_lr() == pytest.approx([expected_lr])
        manager.close()
        assert (saved / ".metadata").is_file()
    finally:
        if manager is not None:
            manager.close()
        torch.distributed.destroy_process_group()
