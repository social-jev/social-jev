"""Post-training settings for SocialJev checkpoints."""

import os
import socket
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download

from .data import Sample

__all__ = ["Sample", "TrainConfig"]


@dataclass(kw_only=True, frozen=True)
class TrainConfig:
    """Configure supervised post-training."""

    steps: int
    seq_len: int = 2048
    local_batch_size: int = 1
    global_batch_size: int = -1
    data_parallel_replicate_degree: int = 1
    data_parallel_shard_degree: int = -1
    head_lr: float = 1e-3
    backbone_lr: float | None = None
    eval_steps: int | None = None
    val_fraction: float | None = None

    def __post_init__(self):  # noqa: C901 -- check independent public settings together.
        """Validate settings that do not depend on the available devices."""
        for name in ("steps", "seq_len", "local_batch_size", "data_parallel_replicate_degree"):
            if getattr(self, name) < 1:
                message = f"{name} must be positive."
                raise ValueError(message)
        for name in ("global_batch_size", "data_parallel_shard_degree"):
            value = getattr(self, name)
            if value != -1 and value < 1:
                message = f"{name} must be positive or -1."
                raise ValueError(message)
        for name in ("head_lr", "backbone_lr"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                message = f"{name} must be finite and positive."
                raise ValueError(message)
        if self.eval_steps is not None and self.eval_steps < 1:
            message = "eval_steps must be positive."
            raise ValueError(message)
        if self.val_fraction is not None and not 0 < self.val_fraction < 1:
            message = "val_fraction must be between zero and one."
            raise ValueError(message)


def run(repo, *, revision, trainset, valset, config, stop_inference):  # noqa: C901, PLR0913 -- own the training process and cleanup.
    """Load source weights and run TorchTitan on the available GPUs."""
    if not torch.cuda.is_available():
        message = "TorchTitan training requires a CUDA device."
        raise RuntimeError(message)

    from .titan import TrainedModel, Trainer  # noqa: PLC0415 -- optional training dependencies.

    if config.val_fraction is not None:
        if valset is not None:
            message = "Choose valset or val_fraction."
            raise ValueError(message)
        dataset = trainset()
        boundary = int(len(dataset) * (1 - config.val_fraction))

        def trainset():
            return (dataset[index] for index in range(boundary))

        def valset():
            return (dataset[index] for index in range(boundary, len(dataset)))

    checkpoint = (
        Path(repo).resolve()
        if Path(repo).is_dir()
        else Path(snapshot_download(repo, revision=revision))
    )
    environment = {}
    if "LOCAL_RANK" not in os.environ:
        # give a direct single-device call the same process settings as torchrun
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            port = str(server.getsockname()[1])
        environment = {
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
        }

    if stop_inference is not None:
        stop_inference()
        torch.cuda.empty_cache()

    with patch.dict(os.environ, environment), tempfile.TemporaryDirectory() as run_dir:
        owns_process_group = not dist.is_initialized()
        trainer = None
        keep_process_group = False
        try:
            trainer = Trainer.from_checkpoint(checkpoint, config, trainset, valset, run_dir)
            trainer.train()
            if valset is not None and config.eval_steps is None and trainer.step < config.steps:
                trainer.validator.validate(trainer.model_parts, trainer.step)
            model = trainer.model_parts[0].eval()
            model.source_checkpoint = checkpoint
            model.tokenizer = trainer.tokenizer.tokenizer
            if dist.get_world_size() > 1:
                keep_process_group = True
                return TrainedModel(model, owns_process_group)
            return model
        finally:
            if trainer is not None:
                trainer.close()
            if owns_process_group and dist.is_initialized() and not keep_process_group:
                dist.destroy_process_group()
