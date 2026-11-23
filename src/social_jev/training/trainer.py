"""TorchTitan adapters for the packed SocialJev model."""

from __future__ import annotations

import os
import socket
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.fsdp import DataParallelMeshDims, MixedPrecisionPolicy, fully_shard
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.optimizer.optimizer import ParamGroupConfig
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import Validator
from torchtitan.config import Configurable, ParallelismConfig, TrainingConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer as TorchTitanTrainer
from transformers import AutoTokenizer

from ..config import SocialJevConfig
from ..data import SampleStep, packed_batches
from ..transformers import SocialJevModel
from .loss import Loss  # noqa: TC001 -- runtime configuration field type.

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from transformers import PreTrainedTokenizerBase

    from ..data import Sample
    from .config import TrainConfig


class TitanTokenizer(Configurable):
    """Load the tokenizer saved with the source checkpoint."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    def __init__(self, config, tokenizer_path):
        del config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


class TitanLoader(BaseDataLoader):
    """Build packed CPU batches from a callable supplied to train()."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        source: Callable[[], Iterable[Sample]]
        shuffle_buffer_size: int = 256

    def __init__(  # noqa: PLR0913 -- torch titan loader interface.
        self,
        config,
        *,
        dp_rank,
        dp_world_size,
        tokenizer,
        seq_len,
        local_batch_size,
        snapshot_every_n_steps=None,
    ):
        del snapshot_every_n_steps
        self.config = config
        self.rank = dp_rank
        self.world_size = dp_world_size
        self.tokenizer = tokenizer.tokenizer
        self.seq_len = seq_len
        self.batch_size = local_batch_size
        self.consumed = 0

    def __iter__(self):
        batches = packed_batches(
            SampleStep(
                self.config.source,
                self.rank,
                self.world_size,
                shuffle_buffer_size=self.config.shuffle_buffer_size,
            ),
            self.tokenizer,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
        )
        for _ in range(self.consumed):
            batches.next()
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(batches.next)
            while (batch := pending.result()) is not None:
                pending = executor.submit(batches.next)
                self.consumed += 1
                yield {"input": batch.input_ids, "batch": batch}, batch.targets

    def state_dict(self):
        """Save the number of batches consumed by the trainer."""
        return {"consumed": self.consumed}

    def load_state_dict(self, state_dict):
        """Resume from the saved batch position."""
        self.consumed = state_dict["consumed"]


class TitanModel(SocialJevModel, Configurable):
    """Adapt the existing Transformers model to TorchTitan's construction hooks."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        checkpoint: str
        train_backbone: bool

        def update_from_config(self, *, config, **kwargs):
            """Read architecture settings from the checkpoint."""

        def get_nparams_and_flops(self, model, seq_len):
            """Estimate training flops from the model size and attention length."""
            count = sum(parameter.numel() for parameter in model.parameters())
            config = model.config.backbone_config
            return count, 6 * count + 12 * config.num_hidden_layers * config.hidden_size * seq_len

    def __init__(self, config):
        super().__init__(SocialJevConfig.from_pretrained(config.checkpoint))
        self.backbone.requires_grad_(config.train_backbone)

    def verify_module_protocol(self):
        """The Transformers layers use the torch module interface."""

    def init_weights(self, *, buffer_device=None):
        """Restore the rotary buffers discarded by meta construction."""
        rotary = self.backbone.rotary_emb
        with torch.device(buffer_device or rotary.inv_freq.device):
            self.backbone.rotary_emb = type(rotary)(rotary.config)

    def forward(self, input_ids, attention_mask=None):
        """Score a packed batch or the base model input tensors."""
        if attention_mask is None:
            return self.forward_packed(input_ids)
        return super().forward(input_ids, attention_mask)

    def to_hf(self, tokenizer) -> SocialJevModel | None:
        """Collect an ordinary Transformers model on rank zero."""
        weights = get_model_state_dict(
            self, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        if dist.get_rank() != 0:
            return None
        # attach collected tensors without allocating another set of weights
        with torch.device("meta"):
            model = SocialJevModel(deepcopy(self.config))
        model.load_state_dict(weights, assign=True)
        # checkpoints omit these nonpersistent rotary buffers
        rotary = model.backbone.rotary_emb
        with torch.device("cpu"):
            model.backbone.rotary_emb = type(rotary)(model.backbone.config)
        model.tokenizer = tokenizer
        return model.eval()


def parallelize(model, *, parallel_dims, reduce_dtype, **kwargs):
    """Apply replication or sharding to the Transformers layers."""
    del kwargs
    if not parallel_dims.dp_enabled:
        return model
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=reduce_dtype)
    if parallel_dims.dp_shard == 1:
        # replication only: every rank retains the complete model
        fully_shard(
            model,
            mesh=parallel_dims.get_mesh("dp_replicate"),
            dp_mesh_dims=DataParallelMeshDims(replicate="dp_replicate"),
            mp_policy=policy,
        )
    else:
        dims = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        mesh = parallel_dims.get_mesh(dims)
        for layer in model.backbone.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
    disable_fsdp_gradient_division(model)
    return model


class Trainer(TorchTitanTrainer):
    """Run TorchTitan with one loss item per query."""

    @dataclass(kw_only=True, slots=True)
    class Config(TorchTitanTrainer.Config):
        loss_fn: Loss
        run_dir: str | None

    config: Config

    def __init__(self, config):
        super().__init__(config)
        model = self.model_parts[0]
        weights = get_model_state_dict(model)
        dcp.load(weights, storage_reader=dcp.HuggingFaceStorageReader(config.hf_assets_path))
        set_model_state_dict(model, weights)

    def init_distributed(self):
        """Initialize ranks and resolve their shared output directory."""
        parallel_dims = super().init_distributed()
        if self.config.run_dir is not None:
            directory = [
                datetime.now().astimezone().strftime(self.config.run_dir)
                if dist.get_rank() == 0
                else None
            ]
            dist.broadcast_object_list(directory, src=0)
            assert directory[0] is not None
            self.config.dump_folder = directory[0]
        return parallel_dims

    def train_step(self, data_iterator):
        """Accumulate per-query losses normalized across the complete distributed update."""
        should_log = self.metrics_processor.should_log(self.step)
        observations = []
        group = (
            self.parallel_dims.get_mesh("batch").get_group()
            if self.parallel_dims.dp_enabled
            else None
        )
        batches = [next(data_iterator) for _ in range(self.gradient_accumulation_steps)]

        count = torch.tensor(sum(len(targets) for _, targets in batches), device=self.device)
        if group is not None:
            dist.all_reduce(count, group=group)
        self.optimizers.zero_grad(set_to_none=True)
        model = self.model_parts[0]
        total = torch.zeros((), device=self.device)
        for inputs, _targets in batches:
            batch = inputs["batch"].to(self.device)
            with self.train_context():
                logits = model(batch).float()
                assert batch.targets is not None
                losses = self.config.loss_fn(logits, batch.targets, batch.option_indices >= 0)
                loss = losses.sum() / count
                loss.backward()
            if should_log:
                observations.append((losses.detach(), batch.metadata))
            total += loss.detach()
            self.ntokens_seen += batch.input_ids.numel()
            self.metrics_processor.ntokens_since_last_log += batch.input_ids.numel()
        norm = dist_utils.clip_grad_norm_(list(model.parameters()), self.config.training.max_norm)
        self.optimizers.step()
        self.lr_schedulers.step()
        if should_log:
            mesh = self.parallel_dims.get_optional_mesh("loss")
            mean = dist_utils.dist_sum(total, mesh)
            maximum = dist_utils.dist_max(total * count / sum(len(y) for _, y in batches), mesh)
            extras = {"train/loss": mean, **self.lr_schedulers.get_metrics()}
            extras.update(metadata_metrics(observations, "train", group))
            self.metrics_processor.log(self.step, mean, maximum, float(norm.item()), extras)

    @classmethod
    def from_checkpoint(cls, checkpoint, config: TrainConfig, trainset, valset, run_dir):
        """Build TorchTitan with the source checkpoint and two learning rates."""
        world_size = (
            dist.get_world_size()
            if dist.is_initialized()
            else int(os.environ.get("WORLD_SIZE", "1"))
        )
        replicate_degree = config.data_parallel_replicate_degree
        if replicate_degree is None:
            replicate_degree = world_size // config.data_parallel_shard_degree
        groups = [
            ParamGroupConfig(
                pattern=r"^decision_head\.",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": config.head_lr},
            ),
        ]
        if config.backbone_lr is not None:
            groups.append(
                ParamGroupConfig(
                    pattern=r"^backbone\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": config.backbone_lr},
                ),
            )
        return cls.Config(
            loss_fn=config.loss_fn,
            model_spec=ModelSpec(
                name="social_jev",
                flavor="post-training",
                model=TitanModel.Config(
                    checkpoint=str(checkpoint),
                    train_backbone=config.backbone_lr is not None,
                ),
                parallelize_fn=partial(parallelize, reduce_dtype=config.reduce_dtype),
                pipelining_fn=None,
                post_optimizer_build_fn=None,
                state_dict_adapter=None,
            ),
            hf_assets_path=str(checkpoint),
            run_dir=None if run_dir is None else str(run_dir),
            dump_folder="." if run_dir is None else str(run_dir),
            checkpoint=CheckpointManager.Config(
                enable=run_dir is not None,
                folder=".",
                interval=config.checkpoint_steps or config.steps,
                last_save_model_only=False,
                last_save_in_hf=False,
                keep_latest_k=0,
            ),
            tokenizer=TitanTokenizer.Config(),
            dataloader=TitanLoader.Config(
                source=trainset, shuffle_buffer_size=config.shuffle_buffer_size
            ),
            training=TrainingConfig(
                steps=config.steps,
                seq_len=config.seq_len,
                local_batch_size=config.local_batch_size,
                global_batch_size=config.global_batch_size,
                disable_cuda_graphs=True,
            ),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=replicate_degree,
                data_parallel_shard_degree=config.data_parallel_shard_degree,
                spmd_backend="partial_dtensor",
            ),
            optimizer=OptimizersContainer.Config(param_groups=groups),
            loss=CrossEntropyLoss.Config(),
            validator=Validation.Config(
                enable=valset is not None,
                source=valset,
                loss_fn=config.loss_fn,
                interval=config.val_steps,
                seq_len=config.seq_len,
                batch_size=config.local_batch_size,
                steps=config.steps,
            ),
        ).build()


class Validation(Validator):
    """Evaluate packed queries without modifying the model weights."""

    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        source: Callable[[], Iterable[Sample]] | None
        loss_fn: Loss
        interval: int | None
        seq_len: int
        batch_size: int
        steps: int

    def __init__(  # noqa: PLR0913 -- torch titan validator interface.
        self,
        config,
        *,
        dp_world_size,
        dp_rank,
        tokenizer,
        parallel_dims,
        metrics_processor,
        **kwargs,
    ):
        del kwargs
        self.config: Validation.Config = config
        self.last_step: int | None = None
        del dp_rank, dp_world_size, parallel_dims
        self.tokenizer: PreTrainedTokenizerBase = tokenizer.tokenizer
        self.metrics_processor = metrics_processor

    def should_validate(self, step):
        """Validate at step zero, the requested intervals, and the final update."""
        return step in (0, self.config.steps) or (
            self.config.interval is not None and step % self.config.interval == 0
        )

    @torch.no_grad()
    def validate(self, model_parts, step):
        """Evaluate the same sequence of validation batches on every rank."""
        model = model_parts[0].eval()
        source = self.config.source
        assert source is not None
        batches = packed_batches(
            SampleStep(source, 0, 1, infinite=False),
            self.tokenizer,
            seq_len=self.config.seq_len,
            batch_size=self.config.batch_size,
        )
        total = torch.zeros((), device=model.device)
        queries = 0
        observations = []
        try:
            while (batch := batches.next()) is not None:
                batch = batch.to(model.device)
                logits = model(batch).float()
                targets = batch.targets
                assert targets is not None
                losses = self.config.loss_fn(logits, targets, batch.option_indices >= 0)
                total += losses.sum()
                queries += len(targets)
                observations.append((losses.to("cpu"), batch.metadata))
            if queries:
                mean = float(total.item()) / queries
                extras = {"eval/loss": mean, **metadata_metrics(observations, "eval", None)}
                self.metrics_processor.log_validation(mean, step, extras)
            self.last_step = step
        finally:
            model.train()


def metadata_metrics(observations, prefix, group):
    """Average detached losses for each metadata value."""
    sums: Counter[tuple[str, str]] = Counter()
    counts: Counter[tuple[str, str]] = Counter()
    for losses, metadata in observations:
        for loss, values in zip(losses.tolist(), metadata, strict=True):
            for key_value in values.items():
                sums[key_value] += loss
                counts[key_value] += 1
    if group is not None:
        parts = [(Counter(), Counter()) for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(parts, (sums, counts), group=group)
        sums = sum((item[0] for item in parts), Counter())
        counts = sum((item[1] for item in parts), Counter())
    return {
        f"{prefix}/{name}/{value}/{metric}": result
        for (name, value), count in counts.items()
        for metric, result in (("loss", sums[name, value] / count), ("queries", count))
    }


def run(repo, *, revision, trainset, valset, config, run_dir, save, stop_inference):  # noqa: C901, PLR0912, PLR0913 -- own the training process and cleanup.
    """Load source weights and run TorchTitan on the available GPUs."""
    if run_dir is None and config.checkpoint_steps is not None:
        message = "checkpoint_steps requires run_dir."
        raise ValueError(message)
    if not torch.cuda.is_available():
        message = "TorchTitan training requires a CUDA device."
        raise RuntimeError(message)

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

    with patch.dict(os.environ, environment):
        owns_process_group = not dist.is_initialized()
        trainer = None
        trained = None
        try:
            trainer = Trainer.from_checkpoint(checkpoint, config, trainset, valset, run_dir)
            if valset is not None:
                trainer.validator.validate(trainer.model_parts, trainer.step)
            trainer.train()
            if valset is not None and trainer.validator.last_step != trainer.step:
                trainer.validator.validate(trainer.model_parts, trainer.step)
            if save is not None:
                trained = trainer.model_parts[0].to_hf(trainer.tokenizer.tokenizer)
                if trained is not None:
                    if isinstance(save, Path):
                        trained.save_pretrained(save.expanduser(), max_shard_size="5GB")
                    else:
                        trained.push_to_hub(save, max_shard_size="5GB")
        finally:
            if trainer is not None:
                trainer.close()
            del trained
            del trainer
            if owns_process_group and dist.is_initialized():
                dist.destroy_process_group()
            torch.cuda.empty_cache()
