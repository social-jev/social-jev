"""TorchTitan adapters for the packed SocialJev model."""

from __future__ import annotations

import os
import socket
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast
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
from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer
from torchtitan.components.optimizer.optimizer import ParamGroupConfig
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import Validator
from torchtitan.config import Configurable, ParallelismConfig, TrainingConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer as TorchTitanTrainer
from transformers import AutoTokenizer

from ..config import SocialJevConfig
from ..data import SampleStep, packed_batches
from ..transformers import SocialJevModel, is_dcp_checkpoint
from .loss import Loss  # noqa: TC001 -- runtime configuration field type.
from .lr_schedule import LRSchedule, WarmupStableDecay

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from transformers import PreTrainedTokenizerBase

    from ..data import Sample
    from .config import TrainConfig


@dataclass(kw_only=True, slots=True)
class ScheduleConfig(LRSchedulersContainer.Config):
    """Build TorchTitan schedulers from the caller's learning-rate function."""

    lr_schedule: LRSchedule = field(default_factory=WarmupStableDecay)

    def build(self, *, optimizers, training_steps):
        def multiplier(current_step):
            return self.lr_schedule(current_step, training_steps)

        return LRSchedulersContainer(optimizers, multiplier)


class TitanTokenizer(Configurable):
    """Load the tokenizer saved with the source checkpoint."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    def __init__(self, config, tokenizer_path):
        del config
        self.tokenizer = cast(
            "PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(tokenizer_path)
        )


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
                yield {"input": batch.input_ids, "batch": batch}, batch.labels

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
        self.backbone.set_attn_implementation("flex_attention")
        self.backbone.requires_grad_(config.train_backbone)

    def verify_module_protocol(self):
        """The Transformers layers use the torch module interface."""

    def init_weights(self, *, buffer_device=None):
        """Restore the rotary buffers discarded by meta construction."""
        rotary = self.backbone.rotary_emb
        with torch.device(buffer_device or rotary.inv_freq.device):
            self.backbone.rotary_emb = type(rotary)(rotary.config)

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


class TrainingCheckpointManager(CheckpointManager):
    """Load the requested source once and save complete checkpoints."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        pass

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = kwargs["model_parts"][0]
        self.loaded = False
        self.tokenizer: PreTrainedTokenizerBase

    def load(self, step=-1):
        """Restore the explicit source, regardless of the output directory."""
        del step
        if self.loaded:
            return False
        assert self.initial_load_path is not None
        if self.initial_load_in_hf:
            weights = get_model_state_dict(self.model)
            dcp.load(
                weights,
                storage_reader=dcp.HuggingFaceStorageReader(self.initial_load_path),
            )
            set_model_state_dict(self.model, weights)
        else:
            # titan selects and restores all registered training state
            self.dcp_load(
                self._states_to_load(model_only=False),
                checkpoint_id=self.initial_load_path,
                from_hf=False,
                from_quantized=False,
            )
        self.loaded = True
        return True

    def save(self, curr_step, last_step=False):
        """Save training state and its configuration/tokenizer assets."""
        saved = super().save(curr_step, last_step)
        if saved and dist.get_rank() == 0:
            directory = Path(self.folder) / f"step-{curr_step}"
            self.model.config.save_pretrained(directory)
            self.tokenizer.save_pretrained(directory)
        return saved


class TitanTrainer(TorchTitanTrainer):
    """Run TorchTitan with one loss item per query."""

    @dataclass(kw_only=True, slots=True)
    class Config(TorchTitanTrainer.Config):
        loss_fn: Loss
        run_dir: str

    config: Config
    checkpointer: TrainingCheckpointManager
    tokenizer: TitanTokenizer

    def __init__(self, config):
        super().__init__(config)
        self.checkpointer.tokenizer = self.tokenizer.tokenizer
        self.loss = config.loss_fn

    def init_distributed(self):
        """Initialize ranks and resolve their shared output directory."""
        parallel_dims = super().init_distributed()
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

        count = torch.tensor(sum(len(labels) for _, labels in batches), device=self.device)
        if group is not None:
            dist.all_reduce(count, group=group)
        self.optimizers.zero_grad(set_to_none=True)
        model = self.model_parts[0]
        total = torch.zeros((), device=self.device)
        for inputs, _targets in batches:
            batch = inputs["batch"].to(self.device)
            with self.train_context():
                # the distributed wrapper handles mixed precision; on one gpu, use autocast
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda" and not self.parallel_dims.dp_enabled,
                ):
                    logits = model(batch)
                logits = logits.float()
                assert batch.labels is not None
                losses = self.loss(logits, batch.labels, batch.option_mask)
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
            run_dir=str(run_dir),
            dump_folder=str(run_dir),
            checkpoint=TrainingCheckpointManager.Config(
                enable=True,
                folder=".",
                initial_load_path=str(checkpoint),
                initial_load_in_hf=not is_dcp_checkpoint(checkpoint),
                initial_load_model_only=not is_dcp_checkpoint(checkpoint),
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
            lr_scheduler=ScheduleConfig(
                lr_schedule=config.lr_schedule,
                # torchtitan includes this value in its startup log
                warmup_steps=getattr(config.lr_schedule, "warmup_steps", 0),
            ),
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
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.group = parallel_dims.get_mesh("batch").get_group() if dp_world_size > 1 else None
        self.placeholder = None
        self.parallel_dims = parallel_dims
        self.loss = config.loss_fn
        self.tokenizer: PreTrainedTokenizerBase = tokenizer.tokenizer
        self.metrics_processor = metrics_processor

    def should_validate(self, step):
        """Validate at step zero, the requested intervals, and the final update."""
        return step in (0, self.config.steps) or (
            self.config.interval is not None and step % self.config.interval == 0
        )

    def if_exhausted_fill_placeholder(self, batch):
        """Keep exhausted ranks participating until every rank finishes."""
        if self.group is None:
            return batch
        available = torch.tensor([batch is not None], dtype=torch.int64, device=self.device)
        availability = torch.empty(self.dp_world_size, dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(availability, available, group=self.group)
        active_ranks = availability.nonzero().flatten()
        if active_ranks.numel() == 0:
            return None
        if self.placeholder is None:
            # share one real cpu batch, including with ranks that have no samples
            source_rank = int(active_ranks[0].item())
            shared = [batch if self.dp_rank == source_rank else None]
            dist.broadcast_object_list(
                shared,
                src=dist.get_global_rank(self.group, source_rank),
                group=self.group,
            )
            self.placeholder = shared[0]
        return self.placeholder if batch is None else batch

    @torch.no_grad()
    def validate(self, model_parts, step):
        """Evaluate disjoint samples and combine losses across ranks."""
        model = model_parts[0].eval()
        self.device = model.device
        self.placeholder = None
        source = self.config.source
        assert source is not None
        batches = packed_batches(
            SampleStep(source, self.dp_rank, self.dp_world_size, infinite=False),
            self.tokenizer,
            seq_len=self.config.seq_len,
            batch_size=self.config.batch_size,
        )
        totals = torch.zeros(2, dtype=torch.float64, device=self.device)
        observations = []
        try:
            while True:
                batch = batches.next()
                exhausted = batch is None
                batch = self.if_exhausted_fill_placeholder(batch)
                if batch is None:
                    break
                batch = batch.to(self.device)
                # the distributed wrapper handles mixed precision; on one gpu, use autocast
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda" and not self.parallel_dims.dp_enabled,
                ):
                    logits = model(batch)
                if exhausted:
                    continue
                labels = batch.labels
                assert labels is not None
                losses = self.loss(logits.float(), labels, batch.option_mask)
                totals[0] += losses.double().sum()
                totals[1] += losses.numel()
                observations.append((losses.cpu(), batch.metadata))
            if self.group is not None:
                dist.all_reduce(totals, group=self.group)
            categories = metadata_metrics(observations, "eval", self.group)
            if totals[1].item():
                mean = (totals[0] / totals[1]).item()
                self.metrics_processor.log_validation(mean, step, {"eval/loss": mean, **categories})
            self.last_step = step
        finally:
            self.placeholder = None
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


class Trainer:
    """Own a training session from checkpoint loading through cleanup."""

    def __init__(  # noqa: PLR0913 -- session inputs.
        self,
        config: TrainConfig,
        *,
        repo: str | Path,
        trainset,
        valset=None,
        revision=None,
        run_dir: str | Path | None = None,
    ):
        self.config = config
        self.repo = repo
        self.revision = revision
        self.trainset = trainset
        self.valset = valset
        self.run_dir = "artifacts/train-%Y-%m-%d--%H-%M" if run_dir is None else run_dir

    def run(self, *, final_save_to: str | Path | None = None) -> None:  # noqa: C901 -- own the training process and cleanup.
        """Train, optionally export the final model, and release resources."""
        if not torch.cuda.is_available():
            message = "TorchTitan training requires a CUDA device."
            raise RuntimeError(message)

        checkpoint = (
            Path(self.repo).expanduser().resolve()
            if isinstance(self.repo, Path)
            else Path(snapshot_download(str(self.repo), revision=self.revision))
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

        with patch.dict(os.environ, environment):
            owns_process_group = not dist.is_initialized()
            trainer = None
            trained = None
            try:
                trainer = TitanTrainer.from_checkpoint(
                    checkpoint, self.config, self.trainset, self.valset, self.run_dir
                )
                trainer.checkpointer.load()
                if final_save_to is None and dist.get_rank() == 0:
                    logger.warning(
                        "No final HF export is configured. "
                        "The final checkpoint will be saved in DCP format under %s.",
                        trainer.config.dump_folder,
                    )
                if self.valset is not None:
                    trainer.validator.validate(trainer.model_parts, trainer.step)
                trainer.train()
                if self.valset is not None and trainer.validator.last_step != trainer.step:
                    trainer.validator.validate(trainer.model_parts, trainer.step)
                if final_save_to is not None:
                    trained = trainer.model_parts[0].to_hf(trainer.tokenizer.tokenizer)
                    if trained is not None:
                        if isinstance(final_save_to, Path):
                            trained.save_pretrained(
                                final_save_to.expanduser(), max_shard_size="5GB"
                            )
                        else:
                            trained.push_to_hub(final_save_to, max_shard_size="5GB")
            finally:
                if trainer is not None:
                    trainer.close()
                del trained
                del trainer
                if owns_process_group and dist.is_initialized():
                    dist.destroy_process_group()
                torch.cuda.empty_cache()
