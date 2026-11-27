"""TorchTitan adapters for the packed SocialJev model."""

from __future__ import annotations

import os
import socket
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
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer
from torchtitan.components.optimizer.optimizer import ParamGroupConfig
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable, ParallelismConfig, TrainingConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer as TorchTitanTrainer
from transformers import AutoTokenizer

from ..config import SocialJevConfig
from ..data import SampleStep, packed_batches
from ..transformers import SocialJevModel, is_dcp_checkpoint
from .evaluation import ValAndEvalOrchestrator, autocast
from .logging import Logger, metadata_metrics
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
        trainset: Callable[[], Iterable[Sample]]
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
        self.local_batch_size = local_batch_size
        self.consumed = 0

    def __iter__(self):
        batches = packed_batches(
            SampleStep(
                self.config.trainset,
                self.rank,
                self.world_size,
                shuffle_buffer_size=self.config.shuffle_buffer_size,
            ),
            self.tokenizer,
            seq_len=self.seq_len,
            local_batch_size=self.local_batch_size,
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
        # export bf16 weights while retaining fp32 training and optimizer state
        for name, tensor in weights.items():
            tensor = cast("torch.Tensor", tensor)
            if tensor.is_floating_point():
                weights[name] = tensor.to(dtype=torch.bfloat16)
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


def parallelize(model, *, parallel_dims, compute_dtype, **kwargs):
    """Apply replication or sharding to the Transformers layers."""
    del kwargs
    if not parallel_dims.dp_enabled:
        return model
    policy = MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=torch.float32)
    dims = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    mesh = parallel_dims.get_mesh(dims)
    if parallel_dims.dp_shard > 1:
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
        self.compute_dtype = getattr(torch, config.training.mixed_precision_param)
        if dist.get_rank() == 0:
            self.metrics_processor.logger = Logger(
                directory=Path(self.config.dump_folder),
                metrics_logger=self.metrics_processor.logger,
            )

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
                with autocast(
                    self.device,
                    self.compute_dtype,
                    distributed=self.parallel_dims.dp_enabled,
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
        optimizer_kwargs = {
            "weight_decay": config.weight_decay,
            "betas": config.betas,
            "eps": config.eps,
        }
        groups = [
            ParamGroupConfig(
                pattern=r"^decision_head\.",
                optimizer_name="AdamW",
                optimizer_kwargs={**optimizer_kwargs, "lr": config.head_lr},
            ),
        ]
        if config.backbone_lr is not None:
            groups.append(
                ParamGroupConfig(
                    pattern=r"^backbone\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={**optimizer_kwargs, "lr": config.backbone_lr},
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
                parallelize_fn=partial(parallelize, compute_dtype=getattr(torch, config.dtype)),
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
                trainset=trainset, shuffle_buffer_size=config.shuffle_buffer_size
            ),
            training=TrainingConfig(
                steps=config.steps,
                seq_len=config.seq_len,
                local_batch_size=config.local_batch_size,
                global_batch_size=config.global_batch_size,
                dtype="float32",
                mixed_precision_param=config.dtype,
                mixed_precision_reduce="float32",
                max_norm=config.clip_grad_norm,
                disable_cuda_graphs=True,
            ),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=replicate_degree,
                data_parallel_shard_degree=config.data_parallel_shard_degree,
                spmd_backend="partial_dtensor",
            ),
            metrics=MetricsProcessor.Config(log_freq=config.log_steps),
            optimizer=OptimizersContainer.Config(param_groups=groups),
            lr_scheduler=ScheduleConfig(
                lr_schedule=config.lr_schedule,
                # torchtitan includes this value in its startup log
                warmup_steps=getattr(config.lr_schedule, "warmup_steps", 0),
            ),
            loss=CrossEntropyLoss.Config(),
            validator=ValAndEvalOrchestrator.Config(
                enable=valset is not None or bool(config.evaluations),
                evaluations=config.evaluations,
                eval_steps=config.eval_steps,
                valset=valset,
                loss_function=config.loss_fn,
                val_steps=config.val_steps,
                total_steps=config.steps,
                compute_dtype=getattr(torch, config.dtype),
            ),
        ).build()


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
        init_logger()
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
                if trainer.config.validator.enable:
                    trainer.validator.validate(trainer.model_parts, trainer.step)
                trainer.train()
                if trainer.config.validator.enable and trainer.validator.should_validate(
                    trainer.step
                ):
                    trainer.validator.validate(trainer.model_parts, trainer.step)
                if final_save_to is not None:
                    trained = trainer.model_parts[0].to_hf(trainer.tokenizer.tokenizer)
                    if trained is not None:
                        if isinstance(final_save_to, Path):
                            trained.save_pretrained(
                                final_save_to.expanduser(), max_shard_size="5GB"
                            )
                        else:
                            trained.push_to_hub(final_save_to, private=True, max_shard_size="5GB")
            finally:
                if trainer is not None:
                    trainer.close()
                del trained
                del trainer
                if owns_process_group and dist.is_initialized():
                    dist.destroy_process_group()
                torch.cuda.empty_cache()
