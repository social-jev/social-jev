"""TorchTitan adapters for the packed SocialJev model."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from huggingface_hub import HfApi
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError
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

from .config import SocialJevConfig
from .packing import SampleStep, packed_batches
from .transformers import SocialJevModel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from transformers import PreTrainedTokenizerBase

    from .data import Sample
    from .training import TrainConfig


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
        self.ranks = dp_world_size
        self.tokenizer = tokenizer.tokenizer
        self.seq_len = seq_len
        self.batch_size = local_batch_size
        self.consumed = 0

    def __iter__(self):
        batches = packed_batches(
            SampleStep(self.config.source, self.rank, self.ranks),
            self.tokenizer,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            drop_last=True,
            fixed_length=True,
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

    def save_pretrained(self, save_directory, **kwargs):
        """Save updated weights with the tokenizer and the source checkpoint code."""
        super().save_pretrained(save_directory, **kwargs)
        self.tokenizer.save_pretrained(save_directory)
        for source in self.source_checkpoint.rglob("*.py"):
            target = Path(save_directory) / source.relative_to(self.source_checkpoint)
            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, target)
        copy2(self.source_checkpoint / "config.json", Path(save_directory) / "config.json")


class TrainedModel:
    """Hold the distributed model until caller publication or cleanup."""

    def __init__(self, model, owns_process_group):
        self.model: TitanModel | None = model
        self.owns_process_group = owns_process_group

    def push_to_hub(self, repo_id, *, private=None, revision=None):
        """Collect the weights on rank zero and publish the complete checkpoint."""
        model = self.model
        assert model is not None
        weights = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        status = [None, None]
        if dist.get_rank() == 0:
            try:
                with TemporaryDirectory() as directory:
                    model.save_pretrained(directory, state_dict=weights)
                    api = HfApi()
                    api.create_repo(repo_id, private=private, exist_ok=True)
                    status[0] = api.upload_folder(
                        repo_id=repo_id, folder_path=directory, revision=revision
                    ).commit_url
            except Exception as error:  # noqa: BLE001 -- return publication errors to every rank.
                status[1] = str(error)
        dist.broadcast_object_list(status, src=0)
        if status[1] is not None:
            message = f"Checkpoint publication failed: {status[1]}"
            raise RuntimeError(message)
        return status[0]

    def stop(self):
        """Release the local weight shard and its process group."""
        self.model = None
        if self.owns_process_group:
            dist.destroy_process_group()


def parallelize(model, *, parallel_dims, **kwargs):
    """Apply the requested data-parallel mesh to the Transformers layers."""
    del kwargs
    if not parallel_dims.dp_enabled:
        return model
    dims = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    mesh = parallel_dims.get_mesh(dims)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.backbone.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    disable_fsdp_gradient_division(model)
    return model


class Trainer(TorchTitanTrainer):
    """Run TorchTitan with one loss item per query."""

    @dataclass(kw_only=True, slots=True)
    class Config(TorchTitanTrainer.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        model = self.model_parts[0]
        weights = get_model_state_dict(model)
        dcp.load(weights, storage_reader=dcp.HuggingFaceStorageReader(config.hf_assets_path))
        set_model_state_dict(model, weights)

    def train_step(self, data_iterator):  # noqa: C901 -- coordinate each update across ranks.
        """Accumulate loss over queries, with all ranks stopping together."""
        should_log = self.metrics_processor.should_log(self.step)
        observations = []
        group = (
            self.parallel_dims.get_mesh("batch").get_group()
            if self.parallel_dims.dp_enabled
            else None
        )
        window = []
        for _ in range(self.gradient_accumulation_steps):
            try:
                inputs, targets = next(data_iterator)
                window.append((inputs, targets))
            except DataloaderExhaustedError:
                break
        complete = torch.tensor(
            int(len(window) == self.gradient_accumulation_steps), device=self.device
        )
        if group is not None:
            dist.all_reduce(complete, op=dist.ReduceOp.MIN, group=group)
        if not complete.item():
            self.step -= 1
            raise DataloaderExhaustedError

        count = torch.tensor(sum(len(targets) for _, targets in window), device=self.device)
        if group is not None:
            dist.all_reduce(count, group=group)
        self.optimizers.zero_grad(set_to_none=True)
        model = self.model_parts[0]
        total = torch.zeros((), device=self.device)
        for inputs, targets in window:
            batch = inputs["batch"].to(self.device)
            with self.train_context():
                logits = model(batch).float()
                log_probs = logits.log_softmax(-1).masked_fill(batch.option_indices < 0, 0)
                losses = -(targets.to(self.device) * log_probs).sum(dim=-1)
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
            maximum = dist_utils.dist_max(total * count / sum(len(y) for _, y in window), mesh)
            extras = {"train/loss": mean, **self.lr_schedulers.get_metrics()}
            extras.update(metadata_metrics(observations, "train", group))
            self.metrics_processor.log(self.step, mean, maximum, float(norm.item()), extras)

    @classmethod
    def from_checkpoint(cls, checkpoint, config: TrainConfig, trainset, valset, run_dir):
        """Build TorchTitan with the source checkpoint and two learning rates."""
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
            model_spec=ModelSpec(
                name="social_jev",
                flavor="post-training",
                model=TitanModel.Config(
                    checkpoint=str(checkpoint),
                    train_backbone=config.backbone_lr is not None,
                ),
                parallelize_fn=parallelize,
                pipelining_fn=None,
                post_optimizer_build_fn=None,
                state_dict_adapter=None,
            ),
            hf_assets_path=str(checkpoint),
            dump_folder=run_dir,
            tokenizer=TitanTokenizer.Config(),
            dataloader=TitanLoader.Config(source=trainset),
            training=TrainingConfig(
                steps=config.steps,
                seq_len=config.seq_len,
                local_batch_size=config.local_batch_size,
                global_batch_size=config.global_batch_size,
                disable_cuda_graphs=True,
            ),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=config.data_parallel_replicate_degree,
                data_parallel_shard_degree=config.data_parallel_shard_degree,
                spmd_backend="partial_dtensor",
            ),
            optimizer=OptimizersContainer.Config(param_groups=groups),
            loss=CrossEntropyLoss.Config(),
            validator=Validation.Config(
                enable=valset is not None,
                source=valset,
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
        del dp_rank, dp_world_size, parallel_dims
        self.tokenizer: PreTrainedTokenizerBase = tokenizer.tokenizer
        self.metrics_processor = metrics_processor

    def should_validate(self, step):
        """Use the requested interval, or the final step when omitted."""
        return (
            step % self.config.interval == 0 if self.config.interval else step == self.config.steps
        )

    @torch.no_grad()
    def validate(self, model_parts, step):
        """Evaluate the same sequence of validation batches on every rank."""
        model = model_parts[0].eval()
        source = self.config.source
        assert source is not None
        batches = packed_batches(
            SampleStep(source, 0, 1),
            self.tokenizer,
            seq_len=self.config.seq_len,
            batch_size=self.config.batch_size,
            drop_last=False,
            fixed_length=True,
        )
        total = torch.zeros((), device=model.device)
        queries = 0
        observations = []
        try:
            while (batch := batches.next()) is not None:
                batch = batch.to(model.device)
                logits = model(batch).float()
                log_probs = logits.log_softmax(-1).masked_fill(batch.option_indices < 0, 0)
                targets = batch.targets
                assert targets is not None
                losses = -(targets * log_probs).sum(dim=-1)
                total += losses.sum()
                queries += len(targets)
                observations.append((losses.to("cpu"), batch.metadata))
            if queries:
                mean = float(total.item()) / queries
                extras = {"eval/loss": mean, **metadata_metrics(observations, "eval", None)}
                self.metrics_processor.log_validation(mean, step, extras)
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
