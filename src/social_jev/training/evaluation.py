"""Validation loss and benchmark evaluation on the current training weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torchtitan.components.validate import Validator as TitanValidator
from torchtitan.tools.logging import logger

from ..data import Sample, SampleStep, packed_batches
from ..evaluation import (
    OnlineEval,
    create_eval_requests,
    get_named_metrics,
    score_evaluation,
)
from .logging import metadata_metrics
from .loss import Loss  # noqa: TC001 -- runtime configuration field type.

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from transformers import PreTrainedTokenizerBase


def autocast(device, compute_dtype, *, distributed):
    """Apply mixed precision when no distributed wrapper handles it."""
    return torch.autocast(
        device_type=device.type,
        dtype=compute_dtype,
        enabled=device.type == "cuda" and not distributed and compute_dtype != torch.float32,
    )


class Validation(TitanValidator):
    """Run distributed validation and aggregate losses over real queries."""

    def __init__(  # noqa: PLR0913 -- torch titan validator interface.
        self,
        config,
        *,
        dp_world_size,
        dp_rank,
        tokenizer,
        parallel_dims,
        metrics_processor,
        seq_len,
        local_batch_size,
        **kwargs,
    ):
        del kwargs
        self.config: ValAndEvalOrchestrator.Config = config
        self.valset = config.valset
        self.loss = config.loss_function
        self.interval = config.val_steps
        self.last_step = None
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.group = parallel_dims.get_mesh("batch").get_group() if dp_world_size > 1 else None
        self.placeholder = None
        self.parallel_dims = parallel_dims
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.tokenizer: PreTrainedTokenizerBase = tokenizer.tokenizer
        self.metrics_processor = metrics_processor

    def should_validate(self, step):
        """Run once before training, at the configured interval, and at completion."""
        if self.last_step == step:
            return False
        return (
            self.last_step is None
            or step == self.config.total_steps
            or (self.interval is not None and step % self.interval == 0)
        )

    def samples(self):
        """Read a fresh pass over the validation dataset."""
        assert self.valset is not None
        return self.valset()

    def reset_results(self):
        """Start a fresh loss total and metadata observations."""
        self.totals = torch.zeros(2, dtype=torch.float64, device=self.device)
        self.observations = []

    def accumulate_batch_results(self, batch, logits):
        """Compute validation losses for the supplied real queries."""
        assert batch.labels is not None
        losses = self.loss(logits.float(), batch.labels, batch.option_mask)
        self.totals[0] += losses.double().sum()
        self.totals[1] += losses.numel()
        self.observations.append((losses.cpu(), batch.metadata))

    def log_metrics(self, step):
        """Combine rank totals and log validation loss and supplied categories."""
        if self.group is not None:
            dist.all_reduce(self.totals, group=self.group)
        categories = metadata_metrics(self.observations, "validation_metrics", self.group)
        if self.totals[1].item():
            mean = (self.totals[0] / self.totals[1]).item()
            self.metrics_processor.log_validation(mean, step, categories)

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
        """Score disjoint DP samples while exhausted ranks perform placeholder forwards."""
        if not self.should_validate(step):
            return
        model = model_parts[0].eval()
        self.device = model.device
        try:
            self.reset_results()
            batches = packed_batches(
                SampleStep(self.samples, self.dp_rank, self.dp_world_size, infinite=False),
                self.tokenizer,
                seq_len=self.seq_len,
                local_batch_size=self.local_batch_size,
            )
            while True:
                batch = batches.next()
                exhausted = batch is None
                batch = self.if_exhausted_fill_placeholder(batch)
                if batch is None:
                    break
                batch = batch.to(self.device)
                with autocast(
                    self.device,
                    self.config.compute_dtype,
                    distributed=self.parallel_dims.dp_enabled,
                ):
                    logits = model(batch)
                if not exhausted:
                    self.accumulate_batch_results(batch, logits)
            self.log_metrics(step)
            self.last_step = step
        finally:
            self.placeholder = None
            model.train()


class Evaluation(Validation):
    """Use the validation execution loop to score a benchmark with shared evaluation metrics."""

    def __init__(self, config, name, evaluation, **runtime):
        super().__init__(config, **runtime)
        self.name = name
        self.evaluation = evaluation
        self.interval = config.eval_steps

    def reset_results(self):
        """Prepare fresh benchmark requests and clear predictions for this pass."""
        evaluation = self.evaluation
        data = evaluation.data() if callable(evaluation.data) else evaluation.data
        self.requests, self.labels, self.selected_metrics = create_eval_requests(
            data,
            queries=evaluation.queries,
            metric_names=get_named_metrics(evaluation.metrics),
        )
        self.predictions = []

    def samples(self):
        """Attach query positions before partitioning, splitting, and packing."""
        index = 0
        for request in self.requests:
            sample = Sample.from_request(request)
            sample.query_ids = list(range(index, index + len(request.queries)))
            index += len(request.queries)
            yield sample

    @staticmethod
    def option_probabilities(batch, logits):
        """Normalize real option scores and return one probability vector per query."""
        is_option = batch.option_mask
        probabilities = logits.float().masked_fill(~is_option, -torch.inf).softmax(-1)
        return [
            row[mask].tolist()
            for row, mask in zip(probabilities.cpu(), is_option.cpu(), strict=True)
        ]

    def accumulate_batch_results(self, batch, logits):
        """Retain each query's probabilities with its original position."""
        self.predictions.extend(
            zip(batch.query_ids, self.option_probabilities(batch, logits), strict=True)
        )

    def log_metrics(self, step):
        """Restore query order across ranks and use standalone evaluation scoring."""
        predictions = self.predictions
        if self.group is not None:
            parts = [[] for _ in range(dist.get_world_size(self.group))]
            dist.all_gather_object(parts, predictions, group=self.group)
            predictions = [prediction for part in parts for prediction in part]
        predictions = sorted(predictions)
        if [index for index, _ in predictions] != list(range(len(self.labels))):
            message = (
                f"Evaluation {self.name!r} did not score every query. "
                "Increase seq_len to include longer queries."
            )
            raise ValueError(message)
        if self.group is not None and dist.get_rank(self.group) != 0:
            return
        result = score_evaluation(
            [values for _, values in predictions],
            requests=self.requests,
            labels=self.labels,
            selected_metrics=self.selected_metrics,
            metrics=self.evaluation.metrics,
            calibration_bins=self.evaluation.calibration_bins,
        )
        metrics = {
            f"evaluation_metrics/{self.name}/{name}": score for name, score in result.scores.items()
        }
        metrics[f"evaluation_metrics/{self.name}/ece"] = result.calibration.ece
        self.metrics_processor.logger.log(metrics, step)
        logger.info("evaluate step: %s  %s", step, metrics)


class ValAndEvalOrchestrator(TitanValidator):
    """Delegate scheduled validation and benchmark evaluation to their validators."""

    @dataclass(kw_only=True, slots=True)
    class Config(TitanValidator.Config):
        valset: Callable[[], Iterable[Sample]] | None
        loss_function: Loss
        val_steps: int | None
        total_steps: int
        compute_dtype: torch.dtype = torch.bfloat16
        evaluations: dict[str, OnlineEval] = field(default_factory=dict)
        eval_steps: int | None = 10

    def __init__(self, config, **runtime):
        self.tasks: list[Validation] = []
        if config.valset is not None:
            self.tasks.append(Validation(config, **runtime))
        self.tasks.extend(
            Evaluation(config, name, evaluation, **runtime)
            for name, evaluation in config.evaluations.items()
        )

    def should_validate(self, step):
        """Tell TorchTitan whether any dataset is due."""
        return any(task.should_validate(step) for task in self.tasks)

    def validate(self, model_parts, step):
        """Run each due dataset once at this step."""
        for task in self.tasks:
            if task.should_validate(step):
                task.validate(model_parts, step)
