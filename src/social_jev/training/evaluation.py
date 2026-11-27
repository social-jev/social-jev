"""Validation loss and benchmark evaluation on the current training weights."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torchtitan.components.validate import Validator
from torchtitan.tools.logging import logger

from ..data import Sample, SampleStep, packed_batches
from ..evaluation import OnlineEvaluation  # noqa: TC001 -- runtime configuration field type.
from ..evaluation.common import score_evaluation
from .loss import Loss  # noqa: TC001 -- runtime configuration field type.

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from transformers import PreTrainedTokenizerBase


class ValidationRunner:
    """Accumulate query losses for the validation dataset."""

    def __init__(self, source, loss_function, device, group, metrics_processor):
        self.source = source
        self.loss = loss_function
        self.group = group
        self.metrics_processor = metrics_processor
        self.totals = torch.zeros(2, dtype=torch.float64, device=device)
        self.observations = []

    def score_batch(self, batch, logits):
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


class EvaluationRunner:
    """Score a benchmark using probabilities from the current training model."""

    def __init__(self, name, evaluation, group, metrics_processor):
        self.name = name
        self.evaluation = evaluation
        self.group = group
        self.metrics_processor = metrics_processor
        self.requests, self.labels, self.selected_metrics = evaluation.prepare()
        self.predictions = []

    def source(self):
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

    def score_batch(self, batch, logits):
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
        metrics[f"evaluation_metrics/{self.name}/calibration/ece"] = result.calibration.ece
        self.metrics_processor.logger.log(metrics, step)
        logger.info("evaluate step: %s  %s", step, metrics)


class ValAndEvalSchedule(Validator):
    """Schedule validation and benchmarks through one distributed forward loop."""

    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        source: Callable[[], Iterable[Sample]] | None
        loss_function: Loss
        val_steps: int | None
        total_steps: int
        compute_dtype: torch.dtype = torch.bfloat16
        evaluations: dict[str, OnlineEvaluation] = field(default_factory=dict)
        eval_steps: int | None = 10

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
        self.config: ValAndEvalSchedule.Config = config
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

    def is_due(self, step, interval):
        """Run before training, at the configured interval, and at completion."""
        return step in (0, self.config.total_steps) or (
            interval is not None and step % interval == 0
        )

    def should_validate(self, step):
        """Tell TorchTitan when either validation or evaluation is due."""
        return (self.config.source is not None and self.is_due(step, self.config.val_steps)) or (
            bool(self.config.evaluations) and self.is_due(step, self.config.eval_steps)
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
    def validate(self, model_parts, step):  # noqa: C901 -- share the distributed loop across both runners.
        """Run due datasets on disjoint DP samples, excluding placeholder predictions."""
        if self.last_step == step:
            return
        model = model_parts[0].eval()
        self.device = model.device
        try:
            runners = []
            if self.config.source is not None and (
                self.last_step is None or self.is_due(step, self.config.val_steps)
            ):
                runners.append(
                    ValidationRunner(
                        self.config.source,
                        self.config.loss_function,
                        self.device,
                        self.group,
                        self.metrics_processor,
                    )
                )
            if self.last_step is None or self.is_due(step, self.config.eval_steps):
                runners.extend(
                    EvaluationRunner(name, evaluation, self.group, self.metrics_processor)
                    for name, evaluation in self.config.evaluations.items()
                )
            for runner in runners:
                self.placeholder = None
                batches = packed_batches(
                    SampleStep(runner.source, self.dp_rank, self.dp_world_size, infinite=False),
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
                    # the distributed wrapper handles mixed precision; on one gpu, use autocast
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.config.compute_dtype,
                        enabled=(
                            self.device.type == "cuda"
                            and not self.parallel_dims.dp_enabled
                            and self.config.compute_dtype != torch.float32
                        ),
                    ):
                        logits = model(batch)
                    if not exhausted:
                        runner.score_batch(batch, logits)
                runner.log_metrics(step)
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
