"""Persistent output for TorchTitan training logs and metrics."""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch.distributed as dist
from torchtitan.components.metrics import BaseLogger
from torchtitan.tools.logging import logger


class FileFormatter(logging.Formatter):
    """Remove terminal colour codes from file output."""

    def format(self, record: logging.LogRecord) -> str:
        return re.sub(r"\x1b\[[0-9;]*m", "", super().format(record))


class Logger(BaseLogger):
    """Save run messages and structured metrics in the run directory."""

    def __init__(self, directory: Path, metrics_logger: BaseLogger):
        directory.mkdir(parents=True, exist_ok=True)
        self.metrics_logger = metrics_logger
        self.metrics_file = (directory / "metrics.jsonl").open("a", encoding="utf-8")
        self.file_handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
        self.file_handler.setFormatter(
            FileFormatter("[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(self.file_handler)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Append one record for each TorchTitan metrics event."""
        self.metrics_logger.log(metrics, step)
        self.metrics_file.write(json.dumps({"step": step, **metrics}) + "\n")
        self.metrics_file.flush()

    def close(self) -> None:
        """Close metrics outputs and detach the log-file handler."""
        self.metrics_logger.close()
        self.metrics_file.close()
        logger.removeHandler(self.file_handler)
        self.file_handler.close()


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
