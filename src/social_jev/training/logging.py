"""Persistent output for TorchTitan training logs and metrics."""

import json
import logging
import re
from pathlib import Path
from typing import Any

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
