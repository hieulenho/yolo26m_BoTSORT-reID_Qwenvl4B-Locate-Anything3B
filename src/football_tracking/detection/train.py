"""Entrypoints for detector training.

This module intentionally does not start training at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.detection.trainer import YOLOv8Trainer
from football_tracking.detection.training_config import load_training_config


def train_detector(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_training_config(config_path, overrides=overrides)
    return YOLOv8Trainer(config).train(dry_run=dry_run)


if __name__ == "__main__":
    from football_tracking.cli import main

    raise SystemExit(main())
