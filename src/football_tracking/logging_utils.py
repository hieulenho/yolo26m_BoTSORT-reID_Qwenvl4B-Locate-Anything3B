"""Logging setup helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOGGER_NAME = "football_tracking"
HANDLER_MARKER = "_football_tracking_handler"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level}")
    return numeric_level


def _has_handler(
    logger: logging.Logger,
    handler_type: type[logging.Handler],
    path: Path | None,
) -> bool:
    for handler in logger.handlers:
        if not getattr(handler, HANDLER_MARKER, False):
            continue
        if not isinstance(handler, handler_type):
            continue
        if path is None:
            return True
        handler_path = Path(getattr(handler, "baseFilename", "")).resolve()
        if handler_path == path.resolve():
            return True
    return False


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure package logging without adding duplicate handlers."""

    numeric_level = _coerce_level(level)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if not _has_handler(logger, logging.StreamHandler, None):
        console_handler = logging.StreamHandler(sys.stdout)
        setattr(console_handler, HANDLER_MARKER, True)
        logger.addHandler(console_handler)

    for handler in logger.handlers:
        if getattr(handler, HANDLER_MARKER, False):
            handler.setLevel(numeric_level)
            handler.setFormatter(formatter)

    if log_file is not None:
        file_path = Path(log_file).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not _has_handler(logger, logging.FileHandler, file_path):
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            setattr(file_handler, HANDLER_MARKER, True)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
