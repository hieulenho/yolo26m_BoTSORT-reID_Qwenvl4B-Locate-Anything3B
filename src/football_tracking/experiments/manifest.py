"""Experiment manifest helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"created_at": datetime.now(UTC).isoformat(), **payload}, indent=2, default=str),
        encoding="utf-8",
    )
    return path
