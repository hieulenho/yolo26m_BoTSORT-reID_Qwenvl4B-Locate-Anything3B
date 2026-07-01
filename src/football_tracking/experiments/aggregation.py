"""Aggregate persisted tracker experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_result_files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("result.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return rows
