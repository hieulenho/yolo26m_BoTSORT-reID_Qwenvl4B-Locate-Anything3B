"""Small helpers to export detector result payloads."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json_and_csv(payload: dict[str, Any], json_path: Path, csv_path: Path) -> dict[str, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in payload.items():
            writer.writerow({"metric": key, "value": value})
    return {"json": json_path, "csv": csv_path}
