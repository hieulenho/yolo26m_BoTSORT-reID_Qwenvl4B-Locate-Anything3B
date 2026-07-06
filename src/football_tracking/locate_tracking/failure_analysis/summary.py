"""Write language benchmark failure analysis summaries."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.failure_analysis.case_builder import build_failure_cases


class FailureAnalysisError(RuntimeError):
    """Raised when failure analysis cannot be generated."""


def analyze_failures(
    *,
    evaluation: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    rows = _load_per_query_rows(evaluation)
    cases = build_failure_cases(rows)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "failure_cases.json"
    csv_path = root / "failure_cases.csv"
    md_path = root / "failure_summary.md"
    for path in (json_path, csv_path, md_path):
        if path.exists() and not overwrite:
            raise FailureAnalysisError(f"Output exists and overwrite=false: {path}")
    payload = {
        "failure_count": len(cases),
        "category_counts": dict(sorted(Counter(case.category for case in cases).items())),
        "cases": [case.to_dict() for case in cases],
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_csv(payload["cases"], csv_path)
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {
        "status": "ok",
        "failure_count": len(cases),
        "category_counts": payload["category_counts"],
        "paths": {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)},
    }


def _load_per_query_rows(evaluation: str | Path) -> list[dict[str, Any]]:
    path = Path(evaluation)
    if path.is_dir():
        path = path / "per_query_metrics.json"
    if not path.is_file():
        raise FailureAnalysisError(f"Per-query metrics do not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise FailureAnalysisError("Per-query metrics JSON must be a list.")
    return [dict(row) for row in payload]


def _write_csv(cases: list[dict[str, Any]], path: Path) -> None:
    fields = ["sequence_name", "query_id", "category", "reason", "severity"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({field: case.get(field) for field in fields})


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Language Tracking Failure Analysis", ""]
    lines.append(f"Failure cases: `{payload['failure_count']}`")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for category, count in payload["category_counts"].items():
        lines.append(f"| {category} | {count} |")
    lines.append("")
    return "\n".join(lines)
