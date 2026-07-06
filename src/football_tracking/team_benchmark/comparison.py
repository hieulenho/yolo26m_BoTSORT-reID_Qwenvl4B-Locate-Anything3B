"""Compare evaluated team benchmark variants."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class TeamBenchmarkComparisonError(RuntimeError):
    """Raised when team benchmark comparisons cannot be built."""


COMPARISON_METRICS = (
    "track_team_accuracy",
    "macro_team_f1",
    "track_unknown_rate",
    "query_resolved_rate",
    "query_selected_track_exact_accuracy",
    "query_team_accuracy",
    "correct_id_correct_team_rate",
    "grounding_calls_per_query",
    "runtime_seconds_per_query",
)


def compare_team_benchmark_evaluations(
    *,
    evaluations: tuple[str | Path, ...],
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if len(evaluations) < 2:
        raise TeamBenchmarkComparisonError("At least two evaluations are required.")
    rows = [_load_aggregate(path) for path in evaluations]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output / "team_benchmark_comparison.json",
        "csv": output / "team_benchmark_comparison.csv",
        "markdown": output / "team_benchmark_comparison.md",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise TeamBenchmarkComparisonError(f"Output exists and overwrite=false: {path}")
    result = {
        "variant_count": len(rows),
        "rows": rows,
        "metric_keys": list(COMPARISON_METRICS),
    }
    paths["json"].write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _write_csv(rows, paths["csv"])
    paths["markdown"].write_text(_comparison_markdown(rows), encoding="utf-8")
    return {
        **result,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _load_aggregate(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    aggregate_path = candidate / "aggregate_metrics.json" if candidate.is_dir() else candidate
    if not aggregate_path.is_file():
        raise TeamBenchmarkComparisonError(f"Missing aggregate metrics: {aggregate_path}")
    data = json.loads(aggregate_path.read_text(encoding="utf-8"))
    row = {
        "variant_id": data.get("variant_id"),
        "variant_name": data.get("variant_name"),
        "pipeline_type": data.get("pipeline_type"),
    }
    row.update({metric: data.get(metric) for metric in COMPARISON_METRICS})
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["variant_id", "variant_name", "pipeline_type", *COMPARISON_METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _comparison_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Team Benchmark Comparison",
        "",
        (
            "| Variant | Pipeline | Track Team Acc | Macro F1 | Query Exact Acc | "
            "Query Team Acc | Correct ID+Team | Calls/Query |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("variant_id")),
                    str(row.get("pipeline_type")),
                    _fmt(row.get("track_team_accuracy")),
                    _fmt(row.get("macro_team_f1")),
                    _fmt(row.get("query_selected_track_exact_accuracy")),
                    _fmt(row.get("query_team_accuracy")),
                    _fmt(row.get("correct_id_correct_team_rate")),
                    _fmt(row.get("grounding_calls_per_query")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
