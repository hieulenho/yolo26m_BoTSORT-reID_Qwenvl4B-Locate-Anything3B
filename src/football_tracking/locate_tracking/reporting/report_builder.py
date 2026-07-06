"""Build research-ready Markdown reports from saved language benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.reporting.charts import write_chart_data
from football_tracking.locate_tracking.reporting.research_summary import (
    future_work_section,
    limitations_section,
)
from football_tracking.locate_tracking.reporting.tables import metric_table, rows_table


class LanguageReportError(RuntimeError):
    """Raised when a language benchmark report cannot be generated."""


def generate_language_report(
    *,
    evaluation: str | Path,
    output: str | Path,
    ablation: str | Path | None = None,
    failures: str | Path | None = None,
    mot_metrics: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise LanguageReportError(f"Report exists and overwrite=false: {output_path}")
    aggregate, per_query = _load_evaluation(evaluation)
    ablation_payload = _load_optional_json(ablation)
    failure_payload = _load_optional_json(failures)
    mot_payload = _load_optional_json(mot_metrics)
    chart_paths = write_chart_data(
        ablation_rows=list((ablation_payload or {}).get("rows", [])),
        output_dir=output_path.parent / "charts",
        overwrite=overwrite,
    )
    markdown = _render_report(
        aggregate=aggregate,
        per_query=per_query,
        ablation_payload=ablation_payload,
        failure_payload=failure_payload,
        mot_payload=mot_payload,
        chart_paths=chart_paths,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "report": str(output_path),
                "evaluation": str(evaluation),
                "ablation": str(ablation) if ablation else None,
                "failures": str(failures) if failures else None,
                "mot_metrics": str(mot_metrics) if mot_metrics else None,
                "charts": chart_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "paths": {
            "markdown": str(output_path),
            "metadata": str(metadata_path),
            **chart_paths,
        },
    }


def _load_evaluation(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(path)
    if root.is_dir():
        aggregate_path = root / "aggregate_metrics.json"
        per_query_path = root / "per_query_metrics.json"
    else:
        aggregate_path = root
        per_query_path = root.parent / "per_query_metrics.json"
    if not aggregate_path.is_file() or not per_query_path.is_file():
        raise LanguageReportError("Evaluation aggregate/per-query files are missing.")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    per_query = json.loads(per_query_path.read_text(encoding="utf-8"))
    return dict(aggregate), [dict(row) for row in per_query]


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_dir():
        for filename in ("ablation_results.json", "failure_cases.json", "best_tracker_result.json"):
            candidate = resolved / filename
            if candidate.is_file():
                resolved = candidate
                break
    if not resolved.is_file():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _render_report(
    *,
    aggregate: dict[str, Any],
    per_query: list[dict[str, Any]],
    ablation_payload: dict[str, Any] | None,
    failure_payload: dict[str, Any] | None,
    mot_payload: dict[str, Any] | None,
    chart_paths: dict[str, str],
) -> str:
    metric_keys = (
        "query_count",
        "query_resolution_rate",
        "initial_selection_accuracy_strict",
        "micro_target_precision",
        "micro_target_recall",
        "micro_target_f1",
        "macro_continuity_ratio",
        "reacquisition_success_rate",
        "false_reacquisition_rate",
        "grounding_calls_per_1000_frames",
    )
    ablation_rows = list((ablation_payload or {}).get("rows", []))
    failure_counts = (failure_payload or {}).get("category_counts", {})
    lines = [
        "# Language-Guided Semantic Tracking Technical Report",
        "",
        "## Abstract",
        "",
        "This report is generated from saved benchmark artifacts. It separates raw MOT quality "
        "from language-guided semantic tracking quality.",
        "",
        "## Raw MOT Context",
        "",
        _mot_summary(mot_payload),
        "",
        "## Language Benchmark Metrics",
        "",
        metric_table(aggregate, metric_keys),
        "",
        "## Per-Query Results",
        "",
        rows_table(
            per_query,
            (
                "query_id",
                "status",
                "target_f1",
                "target_continuity_ratio",
                "reacquisition_success_count",
            ),
        ),
        "",
        "## Ablation Study",
        "",
        rows_table(
            ablation_rows,
            (
                "variant_id",
                "micro_target_f1",
                "macro_continuity_ratio",
                "grounding_calls_per_1000_frames",
            ),
        ),
        "",
        "## Failure Analysis",
        "",
        json.dumps(failure_counts, indent=2, default=str),
        "",
        "## Charts",
        "",
        "\n".join(f"- `{path}`" for path in chart_paths.values()) or "No chart data available.",
        "",
        limitations_section(),
        "",
        future_work_section(),
        "",
    ]
    return "\n".join(lines)


def _mot_summary(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "Raw MOT metrics were not provided to this language report."
    metrics = payload.get("metrics", payload)
    keys = ("selected_tracker", "HOTA", "IDF1", "IDSW", "MOTA", "tracker_fps")
    return metric_table(metrics, keys)
