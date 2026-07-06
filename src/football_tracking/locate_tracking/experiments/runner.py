"""Run language tracking ablation evaluations from saved prediction artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.benchmark.evaluator import evaluate_language_benchmark
from football_tracking.locate_tracking.experiments.artifact_reuse import check_artifact_reuse
from football_tracking.locate_tracking.experiments.fingerprint import file_sha256
from football_tracking.locate_tracking.experiments.variants import load_language_ablation_config


class LanguageAblationRunError(RuntimeError):
    """Raised when a language ablation run fails."""


def run_language_ablation(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_language_ablation_config(config_path)
    root = Path(output_dir) if output_dir is not None else config.output_dir
    reuse = check_artifact_reuse(config.variants)
    plan = {
        "dry_run": dry_run,
        "benchmark_manifest": str(config.benchmark_manifest),
        "benchmark_manifest_sha256": file_sha256(config.benchmark_manifest),
        "variant_count": len(config.variants),
        "variants": [variant.to_dict() for variant in config.variants],
        "artifact_reuse": reuse.to_dict(),
        "output_dir": str(root),
    }
    if dry_run:
        return plan
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    evaluations: dict[str, Any] = {}
    for variant in config.variants:
        variant_dir = root / variant.variant_id
        evaluation = evaluate_language_benchmark(
            manifest_path=config.benchmark_manifest,
            prediction_manifest_path=variant.prediction_manifest,
            output_dir=variant_dir,
            iou_threshold=config.iou_threshold,
            overwrite=overwrite,
        )
        row = {
            "variant_id": variant.variant_id,
            "name": variant.name,
            "description": variant.description,
            "prediction_manifest_sha256": file_sha256(variant.prediction_manifest),
            **evaluation.aggregate,
        }
        rows.append(row)
        evaluations[variant.variant_id] = evaluation.to_dict()
    paths = _write_outputs(root, plan, rows, evaluations, overwrite=overwrite)
    return {**plan, "dry_run": False, "rows": rows, "paths": paths}


def _write_outputs(
    root: Path,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    evaluations: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, str]:
    json_path = root / "ablation_results.json"
    csv_path = root / "ablation_results.csv"
    md_path = root / "ablation_summary.md"
    for path in (json_path, csv_path, md_path):
        if path.exists() and not overwrite:
            raise LanguageAblationRunError(f"Output exists and overwrite=false: {path}")
    json_path.write_text(
        json.dumps({**plan, "rows": rows, "evaluations": evaluations}, indent=2, default=str),
        encoding="utf-8",
    )
    _write_csv(rows, csv_path)
    md_path.write_text(_markdown(rows), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "variant_id",
        "name",
        "query_count",
        "query_resolution_rate",
        "micro_target_f1",
        "macro_continuity_ratio",
        "reacquisition_success_rate",
        "false_reacquisition_rate",
        "grounding_calls_per_1000_frames",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _markdown(rows: list[dict[str, Any]]) -> str:
    fields = [
        "variant_id",
        "micro_target_f1",
        "macro_continuity_ratio",
        "reacquisition_success_rate",
        "grounding_calls_per_1000_frames",
    ]
    lines = [
        "# Language Ablation Summary",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _field in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
