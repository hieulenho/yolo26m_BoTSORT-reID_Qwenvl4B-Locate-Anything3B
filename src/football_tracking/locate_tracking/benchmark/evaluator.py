"""Evaluate saved language-guided semantic tracking predictions."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.benchmark.aggregate import aggregate_query_metrics
from football_tracking.locate_tracking.benchmark.continuity_metrics import (
    continuity_ratio,
    longest_correct_run,
    semantic_target_switches,
)
from football_tracking.locate_tracking.benchmark.efficiency_metrics import (
    efficiency_for_prediction,
)
from football_tracking.locate_tracking.benchmark.ground_truth_loader import (
    ground_truth_observations_for_query,
)
from football_tracking.locate_tracking.benchmark.manifest import (
    load_benchmark_manifest,
    load_prediction_manifest,
)
from football_tracking.locate_tracking.benchmark.prediction_loader import (
    predicted_observations_for_query,
    prediction_index,
    raw_id_transitions_for_prediction,
)
from football_tracking.locate_tracking.benchmark.query_metrics import evaluate_query_frames
from football_tracking.locate_tracking.benchmark.reacquisition_metrics import (
    evaluate_reacquisition,
)
from football_tracking.locate_tracking.benchmark.validation import (
    validate_benchmark_manifest,
)


class LanguageBenchmarkEvaluationError(RuntimeError):
    """Raised when language benchmark evaluation fails."""


@dataclass(frozen=True)
class LanguageBenchmarkEvaluation:
    variant_id: str
    per_query: tuple[dict[str, Any], ...]
    aggregate: dict[str, Any]
    paths: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "per_query": list(self.per_query),
            "aggregate": dict(self.aggregate),
            "paths": {key: str(value) for key, value in self.paths.items()},
        }


def evaluate_language_benchmark(
    *,
    manifest_path: str | Path,
    prediction_manifest_path: str | Path,
    output_dir: str | Path,
    iou_threshold: float = 0.5,
    overwrite: bool = False,
) -> LanguageBenchmarkEvaluation:
    validation = validate_benchmark_manifest(manifest_path)
    if validation.has_errors:
        raise LanguageBenchmarkEvaluationError("Benchmark manifest validation failed.")
    manifest = load_benchmark_manifest(manifest_path)
    predictions = load_prediction_manifest(prediction_manifest_path)
    pred_index = prediction_index(predictions)
    rows: list[dict[str, Any]] = []
    for sequence in manifest.sequences:
        for query in sequence.queries:
            prediction = pred_index.get((sequence.sequence_name, query.query_id))
            gt_by_frame = ground_truth_observations_for_query(sequence, query)
            pred_by_frame = predicted_observations_for_query(prediction)
            status = prediction.status if prediction is not None else "missing_prediction"
            frame_metrics = evaluate_query_frames(
                query_id=query.query_id,
                status=status,
                gt_by_frame=gt_by_frame,
                pred_by_frame=pred_by_frame,
                start_frame=query.evaluation_start_frame,
                end_frame=query.evaluation_end_frame,
                iou_threshold=iou_threshold,
            )
            reaq = evaluate_reacquisition(
                opportunities=query.reacquisition_events,
                prediction=prediction,
                frame_metrics=frame_metrics,
            )
            efficiency = efficiency_for_prediction(
                prediction,
                eval_frame_count=query.evaluation_end_frame - query.evaluation_start_frame + 1,
            )
            rows.append(
                {
                    "sequence_name": sequence.sequence_name,
                    "query_id": query.query_id,
                    "query_text": query.query_text,
                    "query_mode": query.query_mode,
                    "query_category": query.query_category,
                    "difficulty": query.difficulty,
                    **frame_metrics.to_dict(),
                    "longest_correct_target_run": longest_correct_run(
                        frame_metrics.frame_results
                    ),
                    "target_continuity_ratio": continuity_ratio(
                        frame_metrics.frame_results,
                        frame_metrics.gt_frame_count,
                    ),
                    "semantic_target_switches": semantic_target_switches(
                        frame_metrics.frame_results
                    ),
                    "raw_id_transitions_along_semantic_target": (
                        raw_id_transitions_for_prediction(prediction)
                    ),
                    "reacquisition_opportunity_count": reaq.opportunity_count,
                    "reacquisition_success_count": reaq.confirmed_success_count,
                    "false_reacquisition_count": reaq.false_reacquisition_count,
                    "committed_reacquisition_count": reaq.committed_count,
                    "frames_to_reacquire": list(reaq.frames_to_reacquire),
                    **efficiency.to_dict(),
                }
            )
    aggregate = aggregate_query_metrics(rows)
    output = Path(output_dir)
    paths = _write_outputs(
        rows=rows,
        aggregate=aggregate,
        output_dir=output,
        variant_id=predictions.variant_id,
        overwrite=overwrite,
    )
    return LanguageBenchmarkEvaluation(
        variant_id=predictions.variant_id,
        per_query=tuple(rows),
        aggregate=aggregate,
        paths=paths,
    )


def _write_outputs(
    *,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    output_dir: Path,
    variant_id: str,
    overwrite: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_query_json = output_dir / "per_query_metrics.json"
    per_query_csv = output_dir / "per_query_metrics.csv"
    aggregate_json = output_dir / "aggregate_metrics.json"
    aggregate_csv = output_dir / "aggregate_metrics.csv"
    summary_md = output_dir / "language_benchmark_summary.md"
    for path in (per_query_json, per_query_csv, aggregate_json, aggregate_csv, summary_md):
        if path.exists() and not overwrite:
            raise LanguageBenchmarkEvaluationError(f"Output exists and overwrite=false: {path}")
    per_query_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    _write_per_query_csv(rows, per_query_csv)
    aggregate_json.write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    _write_aggregate_csv(aggregate, aggregate_csv)
    summary_md.write_text(_summary_markdown(variant_id, aggregate), encoding="utf-8")
    return {
        "per_query_json": per_query_json,
        "per_query_csv": per_query_csv,
        "aggregate_json": aggregate_json,
        "aggregate_csv": aggregate_csv,
        "summary_md": summary_md,
    }


def _write_per_query_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "sequence_name",
        "query_id",
        "query_mode",
        "query_category",
        "difficulty",
        "status",
        "initial_selection_correct",
        "target_precision",
        "target_recall",
        "target_f1",
        "target_continuity_ratio",
        "semantic_target_switches",
        "raw_id_transitions_along_semantic_target",
        "reacquisition_opportunity_count",
        "reacquisition_success_count",
        "false_reacquisition_count",
        "grounding_call_count",
        "grounding_calls_per_1000_frames",
        "runtime_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_aggregate_csv(aggregate: dict[str, Any], path: Path) -> None:
    fields = ["metric", "value"]
    excluded = {"status_counts", "by_category", "by_difficulty"}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, value in aggregate.items():
            if key in excluded:
                continue
            writer.writerow({"metric": key, "value": value})


def _summary_markdown(variant_id: str, aggregate: dict[str, Any]) -> str:
    lines = [
        f"# Language Benchmark Summary - {variant_id}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
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
    ):
        lines.append(f"| {key} | {_fmt(aggregate.get(key))} |")
    lines.append("")
    lines.append("Raw MOT/TrackEval metrics are reported separately.")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
