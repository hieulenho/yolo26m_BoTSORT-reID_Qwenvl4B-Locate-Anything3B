"""Evaluate team attribution and language-target prediction manifests."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_tracking.team_benchmark.manifest import (
    load_team_benchmark_manifest,
    load_team_prediction_manifest,
)
from football_tracking.team_benchmark.schemas import (
    TeamBenchmarkManifest,
    TeamPredictionManifest,
    TrackTeamPrediction,
)
from football_tracking.team_benchmark.validation import (
    validate_team_benchmark_manifest,
    validate_team_prediction_manifest,
)


class TeamBenchmarkEvaluationError(RuntimeError):
    """Raised when team benchmark evaluation cannot run."""


@dataclass(frozen=True)
class TeamBenchmarkEvaluation:
    variant_id: str
    pipeline_type: str
    track_rows: tuple[dict[str, Any], ...]
    query_rows: tuple[dict[str, Any], ...]
    aggregate: dict[str, Any]
    paths: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "pipeline_type": self.pipeline_type,
            "track_rows": list(self.track_rows),
            "query_rows": list(self.query_rows),
            "aggregate": dict(self.aggregate),
            "paths": {key: str(value) for key, value in self.paths.items()},
        }


def evaluate_team_benchmark(
    *,
    manifest_path: str | Path,
    prediction_manifest_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> TeamBenchmarkEvaluation:
    manifest_validation = validate_team_benchmark_manifest(manifest_path)
    if manifest_validation.has_errors:
        raise TeamBenchmarkEvaluationError("Team benchmark manifest validation failed.")
    prediction_validation = validate_team_prediction_manifest(prediction_manifest_path)
    if prediction_validation.has_errors:
        raise TeamBenchmarkEvaluationError("Team prediction manifest validation failed.")
    manifest = load_team_benchmark_manifest(manifest_path)
    predictions = load_team_prediction_manifest(prediction_manifest_path)
    track_rows = _evaluate_track_predictions(manifest, predictions)
    query_rows = _evaluate_query_predictions(manifest, predictions)
    aggregate = _aggregate_metrics(manifest, predictions, track_rows, query_rows)
    paths = _write_outputs(
        output_dir=Path(output_dir),
        variant_id=predictions.variant_id,
        pipeline_type=predictions.pipeline_type,
        track_rows=track_rows,
        query_rows=query_rows,
        aggregate=aggregate,
        overwrite=overwrite,
    )
    return TeamBenchmarkEvaluation(
        variant_id=predictions.variant_id,
        pipeline_type=predictions.pipeline_type,
        track_rows=tuple(track_rows),
        query_rows=tuple(query_rows),
        aggregate=aggregate,
        paths=paths,
    )


def _evaluate_track_predictions(
    manifest: TeamBenchmarkManifest,
    predictions: TeamPredictionManifest,
) -> list[dict[str, Any]]:
    pred_index = {
        (prediction.sequence_name, prediction.track_id): prediction
        for prediction in predictions.track_predictions
    }
    rows: list[dict[str, Any]] = []
    for sequence in manifest.sequences:
        for annotation in sequence.track_annotations:
            prediction = pred_index.get((sequence.sequence_name, annotation.track_id))
            predicted_team = _team_label(prediction)
            status = "missing_prediction" if prediction is None else prediction.status
            correct = (
                prediction is not None
                and prediction.status == "resolved"
                and predicted_team == annotation.team_label
            )
            role_correct = (
                None
                if annotation.role_label is None
                else prediction is not None
                and prediction.status == "resolved"
                and prediction.role_label == annotation.role_label
            )
            rows.append(
                {
                    "sequence_name": sequence.sequence_name,
                    "track_id": annotation.track_id,
                    "gt_team_label": annotation.team_label,
                    "gt_role_label": annotation.role_label,
                    "predicted_team_label": predicted_team,
                    "predicted_role_label": None if prediction is None else prediction.role_label,
                    "status": status,
                    "confidence": None if prediction is None else prediction.confidence,
                    "correct_team": correct,
                    "role_correct": role_correct,
                    "error_type": _track_error_type(
                        status=status,
                        correct_team=correct,
                        role_correct=role_correct,
                    ),
                    "evidence_frame_count": (
                        0 if prediction is None else len(prediction.evidence_frames)
                    ),
                }
            )
    return rows


def _evaluate_query_predictions(
    manifest: TeamBenchmarkManifest,
    predictions: TeamPredictionManifest,
) -> list[dict[str, Any]]:
    pred_index = {
        (prediction.sequence_name, prediction.query_id): prediction
        for prediction in predictions.query_predictions
    }
    rows: list[dict[str, Any]] = []
    for sequence in manifest.sequences:
        for annotation in sequence.query_annotations:
            prediction = pred_index.get((sequence.sequence_name, annotation.query_id))
            selected = set(() if prediction is None else prediction.selected_track_ids)
            expected = set(annotation.expected_track_ids)
            exact_match = bool(selected) and selected == expected
            hit = bool(selected & expected)
            predicted_team = None if prediction is None else prediction.team_label
            team_correct = (
                prediction is not None
                and prediction.status == "resolved"
                and predicted_team == annotation.expected_team_label
            )
            rows.append(
                {
                    "sequence_name": sequence.sequence_name,
                    "query_id": annotation.query_id,
                    "query_text": annotation.query_text,
                    "difficulty": annotation.difficulty,
                    "expected_track_ids": sorted(expected),
                    "expected_team_label": annotation.expected_team_label,
                    "selected_track_ids": sorted(selected),
                    "predicted_team_label": predicted_team,
                    "status": "missing_prediction" if prediction is None else prediction.status,
                    "confidence": None if prediction is None else prediction.confidence,
                    "support_ratio": None if prediction is None else prediction.support_ratio,
                    "grounding_call_count": (
                        0 if prediction is None else prediction.grounding_call_count
                    ),
                    "runtime_seconds": None if prediction is None else prediction.runtime_seconds,
                    "selected_track_exact_match": exact_match,
                    "selected_track_hit": hit,
                    "team_correct": team_correct,
                    "correct_id_correct_team": exact_match and team_correct,
                    "error_type": _query_error_type(
                        status="missing_prediction" if prediction is None else prediction.status,
                        exact_match=exact_match,
                        hit=hit,
                        team_correct=team_correct,
                    ),
                }
            )
    return rows


def _aggregate_metrics(
    manifest: TeamBenchmarkManifest,
    predictions: TeamPredictionManifest,
    track_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    team_prf = _per_team_prf(track_rows)
    confusion = _confusion_matrix(track_rows)
    role_prf = _per_role_prf(track_rows)
    role_confusion = _role_confusion_matrix(track_rows)
    aggregate = {
        "benchmark_name": manifest.benchmark_name,
        "variant_id": predictions.variant_id,
        "variant_name": predictions.variant_name,
        "pipeline_type": predictions.pipeline_type,
        "sequence_count": manifest.sequence_count,
        "annotated_track_count": len(track_rows),
        "query_count": len(query_rows),
        "track_prediction_count": len(predictions.track_predictions),
        "query_prediction_count": len(predictions.query_predictions),
        "track_team_accuracy": _mean(row["correct_team"] for row in track_rows),
        "track_team_coverage": _mean(row["status"] != "missing_prediction" for row in track_rows),
        "track_unknown_rate": _mean(
            row["status"] in {"unknown", "not_found", "missing_prediction"}
            for row in track_rows
        ),
        "wrong_team_rate": _mean(
            row["status"] == "resolved" and not row["correct_team"] for row in track_rows
        ),
        "macro_team_f1": _macro_f1(team_prf),
        "role_accuracy": _mean(
            row["role_correct"] for row in track_rows if row["role_correct"] is not None
        ),
        "macro_role_f1": _macro_f1(role_prf),
        "query_resolved_rate": _mean(row["status"] == "resolved" for row in query_rows),
        "query_selected_track_exact_accuracy": _mean(
            row["selected_track_exact_match"] for row in query_rows
        ),
        "query_selected_track_hit_rate": _mean(row["selected_track_hit"] for row in query_rows),
        "query_team_accuracy": _mean(row["team_correct"] for row in query_rows),
        "correct_id_correct_team_rate": _mean(
            row["correct_id_correct_team"] for row in query_rows
        ),
        "query_ambiguous_rate": _mean(row["status"] == "ambiguous" for row in query_rows),
        "query_not_found_rate": _mean(
            row["status"] in {"not_found", "unknown", "missing_prediction"}
            for row in query_rows
        ),
        "mean_query_support_ratio": _mean(
            row["support_ratio"] for row in query_rows if row["support_ratio"] is not None
        ),
        "grounding_calls_per_query": _mean(
            row["grounding_call_count"] for row in query_rows
        ),
        "runtime_seconds_per_query": _mean(
            row["runtime_seconds"] for row in query_rows if row["runtime_seconds"] is not None
        ),
        "per_team": team_prf,
        "team_confusion_matrix": confusion,
        "per_role": role_prf,
        "role_confusion_matrix": role_confusion,
    }
    return aggregate


def _per_team_prf(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    labels = sorted(
        {
            str(row["gt_team_label"])
            for row in rows
        }
        | {
            str(row["predicted_team_label"])
            for row in rows
            if row["predicted_team_label"] is not None
        }
    )
    result: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(
            1
            for row in rows
            if row["gt_team_label"] == label and row["predicted_team_label"] == label
        )
        fp = sum(
            1
            for row in rows
            if row["gt_team_label"] != label and row["predicted_team_label"] == label
        )
        fn = sum(
            1
            for row in rows
            if row["gt_team_label"] == label and row["predicted_team_label"] != label
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        support = sum(1 for row in rows if row["gt_team_label"] == label)
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return result


def _confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        gt = str(row["gt_team_label"])
        pred = str(row["predicted_team_label"] or "__missing__")
        matrix[gt][pred] += 1
    return {gt: dict(counter) for gt, counter in sorted(matrix.items())}


def _per_role_prf(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    role_rows = [row for row in rows if row["gt_role_label"] is not None]
    labels = sorted(
        {
            str(row["gt_role_label"])
            for row in role_rows
        }
        | {
            str(row["predicted_role_label"])
            for row in role_rows
            if row["predicted_role_label"] is not None
        }
    )
    result: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(
            1
            for row in role_rows
            if row["gt_role_label"] == label and row["predicted_role_label"] == label
        )
        fp = sum(
            1
            for row in role_rows
            if row["gt_role_label"] != label and row["predicted_role_label"] == label
        )
        fn = sum(
            1
            for row in role_rows
            if row["gt_role_label"] == label and row["predicted_role_label"] != label
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        support = sum(1 for row in role_rows if row["gt_role_label"] == label)
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return result


def _role_confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["gt_role_label"] is None:
            continue
        gt = str(row["gt_role_label"])
        pred = str(row["predicted_role_label"] or "__missing__")
        matrix[gt][pred] += 1
    return {gt: dict(counter) for gt, counter in sorted(matrix.items())}


def _write_outputs(
    *,
    output_dir: Path,
    variant_id: str,
    pipeline_type: str,
    track_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    overwrite: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "track_json": output_dir / "track_team_metrics.json",
        "track_csv": output_dir / "track_team_metrics.csv",
        "query_json": output_dir / "query_target_metrics.json",
        "query_csv": output_dir / "query_target_metrics.csv",
        "aggregate_json": output_dir / "aggregate_metrics.json",
        "aggregate_csv": output_dir / "aggregate_metrics.csv",
        "confusion_csv": output_dir / "team_confusion_matrix.csv",
        "role_confusion_csv": output_dir / "role_confusion_matrix.csv",
        "summary_md": output_dir / "team_benchmark_summary.md",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise TeamBenchmarkEvaluationError(f"Output exists and overwrite=false: {path}")
    paths["track_json"].write_text(
        json.dumps(track_rows, indent=2, default=str),
        encoding="utf-8",
    )
    paths["query_json"].write_text(
        json.dumps(query_rows, indent=2, default=str),
        encoding="utf-8",
    )
    paths["aggregate_json"].write_text(
        json.dumps(aggregate, indent=2, default=str),
        encoding="utf-8",
    )
    _write_csv(track_rows, paths["track_csv"])
    _write_csv(query_rows, paths["query_csv"])
    _write_aggregate_csv(aggregate, paths["aggregate_csv"])
    _write_confusion_csv(
        aggregate["team_confusion_matrix"],
        paths["confusion_csv"],
        label_field="gt_team_label",
    )
    _write_confusion_csv(
        aggregate["role_confusion_matrix"],
        paths["role_confusion_csv"],
        label_field="gt_role_label",
    )
    paths["summary_md"].write_text(
        _summary_markdown(variant_id, pipeline_type, aggregate),
        encoding="utf-8",
    )
    return paths


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_aggregate_csv(aggregate: dict[str, Any], path: Path) -> None:
    excluded = {
        "per_team",
        "team_confusion_matrix",
        "per_role",
        "role_confusion_matrix",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in aggregate.items():
            if key not in excluded:
                writer.writerow({"metric": key, "value": _csv_value(value)})


def _write_confusion_csv(
    matrix: dict[str, dict[str, int]],
    path: Path,
    *,
    label_field: str,
) -> None:
    pred_labels = sorted({label for row in matrix.values() for label in row})
    fields = [label_field, *pred_labels]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for gt_label, counts in matrix.items():
            row = {label_field: gt_label}
            row.update({label: counts.get(label, 0) for label in pred_labels})
            writer.writerow(row)


def _summary_markdown(
    variant_id: str,
    pipeline_type: str,
    aggregate: dict[str, Any],
) -> str:
    rows = [
        ("pipeline_type", pipeline_type),
        ("annotated_track_count", aggregate.get("annotated_track_count")),
        ("query_count", aggregate.get("query_count")),
        ("track_team_accuracy", aggregate.get("track_team_accuracy")),
        ("macro_team_f1", aggregate.get("macro_team_f1")),
        ("role_accuracy", aggregate.get("role_accuracy")),
        ("macro_role_f1", aggregate.get("macro_role_f1")),
        ("track_unknown_rate", aggregate.get("track_unknown_rate")),
        ("query_resolved_rate", aggregate.get("query_resolved_rate")),
        (
            "query_selected_track_exact_accuracy",
            aggregate.get("query_selected_track_exact_accuracy"),
        ),
        ("query_team_accuracy", aggregate.get("query_team_accuracy")),
        ("correct_id_correct_team_rate", aggregate.get("correct_id_correct_team_rate")),
        ("grounding_calls_per_query", aggregate.get("grounding_calls_per_query")),
    ]
    lines = [
        f"# Team Benchmark Summary - {variant_id}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in rows:
        lines.append(f"| {key} | {_fmt(value)} |")
    lines.extend(
        [
            "",
            (
                "Tracking metrics such as HOTA, IDF1, IDSW, and MOTA remain "
                "reported by the MOT/TrackEval benchmark."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _team_label(prediction: TrackTeamPrediction | None) -> str | None:
    if prediction is None or prediction.status != "resolved":
        return None
    return prediction.team_label


def _track_error_type(
    *,
    status: str,
    correct_team: bool,
    role_correct: bool | None,
) -> str:
    if status == "missing_prediction":
        return "missing_prediction"
    if status in {"unknown", "not_found"}:
        return status
    if status == "ambiguous":
        return "ambiguous_prediction"
    wrong_team = not correct_team
    wrong_role = role_correct is False
    if wrong_team and wrong_role:
        return "wrong_team_and_role"
    if wrong_team:
        return "wrong_team"
    if wrong_role:
        return "wrong_role"
    return "correct"


def _query_error_type(
    *,
    status: str,
    exact_match: bool,
    hit: bool,
    team_correct: bool,
) -> str:
    if status == "missing_prediction":
        return "missing_prediction"
    if status in {"unknown", "not_found"}:
        return status
    if status == "ambiguous":
        return "ambiguous_prediction"
    if exact_match and team_correct:
        return "correct"
    if not hit and not team_correct:
        return "wrong_track_and_team"
    if not hit:
        return "wrong_track"
    if not exact_match and team_correct:
        return "partial_track_match"
    return "wrong_team"


def _macro_f1(per_team: dict[str, dict[str, float | int]]) -> float | None:
    return _mean(team["f1"] for team in per_team.values())


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _mean(values: Any) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _csv_value(value: Any) -> Any:
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
