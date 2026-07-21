"""Ground-truth evaluation for adaptive domain, vocabulary, and track semantics."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from football_tracking.benchmarking.semantic_annotation import (
    SemanticAnnotationError,
    validate_review_metadata,
)
from football_tracking.detection.serialization import file_sha256
from football_tracking.paths import get_project_root


class SemanticEvaluationError(RuntimeError):
    """Raised when semantic benchmark inputs are missing or incompatible."""


def evaluate_semantic_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    artifact_overrides: dict[str, str | Path | None] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise SemanticEvaluationError(f"Semantic GT manifest does not exist: {manifest_file}")
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SemanticEvaluationError("Semantic GT manifest root must be a mapping.")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SemanticEvaluationError("Semantic GT manifest must contain non-empty samples.")
    if artifact_overrides and len(samples) != 1:
        raise SemanticEvaluationError(
            "Artifact overrides require a manifest with exactly one sample."
        )
    require_review_metadata = bool(manifest.get("require_review_metadata", False))

    rows: list[dict[str, Any]] = []
    class_counts = Counter()
    semantic_pairs: list[tuple[str, str, bool]] = []
    action_pairs: list[tuple[str, str]] = []
    artifact_manifest: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for index, value in enumerate(samples):
        if not isinstance(value, dict):
            raise SemanticEvaluationError(f"samples[{index}] must be a mapping.")
        sample_id = str(value.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen_samples:
            raise SemanticEvaluationError(
                f"samples[{index}].sample_id is missing or duplicated: {sample_id!r}"
            )
        seen_samples.add(sample_id)
        artifacts = _mapping(value.get("artifacts"), f"samples[{index}].artifacts")
        if artifact_overrides:
            for key, path in artifact_overrides.items():
                if path is None:
                    artifacts.pop(key, None)
                else:
                    artifacts[key] = str(Path(path).resolve())
        ground_truth = _mapping(
            value.get("ground_truth"), f"samples[{index}].ground_truth"
        )
        if require_review_metadata:
            try:
                validate_review_metadata(
                    ground_truth.get("review"),
                    f"samples[{index}].ground_truth.review",
                )
            except SemanticAnnotationError as exc:
                raise SemanticEvaluationError(str(exc)) from exc
        discovery_path = _resolve_artifact(
            artifacts.get("discovery"), manifest_file.parent, "discovery"
        )
        route_path = _resolve_artifact(
            artifacts.get("route"), manifest_file.parent, "route", required=False
        )
        semantics_path = _resolve_artifact(
            artifacts.get("semantics"), manifest_file.parent, "semantics"
        )
        report_path = _resolve_artifact(
            artifacts.get("run_report"), manifest_file.parent, "run_report", required=False
        )
        tracking_metadata_path = _resolve_artifact(
            artifacts.get("tracking_metadata"),
            manifest_file.parent,
            "tracking_metadata",
            required=False,
        )
        qwen_answer_path = _resolve_artifact(
            artifacts.get("qwen_answer"),
            manifest_file.parent,
            "qwen_answer",
            required=False,
        )
        locate_result_path = _resolve_artifact(
            artifacts.get("locate_result"),
            manifest_file.parent,
            "locate_result",
            required=False,
        )
        discovery = _read_json(discovery_path)
        route = _read_json(route_path) if route_path is not None else {}
        semantics = _read_json(semantics_path)
        run_report = _read_json(report_path) if report_path is not None else {}
        run_report = _merge_direct_performance_artifacts(
            run_report,
            tracking_metadata=(
                _read_json(tracking_metadata_path)
                if tracking_metadata_path is not None
                else {}
            ),
            qwen_answer=(
                _read_json(qwen_answer_path) if qwen_answer_path is not None else {}
            ),
            locate_result=(
                _read_json(locate_result_path) if locate_result_path is not None else {}
            ),
        )
        gt_objects = _ground_truth_objects(ground_truth, sample_id)
        gt_tracks = _ground_truth_tracks(ground_truth, sample_id)
        if not gt_objects and not gt_tracks:
            raise SemanticEvaluationError(
                f"Sample '{sample_id}' has neither object-vocabulary nor track GT."
            )
        row, sample_pairs, sample_actions, sample_class_counts = _evaluate_sample(
            sample_id=sample_id,
            discovery=discovery,
            route=route,
            semantics=semantics,
            run_report=run_report,
            gt_domain=_canonical(ground_truth.get("domain", "unknown")),
            gt_route=_canonical(ground_truth.get("detector_route", "")),
            gt_objects=gt_objects,
            gt_tracks=gt_tracks,
        )
        rows.append(row)
        semantic_pairs.extend(sample_pairs)
        action_pairs.extend(sample_actions)
        class_counts.update(sample_class_counts)
        for name, path in (
            ("discovery", discovery_path),
            ("route", route_path),
            ("semantics", semantics_path),
            ("run_report", report_path),
            ("tracking_metadata", tracking_metadata_path),
            ("qwen_answer", qwen_answer_path),
            ("locate_result", locate_result_path),
        ):
            if path is not None:
                artifact_manifest.append(
                    {
                        "sample_id": sample_id,
                        "artifact": name,
                        "path": str(path),
                        "sha256": file_sha256(path),
                    }
                )

    summary = _aggregate(rows, class_counts, semantic_pairs, action_pairs)
    output_root = Path(output_dir).resolve()
    paths = _output_paths(output_root)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise SemanticEvaluationError(
            "Semantic benchmark output exists and overwrite=false: "
            + ", ".join(str(path) for path in existing)
        )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_file),
        "manifest_sha256": file_sha256(manifest_file),
        "sample_count": len(rows),
        "summary": summary,
        "per_sample": rows,
        "artifacts": artifact_manifest,
        "metric_scope": {
            "accuracy_source": "human ground-truth manifest",
            "coverage_definition": "accepted non-unknown predictions / GT tracks",
            "selective_accuracy_definition": "correct predictions / accepted predictions",
            "unknown_predictions_count_as_errors": True,
            "fine_label_definition": (
                "human-annotated subtype/species/breed/role/make/model; evaluated only "
                "for GT tracks that provide fine_label"
            ),
        },
    }
    _write_json_atomic(paths["summary_json"], payload)
    _write_csv(paths["per_sample_csv"], rows)
    _write_text_atomic(paths["report_md"], _markdown(payload))
    figures = _write_figures(summary, output_root / "figures")
    return {
        "status": "ok",
        "sample_count": len(rows),
        "summary": summary,
        "paths": {key: str(value) for key, value in paths.items()},
        "figures": [str(path) for path in figures],
    }


def _merge_direct_performance_artifacts(
    run_report: dict[str, Any],
    *,
    tracking_metadata: dict[str, Any],
    qwen_answer: dict[str, Any],
    locate_result: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(run_report)
    if tracking_metadata:
        tracking = dict(merged.get("tracking", {}))
        tracking.setdefault("timing", tracking_metadata.get("timing", {}))
        tracking.setdefault("runtime", tracking_metadata.get("runtime", {}))
        merged["tracking"] = tracking
    if qwen_answer:
        qwen = dict(merged.get("qwen_track_semantics", {}))
        qwen.setdefault("timing", qwen_answer.get("timing", {}))
        qwen.setdefault("cuda_memory", qwen_answer.get("cuda_memory", {}))
        merged["qwen_track_semantics"] = qwen
    if locate_result:
        locate = dict(merged.get("locateanything_verification", {}))
        locate.setdefault("timing", locate_result.get("timing", {}))
        locate.setdefault("cuda_memory", locate_result.get("cuda_memory", {}))
        merged["locateanything_verification"] = locate
    return merged


def _evaluate_sample(
    *,
    sample_id: str,
    discovery: dict[str, Any],
    route: dict[str, Any],
    semantics: dict[str, Any],
    run_report: dict[str, Any],
    gt_domain: str,
    gt_route: str,
    gt_objects: dict[str, str],
    gt_tracks: dict[int, dict[str, str]],
) -> tuple[dict[str, Any], list[tuple[str, str, bool]], list[tuple[str, str]], Counter]:
    predicted_domain = _domain_name(discovery.get("domain"))
    predicted_route = _canonical(route.get("route_name", ""))
    route_evaluated = bool(gt_route)
    predicted_objects = {
        _canonical(row.get("canonical_name", row.get("name", ""))): str(
            row.get("action", "detect")
        ).lower()
        for row in discovery.get("objects", [])
        if isinstance(row, dict)
        and _canonical(row.get("canonical_name", row.get("name", "")))
    }
    gt_classes = set(gt_objects)
    predicted_classes = set(predicted_objects)
    true_positive = len(gt_classes & predicted_classes)
    false_positive = len(predicted_classes - gt_classes)
    false_negative = len(gt_classes - predicted_classes)
    class_precision = _safe_div(true_positive, true_positive + false_positive)
    class_recall = _safe_div(true_positive, true_positive + false_negative)
    class_f1 = _f1(class_precision, class_recall)
    action_pairs = [
        (gt_objects[name], predicted_objects.get(name, "missing"))
        for name in sorted(gt_classes)
    ]
    action_correct = sum(expected == predicted for expected, predicted in action_pairs)

    prediction_by_track = {
        int(row["track_id"]): row
        for row in semantics.get("tracks", [])
        if isinstance(row, dict) and row.get("track_id") is not None
    }
    semantic_pairs: list[tuple[str, str, bool]] = []
    accepted_count = 0
    correct_count = 0
    fine_total = 0
    fine_accepted = 0
    fine_correct = 0
    fine_accepted_correct = 0
    for track_id, expected_labels in sorted(gt_tracks.items()):
        expected = expected_labels["class_label"]
        prediction = prediction_by_track.get(track_id, {})
        accepted = bool(prediction.get("accepted", False))
        predicted = (
            _canonical(prediction.get("class_label", "unknown"))
            if accepted
            else "unknown"
        )
        accepted_count += int(accepted and predicted != "unknown")
        correct_count += int(predicted == expected)
        semantic_pairs.append((expected, predicted, accepted and predicted != "unknown"))
        expected_fine = expected_labels.get("fine_label", "")
        if expected_fine:
            predicted_fine_accepted = bool(
                accepted and prediction.get("fine_accepted", False)
            )
            predicted_fine = (
                _canonical(prediction.get("fine_label", "unknown"))
                if predicted_fine_accepted
                else "unknown"
            )
            fine_total += 1
            fine_accepted += int(predicted_fine_accepted and predicted_fine != "unknown")
            fine_correct += int(predicted_fine == expected_fine)
            fine_accepted_correct += int(
                predicted_fine_accepted and predicted_fine == expected_fine
            )

    semantic_total = len(gt_tracks)
    end_to_end_fps = _nested(run_report, "tracking", "timing", "end_to_end_fps")
    qwen_seconds = _nested(
        run_report, "qwen_track_semantics", "timing", "inference_seconds"
    )
    qwen_load_seconds = _nested(
        run_report, "qwen_track_semantics", "timing", "model_load_seconds"
    )
    locate_execution_seconds = _first_non_none(
        _nested(
            run_report,
            "locateanything_verification",
            "timing",
            "execution_seconds",
        ),
        _nested(run_report, "locateanything_verification", "timing", "total_seconds"),
    )
    locate_load_seconds = _nested(
        run_report,
        "locateanything_verification",
        "timing",
        "model_load_seconds",
    )
    locate_cold_seconds = _first_non_none(
        _nested(
            run_report,
            "locateanything_verification",
            "timing",
            "cold_start_total_seconds",
        ),
        _nested(run_report, "locateanything_verification", "timing", "total_seconds"),
    )
    qwen_peak = _nested(
        run_report, "qwen_track_semantics", "cuda_memory", "peak_allocated_bytes"
    )
    locate_peak = _nested(
        run_report,
        "locateanything_verification",
        "cuda_memory",
        "peak_allocated_bytes",
    )
    return (
        {
            "sample_id": sample_id,
            "gt_domain": gt_domain,
            "predicted_domain": predicted_domain,
            "domain_correct": int(predicted_domain == gt_domain),
            "gt_detector_route": gt_route or None,
            "predicted_detector_route": predicted_route or None,
            "router_evaluated": int(route_evaluated),
            "router_correct": int(route_evaluated and predicted_route == gt_route),
            "gt_class_count": len(gt_classes),
            "predicted_class_count": len(predicted_classes),
            "class_tp": true_positive,
            "class_fp": false_positive,
            "class_fn": false_negative,
            "class_precision": round(class_precision, 6),
            "class_recall": round(class_recall, 6),
            "class_f1": round(class_f1, 6),
            "action_correct": action_correct,
            "action_total": len(action_pairs),
            "action_accuracy": round(_safe_div(action_correct, len(action_pairs)), 6),
            "semantic_correct": correct_count,
            "semantic_total": semantic_total,
            "semantic_accuracy": round(_safe_div(correct_count, semantic_total), 6),
            "semantic_accepted": accepted_count,
            "semantic_coverage": round(_safe_div(accepted_count, semantic_total), 6),
            "semantic_selective_accuracy": round(
                _safe_div(
                    sum(
                        expected == predicted
                        for expected, predicted, accepted in semantic_pairs
                        if accepted
                    ),
                    accepted_count,
                ),
                6,
            ),
            "fine_semantic_correct": fine_correct,
            "fine_semantic_total": fine_total,
            "fine_semantic_accuracy": (
                round(_safe_div(fine_correct, fine_total), 6) if fine_total else None
            ),
            "fine_semantic_accepted": fine_accepted,
            "fine_semantic_coverage": (
                round(_safe_div(fine_accepted, fine_total), 6) if fine_total else None
            ),
            "fine_semantic_selective_accuracy": (
                round(_safe_div(fine_accepted_correct, fine_accepted), 6)
                if fine_accepted
                else None
            ),
            "tracking_end_to_end_fps": end_to_end_fps,
            "qwen_model_load_seconds": qwen_load_seconds,
            "qwen_inference_seconds": qwen_seconds,
            "locate_model_load_seconds": locate_load_seconds,
            "locate_inference_seconds": locate_execution_seconds,
            "locate_cold_start_seconds": locate_cold_seconds,
            "qwen_peak_allocated_bytes": qwen_peak,
            "locate_peak_allocated_bytes": locate_peak,
        },
        semantic_pairs,
        action_pairs,
        Counter(
            {
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
            }
        ),
    )


def _aggregate(
    rows: list[dict[str, Any]],
    class_counts: Counter,
    semantic_pairs: list[tuple[str, str, bool]],
    action_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    class_precision = _safe_div(class_counts["tp"], class_counts["tp"] + class_counts["fp"])
    class_recall = _safe_div(class_counts["tp"], class_counts["tp"] + class_counts["fn"])
    accepted_pairs = [pair for pair in semantic_pairs if pair[2]]
    semantic_correct = sum(expected == predicted for expected, predicted, _ in semantic_pairs)
    accepted_correct = sum(expected == predicted for expected, predicted, _ in accepted_pairs)
    labels = sorted({expected for expected, _, _ in semantic_pairs if expected != "unknown"})
    per_class = {}
    for label in labels:
        tp = sum(
            expected == label and predicted == label
            for expected, predicted, _ in semantic_pairs
        )
        fp = sum(
            expected != label and predicted == label
            for expected, predicted, _ in semantic_pairs
        )
        fn = sum(
            expected == label and predicted != label
            for expected, predicted, _ in semantic_pairs
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        per_class[label] = {
            "support": sum(expected == label for expected, _, _ in semantic_pairs),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(_f1(precision, recall), 6),
        }
    numeric_means = {
        key: _mean([row.get(key) for row in rows])
        for key in (
            "tracking_end_to_end_fps",
            "qwen_model_load_seconds",
            "qwen_inference_seconds",
            "locate_model_load_seconds",
            "locate_inference_seconds",
            "locate_cold_start_seconds",
            "qwen_peak_allocated_bytes",
            "locate_peak_allocated_bytes",
        )
    }
    router_rows = [row for row in rows if row.get("router_evaluated")]
    fine_total = sum(int(row.get("fine_semantic_total", 0)) for row in rows)
    fine_correct = sum(int(row.get("fine_semantic_correct", 0)) for row in rows)
    fine_accepted = sum(int(row.get("fine_semantic_accepted", 0)) for row in rows)
    fine_accepted_correct = sum(
        int(row.get("fine_semantic_correct", 0))
        for row in rows
        if row.get("fine_semantic_accepted", 0)
    )
    return {
        "domain_accuracy": round(
            _safe_div(sum(row["domain_correct"] for row in rows), len(rows)), 6
        ),
        "router_accuracy": (
            round(
                _safe_div(
                    sum(row["router_correct"] for row in router_rows),
                    len(router_rows),
                ),
                6,
            )
            if router_rows
            else None
        ),
        "router_gt_sample_count": len(router_rows),
        "class_precision": round(class_precision, 6),
        "class_recall": round(class_recall, 6),
        "class_f1": round(_f1(class_precision, class_recall), 6),
        "action_accuracy": round(
            _safe_div(
                sum(expected == predicted for expected, predicted in action_pairs),
                len(action_pairs),
            ),
            6,
        ),
        "semantic_track_accuracy": round(
            _safe_div(semantic_correct, len(semantic_pairs)), 6
        ),
        "semantic_macro_f1": round(
            _mean([value["f1"] for value in per_class.values()]) or 0.0, 6
        ),
        "semantic_coverage": round(_safe_div(len(accepted_pairs), len(semantic_pairs)), 6),
        "semantic_selective_accuracy": round(
            _safe_div(accepted_correct, len(accepted_pairs)), 6
        ),
        "semantic_gt_track_count": len(semantic_pairs),
        "semantic_accepted_track_count": len(accepted_pairs),
        "fine_semantic_track_accuracy": (
            round(_safe_div(fine_correct, fine_total), 6) if fine_total else None
        ),
        "fine_semantic_coverage": (
            round(_safe_div(fine_accepted, fine_total), 6) if fine_total else None
        ),
        "fine_semantic_selective_accuracy": (
            round(_safe_div(fine_accepted_correct, fine_accepted), 6)
            if fine_accepted
            else None
        ),
        "fine_semantic_gt_track_count": fine_total,
        "fine_semantic_accepted_track_count": fine_accepted,
        "per_class": per_class,
        "performance_means": numeric_means,
    }


def _ground_truth_objects(data: dict[str, Any], sample_id: str) -> dict[str, str]:
    rows = data.get("objects", [])
    if not isinstance(rows, list):
        raise SemanticEvaluationError(f"Sample '{sample_id}' ground_truth.objects must be a list.")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _canonical(row.get("canonical_name", row.get("name", "")))
        if name:
            result[name] = str(row.get("action", "detect")).lower()
    return result


def _ground_truth_tracks(
    data: dict[str, Any], sample_id: str
) -> dict[int, dict[str, str]]:
    rows = data.get("tracks", [])
    if not isinstance(rows, list):
        raise SemanticEvaluationError(f"Sample '{sample_id}' ground_truth.tracks must be a list.")
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or bool(row.get("ignore", False)):
            continue
        track_id = int(row.get("track_id", 0))
        label = _canonical(row.get("class_label", ""))
        if track_id <= 0 or not label:
            raise SemanticEvaluationError(
                f"Sample '{sample_id}' has invalid semantic GT row: {row}"
            )
        if track_id in result:
            raise SemanticEvaluationError(
                f"Sample '{sample_id}' duplicates GT track_id={track_id}."
            )
        result[track_id] = {
            "class_label": label,
            "fine_label": _canonical(row.get("fine_label", "")),
        }
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SemanticEvaluationError(f"{name} must be a mapping.")
    return value


def _resolve_artifact(
    value: Any,
    manifest_dir: Path,
    name: str,
    *,
    required: bool = True,
) -> Path | None:
    if value in (None, "") and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SemanticEvaluationError(f"Artifact '{name}' must be a path string.")
    candidate = Path(value)
    if not candidate.is_absolute():
        local = (manifest_dir / candidate).resolve()
        project = (get_project_root() / candidate).resolve()
        candidate = local if local.is_file() else project
    else:
        candidate = candidate.resolve()
    if not candidate.is_file():
        raise SemanticEvaluationError(f"Artifact '{name}' does not exist: {candidate}")
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SemanticEvaluationError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise SemanticEvaluationError(f"JSON artifact root must be an object: {path}")
    return value


def _domain_name(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name", "unknown")
    return _canonical(value or "unknown")


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_non_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _output_paths(root: Path) -> dict[str, Path]:
    return {
        "summary_json": root / "semantic_benchmark_summary.json",
        "per_sample_csv": root / "semantic_benchmark_per_sample.csv",
        "report_md": root / "semantic_benchmark_report.md",
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join(
        [
            "# Adaptive semantic benchmark",
            "",
            f"- Human-annotated samples: **{payload['sample_count']}**",
            f"- Domain accuracy: **{summary['domain_accuracy']:.3f}**",
            "- Detector router accuracy: **{}**".format(
                f"{summary['router_accuracy']:.3f}"
                if summary["router_accuracy"] is not None
                else "n/a (no route GT)"
            ),
            f"- Dynamic class F1: **{summary['class_f1']:.3f}**",
            f"- Detect/track/context action accuracy: **{summary['action_accuracy']:.3f}**",
            f"- Semantic track accuracy: **{summary['semantic_track_accuracy']:.3f}**",
            f"- Semantic macro-F1: **{summary['semantic_macro_f1']:.3f}**",
            f"- Semantic coverage: **{summary['semantic_coverage']:.3f}**",
            f"- Selective accuracy: **{summary['semantic_selective_accuracy']:.3f}**",
            "- Fine-grained track accuracy: **{}**".format(
                f"{summary['fine_semantic_track_accuracy']:.3f}"
                if summary["fine_semantic_track_accuracy"] is not None
                else "n/a (no fine-label GT)"
            ),
            "- Fine-grained coverage: **{}**".format(
                f"{summary['fine_semantic_coverage']:.3f}"
                if summary["fine_semantic_coverage"] is not None
                else "n/a (no fine-label GT)"
            ),
            "",
            "Unknown or rejected predictions count as errors in semantic track accuracy. "
            "Selective accuracy is reported separately and must be read together with coverage.",
            "",
        ]
    )


def _write_figures(summary: dict[str, Any], directory: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    values = {
        "Domain acc.": summary["domain_accuracy"],
        "Class F1": summary["class_f1"],
        "Action acc.": summary["action_accuracy"],
        "Track acc.": summary["semantic_track_accuracy"],
        "Macro F1": summary["semantic_macro_f1"],
        "Coverage": summary["semantic_coverage"],
        "Selective acc.": summary["semantic_selective_accuracy"],
    }
    if summary["router_accuracy"] is not None:
        values = {
            "Domain acc.": summary["domain_accuracy"],
            "Router acc.": summary["router_accuracy"],
            **{key: value for key, value in values.items() if key != "Domain acc."},
        }
    if summary["fine_semantic_track_accuracy"] is not None:
        values["Fine acc."] = summary["fine_semantic_track_accuracy"]
        values["Fine coverage"] = summary["fine_semantic_coverage"]
    path = directory / "semantic_quality.png"
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(values.keys(), values.values(), color="#2f5d2a")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Adaptive semantic benchmark (human ground truth)")
    axis.tick_params(axis="x", rotation=25)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return [path]
