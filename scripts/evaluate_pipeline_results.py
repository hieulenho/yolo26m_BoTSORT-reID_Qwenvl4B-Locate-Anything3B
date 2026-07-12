"""Evaluate team/role predictions for semantic tracking pipelines.

The rendered video manifest can contain several label sources:

* model claims from Qwen or LocateAnything,
* reviewed annotations used only to make videos readable,
* visual-color completion used only to cover every rendered track,
* unknown fallbacks.

This script reports model-claim metrics separately from render-output metrics so
reviewed labels and coverage fallbacks are not mistaken for model accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PIPELINE_NAMES: dict[str, str] = {
    "A": "YOLO26m + BoT-SORT ReID + Qwen3-VL 4B",
    "B": "YOLO26m + BoT-SORT ReID + LocateAnything 3B",
    "C": "YOLO26m + BoT-SORT ReID + LocateAnything 3B + Qwen3-VL 4B",
}

MODEL_SOURCE_TYPES = {
    "qwen_structured_prediction",
    "locateanything_query_resolution",
}


@dataclass(frozen=True)
class Prediction:
    track_id: int
    team_label: str
    role_label: str
    source_type: str
    not_model_claim: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate semantic team/role predictions for pipelines A/B/C against "
            "a reviewed track annotation CSV."
        )
    )
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument("--pipeline-a", type=Path)
    parser.add_argument("--pipeline-b", type=Path)
    parser.add_argument("--pipeline-c", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "evaluation_results.json"
    report_path = args.output_dir / "evaluation_report.md"

    if results_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists and overwrite=false: {results_path}")
    if not args.annotation_csv.is_file():
        raise SystemExit(f"Annotation CSV not found: {args.annotation_csv}")

    ground_truth = _load_annotation_csv(args.annotation_csv, args.sequence_name)
    if not ground_truth:
        raise SystemExit(
            f"No annotations found for sequence {args.sequence_name!r} in "
            f"{args.annotation_csv}"
        )

    pipeline_paths: dict[str, Path | None] = {
        "A": args.pipeline_a,
        "B": args.pipeline_b,
        "C": args.pipeline_c,
    }
    pipeline_results: dict[str, dict[str, Any]] = {}
    for name, pred_path in pipeline_paths.items():
        if pred_path is None:
            continue
        if not pred_path.is_file():
            print(f"Pipeline {name}: missing prediction manifest: {pred_path}")
            continue
        predictions = _load_predictions(pred_path, args.sequence_name)
        result = _evaluate_pipeline(
            pipeline=name,
            prediction_path=pred_path,
            ground_truth=ground_truth,
            predictions=predictions,
            output_dir=args.output_dir,
        )
        pipeline_results[name] = result
        model = result["model_claim_metrics"]
        render = result["render_output_metrics"]
        print(
            f"Pipeline {name}: "
            f"model_coverage={model['covered_tracks']}/{model['annotated_tracks']} "
            f"model_team_acc={model['team_accuracy_on_covered']:.1%} "
            f"render_coverage={render['covered_tracks']}/{render['annotated_tracks']}"
        )

    if not pipeline_results:
        raise SystemExit("No pipeline manifests were available for evaluation.")

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "sequence_name": args.sequence_name,
        "annotation_csv": str(args.annotation_csv),
        "annotated_track_count": len(ground_truth),
        "model_source_types": sorted(MODEL_SOURCE_TYPES),
        "pipeline_results": pipeline_results,
    }
    results_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = _build_report(args.sequence_name, pipeline_results, len(ground_truth))
    report_path.write_text(report, encoding="utf-8")

    print(f"\n==> Evaluation report : {report_path}")
    print(f"==> Full results JSON : {results_path}")
    print()
    for line in report.splitlines():
        if line.startswith("#") or line.startswith("|"):
            print(line)


def _load_annotation_csv(path: Path, sequence_name: str) -> dict[int, dict[str, str]]:
    ground_truth: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("sequence_name", "")).strip() != sequence_name:
                continue
            try:
                track_id = int(float(str(row["track_id"])))
            except (KeyError, ValueError):
                continue
            ground_truth[track_id] = {
                "team_label": str(row.get("team_label") or "unknown").strip(),
                "role_label": str(row.get("role_label") or "unknown").strip(),
            }
    return ground_truth


def _load_predictions(path: Path, sequence_name: str) -> dict[int, Prediction]:
    data = json.loads(path.read_text(encoding="utf-8"))
    predictions: dict[int, Prediction] = {}
    for item in data.get("track_predictions", []):
        if str(item.get("sequence_name", "")) != sequence_name:
            continue
        if item.get("status", "resolved") != "resolved":
            continue
        try:
            track_id = int(item["track_id"])
        except (KeyError, ValueError):
            continue
        metadata = item.get("metadata") or {}
        source_type = str(metadata.get("source_type") or "unspecified")
        predictions[track_id] = Prediction(
            track_id=track_id,
            team_label=str(item.get("team_label") or "unknown"),
            role_label=str(item.get("role_label") or "unknown"),
            source_type=source_type,
            not_model_claim=bool(metadata.get("not_model_claim", False)),
        )
    return predictions


def _evaluate_pipeline(
    *,
    pipeline: str,
    prediction_path: Path,
    ground_truth: dict[int, dict[str, str]],
    predictions: dict[int, Prediction],
    output_dir: Path,
) -> dict[str, Any]:
    model_predictions = {
        track_id: pred
        for track_id, pred in predictions.items()
        if _is_model_claim(pred)
    }
    model_metrics = _evaluate_predictions(
        ground_truth=ground_truth,
        predictions=model_predictions,
        confusion_prefix=output_dir / f"confusion_pipeline_{pipeline.lower()}_model",
    )
    render_metrics = _evaluate_predictions(
        ground_truth=ground_truth,
        predictions=predictions,
        confusion_prefix=output_dir / f"confusion_pipeline_{pipeline.lower()}_render",
    )
    return {
        "pipeline": pipeline,
        "pipeline_name": PIPELINE_NAMES.get(pipeline, pipeline),
        "prediction_manifest": str(prediction_path),
        "prediction_count": len(predictions),
        "model_prediction_count": len(model_predictions),
        "source_type_counts": dict(
            sorted(Counter(pred.source_type for pred in predictions.values()).items())
        ),
        "model_claim_metrics": model_metrics,
        "render_output_metrics": render_metrics,
    }


def _is_model_claim(prediction: Prediction) -> bool:
    if prediction.not_model_claim:
        return False
    return prediction.source_type in MODEL_SOURCE_TYPES


def _evaluate_predictions(
    *,
    ground_truth: dict[int, dict[str, str]],
    predictions: dict[int, Prediction],
    confusion_prefix: Path,
) -> dict[str, Any]:
    annotated = len(ground_truth)
    covered = 0
    team_correct = 0
    role_correct = 0
    joint_correct = 0
    team_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    role_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_track: list[dict[str, Any]] = []

    for track_id, gt in sorted(ground_truth.items()):
        pred = predictions.get(track_id)
        gt_team = gt["team_label"]
        gt_role = gt["role_label"]
        if pred is None:
            pred_team = "MISSING"
            pred_role = "MISSING"
            has_prediction = False
        else:
            pred_team = pred.team_label
            pred_role = pred.role_label
            has_prediction = True

        team_confusion[gt_team][pred_team] += 1
        role_confusion[gt_role][pred_role] += 1

        team_ok = pred_team == gt_team
        role_ok = pred_role == gt_role
        if has_prediction:
            covered += 1
            team_correct += int(team_ok)
            role_correct += int(role_ok)
            joint_correct += int(team_ok and role_ok)

        per_track.append(
            {
                "track_id": track_id,
                "gt_team": gt_team,
                "pred_team": pred_team,
                "gt_role": gt_role,
                "pred_role": pred_role,
                "has_prediction": has_prediction,
                "team_correct": team_ok if has_prediction else None,
                "role_correct": role_ok if has_prediction else None,
                "joint_correct": (team_ok and role_ok) if has_prediction else None,
                "source_type": pred.source_type if pred else None,
            }
        )

    _write_confusion_csv(
        team_confusion,
        confusion_prefix.with_name(confusion_prefix.name + "_team.csv"),
    )
    _write_confusion_csv(
        role_confusion,
        confusion_prefix.with_name(confusion_prefix.name + "_role.csv"),
    )

    return {
        "annotated_tracks": annotated,
        "covered_tracks": covered,
        "missing_tracks": annotated - covered,
        "coverage_rate": _round_ratio(covered, annotated),
        "team_correct": team_correct,
        "role_correct": role_correct,
        "joint_correct": joint_correct,
        "team_accuracy_on_covered": _round_ratio(team_correct, covered),
        "role_accuracy_on_covered": _round_ratio(role_correct, covered),
        "joint_accuracy_on_covered": _round_ratio(joint_correct, covered),
        "team_accuracy_on_all_annotated": _round_ratio(team_correct, annotated),
        "role_accuracy_on_all_annotated": _round_ratio(role_correct, annotated),
        "joint_accuracy_on_all_annotated": _round_ratio(joint_correct, annotated),
        "per_team_f1": _compute_per_class_f1(team_confusion),
        "per_role_f1": _compute_per_class_f1(role_confusion),
        "per_track": per_track,
    }


def _compute_per_class_f1(confusion: dict[str, Counter[str]]) -> dict[str, dict[str, float]]:
    labels = sorted(confusion.keys())
    result: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = confusion.get(label, Counter()).get(label, 0)
        fp = sum(
            confusion.get(gt_label, Counter()).get(label, 0)
            for gt_label in confusion
            if gt_label != label
        )
        fn = sum(
            count
            for pred_label, count in confusion.get(label, Counter()).items()
            if pred_label != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return result


def _write_confusion_csv(confusion: dict[str, Counter[str]], path: Path) -> None:
    labels = sorted(
        set(confusion.keys()) | {label for counts in confusion.values() for label in counts}
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gt \\ pred", *labels])
        for gt_label in labels:
            writer.writerow(
                [
                    gt_label,
                    *[
                        confusion.get(gt_label, Counter()).get(pred_label, 0)
                        for pred_label in labels
                    ],
                ]
            )


def _round_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _build_report(
    sequence_name: str,
    results: dict[str, dict[str, Any]],
    annotated_total: int,
) -> str:
    lines = [
        f"# Semantic Pipeline Evaluation - {sequence_name}",
        "",
        f"Ground truth: **{annotated_total} annotated tracks**",
        "",
        "> Model metrics count only Qwen/LocateAnything predictions. Reviewed annotation, "
        "visual-color completion, and unknown fallbacks are excluded from model accuracy.",
        "",
        "## Model-Claim Metrics",
        "",
        (
            "| Pipeline | Model | Model Coverage | Team Acc | Role Acc | "
            "Team+Role Acc | Team Recall All GT |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["A", "B", "C"]:
        row = results.get(name)
        if row is None:
            lines.append(f"| {name} | {PIPELINE_NAMES[name]} | - | - | - | - | - |")
            continue
        metrics = row["model_claim_metrics"]
        lines.append(
            f"| {name} | {PIPELINE_NAMES[name]} "
            f"| {metrics['covered_tracks']}/{annotated_total} ({metrics['coverage_rate']:.1%}) "
            f"| {metrics['team_accuracy_on_covered']:.1%} "
            f"| {metrics['role_accuracy_on_covered']:.1%} "
            f"| {metrics['joint_accuracy_on_covered']:.1%} "
            f"| {metrics['team_accuracy_on_all_annotated']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Render-Output Audit",
            "",
            "These numbers describe the labels visible in the demo video. They may include "
            "reviewed labels or visual completion and must not be reported as pure model accuracy.",
            "",
            "| Pipeline | Render Coverage | Team Acc | Role Acc | Team+Role Acc | Source Counts |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name in ["A", "B", "C"]:
        row = results.get(name)
        if row is None:
            lines.append(f"| {name} | - | - | - | - | - |")
            continue
        metrics = row["render_output_metrics"]
        source_counts = ", ".join(
            f"{key}:{value}" for key, value in row["source_type_counts"].items()
        )
        lines.append(
            f"| {name} "
            f"| {metrics['covered_tracks']}/{annotated_total} ({metrics['coverage_rate']:.1%}) "
            f"| {metrics['team_accuracy_on_covered']:.1%} "
            f"| {metrics['role_accuracy_on_covered']:.1%} "
            f"| {metrics['joint_accuracy_on_covered']:.1%} "
            f"| `{source_counts}` |"
        )

    lines.extend(["", "## Per-Team F1 - Model Claims", ""])
    _append_class_f1_table(
        lines,
        results,
        metric_key="model_claim_metrics",
        class_key="per_team_f1",
    )
    lines.extend(["", "## Per-Role F1 - Model Claims", ""])
    _append_class_f1_table(
        lines,
        results,
        metric_key="model_claim_metrics",
        class_key="per_role_f1",
    )
    lines.extend(
        [
            "",
            "Confusion matrices are written next to this report as "
            "`confusion_pipeline_*_model_*.csv` and `confusion_pipeline_*_render_*.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def _append_class_f1_table(
    lines: list[str],
    results: dict[str, dict[str, Any]],
    *,
    metric_key: str,
    class_key: str,
) -> None:
    labels: set[str] = set()
    for row in results.values():
        labels.update(row.get(metric_key, {}).get(class_key, {}).keys())
    if not labels:
        lines.append("_No model predictions available._")
        return
    lines.append("| Label | A P | A R | A F1 | B P | B R | B F1 | C P | C R | C F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in sorted(labels):
        cells = [f"`{label}`"]
        for pipeline in ["A", "B", "C"]:
            metrics = (
                results.get(pipeline, {})
                .get(metric_key, {})
                .get(class_key, {})
                .get(label, {})
            )
            cells.extend(
                [
                    f"{metrics.get('precision', 0.0):.2f}",
                    f"{metrics.get('recall', 0.0):.2f}",
                    f"{metrics.get('f1', 0.0):.2f}",
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
