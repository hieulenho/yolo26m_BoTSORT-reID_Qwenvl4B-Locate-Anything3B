from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from football_tracking.team_benchmark.evaluator import evaluate_team_benchmark
from football_tracking.team_benchmark.manifest import load_team_benchmark_manifest
from football_tracking.team_benchmark.visual_color_classifier import (
    TrackColorClassifier,
    build_samples_for_all_tracks,
    build_samples_from_manifest,
    leave_one_track_out_metrics,
    make_query_predictions_from_track_predictions,
    prediction_rows,
    predictions_to_manifest_dict,
    track_observation_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real visual-color team attribution experiment for video_1.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/team_benchmark/video_1/benchmark_manifest_expanded.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/team_benchmark/video_1_visual_color_real"),
    )
    parser.add_argument("--samples-per-track", type=int, default=7)
    parser.add_argument("--min-observations", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and overwrite=false: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_team_benchmark_manifest(args.manifest)
    sequence = manifest.sequences[0]
    if sequence.tracks_path is None:
        raise ValueError("The expanded video_1 manifest must contain tracks_path.")

    gt_samples = build_samples_from_manifest(
        manifest_path=args.manifest,
        samples_per_track=args.samples_per_track,
    )
    classifier = TrackColorClassifier.fit(gt_samples)
    gt_predictions = [
        classifier.predict(
            sequence_name=sample.sequence_name,
            track_id=sample.track_id,
            role_label=sample.role_label,
            feature=sample.feature,
            evidence_frames=sample.evidence_frames,
            crop_count=sample.crop_count,
        )
        for sample in gt_samples
    ]
    observation_counts = track_observation_counts(sequence.tracks_path)
    query_predictions = make_query_predictions_from_track_predictions(
        manifest_path=args.manifest,
        predictions=gt_predictions,
        track_observation_counts=observation_counts,
    )
    prediction_manifest = predictions_to_manifest_dict(
        variant_id=f"visual_color_{sequence.sequence_name}_real",
        variant_name=(
            f"Visual color prototype classifier - {sequence.sequence_name} real predictions"
        ),
        benchmark_name=manifest.benchmark_name,
        predictions=gt_predictions,
        query_predictions=query_predictions,
        metadata={
            "experiment_type": "real_visual_color_prediction",
            "samples_per_track": args.samples_per_track,
            "note": (
                "Track labels are predicted from crop color features. The classifier is "
                "calibrated on the verified benchmark tracks; leave-one-track-out metrics "
                "are reported separately for non-singleton classes."
            ),
        },
    )
    prediction_manifest_path = args.output_dir / "visual_color_predictions_manifest.json"
    _write_json(prediction_manifest_path, prediction_manifest)

    evaluation = evaluate_team_benchmark(
        manifest_path=args.manifest,
        prediction_manifest_path=prediction_manifest_path,
        output_dir=args.output_dir / "evaluation",
        overwrite=True,
    )

    loo = leave_one_track_out_metrics(gt_samples)
    _write_json(args.output_dir / "leave_one_track_out_metrics.json", loo)
    _write_csv(args.output_dir / "leave_one_track_out_metrics.csv", loo["rows"])

    all_samples = build_samples_for_all_tracks(
        sequence_name=sequence.sequence_name,
        source_video=sequence.source_video,
        tracks_path=sequence.tracks_path,
        samples_per_track=args.samples_per_track,
        min_observations=args.min_observations,
    )
    all_predictions = [
        classifier.predict(
            sequence_name=sample.sequence_name,
            track_id=sample.track_id,
            role_label=None,
            feature=sample.feature,
            evidence_frames=sample.evidence_frames,
            crop_count=sample.crop_count,
        )
        for sample in all_samples
    ]
    all_rows = prediction_rows(all_predictions)
    _write_json(args.output_dir / "all_track_predictions.json", all_rows)
    _write_csv(args.output_dir / "all_track_predictions.csv", all_rows)

    summary = _summary_markdown(
        benchmark_name=manifest.benchmark_name,
        sequence_name=sequence.sequence_name,
        aggregate=evaluation.aggregate,
        loo=loo,
        all_track_count=len(all_rows),
        output_dir=args.output_dir,
    )
    summary_path = args.output_dir / "visual_color_experiment_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "prediction_manifest": str(prediction_manifest_path),
                "evaluation_dir": str(args.output_dir / "evaluation"),
                "summary": str(summary_path),
                "annotated_track_count": evaluation.aggregate["annotated_track_count"],
                "all_predicted_track_count": len(all_rows),
                "track_team_accuracy": evaluation.aggregate["track_team_accuracy"],
                "leave_one_track_out_accuracy": loo["accuracy"],
            },
            indent=2,
        )
    )


def _summary_markdown(
    *,
    benchmark_name: str,
    sequence_name: str,
    aggregate: dict[str, Any],
    loo: dict[str, Any],
    all_track_count: int,
    output_dir: Path,
) -> str:
    return "\n".join(
        [
            "# Visual Color Team Attribution Experiment",
            "",
            f"This is a real automated prediction run for `{benchmark_name}`.",
            f"Sequence: `{sequence_name}`.",
            "It predicts team labels from track crops, then evaluates against verified labels.",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Verified GT tracks | {aggregate['annotated_track_count']} |",
            f"| Language queries | {aggregate['query_count']} |",
            f"| Track team accuracy | {_fmt(aggregate['track_team_accuracy'])} |",
            f"| Macro team F1 | {_fmt(aggregate['macro_team_f1'])} |",
            f"| Wrong team rate | {_fmt(aggregate['wrong_team_rate'])} |",
            f"| Query exact accuracy | {_fmt(aggregate['query_selected_track_exact_accuracy'])} |",
            f"| Correct ID + team rate | {_fmt(aggregate['correct_id_correct_team_rate'])} |",
            f"| LOO evaluated tracks | {loo['evaluated_track_count']} |",
            f"| LOO skipped singleton tracks | {loo['skipped_track_count']} |",
            f"| LOO accuracy | {_fmt(loo['accuracy'])} |",
            f"| All tracks predicted | {all_track_count} |",
            "",
            "LOO means leave-one-track-out. Singleton classes, such as one-off goalkeeper",
            "colors, are skipped because no same-class prototype remains after holding",
            "that track out.",
            "",
            "Artifacts:",
            f"- `{output_dir / 'visual_color_predictions_manifest.json'}`",
            f"- `{output_dir / 'evaluation' / 'aggregate_metrics.json'}`",
            f"- `{output_dir / 'all_track_predictions.csv'}`",
            f"- `{output_dir / 'leave_one_track_out_metrics.csv'}`",
            "",
        ]
    )


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(value, ensure_ascii=False, default=str)
                    if isinstance(value, list | tuple | dict)
                    else value
                    for field, value in row.items()
                }
            )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
