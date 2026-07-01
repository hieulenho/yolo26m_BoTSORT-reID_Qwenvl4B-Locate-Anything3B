"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import yaml

from football_tracking.benchmarking.benchmark import BenchmarkError, generate_benchmark
from football_tracking.config import ConfigError, load_config
from football_tracking.data.audit import AuditConfigError, run_dataset_audit
from football_tracking.data.prepare import (
    DataConfigError,
    audit_data,
    prepare_data,
    validate_data,
    visualize_annotations,
)
from football_tracking.data.sportsmot_adapter import (
    SportsMotError,
    audit_sportsmot,
    prepare_sportsmot,
    validate_sportsmot,
)
from football_tracking.detection.baseline import (
    BaselineConfigError,
    evaluate_baseline,
    run_baseline,
)
from football_tracking.detection.cache import (
    DetectionCacheError,
    create_detection_cache,
    validate_detection_cache,
)
from football_tracking.detection.checkpoint import CheckpointError, validate_checkpoint
from football_tracking.detection.compare_models import compare_detector_reports
from football_tracking.detection.detector import DetectorError
from football_tracking.detection.evaluate import evaluate_detector
from football_tracking.detection.trainer import TrainingError, YOLOv8Trainer, run_training_preflight
from football_tracking.detection.training_config import (
    TrainingConfigError,
    load_training_config,
)
from football_tracking.experiments.ablation import AblationError, run_tracker_ablation
from football_tracking.experiments.experiment_config import ExperimentConfigError
from football_tracking.experiments.experiment_runner import (
    ExperimentRunnerError,
    compare_trackers,
    evaluate_tracking_outputs,
    run_tracker_from_cache,
    summarize_experiments,
)
from football_tracking.logging_utils import setup_logging
from football_tracking.rendering.video_renderer import RenderVideoError, render_videos
from football_tracking.reporting.detector_report import write_finetuned_report
from football_tracking.reporting.final_report import FinalReportError, generate_final_report
from football_tracking.tracking.checkpoint_resolver import CheckpointResolutionError
from football_tracking.tracking.deepsort_adapter import DeepSortConfigError
from football_tracking.tracking.pipeline import (
    TrackingPipelineError,
    run_tracking,
    validate_tracking_outputs,
)
from football_tracking.tracking.sequence_runner import SequenceRunnerError
from football_tracking.tracking.sort_adapter import SortConfigError
from football_tracking.tracking.tracker_factory import TrackerFactoryError
from football_tracking.utils.environment import format_doctor_report, run_doctor


def _add_data_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the data pipeline YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing outputs.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first sequence-level error.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Limit the number of sequences.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default=None,
        help="Reserved split filter for inspection commands.",
    )
    parser.add_argument("--log-level", default=None, help="Override logging level.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show tracebacks for programming errors.",
    )


def _add_baseline_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/yolov8m_baseline.yaml"),
        help="Path to the YOLOv8m baseline YAML config.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _parse_batch(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


def _add_training_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/yolov8m_train.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=_parse_batch, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_tracking_common_options(
    parser: argparse.ArgumentParser,
    default_config: Path,
) -> None:
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render", action="store_true")
    render_group.add_argument("--no-render", action="store_true")
    parser.add_argument("--save-mot", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_detection_cache_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/detection_cache.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_experiment_common_options(
    parser: argparse.ArgumentParser,
    default_config: Path,
) -> None:
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_render_video_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/render_video.yaml"))
    parser.add_argument("--tracker", choices=("sort", "deepsort"), default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_benchmark_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_report_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/report.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check the local project environment.")
    doctor_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a YAML config file.",
    )

    prepare_parser = subparsers.add_parser("prepare-data", help="Prepare YOLO and MOT datasets.")
    _add_data_common_options(prepare_parser)
    prepare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and plan without writing files.",
    )

    sportsmot_parser = subparsers.add_parser(
        "prepare-sportsmot",
        help="Prepare SportsMOT football-only YOLO and MOT datasets.",
    )
    _add_data_common_options(sportsmot_parser)
    sportsmot_parser.set_defaults(config=Path("configs/sportsmot_data.yaml"))
    sportsmot_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate SportsMOT and plan splits without writing converted outputs.",
    )

    prepare_dataset_parser = subparsers.add_parser(
        "prepare-dataset",
        help="Prepare the recommended SportsMOT football dataset.",
    )
    _add_data_common_options(prepare_dataset_parser)
    prepare_dataset_parser.set_defaults(config=Path("configs/sportsmot_data.yaml"))
    prepare_dataset_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate SportsMOT and plan splits without writing converted outputs.",
    )

    validate_parser = subparsers.add_parser("validate-data", help="Validate raw data annotations.")
    _add_data_common_options(validate_parser)

    audit_parser = subparsers.add_parser("audit-data", help="Create basic dataset audit outputs.")
    _add_data_common_options(audit_parser)

    visualize_parser = subparsers.add_parser(
        "visualize-annotations",
        help="Draw sample annotation overlays.",
    )
    _add_data_common_options(visualize_parser)
    visualize_parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Frames per sequence.",
    )

    baseline_detect_parser = subparsers.add_parser(
        "baseline-detect",
        help="Run YOLOv8m pretrained baseline inference and save predictions.",
    )
    _add_baseline_common_options(baseline_detect_parser)

    evaluate_baseline_parser = subparsers.add_parser(
        "evaluate-baseline",
        help="Evaluate the YOLOv8m pretrained baseline with Ultralytics validator.",
    )
    _add_baseline_common_options(evaluate_baseline_parser)

    run_baseline_parser = subparsers.add_parser(
        "run-baseline",
        help="Run YOLOv8m pretrained baseline inference and evaluation.",
    )
    _add_baseline_common_options(run_baseline_parser)

    preflight_parser = subparsers.add_parser(
        "preflight-training",
        help="Validate YOLO training config and dataset before training.",
    )
    _add_training_common_options(preflight_parser)

    train_parser = subparsers.add_parser(
        "train-detector",
        help="Fine-tune YOLOv8m detector.",
    )
    _add_training_common_options(train_parser)

    resume_parser = subparsers.add_parser(
        "resume-detector",
        help="Resume detector training from last.pt.",
    )
    _add_training_common_options(resume_parser)

    eval_parser = subparsers.add_parser(
        "evaluate-detector",
        help="Evaluate fine-tuned detector on val or test split.",
    )
    _add_training_common_options(eval_parser)
    eval_parser.set_defaults(config=Path("configs/yolov8m_eval.yaml"))

    compare_parser = subparsers.add_parser(
        "compare-detectors",
        help="Compare pretrained baseline and fine-tuned detector metrics.",
    )
    compare_parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    compare_parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures/yolov8m_finetuned/comparison"),
    )
    compare_parser.add_argument("--debug", action="store_true")

    track_video_parser = subparsers.add_parser(
        "track-video",
        help="Track players in a video with YOLOv8m and DeepSORT.",
    )
    _add_tracking_common_options(track_video_parser, Path("configs/track_video.yaml"))

    track_parser = subparsers.add_parser(
        "track",
        help="Track the configured SportsMOT football split with YOLOv8m and DeepSORT.",
    )
    _add_tracking_common_options(track_parser, Path("configs/track_sportsmot.yaml"))

    track_sportsmot_parser = subparsers.add_parser(
        "track-sportsmot",
        help="Track SportsMOT football sequences with YOLOv8m and DeepSORT.",
    )
    _add_tracking_common_options(track_sportsmot_parser, Path("configs/track_sportsmot.yaml"))

    validate_tracks_parser = subparsers.add_parser(
        "validate-tracks",
        help="Validate DeepSORT MOT prediction outputs.",
    )
    _add_tracking_common_options(validate_tracks_parser, Path("configs/track_sportsmot_smoke.yaml"))

    cache_parser = subparsers.add_parser(
        "cache-detections",
        help="Run YOLOv8m once and save shared detection cache.",
    )
    _add_detection_cache_common_options(cache_parser)

    validate_cache_parser = subparsers.add_parser(
        "validate-detection-cache",
        help="Validate shared detection cache files.",
    )
    _add_detection_cache_common_options(validate_cache_parser)

    track_from_cache_parser = subparsers.add_parser(
        "track-from-cache",
        help="Run one tracker from shared cached detections.",
    )
    track_from_cache_parser.add_argument(
        "--tracker",
        choices=("sort", "deepsort"),
        required=True,
    )
    track_from_cache_parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/compare_trackers.yaml"),
    )
    track_from_cache_parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    track_from_cache_parser.add_argument("--confidence", type=float, default=None)
    track_from_cache_parser.add_argument("--max-sequences", type=int, default=None)
    track_from_cache_parser.add_argument("--max-frames", type=int, default=None)
    track_from_cache_parser.add_argument("--resume", action="store_true")
    track_from_cache_parser.add_argument("--overwrite", action="store_true")
    track_from_cache_parser.add_argument("--skip-completed", action="store_true")
    track_from_cache_parser.add_argument("--dry-run", action="store_true")
    track_from_cache_parser.add_argument("--debug", action="store_true")

    compare_trackers_parser = subparsers.add_parser(
        "compare-trackers",
        help="Compare SORT and DeepSORT from shared cached detections.",
    )
    _add_experiment_common_options(compare_trackers_parser, Path("configs/compare_trackers.yaml"))

    evaluate_tracking_parser = subparsers.add_parser(
        "evaluate-tracking",
        help="Evaluate existing MOT tracker outputs with TrackEval.",
    )
    _add_experiment_common_options(evaluate_tracking_parser, Path("configs/compare_trackers.yaml"))

    render_video_parser = subparsers.add_parser(
        "render-video",
        help="Render annotated MP4 videos from MOT tracker outputs.",
    )
    _add_render_video_options(render_video_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Generate benchmark CSV, JSON, Markdown, and figures.",
    )
    _add_benchmark_options(benchmark_parser)

    report_parser = subparsers.add_parser(
        "generate-report",
        help="Generate the final Markdown report.",
    )
    _add_report_options(report_parser)

    ablation_parser = subparsers.add_parser(
        "run-tracker-ablation",
        help="Generate or run tracker ablation experiments.",
    )
    ablation_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tracker_ablation.yaml"),
    )
    ablation_parser.add_argument("--max-experiments", type=int, default=None)
    ablation_parser.add_argument("--resume", action="store_true")
    ablation_parser.add_argument("--dry-run", action="store_true")
    ablation_parser.add_argument("--debug", action="store_true")

    summarize_parser = subparsers.add_parser(
        "summarize-experiments",
        help="Summarize persisted experiment result.json files.",
    )
    summarize_parser.add_argument("--root", type=Path, default=Path("outputs/experiments"))
    summarize_parser.add_argument("--debug", action="store_true")

    return parser


def _configure_logging(config_path: Path | None) -> None:
    try:
        config = load_config(config_path=config_path)
    except ConfigError:
        setup_logging("INFO")
        return

    log_level = str(config.runtime.get("log_level", "INFO"))
    log_file = config.paths.get("logs_dir")
    setup_logging(log_level, log_file=log_file / "app.log" if log_file else None)


def _looks_like_audit_config(path: Path) -> bool:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return False
    return (
        isinstance(raw, dict)
        and isinstance(raw.get("dataset"), dict)
        and "config" in raw["dataset"]
    )


def _looks_like_sportsmot_config(path: Path) -> bool:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return False
    return (
        isinstance(raw, dict)
        and isinstance(raw.get("dataset"), dict)
        and raw["dataset"].get("adapter") == "sportsmot"
    )


def _baseline_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "split": args.split,
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "batch": args.batch,
        "max_images": args.max_images,
        "max_sequences": args.max_sequences,
        "overwrite": True if args.overwrite else None,
        "no_visualization": args.no_visualization,
    }


def _training_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "device": getattr(args, "device", None),
        "epochs": getattr(args, "epochs", None),
        "batch": getattr(args, "batch", None),
        "imgsz": getattr(args, "imgsz", None),
        "workers": getattr(args, "workers", None),
        "checkpoint": getattr(args, "checkpoint", None),
        "resume": getattr(args, "resume", None) or None,
        "overwrite": True if getattr(args, "overwrite", False) else None,
        "dry_run": True if getattr(args, "dry_run", False) else None,
        "split": getattr(args, "split", None),
        "max_images": getattr(args, "max_images", None),
    }


def _tracking_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source": getattr(args, "source", None),
        "checkpoint": getattr(args, "checkpoint", None),
        "device": getattr(args, "device", None),
        "conf": getattr(args, "conf", None),
        "imgsz": getattr(args, "imgsz", None),
        "max_sequences": getattr(args, "max_sequences", None),
        "max_frames": getattr(args, "max_frames", None),
        "render": True if getattr(args, "render", False) else None,
        "no_render": True if getattr(args, "no_render", False) else None,
        "save_mot": True if getattr(args, "save_mot", None) else None,
        "overwrite": True if getattr(args, "overwrite", False) else None,
        "show_window": True if getattr(args, "show", False) else None,
    }


def _detection_cache_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "device": getattr(args, "device", None),
        "max_sequences": getattr(args, "max_sequences", None),
        "max_frames": getattr(args, "max_frames", None),
        "overwrite": True if getattr(args, "overwrite", False) else None,
    }


def _experiment_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "split": getattr(args, "split", None),
        "confidence": getattr(args, "confidence", None),
        "max_sequences": getattr(args, "max_sequences", None),
        "max_frames": getattr(args, "max_frames", None),
        "overwrite": True if getattr(args, "overwrite", False) else None,
        "resume": True if getattr(args, "resume", False) else None,
        "skip_completed": True if getattr(args, "skip_completed", False) else None,
    }


def _render_video_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "tracker": getattr(args, "tracker", None),
        "max_sequences": getattr(args, "max_sequences", None),
        "max_frames": getattr(args, "max_frames", None),
        "overwrite": True if getattr(args, "overwrite", False) else None,
    }


def _benchmark_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {"overwrite": True if getattr(args, "overwrite", False) else None}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        _configure_logging(args.config)
        report = run_doctor(config_path=args.config)
        sys.stdout.write(format_doctor_report(report))
        sys.stdout.write("\n")
        return report.exit_code

    if args.command in {
        "prepare-data",
        "prepare-sportsmot",
        "prepare-dataset",
        "validate-data",
        "audit-data",
        "visualize-annotations",
    }:
        setup_logging(args.log_level or "INFO")
        try:
            is_sportsmot = args.command in {
                "prepare-sportsmot",
                "prepare-dataset",
            } or _looks_like_sportsmot_config(args.config)
            if args.command in {"prepare-data", "prepare-sportsmot", "prepare-dataset"}:
                if is_sportsmot:
                    result = prepare_sportsmot(
                        args.config,
                        dry_run=args.dry_run,
                        overwrite=args.overwrite or None,
                    )
                    sys.stdout.write(json.dumps(result, indent=2, default=str))
                    sys.stdout.write("\n")
                    return 0
                result = prepare_data(
                    args.config,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite or None,
                    fail_fast=args.fail_fast or None,
                    max_sequences=args.max_sequences,
                )
                sys.stdout.write(json.dumps(result.summary(), indent=2))
                sys.stdout.write("\n")
                return 1 if result.validation_report.has_errors else 0
            if args.command == "validate-data":
                report = (
                    validate_sportsmot(args.config)
                    if is_sportsmot
                    else validate_data(args.config, max_sequences=args.max_sequences)
                )
                sys.stdout.write(json.dumps(report.to_dict(), indent=2))
                sys.stdout.write("\n")
                return 1 if report.has_errors else 0
            if args.command == "audit-data":
                audit = (
                    audit_sportsmot(args.config)
                    if is_sportsmot
                    else
                    run_dataset_audit(args.config, max_sequences=args.max_sequences)
                    if _looks_like_audit_config(args.config)
                    else audit_data(args.config, max_sequences=args.max_sequences)
                )
                sys.stdout.write(json.dumps(audit, indent=2))
                sys.stdout.write("\n")
                return 0
            paths = visualize_annotations(
                args.config,
                num_samples=args.num_samples,
                max_sequences=args.max_sequences,
            )
            sys.stdout.write(json.dumps({"written": [str(path) for path in paths]}, indent=2))
            sys.stdout.write("\n")
            return 0
        except (
            SportsMotError,
            AuditConfigError,
            DataConfigError,
            ConfigError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            if args.debug:
                traceback.print_exc()
            return 2

    if args.command in {"baseline-detect", "evaluate-baseline", "run-baseline"}:
        setup_logging("INFO")
        try:
            overrides = _baseline_overrides(args)
            if args.command == "baseline-detect":
                result = run_baseline(
                    args.config,
                    overrides=overrides,
                    dry_run=args.dry_run,
                    evaluate=False,
                )
            elif args.command == "evaluate-baseline":
                result = evaluate_baseline(
                    args.config,
                    overrides=overrides,
                    dry_run=args.dry_run,
                )
            else:
                result = run_baseline(
                    args.config,
                    overrides=overrides,
                    dry_run=args.dry_run,
                    evaluate=True,
                )
            sys.stdout.write(json.dumps(result, indent=2))
            sys.stdout.write("\n")
            return 0
        except (
            BaselineConfigError,
            DetectorError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            if args.debug:
                traceback.print_exc()
            return 2

    if args.command in {
        "preflight-training",
        "train-detector",
        "resume-detector",
        "evaluate-detector",
        "compare-detectors",
    }:
        setup_logging("INFO")
        try:
            if args.command == "compare-detectors":
                result = compare_detector_reports(args.metrics_dir, args.figures_dir)
                write_finetuned_report(args.metrics_dir, comparison=result)
                sys.stdout.write(json.dumps(result, indent=2))
                sys.stdout.write("\n")
                return 0
            overrides = _training_overrides(args)
            if args.command == "preflight-training":
                config = load_training_config(args.config, overrides=overrides)
                report = run_training_preflight(config)
                sys.stdout.write(json.dumps(report.to_dict(), indent=2))
                sys.stdout.write("\n")
                return 1 if report.has_errors else 0
            if args.command == "train-detector":
                config = load_training_config(args.config, overrides=overrides)
                result = YOLOv8Trainer(config).train(dry_run=args.dry_run)
                sys.stdout.write(json.dumps(result, indent=2, default=str))
                sys.stdout.write("\n")
                return 0
            if args.command == "resume-detector":
                checkpoint = args.checkpoint
                if checkpoint is None:
                    config = load_training_config(args.config, overrides=overrides)
                    checkpoint = config.run_dir / "weights" / "last.pt"
                validate_checkpoint(checkpoint)
                config = load_training_config(args.config, overrides={**overrides, "resume": True})
                result = YOLOv8Trainer(config).resume(checkpoint)
                sys.stdout.write(json.dumps(result, indent=2, default=str))
                sys.stdout.write("\n")
                return 0
            result = evaluate_detector(
                args.config,
                overrides=overrides,
                dry_run=args.dry_run,
            )
            sys.stdout.write(json.dumps(result, indent=2, default=str))
            sys.stdout.write("\n")
            return 0
        except (
            CheckpointError,
            TrainingConfigError,
            TrainingError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            if getattr(args, "debug", False):
                traceback.print_exc()
            return 2

    if args.command in {"cache-detections", "validate-detection-cache"}:
        setup_logging("INFO")
        try:
            overrides = _detection_cache_overrides(args)
            if args.command == "cache-detections":
                result = create_detection_cache(
                    args.config,
                    overrides=overrides,
                    dry_run=args.dry_run,
                )
                sys.stdout.write(json.dumps(result, indent=2, default=str))
                sys.stdout.write("\n")
                return 0
            result = validate_detection_cache(args.config, overrides=overrides)
            sys.stdout.write(json.dumps(result, indent=2, default=str))
            sys.stdout.write("\n")
            return 1 if result["report"]["summary"]["errors"] else 0
        except (
            DetectionCacheError,
            CheckpointResolutionError,
            SequenceRunnerError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            if getattr(args, "debug", False):
                traceback.print_exc()
            return 2

    if args.command in {
        "track-from-cache",
        "compare-trackers",
        "evaluate-tracking",
        "run-tracker-ablation",
        "summarize-experiments",
    }:
        setup_logging("INFO")
        try:
            if args.command == "track-from-cache":
                result = run_tracker_from_cache(
                    args.experiment_config,
                    tracker_name=args.tracker,
                    overrides=_experiment_overrides(args),
                    dry_run=args.dry_run,
                )
            elif args.command == "compare-trackers":
                result = compare_trackers(
                    args.config,
                    overrides=_experiment_overrides(args),
                    dry_run=args.dry_run,
                )
            elif args.command == "evaluate-tracking":
                result = evaluate_tracking_outputs(
                    args.config,
                    overrides=_experiment_overrides(args),
                    dry_run=args.dry_run,
                )
            elif args.command == "run-tracker-ablation":
                result = run_tracker_ablation(
                    args.config,
                    dry_run=args.dry_run,
                    max_experiments=args.max_experiments,
                )
            else:
                result = summarize_experiments(args.root)
            sys.stdout.write(json.dumps(result, indent=2, default=str))
            sys.stdout.write("\n")
            if args.command == "evaluate-tracking" and result.get("status") == "validation_failed":
                return 1
            return 0
        except (
            AblationError,
            DetectionCacheError,
            ExperimentConfigError,
            ExperimentRunnerError,
            TrackerFactoryError,
            SortConfigError,
            DeepSortConfigError,
            SequenceRunnerError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            if getattr(args, "debug", False):
                traceback.print_exc()
            return 2

    if args.command == "render-video":
        setup_logging("INFO")
        try:
            result = render_videos(
                args.config,
                overrides=_render_video_overrides(args),
                dry_run=args.dry_run,
            )
            sys.stdout.write(json.dumps(result, indent=2, default=str))
            sys.stdout.write("\n")
            return 0
        except (RenderVideoError, FileNotFoundError, RuntimeError, ValueError) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            if getattr(args, "debug", False):
                traceback.print_exc()
            return 2

    if args.command == "benchmark":
        setup_logging("INFO")
        try:
            result = generate_benchmark(
                args.config,
                overrides=_benchmark_overrides(args),
                dry_run=args.dry_run,
            )
            sys.stdout.write(json.dumps(result, indent=2, default=str))
            sys.stdout.write("\n")
            return 0
        except (BenchmarkError, FileNotFoundError, RuntimeError, ValueError) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            if getattr(args, "debug", False):
                traceback.print_exc()
            return 2

    if args.command == "generate-report":
        setup_logging("INFO")
        try:
            result = generate_final_report(
                args.config,
                output_override=args.output,
                dry_run=args.dry_run,
            )
            sys.stdout.write(json.dumps(result, indent=2, default=str))
            sys.stdout.write("\n")
            return 0
        except (FinalReportError, FileNotFoundError, RuntimeError, ValueError) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            if getattr(args, "debug", False):
                traceback.print_exc()
            return 2

    if args.command in {"track", "track-video", "track-sportsmot", "validate-tracks"}:
        setup_logging("INFO")
        try:
            overrides = _tracking_overrides(args)
            if args.command == "validate-tracks":
                result = validate_tracking_outputs(args.config, overrides=overrides)
                sys.stdout.write(json.dumps(result, indent=2, default=str))
                sys.stdout.write("\n")
                return 1 if result["report"]["summary"]["errors"] else 0
            result = run_tracking(args.config, overrides=overrides, dry_run=args.dry_run)
            sys.stdout.write(json.dumps(result, indent=2, default=str))
            sys.stdout.write("\n")
            if not args.dry_run:
                return 1 if result["summary"]["validation"]["summary"]["errors"] else 0
            return 0
        except (
            CheckpointResolutionError,
            DeepSortConfigError,
            SequenceRunnerError,
            TrackingPipelineError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            if getattr(args, "debug", False):
                traceback.print_exc()
            return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
