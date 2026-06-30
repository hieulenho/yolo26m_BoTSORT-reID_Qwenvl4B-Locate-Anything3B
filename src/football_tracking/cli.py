"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import yaml

from football_tracking.config import ConfigError, load_config
from football_tracking.data.audit import AuditConfigError, run_dataset_audit
from football_tracking.data.prepare import (
    DataConfigError,
    audit_data,
    prepare_data,
    validate_data,
    visualize_annotations,
)
from football_tracking.detection.baseline import (
    BaselineConfigError,
    evaluate_baseline,
    run_baseline,
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
from football_tracking.logging_utils import setup_logging
from football_tracking.reporting.detector_report import write_finetuned_report
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
        "resume": getattr(args, "resume", None) or None,
        "overwrite": True if getattr(args, "overwrite", False) else None,
        "dry_run": True if getattr(args, "dry_run", False) else None,
        "split": getattr(args, "split", None),
        "max_images": getattr(args, "max_images", None),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        _configure_logging(args.config)
        report = run_doctor(config_path=args.config)
        sys.stdout.write(format_doctor_report(report))
        sys.stdout.write("\n")
        return report.exit_code

    if args.command in {"prepare-data", "validate-data", "audit-data", "visualize-annotations"}:
        setup_logging(args.log_level or "INFO")
        try:
            if args.command == "prepare-data":
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
                report = validate_data(args.config, max_sequences=args.max_sequences)
                sys.stdout.write(json.dumps(report.to_dict(), indent=2))
                sys.stdout.write("\n")
                return 1 if report.has_errors else 0
            if args.command == "audit-data":
                audit = (
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
