"""Dataset-independent command line utilities for traffic tracking."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic tracking utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check the local runtime environment.")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--json", action="store_true")
    for name, help_text in {
        "track-video": "Track a video using an explicit tracking config.",
        "cache-detections": "Create shared detections for tracker comparisons.",
        "compare-trackers": "Run trackers on a common detection cache.",
        "evaluate-tracking": "Evaluate cached MOT predictions with TrackEval.",
        "train-detector": "Fine-tune a detector on an explicitly configured dataset.",
        "evaluate-detector": "Evaluate a detector against annotated images.",
        "preflight-training": "Validate training configuration and dataset.",
    }.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--device")
        command.add_argument("--max-frames", type=int)
        command.add_argument("--max-sequences", type=int)
        command.add_argument("--overwrite", action="store_true", default=None)
        command.add_argument("--dry-run", action="store_true")
        if name == "track-video":
            command.add_argument("--source", type=Path)
            command.add_argument("--output-video", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from football_tracking.utils.environment import format_doctor_report, run_doctor

            report = run_doctor(config_path=args.config)
            output = (
                json.dumps(report.to_dict(), indent=2)
                if args.json else format_doctor_report(report)
            )
            print(output)
            return report.exit_code
        overrides = {
            key: value for key, value in vars(args).items()
            if key not in {"command", "config", "dry_run"} and value is not None
        }
        if args.command == "track-video":
            from football_tracking.tracking.pipeline import run_tracking

            result = run_tracking(args.config, overrides=overrides, dry_run=args.dry_run)
        elif args.command == "cache-detections":
            from football_tracking.detection.cache import create_detection_cache

            result = create_detection_cache(args.config, overrides=overrides, dry_run=args.dry_run)
        elif args.command in {"compare-trackers", "evaluate-tracking"}:
            from football_tracking.experiments.experiment_runner import (
                compare_trackers,
                evaluate_tracking_outputs,
            )

            runner = (
                compare_trackers
                if args.command == "compare-trackers" else evaluate_tracking_outputs
            )
            result = runner(args.config, overrides=overrides, dry_run=args.dry_run)
        elif args.command == "evaluate-detector":
            from football_tracking.detection.evaluate import evaluate_detector

            result = evaluate_detector(args.config, overrides=overrides, dry_run=args.dry_run)
        else:
            from football_tracking.detection.trainer import YOLOv8Trainer, run_training_preflight
            from football_tracking.detection.training_config import load_training_config

            config = load_training_config(args.config, overrides=overrides)
            if args.command == "preflight-training":
                report = run_training_preflight(config)
                print(json.dumps(report.to_dict(), indent=2, default=str))
                return int(report.has_errors)
            result = YOLOv8Trainer(config).train(dry_run=args.dry_run)
        print(json.dumps(result, indent=2, default=str))
        if result.get("status") in {"failed", "validation_failed"}:
            return 1
        if args.command == "track-video" and not args.dry_run:
            return int(bool(result["summary"]["validation"]["summary"]["errors"]))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
