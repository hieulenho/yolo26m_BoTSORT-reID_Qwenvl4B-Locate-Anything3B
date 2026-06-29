"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from football_tracking.config import ConfigError, load_config
from football_tracking.data.prepare import (
    DataConfigError,
    audit_data,
    prepare_data,
    validate_data,
    visualize_annotations,
)
from football_tracking.logging_utils import setup_logging
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
                audit = audit_data(args.config, max_sequences=args.max_sequences)
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
        except (DataConfigError, ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            if args.debug:
                traceback.print_exc()
            return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
