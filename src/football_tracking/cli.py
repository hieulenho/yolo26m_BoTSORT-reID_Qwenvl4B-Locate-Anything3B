"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from football_tracking.config import ConfigError, load_config
from football_tracking.logging_utils import setup_logging
from football_tracking.utils.environment import format_doctor_report, run_doctor


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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
