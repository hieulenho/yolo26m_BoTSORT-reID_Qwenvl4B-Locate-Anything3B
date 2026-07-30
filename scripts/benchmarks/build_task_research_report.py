"""Build the audited report for the explicit single-Qwen task pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.benchmarking.task_research_report import (
    TaskResearchReportError,
    build_task_research_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/task_research_report.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = build_task_research_report(args.config, overwrite=args.overwrite)
    except (TaskResearchReportError, OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
