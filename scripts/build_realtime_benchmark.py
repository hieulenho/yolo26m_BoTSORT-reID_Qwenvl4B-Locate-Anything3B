"""Build charts and tables from measured realtime metadata artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.benchmarking.realtime_report import (
    RealtimeReportError,
    build_realtime_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=path/to/realtime_metrics.json; repeat for each run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    runs = []
    for value in args.run:
        if "=" not in value:
            sys.stderr.write(f"Error: invalid --run value: {value}\n")
            return 2
        name, path = value.split("=", 1)
        runs.append((name.strip(), Path(path.strip())))
    try:
        result = build_realtime_report(runs, args.output_dir, overwrite=args.overwrite)
    except (RealtimeReportError, OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
