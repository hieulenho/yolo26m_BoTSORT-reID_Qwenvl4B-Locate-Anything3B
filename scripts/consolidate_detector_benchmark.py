from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.benchmarking.detector_consolidation import (
    DetectorConsolidationError,
    consolidate_detector_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidate compatible detector reports.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/detector_sportsmot.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = consolidate_detector_benchmark(args.config, overwrite=args.overwrite)
    except (OSError, ValueError, DetectorConsolidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
