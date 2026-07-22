"""Build the audited full-tracker benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_tracking.benchmarking.tracking_consolidation import (
    TrackingConsolidationError,
    consolidate_tracking_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/tracking_full_report.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = consolidate_tracking_benchmark(
            args.config,
            overwrite=True if args.overwrite else None,
        )
    except TrackingConsolidationError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
