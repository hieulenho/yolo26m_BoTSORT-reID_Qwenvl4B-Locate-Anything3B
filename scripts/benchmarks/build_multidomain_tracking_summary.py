"""Build the consolidated GT-based multi-domain tracking report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.benchmarking.multidomain_tracking_summary import (
    MultidomainTrackingSummaryError,
    build_multidomain_tracking_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = build_multidomain_tracking_summary(args.config, overwrite=args.overwrite)
    except (OSError, ValueError, MultidomainTrackingSummaryError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
