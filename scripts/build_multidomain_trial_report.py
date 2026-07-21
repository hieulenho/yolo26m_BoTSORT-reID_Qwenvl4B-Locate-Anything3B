"""Build the public multi-domain trial report from saved run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.benchmarking.multidomain_report import (
    MultidomainReportError,
    build_multidomain_trial_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/samples/multidomain/samples_manifest.json"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/adaptive_runs/multidomain_long"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/adaptive_runs/multidomain_long/summary"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = build_multidomain_trial_report(
            args.manifest,
            args.run_root,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (MultidomainReportError, OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
