"""Build the evidence-based multi-domain benchmark release gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.benchmarking.completion_gate import (
    CompletionGateError,
    build_completion_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-readiness", type=Path, required=True)
    parser.add_argument("--semantic-gt-status", type=Path, required=True)
    parser.add_argument("--idsw-review-status", type=Path, required=True)
    parser.add_argument("--idsw-agreement", type=Path, required=True)
    parser.add_argument("--realtime-report", type=Path, required=True)
    parser.add_argument("--semantic-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = build_completion_gate(
            dataset_readiness=args.dataset_readiness,
            semantic_gt_status=args.semantic_gt_status,
            idsw_review_status=args.idsw_review_status,
            idsw_agreement=args.idsw_agreement,
            realtime_report=args.realtime_report,
            semantic_comparison=args.semantic_comparison,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (CompletionGateError, OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
