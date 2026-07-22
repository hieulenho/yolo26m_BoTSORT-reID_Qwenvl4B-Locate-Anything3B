"""Prepare and validate human review of diagnostic ID-switch categories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.evaluation.idsw_review import (
    IdswReviewError,
    audit_idsw_review,
    prepare_idsw_review,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--events", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("--review", type=Path, required=True)
    status.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_idsw_review(args.events, args.output, overwrite=args.overwrite)
        else:
            result = audit_idsw_review(args.review)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
                temporary.replace(args.output)
    except (IdswReviewError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
