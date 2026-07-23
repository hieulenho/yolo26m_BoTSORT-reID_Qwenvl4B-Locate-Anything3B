"""Build a machine-readable audit of official semantic GT manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.benchmarking.official_semantic_gt import (
    OfficialSemanticGtError,
    audit_official_semantic_gt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--minimum-domains", type=int, default=2)
    parser.add_argument("--minimum-tracks", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_official_semantic_gt(
            args.manifest,
            minimum_domains=args.minimum_domains,
            minimum_tracks=args.minimum_tracks,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except (OfficialSemanticGtError, OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
