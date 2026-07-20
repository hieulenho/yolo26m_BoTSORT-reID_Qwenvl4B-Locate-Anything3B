from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_tracking.benchmarking.semantic_ablation import (
    SemanticAblationError,
    build_semantic_ablation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the semantic ablation report.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = build_semantic_ablation_report(args.config, overwrite=args.overwrite)
    except (SemanticAblationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
