"""Rebuild semantic fusion from saved per-track Qwen results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.adaptive_tracking.semantic_queue import (
    SemanticQueueError,
    rebuild_semantic_cache_from_processed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/ontology/vocabulary_registry.yaml"),
    )
    args = parser.parse_args()
    try:
        result = rebuild_semantic_cache_from_processed(
            processed_dir=args.processed_dir,
            semantic_output=args.output,
            registry_path=args.registry,
        )
    except (OSError, ValueError, SemanticQueueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
