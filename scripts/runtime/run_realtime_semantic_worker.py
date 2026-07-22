"""Process one bounded batch from the realtime semantic queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.adaptive_tracking.semantic_queue import (
    SemanticQueueError,
    process_semantic_queue,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--vlm-config", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/ontology/vocabulary_registry.yaml"),
    )
    args = parser.parse_args()
    try:
        result = process_semantic_queue(
            queue_dir=args.queue_dir,
            vlm_config_path=args.vlm_config,
            semantic_output=args.semantic_output,
            memory_path=args.memory,
            registry_path=args.registry,
            max_events=args.max_events,
        )
    except (SemanticQueueError, RuntimeError, ValueError, OSError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
