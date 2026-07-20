from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_tracking.benchmarking.semantic_pipeline_comparison import (
    SemanticPipelineComparisonError,
    build_semantic_pipeline_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare semantic Pipelines A/B/C.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = build_semantic_pipeline_comparison(
            args.config,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, SemanticPipelineComparisonError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
