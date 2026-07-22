from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.benchmarking.semantic_evaluation import (
    SemanticEvaluationError,
    evaluate_semantic_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive discovery and track semantics against human GT."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, default=None)
    parser.add_argument("--qwen-answer", type=Path, default=None)
    parser.add_argument("--locate-result", type=Path, default=None)
    parser.add_argument("--tracking-metadata", type=Path, default=None)
    parser.add_argument(
        "--without-qwen",
        action="store_true",
        help="Exclude any Qwen artifact inherited from the GT manifest.",
    )
    parser.add_argument(
        "--without-locate",
        action="store_true",
        help="Exclude any LocateAnything artifact inherited from the GT manifest.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.without_qwen and args.qwen_answer is not None:
        parser.error("--without-qwen cannot be combined with --qwen-answer.")
    if args.without_locate and args.locate_result is not None:
        parser.error("--without-locate cannot be combined with --locate-result.")
    artifact_overrides = {
        key: value
        for key, value in {
            "semantics": args.semantics,
            "qwen_answer": args.qwen_answer,
            "locate_result": args.locate_result,
            "tracking_metadata": args.tracking_metadata,
        }.items()
        if value is not None
    }
    if args.without_qwen:
        artifact_overrides["qwen_answer"] = None
    if args.without_locate:
        artifact_overrides["locate_result"] = None
    try:
        result = evaluate_semantic_manifest(
            args.manifest,
            args.output_dir,
            artifact_overrides=artifact_overrides,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, SemanticEvaluationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
