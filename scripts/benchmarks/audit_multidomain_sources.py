"""Audit which official multi-domain benchmark sources are locally runnable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.benchmarking.dataset_registry import (
    DatasetRegistryError,
    audit_dataset_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/benchmarks/multidomain_sources.yaml"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = audit_dataset_registry(args.registry)
    except (DatasetRegistryError, OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
