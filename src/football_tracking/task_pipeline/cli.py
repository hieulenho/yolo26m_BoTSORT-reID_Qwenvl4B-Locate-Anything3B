"""Command-line tools for explicit task-configured realtime runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.task_pipeline.builder import write_task_runtime
from football_tracking.task_pipeline.config import (
    TaskPipelineConfigError,
    load_task_pipeline_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Validate a task and build runtime files.")
    build.add_argument("--task", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--output-video", type=Path, required=True)
    build.add_argument("--device", default="cuda")
    build.add_argument(
        "--preprocessing-mode",
        choices=("none", "auto_low_light", "clahe"),
        default=None,
    )
    build.add_argument("--tracker-config", type=Path, default=None)
    build.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        config = load_task_pipeline_config(args.task)
        paths = write_task_runtime(
            config=config,
            output_dir=args.output_dir,
            output_video=args.output_video,
            device=args.device,
            overwrite=args.overwrite,
            preprocessing_mode=args.preprocessing_mode,
            tracker_config_path=args.tracker_config,
        )
    except (TaskPipelineConfigError, FileExistsError, OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "task_id": config.task_id,
                "semantic_enabled": config.semantic.enabled,
                "paths": {key: str(value.resolve()) for key, value in paths.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
