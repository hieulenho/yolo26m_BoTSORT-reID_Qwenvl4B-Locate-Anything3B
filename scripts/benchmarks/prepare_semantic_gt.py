"""Prepare or finalize reviewed cross-domain semantic ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.benchmarking.semantic_annotation import (
    SemanticAnnotationError,
    finalize_annotation_package,
    merge_reviewed_manifests,
    prepare_annotation_package,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--sample-id", required=True)
    prepare.add_argument("--source-video", type=Path, required=True)
    prepare.add_argument("--tracks", type=Path, required=True)
    prepare.add_argument("--discovery", type=Path, required=True)
    prepare.add_argument("--route", type=Path, required=True)
    prepare.add_argument("--semantics", type=Path, required=True)
    prepare.add_argument("--run-report", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--crops-per-track", type=int, default=3)
    prepare.add_argument("--max-tracks", type=int, default=0)
    prepare.add_argument("--overwrite", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--package-dir", type=Path, required=True)
    finalize.add_argument("--output-manifest", type=Path, default=None)
    finalize.add_argument("--overwrite", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--manifest", type=Path, action="append", required=True)
    merge.add_argument("--output-manifest", type=Path, required=True)
    merge.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_annotation_package(
                sample_id=args.sample_id,
                source_video=args.source_video,
                tracks_path=args.tracks,
                discovery_path=args.discovery,
                route_path=args.route,
                semantics_path=args.semantics,
                run_report_path=args.run_report,
                output_dir=args.output_dir,
                crops_per_track=args.crops_per_track,
                max_tracks=args.max_tracks or None,
                overwrite=args.overwrite,
            )
        elif args.command == "finalize":
            result = finalize_annotation_package(
                package_dir=args.package_dir,
                output_manifest=args.output_manifest,
                overwrite=args.overwrite,
            )
        else:
            result = merge_reviewed_manifests(
                manifest_paths=args.manifest,
                output_manifest=args.output_manifest,
                overwrite=args.overwrite,
            )
    except (SemanticAnnotationError, OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
