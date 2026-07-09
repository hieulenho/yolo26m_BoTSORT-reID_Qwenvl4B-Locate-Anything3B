"""Create a render-only prediction manifest covering every MOT track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_tracking.team_benchmark.label_completion import (
    build_track_label_completion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete team/referee labels for every track in a video.",
    )
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--annotation-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-track", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists and overwrite=false: {args.output}")
    if not args.source_video.is_file():
        raise SystemExit(f"Source video does not exist: {args.source_video}")
    if not args.tracks.is_file():
        raise SystemExit(f"Tracks file does not exist: {args.tracks}")
    if args.samples_per_track < 1:
        raise SystemExit("--samples-per-track must be >= 1")

    payload = build_track_label_completion(
        sequence_name=args.sequence_name,
        source_video=args.source_video,
        tracks_path=args.tracks,
        annotation_csv=args.annotation_csv,
        samples_per_track=args.samples_per_track,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "track_predictions": len(payload["track_predictions"]),
                "metadata": payload["metadata"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
