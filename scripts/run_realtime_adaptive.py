"""Run a generated adaptive plan on a webcam, RTSP stream, or video source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.adaptive_tracking.realtime import (
    RealtimeTrackingError,
    run_realtime_tracking,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", default="0", help="Camera index, RTSP URL, or video path.")
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--output-mot", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = run_realtime_tracking(
            config_path=args.config,
            stream_source=args.source,
            output_video=args.output_video,
            output_mot=args.output_mot,
            metadata_path=args.metadata,
            show_window=not args.no_window,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
    except (RealtimeTrackingError, RuntimeError, ValueError, OSError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
