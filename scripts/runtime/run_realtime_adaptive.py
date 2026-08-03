"""Run a generated adaptive plan on a webcam, RTSP stream, or video source."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from football_tracking.adaptive_tracking.realtime import (
    RealtimeTrackingError,
    run_realtime_tracking,
)
from football_tracking.video_capture import resolve_capture_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source",
        default=None,
        help="Camera index, RTSP URL, or video path.",
    )
    source_group.add_argument(
        "--source-env",
        default=None,
        help="Environment variable containing the camera index, stream URL, or video path.",
    )
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--output-mot", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--semantic-queue-dir", type=Path, default=None)
    parser.add_argument("--semantic-cache", type=Path, default=None)
    parser.add_argument("--task-config", type=Path, default=None)
    parser.add_argument("--semantic-event-interval-frames", type=int, default=90)
    parser.add_argument("--semantic-cache-reload-frames", type=int, default=15)
    parser.add_argument("--semantic-events-per-frame", type=int, default=2)
    parser.add_argument("--semantic-max-pending-events", type=int, default=256)
    parser.add_argument("--semantic-minimum-track-age-frames", type=int, default=None)
    parser.add_argument("--disable-scene-cut-reset", action="store_true")
    parser.add_argument("--scene-cut-threshold", type=float, default=0.65)
    parser.add_argument("--scene-cut-min-gap-frames", type=int, default=15)
    parser.add_argument("--scene-cut-check-interval-frames", type=int, default=5)
    parser.add_argument("--disable-detector-prewarm", action="store_true")
    parser.add_argument("--disable-frame-dropping", action="store_true")
    parser.add_argument("--max-catchup-frames", type=int, default=5)
    parser.add_argument("--synchronous-video-write", action="store_true")
    parser.add_argument("--video-write-queue-size", type=int, default=128)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        stream_source = resolve_capture_source(
            args.source,
            args.source_env,
            environment=os.environ,
        )
        result = run_realtime_tracking(
            config_path=args.config,
            stream_source=stream_source,
            output_video=args.output_video,
            output_mot=args.output_mot,
            metadata_path=args.metadata,
            show_window=not args.no_window,
            max_frames=args.max_frames,
            semantic_queue_dir=args.semantic_queue_dir,
            semantic_cache_path=args.semantic_cache,
            task_config_path=args.task_config,
            semantic_event_interval_frames=args.semantic_event_interval_frames,
            semantic_cache_reload_frames=args.semantic_cache_reload_frames,
            semantic_events_per_frame=args.semantic_events_per_frame,
            semantic_max_pending_events=args.semantic_max_pending_events,
            semantic_minimum_track_age_frames=args.semantic_minimum_track_age_frames,
            reset_on_scene_cut=not args.disable_scene_cut_reset,
            scene_cut_threshold=args.scene_cut_threshold,
            scene_cut_min_gap_frames=args.scene_cut_min_gap_frames,
            scene_cut_check_interval_frames=args.scene_cut_check_interval_frames,
            prewarm_detector=not args.disable_detector_prewarm,
            drop_late_frames=not args.disable_frame_dropping,
            max_catchup_frames=args.max_catchup_frames,
            asynchronous_video_write=not args.synchronous_video_write,
            video_write_queue_size=args.video_write_queue_size,
            overwrite=args.overwrite,
        )
    except (RealtimeTrackingError, RuntimeError, ValueError, OSError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    if not args.quiet:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
