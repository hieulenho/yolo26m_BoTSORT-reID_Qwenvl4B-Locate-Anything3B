"""Open an RTSP or other network stream and verify that readable frames arrive."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402

from football_tracking.video_capture import (  # noqa: E402
    open_video_capture,
    redact_capture_source,
    resolve_capture_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source")
    source_group.add_argument("--source-env")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--read-attempts", type=int, default=30)
    parser.add_argument("--read-timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.read_attempts < args.frames:
        parser.error("--read-attempts must be at least --frames")
    if args.read_timeout_seconds <= 0:
        parser.error("--read-timeout-seconds must be positive")

    try:
        source = resolve_capture_source(
            args.source,
            args.source_env,
            environment=os.environ,
        )
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    display_source = redact_capture_source(source)
    opened_at = time.perf_counter()
    capture, backend = open_video_capture(source)
    open_seconds = time.perf_counter() - opened_at
    if capture is None or backend is None:
        sys.stderr.write(f"Error: Could not open stream: {display_source}\n")
        return 2

    frames = []
    read_started = time.perf_counter()
    try:
        for _ in range(args.read_attempts):
            if time.perf_counter() - read_started > args.read_timeout_seconds:
                break
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                frames.append(frame)
                if len(frames) >= args.frames:
                    break
    finally:
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        capture.release()
    read_seconds = time.perf_counter() - read_started
    if len(frames) < args.frames:
        sys.stderr.write(
            f"Error: Stream opened but delivered only {len(frames)}/{args.frames} "
            f"readable frames: {display_source}\n"
        )
        return 2
    height, width = frames[-1].shape[:2]
    print(
        json.dumps(
            {
                "status": "ok",
                "source": display_source,
                "backend": backend,
                "width": int(width),
                "height": int(height),
                "reported_fps": reported_fps,
                "frames_read": len(frames),
                "open_seconds": open_seconds,
                "read_seconds": read_seconds,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
