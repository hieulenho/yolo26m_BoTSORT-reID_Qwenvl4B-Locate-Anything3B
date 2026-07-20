"""Capture a short webcam, RTSP, or file clip for adaptive scene discovery."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


def _source_value(value: str) -> str | int:
    text = value.strip()
    return int(text) if text.isdigit() else text


def capture_clip(
    source: str,
    output: Path,
    *,
    seconds: float,
    fallback_fps: float,
    overwrite: bool,
) -> dict:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Calibration clip exists: {output}")
    capture = cv2.VideoCapture(_source_value(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open calibration source: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    output_fps = source_fps if source_fps > 1.0 else fallback_fps
    frame_limit = max(int(round(seconds * output_fps)), 1)
    writer = None
    frame_count = 0
    started = time.perf_counter()
    try:
        while frame_count < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open calibration writer: {output}")
            writer.write(frame)
            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frame_count == 0:
        raise RuntimeError("Calibration source returned no frames.")
    return {
        "status": "ok",
        "source": source,
        "output": str(output.resolve()),
        "fps": output_fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / output_fps,
        "capture_wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = capture_clip(
        args.source,
        args.output,
        seconds=args.seconds,
        fallback_fps=args.fallback_fps,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
