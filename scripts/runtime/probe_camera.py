"""Find a readable local webcam index without keeping the device open."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

import cv2

from football_tracking.video_capture import open_video_capture


@dataclass(frozen=True)
class CameraProbe:
    index: int
    backend: str
    width: int
    height: int
    fps: float


def probe_camera(index: int, *, read_attempts: int = 5) -> CameraProbe | None:
    capture, backend_name = open_video_capture(index)
    if capture is None or backend_name is None:
        return None
    try:
        frame = None
        for _ in range(read_attempts):
            ok, candidate = capture.read()
            if ok and candidate is not None and candidate.size:
                frame = candidate
                break
        if frame is None:
            return None
        height, width = frame.shape[:2]
        return CameraProbe(
            index=index,
            backend=backend_name,
            width=int(width),
            height=int(height),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        )
    finally:
        capture.release()


def select_camera(index: int | None, *, max_index: int) -> CameraProbe:
    indices = (index,) if index is not None else tuple(range(max_index + 1))
    for candidate in indices:
        result = probe_camera(candidate)
        if result is not None:
            return result
    requested = str(index) if index is not None else f"0..{max_index}"
    raise RuntimeError(
        f"No readable webcam was found at index {requested}. "
        "Close Camera, Zoom, Teams, or another application using the device."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--max-index", type=int, default=5)
    args = parser.parse_args()
    if args.index is not None and args.index < 0:
        parser.error("--index must be non-negative")
    if args.max_index < 0:
        parser.error("--max-index must be non-negative")
    try:
        selected = select_camera(args.index, max_index=args.max_index)
    except RuntimeError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps({"status": "ok", "selected": asdict(selected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
