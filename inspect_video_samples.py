#!/usr/bin/env python3
"""Create contact sheets for visual inspection of local videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    video = Path(args.video)
    output = Path(args.output)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = np.linspace(0, max(frame_count - 1, 0), args.samples, dtype=int)

    thumbs: list[np.ndarray] = []
    thumb_width = 480
    thumb_height = max(1, round(height * thumb_width / width))
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {index}")
        thumb = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (thumb_width, 42), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            f"frame {index}  t={index / fps:.2f}s",
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)
    capture.release()

    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) < rows * columns:
        thumbs.append(blank.copy())
    sheet = np.vstack(
        [np.hstack(thumbs[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Cannot write {output}")
    print(
        f"video={video} frames={frame_count} fps={fps:.6f} size={width}x{height} "
        f"duration={frame_count / fps:.3f}s output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
