"""Convert a sorted image sequence into a reproducible MP4 benchmark input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.tif")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        payload = image_sequence_to_video(
            input_dir=args.input_dir,
            pattern=args.pattern,
            output=args.output,
            fps=args.fps,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(payload, indent=2))
    return 0


def image_sequence_to_video(
    *,
    input_dir: Path,
    pattern: str,
    output: Path,
    fps: float,
    overwrite: bool,
) -> dict[str, object]:
    source = input_dir.resolve()
    destination = output.resolve()
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    if destination.exists() and not overwrite:
        raise ValueError(f"Output exists and overwrite=false: {destination}")
    images = sorted(path for path in source.glob(pattern) if path.is_file())
    if not images:
        raise ValueError(f"No images match {pattern!r} in {source}")

    first = _read_frame(images[0])
    height, width = first.shape[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError(f"Could not open video writer: {destination}")
    try:
        for image_path in images:
            frame = _read_frame(image_path)
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Image size changed at {image_path}: "
                    f"{frame.shape[1]}x{frame.shape[0]} != {width}x{height}"
                )
            writer.write(frame)
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(destination))
    try:
        encoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if encoded_frames != len(images):
        raise OSError(
            f"Encoded frame count mismatch: {encoded_frames} != {len(images)}"
        )
    metadata = {
        "input_dir": str(source),
        "pattern": pattern,
        "output": str(destination),
        "frame_count": len(images),
        "fps": fps,
        "width": width,
        "height": height,
        "first_image": str(images[0]),
        "last_image": str(images[-1]),
    }
    destination.with_suffix(".conversion.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _read_frame(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    if image.dtype != np.uint8:
        minimum = float(image.min())
        maximum = float(image.max())
        scale = 255.0 / max(maximum - minimum, 1.0)
        image = np.clip((image.astype(np.float32) - minimum) * scale, 0, 255).astype(
            np.uint8
        )
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape at {path}: {image.shape}")
    return image


if __name__ == "__main__":
    raise SystemExit(main())
