"""Build a reproducible 30-60 second multi-domain video suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

from football_tracking.paths import get_project_root, resolve_project_path


class VideoSuiteError(RuntimeError):
    """Raised when a source or generated benchmark clip is invalid."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/multidomain_video_suite.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_input(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    if not resolved.exists():
        raise VideoSuiteError(f"Required input does not exist: {resolved}")
    return resolved


def _probe_video(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoSuiteError(f"Video cannot be opened: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if fps <= 0.0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise VideoSuiteError(f"Video has invalid stream metadata: {path}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "width": width,
        "height": height,
    }


def _scaled_size(width: int, height: int, maximum_dimension: int) -> tuple[int, int]:
    if maximum_dimension <= 0:
        raise VideoSuiteError("maximum_dimension must be positive.")
    scale = min(1.0, maximum_dimension / max(width, height))
    output_width = max(2, int(round(width * scale)))
    output_height = max(2, int(round(height * scale)))
    output_width -= output_width % 2
    output_height -= output_height % 2
    return output_width, output_height


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _transform_metadata(
    source_video: dict[str, float | int],
    output_video: dict[str, float | int],
    *,
    requested_frame_count: int | None,
) -> dict[str, Any]:
    return {
        "requested_frame_count": requested_frame_count,
        "written_frame_count": int(output_video["frame_count"]),
        "strict_frame_count": requested_frame_count is not None,
        "output_fps": float(output_video["fps"]),
        "output_size": [int(output_video["width"]), int(output_video["height"])],
        "scale_x": float(output_video["width"]) / float(source_video["width"]),
        "scale_y": float(output_video["height"]) / float(source_video["height"]),
        "fps_ratio": float(output_video["fps"]) / float(source_video["fps"]),
    }


def _transcode_clip(
    *,
    source: Path,
    destination: Path,
    maximum_dimension: int,
    maximum_duration_seconds: float,
    output_fps: float | None,
    requested_frame_count: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    if destination.exists() and not overwrite:
        source_video = _probe_video(source)
        output_video = _probe_video(destination)
        return {
            "status": "reused",
            "source_video": source_video,
            "output_video": output_video,
            "transform": _transform_metadata(
                source_video,
                output_video,
                requested_frame_count=requested_frame_count,
            ),
        }
    source_video = _probe_video(source)
    effective_fps = float(output_fps or source_video["fps"])
    if effective_fps <= 0.0:
        raise VideoSuiteError(f"Output FPS must be positive for {destination.name}.")
    available_frames = int(source_video["frame_count"])
    duration_bound = max(1, int(math.floor(maximum_duration_seconds * effective_fps)))
    frame_limit = requested_frame_count or duration_bound
    frame_limit = min(frame_limit, available_frames, duration_bound)
    strict_frame_count = requested_frame_count is not None
    if frame_limit <= 0:
        raise VideoSuiteError(f"No frames selected for {destination.name}.")

    width, height = _scaled_size(
        int(source_video["width"]),
        int(source_video["height"]),
        maximum_dimension,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.part{destination.suffix}")
    temporary.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        effective_fps,
        (width, height),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        temporary.unlink(missing_ok=True)
        raise VideoSuiteError(f"Could not initialize transcoding for {destination}.")
    written = 0
    try:
        while written < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written == 0 or (strict_frame_count and written != frame_limit):
        temporary.unlink(missing_ok=True)
        raise VideoSuiteError(
            f"Decoded {written} of {frame_limit} requested frames from {source}."
        )
    os.replace(temporary, destination)
    output_video = _probe_video(destination)
    return {
        "status": "created",
        "source_video": source_video,
        "output_video": output_video,
        "transform": _transform_metadata(
            source_video,
            output_video,
            requested_frame_count=(frame_limit if strict_frame_count else None),
        ),
    }


def prepare_video_suite(
    *,
    config_path: Path,
    output_dir: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    project_root = get_project_root()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("videos"), list):
        raise VideoSuiteError("Suite config must contain a videos list.")
    destination_root = Path(output_dir or config["output_dir"]).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    minimum_duration = float(config.get("minimum_duration_seconds", 30.0))
    maximum_duration = float(config.get("maximum_duration_seconds", 60.0))
    maximum_dimension = int(config.get("maximum_dimension", 960))
    if minimum_duration <= 0.0 or maximum_duration < minimum_duration:
        raise VideoSuiteError("Invalid suite duration bounds.")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_outputs: set[str] = set()
    for row in config["videos"]:
        if not isinstance(row, dict):
            raise VideoSuiteError("Every videos entry must be a mapping.")
        video_id = str(row.get("video_id", "")).strip()
        output_name = str(row.get("output_name", "")).strip()
        if not video_id or video_id in seen_ids:
            raise VideoSuiteError(f"Missing or duplicate video_id: {video_id!r}")
        if not output_name.lower().endswith(".mp4") or output_name in seen_outputs:
            raise VideoSuiteError(f"Output must be a unique MP4 name: {output_name!r}")
        seen_ids.add(video_id)
        seen_outputs.add(output_name)
        source = _resolve_input(str(row.get("source", "")), project_root)
        destination = destination_root / output_name
        result = _transcode_clip(
            source=source,
            destination=destination,
            maximum_dimension=maximum_dimension,
            maximum_duration_seconds=maximum_duration,
            output_fps=(float(row["output_fps"]) if row.get("output_fps") else None),
            requested_frame_count=(
                int(row["frame_count"]) if row.get("frame_count") else None
            ),
            overwrite=overwrite,
        )
        output_video = dict(result["output_video"])
        duration = float(output_video["duration_seconds"])
        if duration < minimum_duration - 0.05 or duration > maximum_duration + 0.05:
            destination.unlink(missing_ok=True)
            raise VideoSuiteError(
                f"Generated duration for {video_id} is {duration:.3f}s; expected "
                f"{minimum_duration:.1f}-{maximum_duration:.1f}s."
            )
        gt_value = row.get("ground_truth_path")
        gt_path = _resolve_input(gt_value, project_root) if gt_value else None
        records.append(
            {
                "sample_id": destination.stem,
                "video_id": video_id,
                "domain": str(row.get("domain", "unknown")),
                "path": str(destination),
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
                "status": result["status"],
                "source": str(source),
                "source_page": row.get("source_page"),
                "license": row.get("license"),
                "selection_reason": row.get("selection_reason"),
                "ground_truth": dict(row.get("expected", {})),
                "ground_truth_path": str(gt_path) if gt_path else None,
                "source_video": result["source_video"],
                "video": output_video,
                "transform": result.get("transform"),
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_path.resolve()),
        "duration_contract_seconds": [minimum_duration, maximum_duration],
        "maximum_dimension": maximum_dimension,
        "video_count": len(records),
        "sample_count": len(records),
        "samples": records,
        "videos": records,
    }
    manifest_path = destination_root / "multidomain_video_suite_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"status": "ok", "manifest": str(manifest_path), "videos": records}


def main() -> int:
    args = _arguments()
    try:
        result = prepare_video_suite(
            config_path=args.config.resolve(),
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, VideoSuiteError, yaml.YAMLError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
