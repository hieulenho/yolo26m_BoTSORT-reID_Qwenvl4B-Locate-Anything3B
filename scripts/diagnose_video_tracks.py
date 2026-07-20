"""Write lightweight diagnostics for a rendered MOT track file."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _video_info(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return {}
    cap = cv2.VideoCapture(str(path))
    try:
        return {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        cap.release()


def _percent(value: float) -> float:
    return round(value * 100.0, 3)


def _safe_quantiles(values: list[int], n: int = 4) -> list[float]:
    if len(values) < n:
        return [float(statistics.median(values))] * (n - 1) if values else [0.0] * (n - 1)
    return [float(item) for item in statistics.quantiles(values, n=n)]


def diagnose_tracks(
    tracks_path: Path,
    metadata_path: Path | None,
    source_video: Path | None,
) -> dict[str, Any]:
    tracks: dict[int, list[int]] = defaultdict(list)
    scores: dict[int, list[float]] = defaultdict(list)
    areas: dict[int, list[float]] = defaultdict(list)
    boxes_per_frame: dict[int, int] = defaultdict(int)

    lines = tracks_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            raise ValueError(f"Invalid MOT row at line {line_number}: expected at least 7 columns")
        frame_id = int(float(parts[0]))
        track_id = int(float(parts[1]))
        width = float(parts[4])
        height = float(parts[5])
        score = float(parts[6])
        tracks[track_id].append(frame_id)
        scores[track_id].append(score)
        areas[track_id].append(width * height)
        boxes_per_frame[frame_id] += 1

    video = _video_info(source_video)
    metadata = _read_json(metadata_path)
    # Prefer the actual processed frame count from tracking metadata. This matters
    # for smoke runs with --max-frames, where the source video can be much longer
    # than the tracked segment.
    frame_count = int(metadata.get("frame_count") or video.get("frame_count") or 0)
    fps = float(video.get("fps") or 0.0)
    total_boxes = sum(len(frames) for frames in tracks.values())
    lengths = [len(frames) for frames in tracks.values()]
    spans = [max(frames) - min(frames) + 1 for frames in tracks.values()] if tracks else []
    gaps = [span - length for span, length in zip(spans, lengths, strict=True)] if tracks else []
    q_lengths = _safe_quantiles(lengths)
    q_boxes = _safe_quantiles(list(boxes_per_frame.values()))
    short_30 = sum(length < 30 for length in lengths)
    short_120 = sum(length < 120 for length in lengths)
    warnings: list[str] = []
    if lengths and statistics.median(lengths) < 30:
        warnings.append(
            "Median track length is below 30 frames; likely heavy fragmentation/ID resets."
        )
    if lengths and short_30 / max(len(lengths), 1) > 0.5:
        warnings.append("More than half of tracks are shorter than 30 frames.")
    if frame_count and len(boxes_per_frame) / frame_count < 0.8:
        warnings.append(
            "Many frames have no emitted tracks; detector or output confirmation may "
            "be too strict."
        )
    if (
        metadata.get("detector_checkpoint")
        and "football" in str(metadata["detector_checkpoint"]).replace("\\", "/")
        and warnings
    ):
        warnings.append(
            "Run used the football fine-tuned detector; if fragmentation persists, try "
            "football_high_recall or general_person."
        )

    duration_seconds = round(frame_count / fps, 3) if fps > 0 and frame_count else None
    return {
        "tracks_path": str(tracks_path),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "source_video": str(source_video) if source_video else None,
        "video": video,
        "detector_checkpoint": metadata.get("detector_checkpoint"),
        "checkpoint_type": metadata.get("checkpoint_type"),
        "detector_config": metadata.get("detector_config"),
        "tracker_config": metadata.get("tracker_config"),
        "duration_seconds": duration_seconds,
        "frame_count": frame_count,
        "frames_with_tracks": len(boxes_per_frame),
        "frame_track_coverage_percent": (
            _percent(len(boxes_per_frame) / frame_count) if frame_count else None
        ),
        "unique_track_count": len(tracks),
        "total_track_boxes": total_boxes,
        "track_length": {
            "min": min(lengths) if lengths else 0,
            "p25": q_lengths[0],
            "median": float(statistics.median(lengths)) if lengths else 0.0,
            "p75": q_lengths[2],
            "max": max(lengths) if lengths else 0,
            "shorter_than_10": sum(length < 10 for length in lengths),
            "shorter_than_30": short_30,
            "shorter_than_60": sum(length < 60 for length in lengths),
            "shorter_than_120": short_120,
            "shorter_than_30_percent": _percent(short_30 / len(lengths)) if lengths else 0.0,
            "shorter_than_120_percent": _percent(short_120 / len(lengths)) if lengths else 0.0,
        },
        "track_gaps": {
            "tracks_with_gaps": sum(gap > 0 for gap in gaps),
            "total_gap_frames": sum(gaps),
            "median_gap_frames": float(statistics.median(gaps)) if gaps else 0.0,
        },
        "boxes_per_frame": {
            "min": min(boxes_per_frame.values()) if boxes_per_frame else 0,
            "p25": q_boxes[0],
            "median": (
                float(statistics.median(boxes_per_frame.values()))
                if boxes_per_frame
                else 0.0
            ),
            "p75": q_boxes[2],
            "max": max(boxes_per_frame.values()) if boxes_per_frame else 0,
        },
        "longest_tracks": [
            {
                "track_id": track_id,
                "length": len(frames),
                "first_frame": min(frames),
                "last_frame": max(frames),
                "mean_score": round(sum(scores[track_id]) / len(scores[track_id]), 4),
                "median_area": round(float(statistics.median(areas[track_id])), 3),
            }
            for track_id, frames in sorted(
                tracks.items(),
                key=lambda item: (len(item[1]), -item[0]),
                reverse=True,
            )[:20]
        ],
        "warnings": warnings,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    track_length = payload["track_length"]
    boxes = payload["boxes_per_frame"]
    lines = [
        "# Tracking Diagnostics",
        "",
        f"- Tracks: `{payload['unique_track_count']}`",
        f"- Boxes: `{payload['total_track_boxes']}`",
        f"- Frames with tracks: `{payload['frames_with_tracks']}` / `{payload['frame_count']}`",
        f"- Track coverage: `{payload['frame_track_coverage_percent']}`%",
        f"- Median track length: `{track_length['median']}` frames",
        "- Tracks shorter than 30 frames: "
        f"`{track_length['shorter_than_30']}` "
        f"(`{track_length['shorter_than_30_percent']}`%)",
        "- Tracks shorter than 120 frames: "
        f"`{track_length['shorter_than_120']}` "
        f"(`{track_length['shorter_than_120_percent']}`%)",
        f"- Median boxes/frame: `{boxes['median']}`",
        "",
        "## Detector",
        "",
        f"- Checkpoint: `{payload.get('detector_checkpoint')}`",
        f"- Type: `{payload.get('checkpoint_type')}`",
        "",
        "## Warnings",
        "",
    ]
    if payload["warnings"]:
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    else:
        lines.append("- No obvious fragmentation warnings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--source-video", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists and overwrite=false: {args.output_json}")
    if args.output_md.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists and overwrite=false: {args.output_md}")
    payload = diagnose_tracks(args.tracks, args.metadata, args.source_video)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        json.dumps(
            {"status": "ok", "json": str(args.output_json), "md": str(args.output_md)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
