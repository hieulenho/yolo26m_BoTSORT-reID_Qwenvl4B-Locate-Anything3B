"""Render tracking video with team and role/position labels from a prediction manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


TEAM_COLORS = {
    "light_blue": (60, 210, 60),
    "dark_blue": (210, 90, 20),
    "goalkeeper_red": (30, 30, 230),
    "goalkeeper_green": (30, 170, 60),
    "yellow_kit": (30, 220, 240),
    "dark_kit": (190, 80, 30),
    "goalkeeper_orange": (20, 140, 240),
    "unknown": (140, 140, 140),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render bbox/ID/team/position labels over a tracked video.",
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_video.exists() and not args.overwrite:
        raise SystemExit(f"Output exists and overwrite=false: {args.output_video}")
    if not args.source_video.is_file():
        raise SystemExit(f"Source video does not exist: {args.source_video}")
    if not args.tracks.is_file():
        raise SystemExit(f"Tracks file does not exist: {args.tracks}")
    if not args.predictions.is_file():
        raise SystemExit(f"Prediction manifest does not exist: {args.predictions}")

    tracks_by_frame = _load_tracks_by_frame(args.tracks)
    labels = _load_prediction_labels(args.predictions, args.sequence_name)
    rendered = _render(
        source_video=args.source_video,
        tracks_by_frame=tracks_by_frame,
        labels=labels,
        output_video=args.output_video,
    )
    metadata_path = args.output_video.with_suffix(".metadata.json")
    metadata = {
        "source_video": str(args.source_video),
        "tracks": str(args.tracks),
        "predictions": str(args.predictions),
        "sequence_name": args.sequence_name,
        "output_video": str(args.output_video),
        "track_count_with_predictions": len(labels),
        **rendered,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({**metadata, "metadata": str(metadata_path)}, indent=2))


def _load_tracks_by_frame(path: Path) -> dict[int, list[dict[str, Any]]]:
    tracks_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            tracks_by_frame[frame].append(
                {
                    "track_id": track_id,
                    "x": float(row[2]),
                    "y": float(row[3]),
                    "w": float(row[4]),
                    "h": float(row[5]),
                }
            )
    return dict(tracks_by_frame)


def _load_prediction_labels(path: Path, sequence_name: str) -> dict[int, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[int, dict[str, str]] = {}
    for item in data.get("track_predictions", []):
        if item.get("sequence_name") != sequence_name:
            continue
        if item.get("status", "resolved") != "resolved":
            continue
        track_id = int(item["track_id"])
        labels[track_id] = {
            "team_label": str(item.get("team_label") or "unknown"),
            "role_label": str(item.get("role_label") or "unknown"),
        }
    return labels


def _render(
    *,
    source_video: Path,
    tracks_by_frame: dict[int, list[dict[str, Any]]],
    labels: dict[int, dict[str, str]],
    output_video: Path,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer: {output_video}")

    rendered_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rendered_frames += 1
        for row in tracks_by_frame.get(rendered_frames, []):
            track_id = int(row["track_id"])
            label = labels.get(track_id, {"team_label": "unknown", "role_label": "unknown"})
            team = label["team_label"]
            role = label["role_label"]
            color = TEAM_COLORS.get(team, _hash_color(team))
            x1 = max(0, int(row["x"]))
            y1 = max(0, int(row["y"]))
            x2 = min(width - 1, int(row["x"] + row["w"]))
            y2 = min(height - 1, int(row["y"] + row["h"]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"ID{track_id} {team} {role}"
            _draw_label(frame, text, x1, y1, color)
        writer.write(frame)

    cap.release()
    writer.release()
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "rendered_frames": rendered_frames,
    }


def _draw_label(
    frame: Any,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_h - baseline - 6)
    bottom = max(text_h + baseline + 6, y)
    cv2.rectangle(frame, (x, top), (x + text_w + 6, bottom), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 3, bottom - baseline - 3),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _hash_color(label: str) -> tuple[int, int, int]:
    value = abs(hash(label))
    return (
        70 + value % 150,
        70 + (value // 11) % 150,
        70 + (value // 37) % 150,
    )


if __name__ == "__main__":
    main()
