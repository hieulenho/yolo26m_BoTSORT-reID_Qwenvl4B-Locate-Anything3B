"""Render tracking video with team and role labels from a prediction manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


TEAM_COLORS: dict[str, tuple[int, int, int]] = {
    "light_blue": (60, 210, 60),
    "dark_blue": (210, 90, 20),
    "goalkeeper_red": (30, 30, 230),
    "goalkeeper_green": (30, 170, 60),
    "yellow_kit": (30, 220, 240),
    "dark_kit": (190, 80, 30),
    "goalkeeper_orange": (20, 140, 240),
    "unknown": (140, 140, 140),
}

# Short scoreboard-style labels for the current demo videos. Keep these separate
# from the benchmark labels so the JSON remains domain-neutral.
TEAM_SHORT_NAMES: dict[str, str] = {
    "light_blue": "MCI",
    "dark_blue": "CHE",
    "goalkeeper_red": "CHE",
    "goalkeeper_green": "MCI",
    "yellow_kit": "YLW",
    "dark_kit": "DRK",
    "goalkeeper_orange": "GK",
    "unknown": "?",
}

ROLE_SHORT_NAMES: dict[str, str] = {
    "goalkeeper": "GK",
    "defender": "DEF",
    "midfielder": "MID",
    "forward": "FWD",
    "player": "P",
    "unknown": "?",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render bbox/ID/team/role labels over a tracked video.",
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="",
        help="Optional title rendered in the top-left corner of every frame.",
    )
    parser.add_argument(
        "--hide-unlabeled",
        action="store_true",
        help=(
            "Skip tracks that do not have a resolved prediction. This is the "
            "recommended mode for verified demo videos because it avoids "
            "labeling referees/false positives as a team."
        ),
    )
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
        hide_unlabeled=args.hide_unlabeled,
        title=args.title,
    )
    metadata_path = args.output_video.with_suffix(".metadata.json")
    metadata = {
        "source_video": str(args.source_video),
        "tracks": str(args.tracks),
        "predictions": str(args.predictions),
        "sequence_name": args.sequence_name,
        "output_video": str(args.output_video),
        "title": args.title,
        "hide_unlabeled": args.hide_unlabeled,
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
    hide_unlabeled: bool,
    title: str,
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
    total_track_boxes = 0
    drawn_boxes = 0
    skipped_unlabeled_boxes = 0
    unlabeled_track_ids: set[int] = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rendered_frames += 1
        if title:
            _draw_title(frame, title)
        for row in tracks_by_frame.get(rendered_frames, []):
            total_track_boxes += 1
            track_id = int(row["track_id"])
            label = labels.get(track_id)
            if label is None:
                unlabeled_track_ids.add(track_id)
                skipped_unlabeled_boxes += int(hide_unlabeled)
                if hide_unlabeled:
                    continue
                label = {"team_label": "unknown", "role_label": "unknown"}

            team = label["team_label"]
            role = label["role_label"]
            color = TEAM_COLORS.get(team, _hash_color(team))
            x1 = max(0, int(row["x"]))
            y1 = max(0, int(row["y"]))
            x2 = min(width - 1, int(row["x"] + row["w"]))
            y2 = min(height - 1, int(row["y"] + row["h"]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            _draw_label(frame, _format_label(track_id, team, role), x1, y1, color)
            drawn_boxes += 1
        writer.write(frame)

    cap.release()
    writer.release()
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "rendered_frames": rendered_frames,
        "total_track_boxes": total_track_boxes,
        "drawn_boxes": drawn_boxes,
        "skipped_unlabeled_boxes": skipped_unlabeled_boxes,
        "unlabeled_track_count": len(unlabeled_track_ids),
        "unlabeled_track_ids": sorted(unlabeled_track_ids),
    }


def _format_label(track_id: int, team: str, role: str) -> str:
    if team == "unknown" and role == "unknown":
        return f"ID{track_id}"
    team_text = TEAM_SHORT_NAMES.get(team, team.upper()[:3])
    role_text = ROLE_SHORT_NAMES.get(role, role.upper()[:3])
    if role_text in {"?", "P"}:
        return f"ID{track_id} | {team_text}"
    return f"ID{track_id} | {team_text} | {role_text}"


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


def _draw_title(frame: Any, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    margin = 10
    x1 = margin
    y1 = margin
    x2 = x1 + text_w + 14
    y2 = y1 + text_h + baseline + 12
    cv2.rectangle(frame, (x1, y1), (x2, y2), (24, 24, 24), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), 1)
    cv2.putText(
        frame,
        text,
        (x1 + 7, y2 - baseline - 5),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _hash_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    return (
        70 + digest[0] % 150,
        70 + digest[1] % 150,
        70 + digest[2] % 150,
    )


if __name__ == "__main__":
    main()
