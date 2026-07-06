from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Detection:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create contact sheets for MOT track annotation.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-tracks", type=int, default=20)
    parser.add_argument("--samples-per-track", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--padding", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_tracks(path: Path) -> dict[int, list[Detection]]:
    by_track: dict[int, list[Detection]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 7:
                continue
            det = Detection(
                frame=int(float(row[0])),
                track_id=int(float(row[1])),
                x=float(row[2]),
                y=float(row[3]),
                w=float(row[4]),
                h=float(row[5]),
                score=float(row[6]),
            )
            by_track[det.track_id].append(det)
    for detections in by_track.values():
        detections.sort(key=lambda item: item.frame)
    return dict(by_track)


def pick_samples(detections: list[Detection], count: int) -> list[Detection]:
    if len(detections) <= count:
        return detections
    if count <= 1:
        return [detections[len(detections) // 2]]
    indexes = [round(i * (len(detections) - 1) / (count - 1)) for i in range(count)]
    return [detections[index] for index in indexes]


def read_frame(cap: cv2.VideoCapture, frame_number: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number - 1, 0))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number}.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def crop_detection(frame, det: Detection, padding: float) -> Image.Image:
    height, width = frame.shape[:2]
    pad_x = det.w * padding
    pad_y = det.h * padding
    x1 = max(int(det.x - pad_x), 0)
    y1 = max(int(det.y - pad_y), 0)
    x2 = min(int(det.x + det.w + pad_x), width)
    y2 = min(int(det.y + det.h + pad_y), height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        y1_fallback = max(int(det.y), 0)
        y2_fallback = min(int(det.y + det.h), height)
        x1_fallback = max(int(det.x), 0)
        x2_fallback = min(int(det.x + det.w), width)
        crop = frame[y1_fallback:y2_fallback, x1_fallback:x2_fallback]
    return Image.fromarray(crop)


def make_sheet(
    track_id: int,
    detections: list[Detection],
    cap: cv2.VideoCapture,
    args: argparse.Namespace,
) -> Image.Image:
    samples = pick_samples(detections, args.samples_per_track)
    thumb_w = args.crop_size
    label_h = 36
    thumb_h = args.crop_size + label_h
    cols = min(args.sheet_cols, max(len(samples), 1))
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, det in enumerate(samples):
        frame = read_frame(cap, det.frame)
        crop = crop_detection(frame, det, args.padding)
        crop.thumbnail((thumb_w, args.crop_size), Image.Resampling.LANCZOS)
        x = (index % cols) * thumb_w
        y = (index // cols) * thumb_h
        paste_x = x + (thumb_w - crop.width) // 2
        sheet.paste(crop, (paste_x, y))
        draw.rectangle((x, y + args.crop_size, x + thumb_w, y + thumb_h), fill=(245, 245, 245))
        draw.text(
            (x + 4, y + args.crop_size + 4),
            f"id={track_id} frame={det.frame}",
            fill=(0, 0, 0),
            font=font,
        )
        draw.text(
            (x + 4, y + args.crop_size + 18),
            f"score={det.score:.2f}",
            fill=(0, 0, 0),
            font=font,
        )
    return sheet


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video does not exist: {args.video}")
    if not args.tracks.exists():
        raise FileNotFoundError(f"Tracks do not exist: {args.tracks}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_track = load_tracks(args.tracks)
    ranked = sorted(
        by_track.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )[: args.max_tracks]
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    try:
        index_rows: list[dict[str, object]] = []
        for track_id, detections in ranked:
            sheet = make_sheet(track_id, detections, cap, args)
            output_path = args.output_dir / f"track_{track_id:04d}.jpg"
            sheet.save(output_path, quality=92)
            index_rows.append(
                {
                    "track_id": track_id,
                    "start_frame": detections[0].frame,
                    "end_frame": detections[-1].frame,
                    "observation_count": len(detections),
                    "sheet": output_path.as_posix(),
                }
            )
    finally:
        cap.release()

    with (args.output_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["track_id", "start_frame", "end_frame", "observation_count", "sheet"],
        )
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote {len(index_rows)} track sheets to {args.output_dir}")


if __name__ == "__main__":
    main()
