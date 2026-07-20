"""
render_vlm_guided_video.py
--------------------------
Bước 3 của Pipeline D: vẽ bounding box + COCO class label lên video gốc.

Không cần prediction manifest (như Pipeline A/B/C).
Chỉ cần:
  - File tracks .txt (MOT format: frame,id,x,y,w,h,conf,class_id,...)
  - File scene_discovery.json (biết COCO class_id -> class_name)
  - Video gốc

Mỗi track được tô màu theo class COCO (car=xanh lá, person=cam, motorcycle=tím, ...)
Label hiển thị: "ID 5 | car"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

# Màu sắc phân biệt theo COCO class ID (BGR)
COCO_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0:  (50,  200, 80),    # person       -> xanh lá nhạt
    1:  (220, 100, 30),    # bicycle      -> xanh dương
    2:  (30,  120, 240),   # car          -> cam đỏ
    3:  (180, 50,  200),   # motorcycle   -> tím
    5:  (30,  220, 220),   # bus          -> vàng
    7:  (60,  60,  220),   # truck        -> đỏ đậm
    9:  (30,  220, 30),    # traffic light-> xanh sáng
    15: (255, 100, 0),     # cat          -> xanh navy
    16: (0,   150, 255),   # dog          -> vàng cam
}
_DEFAULT_COLOR = (200, 200, 50)  # Màu dự phòng cho class không có trong bảng trên

# Chuỗi màu vòng cho các track_id trùng class
_ID_PALETTE = [
    (50,  200, 80),  (30, 120, 240), (180, 50, 200), (30, 220, 220),
    (60,  60,  220), (30, 220, 30),  (255, 100, 0),  (0,  200, 220),
    (220, 160, 30),  (30, 80,  250),
]


def _get_color(class_id: int, track_id: int) -> tuple[int, int, int]:
    if class_id in COCO_CLASS_COLORS:
        return COCO_CLASS_COLORS[class_id]
    return _ID_PALETTE[track_id % len(_ID_PALETTE)]


def _load_mot(tracks_path: Path) -> dict[int, list[dict]]:
    """
    Đọc MOT file .txt
    Trả về: {frame_index -> [{id, x, y, w, h, conf, class_id}, ...]}
    Format MOT: frame,id,x,y,w,h,conf,class_id,visibility (optional)
    """
    by_frame: dict[int, list[dict]] = {}
    with tracks_path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            try:
                frame = int(float(row[0]))
                tid = int(float(row[1]))
                x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                conf = float(row[6]) if len(row) > 6 else 1.0
                class_id = int(float(row[7])) if len(row) > 7 else 0
            except (ValueError, IndexError):
                continue
            by_frame.setdefault(frame, []).append(
                {"id": tid, "x": x, "y": y, "w": w, "h": h, "conf": conf, "class_id": class_id}
            )
    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Pipeline D video: COCO class labels từ VLM-Guided tracking."
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True,
                        help="scene_discovery.json (chứa coco_class_names)")
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--title", default="Pipeline D | VLM-Guided Tracking")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_video.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (use --overwrite): {args.output_video}")

    if not args.source_video.is_file():
        raise SystemExit(f"Source video not found: {args.source_video}")
    if not args.tracks.is_file():
        raise SystemExit(f"Tracks file not found: {args.tracks}")
    if not args.discovery.is_file():
        raise SystemExit(f"scene_discovery.json not found: {args.discovery}")

    # Load discovery
    disc = json.loads(args.discovery.read_text(encoding="utf-8"))
    class_names: dict[str, str] = disc.get("coco_class_names", {})  # "2" -> "car"
    context_short = disc.get("context_short", "")

    # Load tracks
    print(f"Loading tracks from {args.tracks} ...")
    by_frame = _load_mot(args.tracks)
    total_tracks = len({det["id"] for dets in by_frame.values() for det in dets})
    print(f"  {sum(len(v) for v in by_frame.values())} detections, {total_tracks} unique IDs")

    # Open video
    cap = cv2.VideoCapture(str(args.source_video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.source_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output_video), fourcc, fps, (width, height))

    frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    small_scale = 0.45

    print(f"Rendering -> {args.output_video} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Vẽ title
        title_text = args.title
        if context_short:
            title_text += f" [{context_short}]"
        cv2.putText(frame, title_text, (10, 28), font, small_scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, title_text, (10, 28), font, small_scale, (30, 30, 30), 1, cv2.LINE_AA)

        # Vẽ bounding boxes
        for det in by_frame.get(frame_idx, []):
            x1 = int(det["x"])
            y1 = int(det["y"])
            x2 = x1 + int(det["w"])
            y2 = y1 + int(det["h"])
            cid = det["class_id"]
            tid = det["id"]
            cname = class_names.get(str(cid), f"cls{cid}")
            color = _get_color(cid, tid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label: "ID 5 | car"
            label = f"ID {tid} | {cname}"
            lw, lh = cv2.getTextSize(label, font, font_scale, 1)[0]
            label_y = max(y1 - 6, lh + 6)
            cv2.rectangle(frame, (x1, label_y - lh - 4), (x1 + lw + 4, label_y + 2), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, label_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        writer.write(frame)

        if frame_idx % 300 == 0:
            pct = 100 * frame_idx / max(total_frames, 1)
            print(f"  Frame {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    writer.release()
    print(f"\n==> Done. Output: {args.output_video}")


if __name__ == "__main__":
    main()
