"""Heuristic taxonomy for MOT identity switches.

TrackEval reports the total number of ID switches, which is the primary metric
for benchmarking.  This module recomputes frame-level GT-to-pred associations
and assigns each switch to a practical debugging bucket so tracker variants can
be compared by failure mode.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IDSW_TYPES = (
    "fragmentation",
    "identity_swap",
    "re_identification_failure",
    "association_error",
    "appearance_confusion",
)


@dataclass(frozen=True)
class MotObject:
    frame: int
    object_id: int
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass(frozen=True)
class SwitchEvent:
    tracker: str
    sequence: str
    frame: int
    gt_id: int
    old_pred_id: int
    new_pred_id: int
    switch_type: str
    reason: str
    gap_frames: int
    iou: float
    nearby_gt_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker": self.tracker,
            "sequence": self.sequence,
            "frame": self.frame,
            "gt_id": self.gt_id,
            "old_pred_id": self.old_pred_id,
            "new_pred_id": self.new_pred_id,
            "switch_type": self.switch_type,
            "reason": self.reason,
            "gap_frames": self.gap_frames,
            "iou": self.iou,
            "nearby_gt_count": self.nearby_gt_count,
        }


def analyze_tracker_id_switches(
    *,
    tracker_name: str,
    prediction_dir: str | Path,
    mot_root: str | Path,
    seqmap: str | Path,
    iou_threshold: float = 0.5,
    reid_gap: int = 10,
    swap_window: int = 5,
    crowd_scale: float = 1.5,
) -> dict[str, Any]:
    """Analyze one tracker directory and return summary plus event rows."""

    prediction_root = Path(prediction_dir)
    sequence_names = _read_seqmap(seqmap)
    gt_index = _index_gt_paths(mot_root)
    all_events: list[SwitchEvent] = []
    per_sequence: list[dict[str, Any]] = []
    missing_sequences: list[str] = []

    for sequence in sequence_names:
        gt_path = gt_index.get(sequence)
        pred_path = prediction_root / f"{sequence}.txt"
        if gt_path is None or not pred_path.is_file():
            missing_sequences.append(sequence)
            continue
        gt_by_frame = _load_mot_by_frame(gt_path, is_ground_truth=True)
        pred_by_frame = _load_mot_by_frame(pred_path, is_ground_truth=False)
        events = _analyze_sequence(
            tracker_name=tracker_name,
            sequence=sequence,
            gt_by_frame=gt_by_frame,
            pred_by_frame=pred_by_frame,
            iou_threshold=iou_threshold,
            reid_gap=reid_gap,
            swap_window=swap_window,
            crowd_scale=crowd_scale,
        )
        all_events.extend(events)
        per_sequence.append(_summarize_events(sequence, events))

    summary = _summarize_events("__overall__", all_events)
    summary.update(
        {
            "tracker": tracker_name,
            "prediction_dir": str(prediction_root),
            "sequence_count": len(sequence_names) - len(missing_sequences),
            "missing_sequence_count": len(missing_sequences),
            "missing_sequences": missing_sequences,
            "iou_threshold": iou_threshold,
            "reid_gap": reid_gap,
            "swap_window": swap_window,
            "crowd_scale": crowd_scale,
        }
    )
    return {
        "summary": summary,
        "per_sequence": per_sequence,
        "events": [event.to_dict() for event in all_events],
    }


def analyze_many_trackers(
    *,
    trackers: dict[str, str | Path],
    mot_root: str | Path,
    seqmap: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    iou_threshold: float = 0.5,
    reid_gap: int = 10,
    swap_window: int = 5,
    crowd_scale: float = 1.5,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": output / "idsw_taxonomy_summary.json",
        "summary_csv": output / "idsw_taxonomy_summary.csv",
        "per_sequence_csv": output / "idsw_taxonomy_per_sequence.csv",
        "events_csv": output / "idsw_taxonomy_events.csv",
        "report_md": output / "idsw_taxonomy_report.md",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=false: {path}")

    tracker_results = [
        analyze_tracker_id_switches(
            tracker_name=name,
            prediction_dir=path,
            mot_root=mot_root,
            seqmap=seqmap,
            iou_threshold=iou_threshold,
            reid_gap=reid_gap,
            swap_window=swap_window,
            crowd_scale=crowd_scale,
        )
        for name, path in trackers.items()
    ]
    summaries = [item["summary"] for item in tracker_results]
    per_sequence = [
        {"tracker": item["summary"]["tracker"], **row}
        for item in tracker_results
        for row in item["per_sequence"]
    ]
    events = [row for item in tracker_results for row in item["events"]]
    result = {
        "tracker_count": len(tracker_results),
        "parameters": {
            "mot_root": str(mot_root),
            "seqmap": str(seqmap),
            "iou_threshold": iou_threshold,
            "reid_gap": reid_gap,
            "swap_window": swap_window,
            "crowd_scale": crowd_scale,
        },
        "summaries": summaries,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["summary_json"].write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(summaries, paths["summary_csv"])
    _write_csv(per_sequence, paths["per_sequence_csv"])
    _write_csv(events, paths["events_csv"])
    paths["report_md"].write_text(_report_markdown(summaries), encoding="utf-8")
    return result


def default_tracker_roots() -> dict[str, Path]:
    """Return the standard tracker outputs used by the project, if present."""

    candidates = {
        "sort": Path("outputs/tracks/yolo26m_comparison_all/sort/all"),
        "deepsort": Path("outputs/tracks/yolo26m_comparison_all/deepsort/all"),
        "bytetrack": Path("outputs/tracks/yolo26m_bytetrack_all/bytetrack/all"),
        "botsort_no_reid": Path("outputs/tracks/yolo26m_botsort_no_reid_all/botsort/all"),
        "botsort_reid_balanced": Path(
            "outputs/tracks/yolo26m_comparison_all/botsort_reid/all"
        ),
        "botsort_reid_identity_stable": Path(
            "outputs/tracks/yolo26m_botsort_identity_stable_all/botsort_reid/all"
        ),
    }
    return {name: path for name, path in candidates.items() if path.is_dir()}


def _analyze_sequence(
    *,
    tracker_name: str,
    sequence: str,
    gt_by_frame: dict[int, list[MotObject]],
    pred_by_frame: dict[int, list[MotObject]],
    iou_threshold: float,
    reid_gap: int,
    swap_window: int,
    crowd_scale: float,
) -> list[SwitchEvent]:
    gt_state: dict[int, tuple[int, int]] = {}
    pred_recent_owner: dict[int, tuple[int, int]] = {}
    events: list[SwitchEvent] = []
    frame_ids = sorted(set(gt_by_frame) | set(pred_by_frame))
    for frame in frame_ids:
        gt_objects = gt_by_frame.get(frame, [])
        pred_objects = pred_by_frame.get(frame, [])
        matches = _match_frame(gt_objects, pred_objects, iou_threshold)
        current_pred_to_gt = {pred.object_id: gt.object_id for gt, pred, _ in matches}
        gt_by_id = {gt.object_id: gt for gt in gt_objects}

        for gt, pred, iou_value in sorted(matches, key=lambda item: item[0].object_id):
            previous = gt_state.get(gt.object_id)
            if previous is not None:
                old_pred_id, last_frame = previous
                if old_pred_id != pred.object_id:
                    gap_frames = max(0, frame - last_frame - 1)
                    nearby_count = _nearby_gt_count(gt, gt_by_id.values(), crowd_scale)
                    switch_type, reason = _classify_switch(
                        gt_id=gt.object_id,
                        old_pred_id=old_pred_id,
                        new_pred_id=pred.object_id,
                        frame=frame,
                        gap_frames=gap_frames,
                        nearby_gt_count=nearby_count,
                        pred_recent_owner=pred_recent_owner,
                        current_pred_to_gt=current_pred_to_gt,
                        reid_gap=reid_gap,
                        swap_window=swap_window,
                    )
                    events.append(
                        SwitchEvent(
                            tracker=tracker_name,
                            sequence=sequence,
                            frame=frame,
                            gt_id=gt.object_id,
                            old_pred_id=old_pred_id,
                            new_pred_id=pred.object_id,
                            switch_type=switch_type,
                            reason=reason,
                            gap_frames=gap_frames,
                            iou=round(iou_value, 6),
                            nearby_gt_count=nearby_count,
                        )
                    )
            gt_state[gt.object_id] = (pred.object_id, frame)
            pred_recent_owner[pred.object_id] = (gt.object_id, frame)
    return events


def _classify_switch(
    *,
    gt_id: int,
    old_pred_id: int,
    new_pred_id: int,
    frame: int,
    gap_frames: int,
    nearby_gt_count: int,
    pred_recent_owner: dict[int, tuple[int, int]],
    current_pred_to_gt: dict[int, int],
    reid_gap: int,
    swap_window: int,
) -> tuple[str, str]:
    new_owner = pred_recent_owner.get(new_pred_id)
    if (
        new_owner is not None
        and new_owner[0] != gt_id
        and frame - new_owner[1] <= swap_window
    ):
        return (
            "identity_swap",
            (
                f"new predicted id was recently assigned to GT {new_owner[0]} "
                f"{frame - new_owner[1]} frame(s) earlier"
            ),
        )
    old_current_owner = current_pred_to_gt.get(old_pred_id)
    if old_current_owner is not None and old_current_owner != gt_id:
        return (
            "identity_swap",
            f"old predicted id is currently assigned to GT {old_current_owner}",
        )
    if gap_frames >= reid_gap:
        return (
            "re_identification_failure",
            f"GT was unmatched for {gap_frames} frame(s) before receiving a new id",
        )
    if gap_frames > 0:
        return (
            "fragmentation",
            f"short detection/tracking gap of {gap_frames} frame(s) before a new id",
        )
    if nearby_gt_count > 0:
        return (
            "appearance_confusion",
            f"{nearby_gt_count} nearby GT object(s) at switch frame",
        )
    return (
        "association_error",
        "continuous match changed id without a nearby GT crowding cue",
    )


def _match_frame(
    gt_objects: list[MotObject],
    pred_objects: list[MotObject],
    iou_threshold: float,
) -> list[tuple[MotObject, MotObject, float]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(gt_objects):
        for pred_index, pred in enumerate(pred_objects):
            iou_value = _iou(gt.xyxy, pred.xyxy)
            if iou_value >= iou_threshold:
                candidates.append((iou_value, gt_index, pred_index))
    candidates.sort(reverse=True, key=lambda item: item[0])
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[MotObject, MotObject, float]] = []
    for iou_value, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_objects[gt_index], pred_objects[pred_index], iou_value))
    return matches


def _nearby_gt_count(
    target: MotObject,
    gt_objects: Any,
    crowd_scale: float,
) -> int:
    target_cx, target_cy = target.center
    threshold = crowd_scale * max(target.w, target.h)
    count = 0
    for other in gt_objects:
        if other.object_id == target.object_id:
            continue
        other_cx, other_cy = other.center
        distance = ((target_cx - other_cx) ** 2 + (target_cy - other_cy) ** 2) ** 0.5
        if distance <= threshold:
            count += 1
    return count


def _load_mot_by_frame(path: str | Path, *, is_ground_truth: bool) -> dict[int, list[MotObject]]:
    by_frame: dict[int, list[MotObject]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            object_id = int(float(parts[1]))
            if object_id < 1:
                continue
            score = float(parts[6]) if len(parts) >= 7 else 1.0
            if is_ground_truth and score <= 0:
                continue
            by_frame[frame].append(
                MotObject(
                    frame=frame,
                    object_id=object_id,
                    x=float(parts[2]),
                    y=float(parts[3]),
                    w=float(parts[4]),
                    h=float(parts[5]),
                    score=score,
                )
            )
    return dict(by_frame)


def _read_seqmap(path: str | Path) -> list[str]:
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [line for line in lines if line.lower() != "name"]


def _index_gt_paths(mot_root: str | Path) -> dict[str, Path]:
    root = Path(mot_root)
    return {
        path.parent.parent.name: path
        for path in root.glob("*/*/gt/gt.txt")
        if path.is_file()
    }


def _summarize_events(sequence: str, events: list[SwitchEvent]) -> dict[str, Any]:
    counts = Counter(event.switch_type for event in events)
    total = sum(counts.values())
    row: dict[str, Any] = {
        "sequence": sequence,
        "total_id_switches_recomputed": total,
    }
    for switch_type in IDSW_TYPES:
        count = counts.get(switch_type, 0)
        row[f"{switch_type}_count"] = count
        row[f"{switch_type}_percent"] = _safe_percent(count, total)
    return row


def _report_markdown(summaries: list[dict[str, Any]]) -> str:
    headers = [
        "Tracker",
        "IDSW",
        "Fragmentation",
        "Identity Swap",
        "ReID Failure",
        "Association Error",
        "Appearance Confusion",
    ]
    lines = [
        "# ID Switch Taxonomy",
        "",
        (
            "The IDSW total is recomputed from MOT files using frame-level greedy "
            "IoU matching. TrackEval remains the source for official MOT scores; "
            "this taxonomy is a diagnostic breakdown of switch causes."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["tracker"]),
                    str(row["total_id_switches_recomputed"]),
                    _count_pct(row, "fragmentation"),
                    _count_pct(row, "identity_swap"),
                    _count_pct(row, "re_identification_failure"),
                    _count_pct(row, "association_error"),
                    _count_pct(row, "appearance_confusion"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failure Type Definitions",
            "",
            "- `fragmentation`: the GT target briefly disappears from matching and returns with a new predicted ID.",
            "- `identity_swap`: the new predicted ID was recently owned by another GT, or the old ID is assigned to another GT.",
            "- `re_identification_failure`: the GT target has a longer unmatched gap before returning with a new ID.",
            "- `association_error`: continuous tracking changes ID without a nearby crowding cue.",
            "- `appearance_confusion`: continuous switch occurs while other GT players are nearby, usually indicating crowded/similar-appearance ambiguity.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    inter_w = max(0.0, right - left)
    inter_h = max(0.0, bottom - top)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _safe_percent(count: int, total: int) -> float:
    return 0.0 if total == 0 else round(100.0 * count / total, 3)


def _count_pct(row: dict[str, Any], key: str) -> str:
    return f"{row[f'{key}_count']} ({row[f'{key}_percent']:.1f}%)"
