"""Human-review workflow for diagnostic ID-switch failure categories."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from football_tracking.evaluation.idsw_taxonomy import IDSW_TYPES
from football_tracking.vlm.tracking_context import MotTrackRow


class IdswReviewError(RuntimeError):
    """Raised when an ID-switch review artifact is invalid."""


REVIEW_FIELDS = (
    "tracker",
    "sequence",
    "frame",
    "gt_id",
    "old_pred_id",
    "new_pred_id",
    "heuristic_type",
    "reviewed_type",
    "review_status",
    "reviewer",
    "notes",
)


def prepare_idsw_review(
    events_csv: str | Path,
    output_csv: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(events_csv)
    destination = Path(output_csv)
    if not source.is_file():
        raise IdswReviewError(f"IDSW event CSV does not exist: {source}")
    if destination.exists() and not overwrite:
        raise IdswReviewError(f"Review CSV exists and overwrite=false: {destination}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle))
    rows = [
        {
            "tracker": row.get("tracker", ""),
            "sequence": row.get("sequence", ""),
            "frame": row.get("frame", ""),
            "gt_id": row.get("gt_id", ""),
            "old_pred_id": row.get("old_pred_id", ""),
            "new_pred_id": row.get("new_pred_id", ""),
            "heuristic_type": row.get("switch_type", ""),
            "reviewed_type": "",
            "review_status": "draft",
            "reviewer": "",
            "notes": row.get("reason", ""),
        }
        for row in events
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return {
        "status": "manual_review_required",
        "review_csv": str(destination.resolve()),
        "event_count": len(rows),
        "allowed_types": list(IDSW_TYPES),
    }


def audit_idsw_review(review_csv: str | Path) -> dict[str, Any]:
    path = Path(review_csv)
    if not path.is_file():
        raise IdswReviewError(f"IDSW review CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    errors: list[str] = []
    reviewed: list[dict[str, str]] = []
    ignored = 0
    for index, row in enumerate(rows, start=2):
        status = str(row.get("review_status", "")).strip().lower()
        if status == "ignored":
            ignored += 1
            continue
        if status == "draft":
            continue
        if status != "reviewed":
            errors.append(f"row {index}: review_status must be draft, reviewed, or ignored")
            continue
        reviewed_type = str(row.get("reviewed_type", "")).strip()
        if reviewed_type not in IDSW_TYPES:
            errors.append(f"row {index}: invalid reviewed_type '{reviewed_type}'")
        if not str(row.get("reviewer", "")).strip():
            errors.append(f"row {index}: reviewer is required")
        reviewed.append(row)

    counts = Counter(str(row.get("reviewed_type", "")).strip() for row in reviewed)
    agreed = sum(
        str(row.get("reviewed_type", "")).strip()
        == str(row.get("heuristic_type", "")).strip()
        for row in reviewed
    )
    total = len(rows)
    reviewed_count = len(reviewed)
    review_complete = total > 0 and reviewed_count + ignored == total and not errors
    coverage = (
        round(100.0 * (reviewed_count + ignored) / total, 3) if total else 0.0
    )
    agreement = (
        round(100.0 * agreed / reviewed_count, 3) if reviewed_count else None
    )
    return {
        "status": "ready" if review_complete else "review_required",
        "review_csv": str(path.resolve()),
        "event_count": total,
        "reviewed_event_count": reviewed_count,
        "ignored_event_count": ignored,
        "remaining_event_count": max(total - reviewed_count - ignored, 0),
        "review_coverage_percent": coverage,
        "heuristic_agreement_percent": agreement,
        "reviewed_counts": {name: counts.get(name, 0) for name in IDSW_TYPES},
        "errors": errors,
    }


def compare_idsw_reviews(
    review_a: str | Path,
    review_b: str | Path,
) -> dict[str, Any]:
    """Measure independent reviewer agreement before taxonomy adjudication."""

    first = _reviewed_rows(Path(review_a).resolve())
    second = _reviewed_rows(Path(review_b).resolve())
    if set(first) != set(second):
        missing_a = sorted(set(second) - set(first))
        missing_b = sorted(set(first) - set(second))
        raise IdswReviewError(
            "Review event sets differ; "
            f"missing from A={len(missing_a)}, missing from B={len(missing_b)}"
        )
    keys = sorted(first)
    labels_a = [first[key] for key in keys]
    labels_b = [second[key] for key in keys]
    agreements = sum(left == right for left, right in zip(labels_a, labels_b, strict=True))
    observed = _ratio(agreements, len(keys))
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        _ratio(counts_a[name], len(keys)) * _ratio(counts_b[name], len(keys))
        for name in IDSW_TYPES
    )
    kappa = _ratio(observed - expected, 1.0 - expected) if expected < 1.0 else 1.0
    disagreements = [
        {
            "tracker": key[0],
            "sequence": key[1],
            "frame": key[2],
            "gt_id": key[3],
            "old_pred_id": key[4],
            "new_pred_id": key[5],
            "review_a": first[key],
            "review_b": second[key],
        }
        for key in keys
        if first[key] != second[key]
    ]
    return {
        "status": "adjudication_required" if disagreements else "agreed",
        "review_a": str(Path(review_a).resolve()),
        "review_b": str(Path(review_b).resolve()),
        "event_count": len(keys),
        "agreement_count": agreements,
        "agreement_percent": round(100.0 * observed, 3),
        "cohens_kappa": round(kappa, 6),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }


def _reviewed_rows(path: Path) -> dict[tuple[str, str, int, int, int, int], str]:
    audit = audit_idsw_review(path)
    if audit["status"] != "ready":
        raise IdswReviewError(f"Review must be complete before agreement: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if str(row.get("review_status", "")).strip().lower() != "reviewed":
            continue
        key = (
            str(row.get("tracker", "")),
            str(row.get("sequence", "")),
            int(row.get("frame", 0)),
            int(row.get("gt_id", 0)),
            int(row.get("old_pred_id", 0)),
            int(row.get("new_pred_id", 0)),
        )
        result[key] = str(row.get("reviewed_type", "")).strip()
    return result


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def prepare_idsw_evidence(
    events_csv: str | Path,
    dataset_root: str | Path,
    tracks_root: str | Path,
    output_dir: str | Path,
    *,
    frame_offset: int = 2,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render before/current/after evidence sheets for manual IDSW categorization."""

    events_path = Path(events_csv).resolve()
    data_root = Path(dataset_root).resolve()
    prediction_root = Path(tracks_root).resolve()
    destination = Path(output_dir).resolve()
    index_json = destination / "idsw_evidence_index.json"
    index_html = destination / "index.html"
    if frame_offset < 1:
        raise IdswReviewError("frame_offset must be positive.")
    for name, path in (
        ("events", events_path),
        ("dataset root", data_root),
        ("tracks root", prediction_root),
    ):
        if not path.exists():
            raise IdswReviewError(f"IDSW {name} does not exist: {path}")
    if (index_json.exists() or index_html.exists()) and not overwrite:
        raise IdswReviewError(f"IDSW evidence exists and overwrite=false: {destination}")
    with events_path.open("r", encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle))
    destination.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, dict[int, list[MotTrackRow]]] = {}
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for event_index, event in enumerate(events, start=1):
        tracker = str(event.get("tracker", "")).strip()
        sequence = str(event.get("sequence", "")).strip()
        frame = int(event.get("frame", 0))
        sequence_dir = _resolve_sequence_dir(data_root, sequence)
        gt_path = sequence_dir / "gt" / "gt.txt"
        image_dir = sequence_dir / "img1"
        prediction_path = prediction_root / tracker / "all" / f"{sequence}.txt"
        missing = [path for path in (gt_path, image_dir, prediction_path) if not path.exists()]
        if missing:
            errors.append(
                f"event {event_index}: missing " + ", ".join(str(path) for path in missing)
            )
            continue
        gt_by_frame = _rows_by_frame(gt_path, cache)
        prediction_by_frame = _rows_by_frame(prediction_path, cache)
        frame_ids = [max(1, frame - frame_offset), frame, frame + frame_offset]
        panels = []
        for frame_id in frame_ids:
            image_path = image_dir / f"{frame_id:06d}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                errors.append(f"event {event_index}: unreadable image {image_path}")
                panels = []
                break
            _draw_matching_box(
                image,
                gt_by_frame.get(frame_id, []),
                int(event.get("gt_id", 0)),
                (40, 210, 40),
                "GT",
            )
            _draw_matching_box(
                image,
                prediction_by_frame.get(frame_id, []),
                int(event.get("old_pred_id", 0)),
                (30, 30, 230),
                "old",
            )
            _draw_matching_box(
                image,
                prediction_by_frame.get(frame_id, []),
                int(event.get("new_pred_id", 0)),
                (230, 120, 20),
                "new",
            )
            cv2.putText(
                image,
                f"frame {frame_id}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(_resize_panel(image, width=600))
        if not panels:
            continue
        sheet = cv2.hconcat(panels)
        file_name = f"{event_index:04d}_{_file_token(tracker)}_{_file_token(sequence)}_{frame}.jpg"
        sheet_path = destination / file_name
        if not cv2.imwrite(str(sheet_path), sheet):
            errors.append(f"event {event_index}: failed to write {sheet_path}")
            continue
        results.append(
            {
                "event_index": event_index,
                "tracker": tracker,
                "sequence": sequence,
                "frame": frame,
                "gt_id": int(event.get("gt_id", 0)),
                "old_pred_id": int(event.get("old_pred_id", 0)),
                "new_pred_id": int(event.get("new_pred_id", 0)),
                "heuristic_type": event.get("switch_type", ""),
                "reason": event.get("reason", ""),
                "evidence_path": str(sheet_path),
            }
        )
    payload = {
        "status": "manual_review_required" if results else "evidence_failed",
        "event_count": len(events),
        "evidence_count": len(results),
        "error_count": len(errors),
        "legend": {"GT": "green", "old_prediction": "red", "new_prediction": "blue"},
        "events": results,
        "errors": errors,
    }
    _write_atomic(index_json, json.dumps(payload, indent=2))
    _write_atomic(index_html, _evidence_html(results))
    return payload


def _rows_by_frame(
    path: Path, cache: dict[Path, dict[int, list[MotTrackRow]]]
) -> dict[int, list[MotTrackRow]]:
    if path not in cache:
        grouped: dict[int, list[MotTrackRow]] = {}
        for row in _read_mot_rows_allow_zero_id(path):
            grouped.setdefault(row.frame_index, []).append(row)
        cache[path] = grouped
    return cache[path]


def _read_mot_rows_allow_zero_id(path: Path) -> list[MotTrackRow]:
    rows: list[MotTrackRow] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            raise IdswReviewError(f"Invalid MOT row at {path}:{line_number}")
        frame_id = int(float(fields[0]))
        track_id = int(float(fields[1]))
        if frame_id < 1 or track_id < 0:
            raise IdswReviewError(
                f"Frame must be positive and track ID non-negative at {path}:{line_number}"
            )
        rows.append(
            MotTrackRow(
                frame_index=frame_id,
                track_id=track_id,
                x=float(fields[2]),
                y=float(fields[3]),
                width=float(fields[4]),
                height=float(fields[5]),
                confidence=float(fields[6]) if len(fields) > 6 else None,
            )
        )
    return rows


def _resolve_sequence_dir(data_root: Path, sequence: str) -> Path:
    direct = data_root / sequence
    if direct.is_dir():
        return direct
    candidates = [
        data_root / split / sequence for split in ("train", "val", "test")
    ]
    existing = [path for path in candidates if path.is_dir()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise IdswReviewError(
            f"Sequence '{sequence}' exists in multiple dataset splits: {existing}"
        )
    return direct


def _draw_matching_box(
    image: Any,
    rows: list[MotTrackRow],
    track_id: int,
    color: tuple[int, int, int],
    prefix: str,
) -> None:
    for row in rows:
        if row.track_id != track_id:
            continue
        x1, y1, x2, y2 = row.bbox_xyxy()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image,
            f"{prefix}:{track_id}",
            (x1, max(y1 - 7, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def _resize_panel(image: Any, *, width: int) -> Any:
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _file_token(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _evidence_html(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        relative = Path(row["evidence_path"]).name
        title = (
            f"#{row['event_index']} {row['tracker']} / {row['sequence']} / "
            f"frame {row['frame']}"
        )
        cards.append(
            "<article><h2>{}</h2><p>Heuristic: <b>{}</b> - {}</p>"
            '<img src="{}" alt="{}"></article>'.format(
                html.escape(title),
                html.escape(str(row["heuristic_type"])),
                html.escape(str(row["reason"])),
                html.escape(relative),
                html.escape(title),
            )
        )
    return (
        "<!doctype html><meta charset=\"utf-8\"><title>IDSW evidence</title>"
        "<style>body{font-family:Arial;margin:24px}article{margin-bottom:36px}"
        "img{max-width:100%;border:1px solid #bbb}h2{font-size:18px}</style>"
        "<h1>ID-switch human review evidence</h1>"
        "<p>Green: GT; red: previous predicted ID; blue: new predicted ID.</p>"
        + "".join(cards)
    )


def _write_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "IdswReviewError",
    "audit_idsw_review",
    "compare_idsw_reviews",
    "prepare_idsw_evidence",
    "prepare_idsw_review",
]
