"""Human-review workflow for diagnostic ID-switch failure categories."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from football_tracking.evaluation.idsw_taxonomy import IDSW_TYPES


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


__all__ = ["IdswReviewError", "audit_idsw_review", "prepare_idsw_review"]
