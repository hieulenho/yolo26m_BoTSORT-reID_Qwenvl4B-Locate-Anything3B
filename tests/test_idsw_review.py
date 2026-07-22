from __future__ import annotations

import csv
from pathlib import Path

from football_tracking.evaluation.idsw_review import (
    audit_idsw_review,
    prepare_idsw_review,
)


def test_idsw_review_requires_explicit_human_labels(tmp_path: Path) -> None:
    events = tmp_path / "events.csv"
    with events.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "tracker",
                "sequence",
                "frame",
                "gt_id",
                "old_pred_id",
                "new_pred_id",
                "switch_type",
                "reason",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "tracker": "ocsort",
                "sequence": "seq_01",
                "frame": 10,
                "gt_id": 3,
                "old_pred_id": 7,
                "new_pred_id": 9,
                "switch_type": "appearance_confusion",
                "reason": "crowded frame",
            }
        )
    review = tmp_path / "review.csv"
    prepared = prepare_idsw_review(events, review)
    draft = audit_idsw_review(review)

    assert prepared["status"] == "manual_review_required"
    assert draft["status"] == "review_required"
    assert draft["remaining_event_count"] == 1

    with review.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0].update(
        {
            "reviewed_type": "association_error",
            "review_status": "reviewed",
            "reviewer": "human_reviewer",
        }
    )
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    completed = audit_idsw_review(review)
    assert completed["status"] == "ready"
    assert completed["review_coverage_percent"] == 100.0
    assert completed["heuristic_agreement_percent"] == 0.0
    assert completed["reviewed_counts"]["association_error"] == 1
