from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from football_tracking.evaluation.idsw_review import (
    audit_idsw_review,
    compare_idsw_reviews,
    prepare_idsw_evidence,
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


def test_idsw_evidence_renders_three_frame_sheet(tmp_path: Path) -> None:
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
                "frame": 3,
                "gt_id": 1,
                "old_pred_id": 7,
                "new_pred_id": 9,
                "switch_type": "identity_swap",
                "reason": "test",
            }
        )
    sequence = tmp_path / "dataset" / "seq_01"
    image_dir = sequence / "img1"
    gt_dir = sequence / "gt"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir()
    for frame_id in (1, 3, 5):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / f"{frame_id:06d}.jpg"), image)
    gt_dir.joinpath("gt.txt").write_text(
        "1,1,10,10,20,30,1,-1,-1,-1\n"
        "3,1,12,10,20,30,1,-1,-1,-1\n"
        "5,1,14,10,20,30,1,-1,-1,-1\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "tracks" / "ocsort" / "all"
    predictions.mkdir(parents=True)
    predictions.joinpath("seq_01.txt").write_text(
        "1,7,10,10,20,30,1,-1,-1,-1\n"
        "3,9,12,10,20,30,1,-1,-1,-1\n"
        "5,9,14,10,20,30,1,-1,-1,-1\n",
        encoding="utf-8",
    )

    result = prepare_idsw_evidence(
        events,
        tmp_path / "dataset",
        tmp_path / "tracks",
        tmp_path / "evidence",
    )

    assert result["evidence_count"] == 1
    assert result["error_count"] == 0
    assert Path(result["events"][0]["evidence_path"]).is_file()
    assert (tmp_path / "evidence" / "index.html").is_file()


def test_idsw_review_agreement_reports_kappa(tmp_path: Path) -> None:
    fields = (
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
    rows = [
        {
            "tracker": "sort",
            "sequence": "seq",
            "frame": str(index),
            "gt_id": "1",
            "old_pred_id": "2",
            "new_pred_id": "3",
            "heuristic_type": label,
            "reviewed_type": label,
            "review_status": "reviewed",
            "reviewer": "reviewer",
            "notes": "",
        }
        for index, label in enumerate(("fragmentation", "association_error"), 1)
    ]
    paths = (tmp_path / "a.csv", tmp_path / "b.csv")
    for path in paths:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    result = compare_idsw_reviews(*paths)

    assert result["status"] == "agreed"
    assert result["agreement_percent"] == 100.0
    assert result["cohens_kappa"] == 1.0
