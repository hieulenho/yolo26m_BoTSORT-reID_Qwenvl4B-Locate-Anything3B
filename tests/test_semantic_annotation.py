from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from football_tracking.benchmarking.semantic_annotation import (
    SemanticAnnotationError,
    finalize_annotation_package,
    merge_reviewed_manifests,
    prepare_annotation_package,
)


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 48),
    )
    assert writer.isOpened()
    for value in (30, 80, 130):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()


def test_annotation_package_requires_human_review_before_finalize(tmp_path: Path) -> None:
    video = tmp_path / "sample.avi"
    _write_video(video)
    tracks = tmp_path / "tracks.txt"
    tracks.write_text(
        "1,1,8,6,24,30,0.9,-1,-1,-1\n"
        "2,1,9,6,24,30,0.9,-1,-1,-1\n"
        "3,1,10,6,24,30,0.9,-1,-1,-1\n",
        encoding="utf-8",
    )
    artifacts = {}
    for name, payload in {
        "discovery": {
            "domain": {"name": "traffic"},
            "objects": [{"canonical_name": "car", "action": "track"}],
        },
        "route": {"route_name": "coco_pretrained"},
        "semantics": {"tracks": []},
        "run_report": {},
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifacts[name] = path
    package_dir = tmp_path / "annotation"
    result = prepare_annotation_package(
        sample_id="traffic_01",
        source_video=video,
        tracks_path=tracks,
        discovery_path=artifacts["discovery"],
        route_path=artifacts["route"],
        semantics_path=artifacts["semantics"],
        run_report_path=artifacts["run_report"],
        output_dir=package_dir,
    )

    assert result["status"] == "manual_review_required"
    assert len(list((package_dir / "contact_sheets").glob("*.jpg"))) == 1
    with pytest.raises(SemanticAnnotationError, match="reviewed"):
        finalize_annotation_package(package_dir=package_dir)

    review_path = package_dir / "ground_truth_review.yaml"
    review = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    review["domain"] = "traffic"
    review["detector_route"] = "coco_pretrained"
    review["review"] = {
        "status": "reviewed",
        "annotator": "human_reviewer",
        "reviewed_at": "2026-07-21",
        "method": "manual_video_and_contact_sheet",
    }
    review_path.write_text(yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    csv_path = package_dir / "track_annotations.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0].update(
        {
            "class_label": "car",
            "fine_label": "sedan",
            "review_status": "reviewed",
            "annotator": "human_reviewer",
        }
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    finalized = finalize_annotation_package(package_dir=package_dir)
    manifest = yaml.safe_load(Path(finalized["manifest"]).read_text(encoding="utf-8"))

    assert finalized["track_count"] == 1
    assert manifest["require_review_metadata"] is True
    assert manifest["samples"][0]["ground_truth"]["tracks"][0]["fine_label"] == "sedan"
    assert not Path(manifest["samples"][0]["artifacts"]["discovery"]).is_absolute()

    merged_path = tmp_path / "combined.yaml"
    merged = merge_reviewed_manifests(
        manifest_paths=[finalized["manifest"]],
        output_manifest=merged_path,
    )
    assert merged["sample_count"] == 1
    assert merged["track_count"] == 1
    assert yaml.safe_load(merged_path.read_text(encoding="utf-8"))["samples"][0][
        "sample_id"
    ] == "traffic_01"

    with pytest.raises(SemanticAnnotationError, match="Duplicate sample_id"):
        merge_reviewed_manifests(
            manifest_paths=[finalized["manifest"], finalized["manifest"]],
            output_manifest=tmp_path / "duplicate.yaml",
        )
