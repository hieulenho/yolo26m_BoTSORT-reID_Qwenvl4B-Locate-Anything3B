from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_tracking.evaluation.mot_transform import (
    MotTransformError,
    filter_mot_file_by_ignore_regions,
    load_normalized_mot_ignore_regions,
    stage_ignore_filtered_mot_ground_truth,
    stage_scaled_mot_ground_truth,
)


def test_stage_scaled_mot_ground_truth_scales_boxes_and_seqinfo(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "02"
    (source / "gt").mkdir(parents=True)
    (source / "gt" / "gt.txt").write_text(
        "1,7,100,200,40,60,1,1,1,-1\n",
        encoding="utf-8",
    )
    (source / "seqinfo.ini").write_text(
        "[Sequence]\nimWidth=1024\nimHeight=512\nseqLength=1\n",
        encoding="utf-8",
    )

    destination = stage_scaled_mot_ground_truth(
        gt_root=tmp_path / "source",
        sequences=["02"],
        output_root=tmp_path / "scaled",
        scale_x=0.5,
        scale_y=0.25,
    )

    row = (destination / "02" / "gt" / "gt.txt").read_text(encoding="utf-8")
    assert row == "1,7,50.000000,50.000000,20.000000,15.000000,1,1,1,-1\n"
    seqinfo = (destination / "02" / "seqinfo.ini").read_text(encoding="utf-8")
    assert "imWidth=512" in seqinfo
    assert "imHeight=128" in seqinfo


def test_stage_scaled_mot_ground_truth_rejects_invalid_scale(tmp_path: Path) -> None:
    with pytest.raises(MotTransformError, match="scale_x"):
        stage_scaled_mot_ground_truth(
            gt_root=tmp_path,
            sequences=["02"],
            output_root=tmp_path / "scaled",
            scale_x=0.0,
            scale_y=1.0,
        )


def test_ignore_region_filter_uses_intersection_over_box_area(tmp_path: Path) -> None:
    source = tmp_path / "prediction.txt"
    source.write_text(
        "1,1,10,10,20,20,0.9,1,1,-1\n"
        "1,2,70,70,20,20,0.8,1,1,-1\n",
        encoding="utf-8",
    )
    destination = tmp_path / "filtered.txt"

    stats = filter_mot_file_by_ignore_regions(
        source=source,
        destination=destination,
        regions=[(0.0, 0.0, 25.0, 30.0)],
        overlap_threshold=0.5,
    )

    assert stats == {"input_rows": 2, "kept_rows": 1, "removed_rows": 1}
    assert destination.read_text(encoding="utf-8").startswith("1,2,70,70")


def test_load_and_stage_normalized_ignore_regions(tmp_path: Path) -> None:
    source = tmp_path / "normalized" / "MVI_1"
    (source / "gt").mkdir(parents=True)
    (source / "gt" / "gt.txt").write_text(
        "1,1,5,5,10,10,1,1,1,-1\n1,2,50,50,10,10,1,1,1,-1\n",
        encoding="utf-8",
    )
    (source / "seqinfo.ini").write_text("[Sequence]\nseqLength=1\n")
    region_path = source / "ignored_regions.json"
    region_path.write_text(
        '{"regions":[{"x":0,"y":0,"width":20,"height":20}]}',
        encoding="utf-8",
    )
    (source.parent / "normalized_gt_manifest.json").write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "normalized_sequence": "MVI_1",
                        "ignored_regions_path": region_path.as_posix(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    regions = load_normalized_mot_ignore_regions(
        gt_root=source.parent,
        sequences=["MVI_1"],
        scale_x=2.0,
        scale_y=3.0,
    )
    assert regions == {"MVI_1": [(0.0, 0.0, 40.0, 60.0)]}

    staged_root, stats = stage_ignore_filtered_mot_ground_truth(
        gt_root=source.parent,
        sequences=["MVI_1"],
        output_root=tmp_path / "filtered_gt",
        regions_by_sequence={"MVI_1": [(0.0, 0.0, 20.0, 20.0)]},
        overlap_threshold=0.5,
    )
    assert stats["MVI_1"]["removed_rows"] == 1
    assert (staged_root / "MVI_1" / "gt" / "gt.txt").read_text().startswith(
        "1,2,50,50"
    )
