from __future__ import annotations

from pathlib import Path

import pytest

from football_tracking.evaluation.mot_transform import (
    MotTransformError,
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
