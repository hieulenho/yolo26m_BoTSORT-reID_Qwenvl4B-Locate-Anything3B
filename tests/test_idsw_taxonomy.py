from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_tracking.evaluation.idsw_taxonomy import analyze_many_trackers


def test_taxonomy_writes_auditable_tables_and_figures(tmp_path: Path) -> None:
    mot_root = tmp_path / "mot"
    gt_path = mot_root / "val" / "sequence_01" / "gt" / "gt.txt"
    gt_path.parent.mkdir(parents=True)
    gt_path.write_text(
        "1,1,0,0,10,10,1,1,1\n"
        "2,1,1,0,10,10,1,1,1\n"
        "3,1,2,0,10,10,1,1,1\n",
        encoding="utf-8",
    )
    seqmap = mot_root / "seqmaps" / "all.txt"
    seqmap.parent.mkdir(parents=True)
    seqmap.write_text("name\nsequence_01\n", encoding="utf-8")

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    predictions.joinpath("sequence_01.txt").write_text(
        "1,10,0,0,10,10,1,-1,-1,-1\n"
        "2,10,1,0,10,10,1,-1,-1,-1\n"
        "3,11,2,0,10,10,1,-1,-1,-1\n",
        encoding="utf-8",
    )

    output = tmp_path / "report"
    result = analyze_many_trackers(
        trackers={"test_tracker": predictions},
        mot_root=mot_root,
        seqmap=seqmap,
        output_dir=output,
    )

    summary = result["summaries"][0]
    assert summary["total_id_switches_recomputed"] == 1
    assert summary["association_error_count"] == 1
    for path in result["paths"].values():
        artifact = Path(path)
        assert artifact.is_file()
        assert artifact.stat().st_size > 0
    saved = json.loads((output / "idsw_taxonomy_summary.json").read_text())
    assert saved["tracker_count"] == 1
    assert "official IDSW" in (output / "idsw_taxonomy_report.md").read_text()


def test_taxonomy_refuses_to_overwrite_existing_outputs(tmp_path: Path) -> None:
    output = tmp_path / "report"
    output.mkdir()
    (output / "idsw_taxonomy_summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=false"):
        analyze_many_trackers(
            trackers={},
            mot_root=tmp_path,
            seqmap=tmp_path / "missing.txt",
            output_dir=output,
        )
