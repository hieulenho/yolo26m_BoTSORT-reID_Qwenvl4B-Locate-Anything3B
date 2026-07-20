from __future__ import annotations

from football_tracking.evaluation.multi_tracker_trackeval import (
    _common_prediction_frame_limits,
    _trackeval_gt_folder,
    evaluate_trackers_with_trackeval,
)


def test_trackeval_missing_returns_null_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "football_tracking.evaluation.multi_tracker_trackeval.trackeval_available",
        lambda: False,
    )

    result = evaluate_trackers_with_trackeval(
        ["sort"],
        gt_root=tmp_path / "gt",
        trackers_root=tmp_path / "tracks",
        split="val",
        seqmap=None,
        output_root=tmp_path / "metrics",
        metrics=("HOTA", "CLEAR", "Identity"),
    )

    assert result["sort"].available is False
    assert result["sort"].metrics["HOTA"] is None
    assert "not installed" in result["sort"].reason


def test_trackeval_gt_folder_stages_all_split_from_train_val_test(tmp_path) -> None:
    for split, seq_name in (("train", "seq_train"), ("val", "seq_val"), ("test", "seq_test")):
        seq_dir = tmp_path / "gt" / split / seq_name
        (seq_dir / "gt").mkdir(parents=True)
        (seq_dir / "gt" / "gt.txt").write_text("1,1,0,0,10,10,1,1,1\n", encoding="utf-8")
        (seq_dir / "seqinfo.ini").write_text("[Sequence]\nseqLength=1\n", encoding="utf-8")

    staged = _trackeval_gt_folder(
        tmp_path / "gt",
        "all",
        ["seq_test", "seq_train", "seq_val"],
        tmp_path / "metrics",
    )

    assert staged == tmp_path / "metrics" / "staged_gt" / "all"
    assert (staged / "seq_test" / "gt" / "gt.txt").is_file()
    assert (staged / "seq_train" / "seqinfo.ini").is_file()


def test_partial_trackeval_stages_gt_only_to_prediction_frame_limit(tmp_path) -> None:
    seq_dir = tmp_path / "gt" / "val" / "seq_a"
    (seq_dir / "gt").mkdir(parents=True)
    (seq_dir / "gt" / "gt.txt").write_text(
        "1,1,0,0,10,10,1,1,1\n2,1,0,0,10,10,1,1,1\n"
        "3,1,0,0,10,10,1,1,1\n",
        encoding="utf-8",
    )
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=seq_a\nseqLength=3\n",
        encoding="utf-8",
    )
    for tracker in ("sort", "ocsort"):
        pred_dir = tmp_path / "tracks" / tracker / "val"
        pred_dir.mkdir(parents=True)
        (pred_dir / "seq_a.txt").write_text("1,1,0,0,10,10,1,1,1\n", encoding="utf-8")
        (pred_dir / "seq_a.metadata.json").write_text(
            '{"frame_count": 2}',
            encoding="utf-8",
        )

    limits = _common_prediction_frame_limits(
        ["sort", "ocsort"],
        tmp_path / "tracks",
        "val",
        ["seq_a"],
    )
    staged = _trackeval_gt_folder(
        tmp_path / "gt",
        "val",
        ["seq_a"],
        tmp_path / "metrics",
        frame_limits=limits,
    )

    gt_rows = (staged / "seq_a" / "gt" / "gt.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    seqinfo = (staged / "seq_a" / "seqinfo.ini").read_text(encoding="utf-8")
    assert len(gt_rows) == 2
    assert "seqLength=2" in seqinfo
