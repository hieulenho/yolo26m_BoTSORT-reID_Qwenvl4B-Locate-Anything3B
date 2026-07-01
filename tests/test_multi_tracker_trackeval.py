from __future__ import annotations

from football_tracking.evaluation.multi_tracker_trackeval import evaluate_trackers_with_trackeval


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
