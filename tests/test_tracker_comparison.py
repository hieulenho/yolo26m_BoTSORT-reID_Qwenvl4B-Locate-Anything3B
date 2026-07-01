from __future__ import annotations

from football_tracking.visualization.tracker_comparison import write_tracker_comparison_figures


def test_tracker_comparison_skips_null_metric_figures(tmp_path) -> None:
    paths = write_tracker_comparison_figures(
        [
            {"tracker": "sort", "HOTA": None, "tracker_fps": 100.0},
            {"tracker": "deepsort", "HOTA": None, "tracker_fps": 50.0},
        ],
        tmp_path,
    )

    assert str(tmp_path / "tracker_fps_comparison.png") in paths
    assert not (tmp_path / "hota_comparison.png").exists()
