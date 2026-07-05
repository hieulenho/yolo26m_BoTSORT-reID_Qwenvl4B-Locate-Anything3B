from __future__ import annotations

from football_tracking.experiments.experiment_config import load_compare_trackers_config
from football_tracking.experiments.experiment_runner import (
    _best_tracker_payload,
    _comparison_delta,
    compare_trackers,
)
from football_tracking.experiments.schemas import ExperimentResult
from tests.test_experiment_config import _write_config, _write_sequence


def test_compare_trackers_dry_run_lists_sequences(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _write_sequence(tmp_path)
    config = _write_config(tmp_path)

    result = compare_trackers(config, dry_run=True)

    assert result["dry_run"] is True
    assert result["sequence_count"] == 1
    assert result["trackers"] == ["sort"]


def test_best_tracker_payload_uses_stable_metric_priority(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _write_sequence(tmp_path)
    config = load_compare_trackers_config(_write_config(tmp_path))

    payload = _best_tracker_payload(
        config,
        [
            {"tracker": "fast", "HOTA": 60.0, "IDF1": 70.0, "AssA": 50.0, "IDSW": 25},
            {"tracker": "stable", "HOTA": 60.0, "IDF1": 70.0, "AssA": 50.0, "IDSW": 12},
            {"tracker": "weak", "HOTA": 58.0, "IDF1": 80.0, "AssA": 55.0, "IDSW": 4},
        ],
    )

    assert payload["selected_tracker"] == "stable"
    assert payload["metrics"]["tracker"] == "stable"
    assert payload["tracker_count"] == 3


def test_comparison_delta_is_generic_best_minus_reference() -> None:
    baseline = _result("baseline", hota=55.0, idf1=50.0, assa=45.0, idsw=80, fps=30.0)
    tuned = _result("tuned", hota=60.0, idf1=57.0, assa=52.0, idsw=40, fps=20.0)

    delta = _comparison_delta([baseline, tuned])

    assert delta["available"] is True
    assert delta["reference_tracker"] == "baseline"
    assert delta["candidate_tracker"] == "tuned"
    assert delta["direction"] == "candidate_minus_reference"
    assert delta["metrics"]["HOTA"] == 5.0
    assert delta["metrics"]["IDSW"] == -40


def test_comparison_delta_requires_two_trackers() -> None:
    delta = _comparison_delta([_result("only", hota=60.0, idf1=57.0, assa=52.0, idsw=40)])

    assert delta["available"] is False
    assert "At least two tracker results" in delta["reason"]


def _result(
    name: str,
    *,
    hota: float,
    idf1: float,
    assa: float,
    idsw: int,
    fps: float = 10.0,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=name,
        tracker_name=name,
        status="completed",
        sequence_count=1,
        frame_count=10,
        detection_count=10,
        emitted_track_count=10,
        unique_track_count=2,
        tracker_seconds=1.0,
        frame_read_seconds=0.0,
        cache_read_seconds=0.0,
        mot_write_seconds=0.0,
        total_seconds=1.0,
        tracker_fps=fps,
        cached_pipeline_fps=fps,
        metrics={"HOTA": hota, "IDF1": idf1, "AssA": assa, "IDSW": idsw},
    )
