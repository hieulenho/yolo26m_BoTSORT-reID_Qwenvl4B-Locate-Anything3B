from __future__ import annotations

from football_tracking.experiments.experiment_runner import compare_trackers
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
