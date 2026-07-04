from __future__ import annotations

from pathlib import Path

import yaml

from football_tracking.experiments.tracker_grid import plan_tracker_grid


def _write_project(root: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(root))
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")


def _write_base_tracker(root: Path) -> Path:
    path = root / "configs" / "trackers" / "botsort.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
tracker:
  name: botsort_reid
  tracker_type: botsort
  track_high_thresh: 0.35
  new_track_thresh: 0.50
  track_buffer: 90
output:
  min_hits_for_output: 5
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_base_compare(root: Path) -> Path:
    path = root / "configs" / "compare.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
experiment:
  name: base_compare
  seed: 42
  split: all
dataset:
  mot_root: data/mot
  seqmap: data/mot/seqmaps/all.txt
detections:
  cache_config: configs/cache.yaml
  cache_root: outputs/detections/cache
  confidence_threshold: 0.1
trackers:
  - name: botsort_reid
    config: configs/trackers/botsort.yaml
evaluation:
  metrics: [HOTA, CLEAR, Identity]
benchmark:
  render_video: false
output:
  root: outputs/experiments/base
  tracks_root: outputs/tracks/base
  metrics_root: outputs/metrics/base
  figures_root: outputs/figures/base
runtime:
  overwrite: false
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_grid_config(root: Path, tracker: Path, compare: Path) -> Path:
    path = root / "configs" / "grid.yaml"
    path.write_text(
        f"""
grid:
  name: fixture_grid
  tracker_name: botsort_reid
  base_tracker_config: {tracker.relative_to(root).as_posix()}
  base_compare_config: {compare.relative_to(root).as_posix()}
  output_root: outputs/experiments/tracker_grid/fixture_grid
parameters:
  tracker.new_track_thresh: [0.35, 0.45]
  output.min_hits_for_output: [3, 5]
strategy:
  include_baseline: true
runtime:
  command_prefix: .\\.venv\\Scripts\\python.exe -m football_tracking.cli
""".strip(),
        encoding="utf-8",
    )
    return path


def test_tracker_grid_dry_run_limits_variants(tmp_path, monkeypatch) -> None:
    _write_project(tmp_path, monkeypatch)
    tracker = _write_base_tracker(tmp_path)
    compare = _write_base_compare(tmp_path)
    grid = _write_grid_config(tmp_path, tracker, compare)

    result = plan_tracker_grid(grid, dry_run=True, max_experiments=3)

    assert result["dry_run"] is True
    assert result["variant_count"] == 3
    assert result["variants"][0]["name"] == "baseline"


def test_tracker_grid_writes_variant_configs_and_run_script(tmp_path, monkeypatch) -> None:
    _write_project(tmp_path, monkeypatch)
    tracker = _write_base_tracker(tmp_path)
    compare = _write_base_compare(tmp_path)
    grid = _write_grid_config(tmp_path, tracker, compare)

    result = plan_tracker_grid(grid, overwrite=True, max_experiments=2)

    output_root = tmp_path / result["output_root"]
    assert (output_root / "manifest.json").is_file()
    assert (output_root / "manifest.csv").is_file()
    assert (output_root / "run_all.ps1").is_file()
    variant_config = tmp_path / result["variants"][1]["tracker_config"]
    payload = yaml.safe_load(variant_config.read_text(encoding="utf-8"))
    assert payload["tracker"]["new_track_thresh"] == 0.35
    assert payload["output"]["min_hits_for_output"] == 3
    assert "compare-trackers" in (output_root / "run_all.ps1").read_text(encoding="utf-8")
