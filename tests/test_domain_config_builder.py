from __future__ import annotations

from pathlib import Path

import yaml

from football_tracking.domains.config_builder import build_domain_configs, load_domain_profile


def _write_project(root: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(root))
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")


def _write_tracker_preset(root: Path) -> Path:
    path = root / "configs" / "trackers" / "botsort_balanced.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
tracker:
  name: botsort_reid
  tracker_type: botsort
  track_high_thresh: 0.35
  track_low_thresh: 0.10
  new_track_thresh: 0.50
  track_buffer: 90
  match_thresh: 0.85
  fuse_score: true
  gmc_method: sparseOptFlow
  proximity_thresh: 0.30
  appearance_thresh: 0.30
  with_reid: true
  model: yolo26n-cls.pt
output:
  min_hits_for_output: 5
  compact_ids: true
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_domain_profile(root: Path, tracker_preset: Path) -> Path:
    mot_root = root / "data" / "mot" / "fixture"
    mot_root.mkdir(parents=True)
    seqmap = mot_root / "seqmaps" / "all.txt"
    seqmap.parent.mkdir(parents=True)
    seqmap.write_text("name\nseq\n", encoding="utf-8")
    profile = root / "configs" / "domains" / "fixture.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        f"""
domain:
  name: fixture domain
  namespace: fixture
model:
  name: yolo26m
  backend: ultralytics
  checkpoint: models/detector/fixture/yolo26m_best.pt
  fallback_checkpoint: yolo26m.pt
  allow_pretrained_fallback: true
detector:
  imgsz: 640
  conf: 0.2
  conf_floor: 0.001
  iou: 0.7
  max_det: 100
  device: cpu
  half: false
  class_ids: [0]
  target_class_id: 0
  target_class_name: object
tracker:
  default_name: botsort_reid
  default_preset: balanced
  default_config: {tracker_preset.relative_to(root).as_posix()}
  presets:
    balanced: {tracker_preset.relative_to(root).as_posix()}
dataset:
  mot_root: {mot_root.relative_to(root).as_posix()}
  default_split: all
  seqmaps:
    all: {seqmap.relative_to(root).as_posix()}
""".strip(),
        encoding="utf-8",
    )
    return profile


def test_domain_profile_loads_tracker_preset(tmp_path, monkeypatch) -> None:
    _write_project(tmp_path, monkeypatch)
    tracker_preset = _write_tracker_preset(tmp_path)
    profile_path = _write_domain_profile(tmp_path, tracker_preset)

    profile = load_domain_profile(profile_path)

    assert profile.name == "fixture domain"
    assert profile.namespace == "fixture"
    assert profile.tracker_config_for_preset().name == "botsort_balanced.yaml"


def test_build_domain_configs_writes_tracking_cache_and_compare_configs(
    tmp_path,
    monkeypatch,
) -> None:
    _write_project(tmp_path, monkeypatch)
    tracker_preset = _write_tracker_preset(tmp_path)
    profile_path = _write_domain_profile(tmp_path, tracker_preset)

    result = build_domain_configs(profile_path, overwrite=True)

    assert result["domain"] == "fixture domain"
    assert set(result["configs"]) == {
        "track_video",
        "track_dataset",
        "detection_cache",
        "compare_trackers",
    }
    compare_path = tmp_path / result["configs"]["compare_trackers"]
    cache_path = tmp_path / result["configs"]["detection_cache"]
    compare = yaml.safe_load(compare_path.read_text(encoding="utf-8"))
    cache = yaml.safe_load(cache_path.read_text(encoding="utf-8"))

    assert compare["experiment"]["split"] == "all"
    assert compare["trackers"][0]["name"] == "botsort_reid"
    assert cache["inference"]["target_class_name"] == "object"
    assert "compare-trackers" in result["commands"]["compare_trackers"]
