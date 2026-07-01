from __future__ import annotations

from pathlib import Path

import numpy as np

from football_tracking.experiments.experiment_config import load_compare_trackers_config


def _write_sequence(root: Path) -> None:
    import cv2  # type: ignore[import-not-found]

    seq_dir = root / "data" / "mot" / "sportsmot_football" / "val" / "seq"
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "000001.jpg"), np.zeros((10, 12, 3), dtype=np.uint8))
    (seq_dir / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                "name=seq",
                "imDir=img1",
                "frameRate=25",
                "seqLength=1",
                "imWidth=12",
                "imHeight=10",
                "imExt=.jpg",
            ]
        ),
        encoding="utf-8",
    )
    seqmap = root / "data" / "mot" / "sportsmot_football" / "seqmaps" / "val.txt"
    seqmap.parent.mkdir(parents=True)
    seqmap.write_text("name\nseq\n", encoding="utf-8")


def _write_config(root: Path) -> Path:
    sort_config = root / "sort.yaml"
    sort_config.write_text(
        "tracker:\n  max_age: 3\n  min_hits: 1\n  iou_threshold: 0.3\n",
        encoding="utf-8",
    )
    config = root / "compare.yaml"
    config.write_text(
        f"""
experiment:
  name: fixture_compare
  seed: 42
  split: val
dataset:
  mot_root: {(root / "data" / "mot" / "sportsmot_football").as_posix()}
  seqmap: {(root / "data" / "mot" / "sportsmot_football" / "seqmaps" / "val.txt").as_posix()}
detections:
  cache_config: {(root / "cache.yaml").as_posix()}
  cache_root: {(root / "outputs" / "detections" / "cache").as_posix()}
  confidence_threshold: 0.2
trackers:
  - name: sort
    config: {sort_config.as_posix()}
evaluation:
  trackeval_config: null
  metrics: [HOTA, CLEAR, Identity]
benchmark:
  render_video: false
output:
  root: {(root / "outputs" / "experiments").as_posix()}
  tracks_root: {(root / "outputs" / "tracks").as_posix()}
  metrics_root: {(root / "outputs" / "metrics").as_posix()}
  figures_root: {(root / "outputs" / "figures").as_posix()}
runtime:
  max_sequences: 1
  max_frames_per_sequence: 1
  overwrite: true
""".strip(),
        encoding="utf-8",
    )
    return config


def test_compare_trackers_config_loads_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _write_sequence(tmp_path)

    config = load_compare_trackers_config(_write_config(tmp_path))

    assert config.experiment_name == "fixture_compare"
    assert config.confidence_threshold == 0.2
    assert config.trackers[0].name == "sort"
