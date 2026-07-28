from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from football_tracking.experiments.experiment_config import load_compare_trackers_config
from football_tracking.experiments.experiment_runner import _load_completed_sequence
from football_tracking.tracking.sequence_runner import SequenceSource


def test_experiment_resume_loads_compatible_sequence(tmp_path: Path) -> None:
    config = load_compare_trackers_config(
        "configs/benchmarks/tracking_sportsmot_yolo26m.yaml",
        overrides={"resume": True},
    )
    assert config.resume is True
    spec = config.trackers[0]
    config = replace(
        config,
        tracks_root=tmp_path / "tracks",
        detection_cache_root=tmp_path / "cache",
    )
    source = SequenceSource(
        name="sequence_01",
        source_path=tmp_path / "sequence_01",
        source_type="mot",
        fps=30.0,
        width=640,
        height=480,
        frame_count=2,
    )
    cache_dir = config.detection_cache_root / config.split / source.name
    mot_path = config.tracks_root / spec.name / config.split / f"{source.name}.txt"
    metadata_path = mot_path.with_suffix(".metadata.json")
    mot_path.parent.mkdir(parents=True)
    mot_path.write_text(
        "1,7,10.000000,20.000000,30.000000,40.000000,0.900000,1,1.000000\n"
        "2,7,11.000000,20.000000,30.000000,40.000000,0.910000,1,1.000000\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "sequence": source.name,
                "tracker": spec.name,
                "tracker_config": str(spec.config),
                "cache_dir": str(cache_dir),
                "confidence_threshold": config.confidence_threshold,
                "frame_count": 2,
                "detection_count": 2,
                "emitted_track_count": 2,
                "tracker_seconds": 0.2,
                "frame_read_seconds": 0.1,
                "cache_read_seconds": 0.05,
                "mot_write_seconds": 0.01,
                "total_seconds": 0.36,
                "partial_sequence": False,
            }
        ),
        encoding="utf-8",
    )

    resumed = _load_completed_sequence(
        config,
        spec,
        source,
        cache_dir,
        mot_path,
        metadata_path,
    )

    assert resumed is not None
    summary, validation = resumed
    assert not validation.has_errors
    assert summary["resumed"] is True
    assert summary["track_ids"] == [7]
    assert summary["mot_write_seconds"] == 0.01
