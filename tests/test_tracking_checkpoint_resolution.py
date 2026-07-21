from __future__ import annotations

from pathlib import Path

import pytest

from football_tracking.tracking.checkpoint_resolver import (
    CheckpointResolutionError,
    compute_checkpoint_hash,
    identify_checkpoint_type,
    resolve_detector_checkpoint,
)


def test_checkpoint_hash_and_type(tmp_path) -> None:
    checkpoint = tmp_path / "yolov8m_sportsmot_smoke_best.pt"
    checkpoint.write_bytes(b"weights")

    assert compute_checkpoint_hash(checkpoint)
    assert identify_checkpoint_type(checkpoint) == "smoke"


def test_checkpoint_type_distinguishes_yoloe_from_coco_yolo() -> None:
    assert identify_checkpoint_type("yoloe-26s-seg.pt") == "pretrained_open_vocab"
    assert identify_checkpoint_type("yolo26s.pt") == "pretrained_coco"


def test_configured_pretrained_checkpoint_is_not_reported_as_fallback(tmp_path) -> None:
    resolved = resolve_detector_checkpoint(
        {
            "checkpoint": "yolo26n.pt",
            "allow_pretrained_fallback": True,
        },
        tmp_path,
    )

    assert resolved.checkpoint == "yolo26n.pt"
    assert resolved.source == "config.checkpoint"
    assert resolved.fallback_used is False
    assert resolved.warnings == []


def test_resolver_uses_smoke_only_when_allowed(tmp_path) -> None:
    project_root = tmp_path
    smoke_dir = project_root / "models" / "detector"
    smoke_dir.mkdir(parents=True)
    smoke = smoke_dir / "yolov8m_sportsmot_smoke_best.pt"
    smoke.write_bytes(b"weights")

    resolved = resolve_detector_checkpoint(
        {
            "checkpoint": "models/detector/missing.pt",
            "allow_smoke_checkpoint": True,
        },
        project_root,
    )

    assert Path(resolved.checkpoint) == smoke
    assert resolved.checkpoint_type == "smoke"
    assert resolved.smoke_only is True


def test_resolver_fails_when_fallback_disabled(tmp_path) -> None:
    with pytest.raises(CheckpointResolutionError):
        resolve_detector_checkpoint(
            {"checkpoint": "models/detector/missing.pt"},
            tmp_path,
        )
