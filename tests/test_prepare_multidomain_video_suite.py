from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.data.prepare_multidomain_video_suite import prepare_video_suite


def _write_video(path: Path, *, frames: int, fps: float) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.full((48, 64, 3), index % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_prepare_video_suite_enforces_duration_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source, frames=92, fps=10.0)
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "minimum_duration_seconds": 30,
                "maximum_duration_seconds": 60,
                "maximum_dimension": 960,
                "videos": [
                    {
                        "video_id": "cells",
                        "domain": "medical_microscopy",
                        "source": str(source),
                        "output_name": "cells_31s.mp4",
                        "output_fps": 3,
                        "frame_count": 92,
                        "license": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = prepare_video_suite(
        config_path=config_path,
        output_dir=None,
        overwrite=False,
    )

    record = result["videos"][0]
    assert record["domain"] == "medical_microscopy"
    assert record["transform"]["scale_x"] == 1.0
    assert record["transform"]["scale_y"] == 1.0
    assert record["video"]["frame_count"] == 92
    assert 30.0 <= record["video"]["duration_seconds"] <= 60.0
    assert len(record["sha256"]) == 64
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["video_count"] == 1
    assert manifest["sample_count"] == 1
    assert manifest["samples"][0]["sample_id"] == "cells_31s"
    assert manifest["duration_contract_seconds"] == [30.0, 60.0]
