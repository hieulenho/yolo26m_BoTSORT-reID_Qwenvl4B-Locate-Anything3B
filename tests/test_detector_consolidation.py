from __future__ import annotations

import json
from pathlib import Path

import yaml

from football_tracking.benchmarking.detector_consolidation import (
    consolidate_detector_benchmark,
)


def test_detector_consolidation_combines_accuracy_and_timing(tmp_path: Path) -> None:
    accuracy = tmp_path / "accuracy.json"
    timing = tmp_path / "timing.json"
    accuracy.write_text(
        json.dumps(
            {
                "dataset": {"data_yaml": "dataset.yaml", "split": "val"},
                "model": {"weights": "model.pt"},
                "inference": {"imgsz": 640},
                "metrics": {
                    "precision": 0.8,
                    "recall": 0.7,
                    "map50": 0.75,
                    "map50_95": 0.5,
                    "map75": 0.6,
                },
            }
        ),
        encoding="utf-8",
    )
    timing.write_text(
        json.dumps(
            {
                "dataset": {"data_yaml": "dataset.yaml", "split": "val"},
                "model": {"weights": "model.pt"},
                "inference": {"imgsz": 640},
                "timing": {"detector_fps": 50.0, "end_to_end_fps": 40.0},
                "counts": {"image_count": 10},
                "runtime": {"gpu_name": "test gpu"},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "expected": {
                    "split": "val",
                    "imgsz": 640,
                    "timed_image_count": 10,
                    "gpu_name": "test gpu",
                },
                "sources": [
                    {
                        "name": "model",
                        "accuracy": str(accuracy),
                        "timing": str(timing),
                    }
                ],
                "output": {"root": str(tmp_path / "output")},
            }
        ),
        encoding="utf-8",
    )

    result = consolidate_detector_benchmark(config, overwrite=True)

    assert result["status"] == "ok"
    summary = json.loads(Path(result["paths"]["summary_json"]).read_text())
    assert summary["rows"][0]["map50_95"] == 0.5
    assert summary["rows"][0]["detector_fps"] == 50.0
