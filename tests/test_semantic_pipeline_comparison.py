from __future__ import annotations

import json
from pathlib import Path

import yaml

from football_tracking.benchmarking.semantic_pipeline_comparison import (
    build_semantic_pipeline_comparison,
)


def test_semantic_pipeline_comparison_computes_effective_fps(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "summary": {
                    "semantic_track_accuracy": 0.8,
                    "semantic_macro_f1": 0.75,
                    "semantic_coverage": 0.8,
                    "semantic_selective_accuracy": 1.0,
                    "semantic_gt_track_count": 10,
                    "semantic_accepted_track_count": 8,
                    "performance_means": {
                        "tracking_end_to_end_fps": 20.0,
                        "qwen_model_load_seconds": 2.0,
                        "qwen_inference_seconds": 3.0,
                        "locate_model_load_seconds": None,
                        "locate_inference_seconds": None,
                        "qwen_peak_allocated_bytes": 2 * 1024**3,
                        "locate_peak_allocated_bytes": None,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "comparison.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "output"),
                "frame_count": 100,
                "pipelines": [{"id": "A", "evaluation": str(evaluation)}],
            }
        ),
        encoding="utf-8",
    )

    result = build_semantic_pipeline_comparison(config, overwrite=True)

    row = result["pipelines"][0]
    assert row["semantic_cold_seconds"] == 5.0
    assert row["effective_cold_fps"] == 10.0
    assert row["sequential_peak_gib"] == 2.0
    assert Path(result["figures"][0]).is_file()
