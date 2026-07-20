from __future__ import annotations

import json
from pathlib import Path

import yaml

from football_tracking.benchmarking.semantic_ablation import (
    build_semantic_ablation_report,
)


def test_semantic_ablation_consolidates_coverage_timing_and_gt(tmp_path: Path) -> None:
    answer = tmp_path / "answer.json"
    answer.write_text(
        json.dumps(
            {
                "quantization": "4bit",
                "batch_count": 2,
                "image_count": 6,
                "coverage": {"expected_track_count": 4, "predicted_track_count": 3},
                "timing": {"model_load_seconds": 1.0, "inference_seconds": 2.0},
                "cuda_memory": {
                    "peak_allocated_bytes": 1024**3,
                    "peak_reserved_bytes": 2 * 1024**3,
                },
            }
        ),
        encoding="utf-8",
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "summary": {
                    "semantic_track_accuracy": 0.5,
                    "semantic_macro_f1": 0.4,
                    "semantic_selective_accuracy": 2 / 3,
                }
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "output"),
                "runs": [
                    {
                        "name": "run_a",
                        "answer": str(answer),
                        "evaluation": str(evaluation),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_semantic_ablation_report(config, overwrite=True)

    row = result["runs"][0]
    assert row["model_coverage"] == 0.75
    assert row["semantic_accuracy_gt"] == 0.5
    assert row["peak_allocated_gib"] == 1.0
    assert Path(result["paths"]["json"]).is_file()
    assert Path(result["figures"][0]).is_file()
