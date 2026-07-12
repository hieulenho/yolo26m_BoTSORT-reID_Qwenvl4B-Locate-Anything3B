from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_pipeline_results_separates_model_and_render_fallbacks(
    tmp_path: Path,
) -> None:
    annotation_csv = tmp_path / "annotations.csv"
    annotation_csv.write_text(
        "\n".join(
            [
                "sequence_name,track_id,team_label,role_label",
                "video_1,1,light_blue,player",
                "video_1,2,dark_blue,player",
            ]
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "render_predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "track_predictions": [
                    {
                        "sequence_name": "video_1",
                        "track_id": 1,
                        "status": "resolved",
                        "team_label": "light_blue",
                        "role_label": "player",
                        "metadata": {"source_type": "qwen_structured_prediction"},
                    },
                    {
                        "sequence_name": "video_1",
                        "track_id": 2,
                        "status": "resolved",
                        "team_label": "dark_blue",
                        "role_label": "player",
                        "metadata": {
                            "source_type": "reviewed_annotation_csv",
                            "not_model_claim": True,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "eval"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_pipeline_results.py",
            "--sequence-name",
            "video_1",
            "--annotation-csv",
            str(annotation_csv),
            "--pipeline-a",
            str(predictions),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Pipeline A" in completed.stdout
    result = json.loads((output_dir / "evaluation_results.json").read_text())
    pipeline_a = result["pipeline_results"]["A"]
    assert pipeline_a["model_claim_metrics"]["covered_tracks"] == 1
    assert pipeline_a["model_claim_metrics"]["team_correct"] == 1
    assert pipeline_a["render_output_metrics"]["covered_tracks"] == 2
    assert pipeline_a["render_output_metrics"]["team_correct"] == 2
