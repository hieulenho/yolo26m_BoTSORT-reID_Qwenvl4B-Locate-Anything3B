from __future__ import annotations

import json
from pathlib import Path

from football_tracking.benchmarking.multidomain_report import (
    build_multidomain_trial_report,
)


def test_multidomain_report_separates_video_labels_from_track_accuracy(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "samples_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "bird",
                        "path": str(tmp_path / "bird.webm"),
                        "video": {
                            "fps": 10.0,
                            "frame_count": 30,
                            "duration_seconds": 3.0,
                            "width": 640,
                            "height": 360,
                        },
                        "source_page": "https://example.test/bird",
                        "license": "CC BY",
                        "ground_truth": {
                            "domain": "wildlife",
                            "base_classes": ["bird"],
                            "fine_labels": ["common kingfisher"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    run = tmp_path / "runs" / "bird"
    run.mkdir(parents=True)
    mot_path = tmp_path / "bird.txt"
    mot_path.write_text(
        "1,1,0,0,10,10,0.9,1,1\n2,1,0,0,10,10,0.9,1,1\n"
        "10,2,0,0,10,10,0.9,1,1\n",
        encoding="utf-8",
    )
    (run / "adaptive_run_report.json").write_text(
        json.dumps(
            {
                "scene": {
                    "domain": "wildlife",
                    "objects": [{"canonical_name": "kingfisher"}],
                    "inference_seconds": 4.0,
                },
                "tracking": {
                    "output_mot": str(mot_path),
                    "frame_count": 30,
                    "detection_count": 25,
                    "unique_track_count": 1,
                    "timing": {"steady_state_fps": 22.0, "cold_start_fps": 15.0},
                },
                "qwen_track_semantics": {
                    "timing": {"inference_seconds": 10.0},
                    "cuda_memory": {"peak_allocated_bytes": 4 * 1024**3},
                },
                "locateanything_verification": {},
                "semantic_fusion": {"track_count": 1, "coverage": 1.0, "fine_coverage": 1.0},
                "render": {"semantics": {"track_coverage": 1.0}},
                "hardware": {"gpu_name": "fixture GPU"},
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    (run / "fused_track_semantics.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "track_id": 1,
                        "class_label": "bird",
                        "accepted": True,
                        "fine_label": "common kingfisher",
                        "fine_accepted": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_multidomain_trial_report(
        manifest,
        tmp_path / "runs",
        tmp_path / "report",
        overwrite=True,
    )
    payload = json.loads(Path(result["paths"]["json"]).read_text(encoding="utf-8"))

    assert payload["summary"]["domain_family_accuracy"] == 1.0
    assert payload["samples"][0]["video_level_class_recall"] == 1.0
    assert payload["samples"][0]["accepted_fine_labels"] == ["common kingfisher"]
    assert payload["samples"][0]["semantic_track_accuracy"] is None
    assert payload["samples"][0]["accuracy_status"] == ("requires_human_per_track_ground_truth")
    assert payload["samples"][0]["source_duration_seconds"] == 3.0
    assert payload["samples"][0]["short_track_ratio"] == 1.0
    assert payload["samples"][0]["tracking_stability_proxy"]["scope"] == (
        "prediction_only_proxy_not_ground_truth_idsw"
    )
