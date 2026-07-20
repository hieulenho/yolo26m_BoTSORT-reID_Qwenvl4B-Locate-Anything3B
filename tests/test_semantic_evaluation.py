from __future__ import annotations

import json
from pathlib import Path

import yaml

from football_tracking.benchmarking.semantic_evaluation import (
    evaluate_semantic_manifest,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_semantic_evaluation_uses_human_gt_and_penalizes_unknown(tmp_path: Path) -> None:
    discovery = _write(
        tmp_path / "discovery.json",
        {
            "domain": "traffic",
            "objects": [
                {"canonical_name": "car", "action": "track"},
                {"canonical_name": "road", "action": "context"},
            ],
        },
    )
    semantics = _write(
        tmp_path / "semantics.json",
        {
            "tracks": [
                {"track_id": 1, "class_label": "car", "accepted": True},
                {"track_id": 2, "class_label": "unknown", "accepted": False},
            ]
        },
    )
    report = _write(
        tmp_path / "report.json",
        {
            "tracking": {"timing": {"end_to_end_fps": 30.0}},
            "qwen_track_semantics": {
                "timing": {"inference_seconds": 2.0},
                "cuda_memory": {"peak_allocated_bytes": 100},
            },
        },
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "samples": [
                    {
                        "sample_id": "traffic_1",
                        "artifacts": {
                            "discovery": str(discovery),
                            "semantics": str(semantics),
                            "run_report": str(report),
                        },
                        "ground_truth": {
                            "domain": "traffic",
                            "objects": [
                                {"canonical_name": "car", "action": "track"},
                                {"canonical_name": "road", "action": "context"},
                            ],
                            "tracks": [
                                {"track_id": 1, "class_label": "car"},
                                {"track_id": 2, "class_label": "car"},
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_semantic_manifest(
        manifest,
        tmp_path / "output",
        overwrite=True,
    )

    assert result["summary"]["domain_accuracy"] == 1.0
    assert result["summary"]["class_f1"] == 1.0
    assert result["summary"]["semantic_track_accuracy"] == 0.5
    assert result["summary"]["semantic_coverage"] == 0.5
    assert result["summary"]["semantic_selective_accuracy"] == 1.0


def test_semantic_evaluation_reads_direct_performance_artifacts(tmp_path: Path) -> None:
    discovery = _write(
        tmp_path / "discovery.json",
        {"domain": "traffic", "objects": [{"canonical_name": "car", "action": "track"}]},
    )
    semantics = _write(
        tmp_path / "semantics.json",
        {"tracks": [{"track_id": 1, "class_label": "car", "accepted": True}]},
    )
    tracking = _write(
        tmp_path / "tracking.json",
        {"timing": {"end_to_end_fps": 24.0}},
    )
    qwen = _write(
        tmp_path / "qwen.json",
        {
            "timing": {"inference_seconds": 3.0},
            "cuda_memory": {"peak_allocated_bytes": 200},
        },
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "samples": [
                    {
                        "sample_id": "traffic_direct",
                        "artifacts": {
                            "discovery": str(discovery),
                            "semantics": str(semantics),
                            "tracking_metadata": str(tracking),
                            "qwen_answer": str(qwen),
                        },
                        "ground_truth": {
                            "domain": "traffic",
                            "objects": [{"canonical_name": "car", "action": "track"}],
                            "tracks": [{"track_id": 1, "class_label": "car"}],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_semantic_manifest(manifest, tmp_path / "output", overwrite=True)

    performance = result["summary"]["performance_means"]
    assert performance["tracking_end_to_end_fps"] == 24.0
    assert performance["qwen_inference_seconds"] == 3.0
    assert performance["qwen_peak_allocated_bytes"] == 200.0

    rejected = _write(
        tmp_path / "rejected.json",
        {"tracks": [{"track_id": 1, "class_label": "unknown", "accepted": False}]},
    )
    overridden = evaluate_semantic_manifest(
        manifest,
        tmp_path / "overridden",
        artifact_overrides={"semantics": rejected, "qwen_answer": None},
        overwrite=True,
    )
    assert overridden["summary"]["semantic_track_accuracy"] == 0.0
    assert overridden["summary"]["performance_means"]["qwen_inference_seconds"] is None
