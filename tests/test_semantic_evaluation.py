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
    route = _write(
        tmp_path / "route.json",
        {"route_name": "coco_pretrained"},
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
                            "route": str(route),
                            "semantics": str(semantics),
                            "run_report": str(report),
                        },
                        "ground_truth": {
                            "domain": "traffic",
                            "detector_route": "coco_pretrained",
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
    assert result["summary"]["router_accuracy"] == 1.0
    assert result["summary"]["router_gt_sample_count"] == 1
    assert result["summary"]["class_f1"] == 1.0
    assert result["summary"]["semantic_track_accuracy"] == 0.5
    assert result["summary"]["semantic_coverage"] == 0.5
    assert result["summary"]["semantic_selective_accuracy"] == 1.0
    assert result["summary"]["unknown_rejection_f1"] is None
    summary_path = Path(result["paths"]["summary_json"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["metric_scope"]["accuracy_source"] == (
        "ground-truth manifest (review provenance not required)"
    )


def test_semantic_evaluation_preserves_official_gt_provenance(tmp_path: Path) -> None:
    discovery = _write(
        tmp_path / "discovery.json",
        {"domain": "wildlife", "objects": [{"canonical_name": "zebra", "action": "track"}]},
    )
    semantics = _write(
        tmp_path / "semantics.json",
        {"tracks": [{"track_id": 1, "class_label": "zebra", "accepted": True}]},
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "ground_truth_source": "official MOT boxes matched by framewise IoU",
                "samples": [
                    {
                        "sample_id": "official_zebra",
                        "artifacts": {
                            "discovery": str(discovery),
                            "semantics": str(semantics),
                        },
                        "ground_truth": {
                            "domain": "wildlife",
                            "objects": [{"canonical_name": "zebra", "action": "track"}],
                            "tracks": [{"track_id": 1, "class_label": "zebra"}],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_semantic_manifest(manifest, tmp_path / "output", overwrite=True)

    payload = json.loads(Path(result["paths"]["summary_json"]).read_text(encoding="utf-8"))
    assert payload["metric_scope"]["accuracy_source"] == (
        "official MOT boxes matched by framewise IoU"
    )


def test_semantic_evaluation_matches_specific_scene_to_domain_family(
    tmp_path: Path,
) -> None:
    discovery = _write(
        tmp_path / "discovery.json",
        {
            "domain": {"name": "urban_intersection"},
            "objects": [{"canonical_name": "car", "action": "track"}],
        },
    )
    semantics = _write(
        tmp_path / "semantics.json",
        {"tracks": [{"track_id": 1, "class_label": "car", "accepted": True}]},
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "samples": [
                    {
                        "sample_id": "traffic_family",
                        "artifacts": {
                            "discovery": str(discovery),
                            "semantics": str(semantics),
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

    assert result["summary"]["domain_accuracy"] == 1.0
    row = json.loads(
        Path(result["paths"]["summary_json"]).read_text(encoding="utf-8")
    )["per_sample"][0]
    assert row["predicted_domain"] == "traffic"
    assert row["predicted_domain_raw"] == "urban intersection"


def test_semantic_evaluation_reports_fine_grained_accuracy(tmp_path: Path) -> None:
    discovery = _write(
        tmp_path / "discovery.json",
        {"domain": "wildlife", "objects": [{"canonical_name": "bird", "action": "track"}]},
    )
    semantics = _write(
        tmp_path / "semantics.json",
        {
            "tracks": [
                {
                    "track_id": 1,
                    "class_label": "bird",
                    "accepted": True,
                    "fine_label": "common kingfisher",
                    "fine_accepted": True,
                },
                {
                    "track_id": 2,
                    "class_label": "bird",
                    "accepted": True,
                    "fine_label": "unknown",
                    "fine_accepted": False,
                    "fine_label_scores": {"common kingfisher": 0.8},
                },
            ]
        },
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "samples": [
                    {
                        "sample_id": "birds",
                        "artifacts": {
                            "discovery": str(discovery),
                            "semantics": str(semantics),
                        },
                        "ground_truth": {
                            "domain": "wildlife",
                            "objects": [{"canonical_name": "bird", "action": "track"}],
                            "tracks": [
                                {
                                    "track_id": 1,
                                    "class_label": "bird",
                                    "fine_label": "common kingfisher",
                                },
                                {
                                    "track_id": 2,
                                    "class_label": "bird",
                                    "fine_label": "common kingfisher",
                                },
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_semantic_manifest(manifest, tmp_path / "output", overwrite=True)

    assert result["summary"]["fine_semantic_track_accuracy"] == 0.5
    assert result["summary"]["fine_semantic_coverage"] == 0.5
    assert result["summary"]["fine_semantic_selective_accuracy"] == 1.0
    assert result["summary"]["fine_candidate_accuracy"] == 1.0
    assert result["summary"]["fine_candidate_coverage"] == 1.0


def test_semantic_evaluation_reports_unknown_rejection_and_hallucination(
    tmp_path: Path,
) -> None:
    discovery = _write(
        tmp_path / "discovery.json",
        {"domain": "traffic", "objects": [{"canonical_name": "car", "action": "track"}]},
    )
    semantics = _write(
        tmp_path / "semantics.json",
        {
            "tracks": [
                {"track_id": 1, "class_label": "car", "accepted": True},
                {"track_id": 2, "class_label": "truck", "accepted": True},
                {"track_id": 3, "class_label": "car", "accepted": True},
                {"track_id": 4, "class_label": "unknown", "accepted": False},
            ]
        },
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "samples": [
                    {
                        "sample_id": "unknowns",
                        "artifacts": {
                            "discovery": str(discovery),
                            "semantics": str(semantics),
                        },
                        "ground_truth": {
                            "domain": "traffic",
                            "objects": [{"canonical_name": "car", "action": "track"}],
                            "tracks": [
                                {"track_id": 1, "class_label": "car"},
                                {"track_id": 2, "class_label": "car"},
                                {"track_id": 3, "class_label": "unknown"},
                                {"track_id": 4, "class_label": "unknown"},
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_semantic_manifest(manifest, tmp_path / "output", overwrite=True)

    summary = result["summary"]
    assert summary["semantic_hallucination_count"] == 2
    assert summary["semantic_hallucination_rate"] == 0.666667
    assert summary["unknown_rejection_precision"] == 1.0
    assert summary["unknown_rejection_recall"] == 0.5
    assert summary["unknown_rejection_f1"] == 0.666667
    assert summary["unknown_false_accept_rate"] == 0.5
    assert summary["known_false_reject_rate"] == 0.0
    assert summary["per_domain"]["traffic"]["track_count"] == 4


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
