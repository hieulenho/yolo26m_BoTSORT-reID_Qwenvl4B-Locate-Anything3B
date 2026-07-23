import json
from pathlib import Path

from football_tracking.benchmarking.completion_gate import build_completion_gate


def _json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_completion_gate_requires_every_independent_evidence(tmp_path: Path) -> None:
    dataset = _json(
        tmp_path / "dataset.json",
        {"source_count": 2, "ready_count": 2},
    )
    semantic_gt = _json(tmp_path / "semantic_gt.json", {"status": "ready"})
    review = _json(tmp_path / "review.json", {"status": "ready"})
    agreement = _json(tmp_path / "agreement.json", {"status": "agreed"})
    realtime = _json(
        tmp_path / "realtime.json",
        {
            "runs": [
                {
                    "name": f"{profile}_r{repeat}",
                    "source_kind": "webcam",
                    "frames_processed": 900,
                    "p95_latency_ms": 20,
                    "processing_fps": 30,
                    "drop_rate": 0,
                }
                for profile in (
                    "bounded_tracking_only",
                    "bounded_semantic_deferred",
                    "no_drop_semantic_deferred",
                )
                for repeat in range(1, 4)
            ]
        },
    )
    metric_row = {
        "semantic_accuracy": 0.8,
        "semantic_macro_f1": 0.7,
        "fine_label_accuracy": 0.6,
        "unknown_rejection_f1": 0.5,
        "hallucination_rate": 0.1,
        "semantic_cold_seconds": 2.0,
        "sequential_peak_gib": 4.0,
    }
    comparison = _json(
        tmp_path / "comparison.json",
        {"pipelines": [metric_row, metric_row, metric_row]},
    )

    result = build_completion_gate(
        dataset_readiness=dataset,
        semantic_gt_status=semantic_gt,
        idsw_review_status=review,
        idsw_agreement=agreement,
        realtime_report=realtime,
        semantic_comparison=comparison,
        output_dir=tmp_path / "out",
    )

    assert result["complete"] is True
    assert result["passed_check_count"] == 6

    blocked = build_completion_gate(
        dataset_readiness=dataset,
        semantic_gt_status=semantic_gt,
        idsw_review_status=review,
        idsw_agreement=tmp_path / "missing.json",
        realtime_report=realtime,
        semantic_comparison=comparison,
        output_dir=tmp_path / "blocked",
    )
    assert blocked["complete"] is False
    assert blocked["passed_check_count"] == 5


def test_completion_gate_rejects_file_replay_as_physical_realtime(
    tmp_path: Path,
) -> None:
    dataset = _json(tmp_path / "dataset.json", {"source_count": 1, "ready_count": 1})
    semantic_gt = _json(tmp_path / "semantic_gt.json", {"status": "ready"})
    review = _json(tmp_path / "review.json", {"status": "ready"})
    agreement = _json(tmp_path / "agreement.json", {"status": "agreed"})
    realtime = _json(
        tmp_path / "realtime.json",
        {
            "runs": [
                {
                    "source_kind": "file",
                    "frames_processed": 900,
                    "p95_latency_ms": 20,
                    "processing_fps": 30,
                    "drop_rate": 0,
                }
                for _ in range(3)
            ]
        },
    )
    metric_row = {
        "semantic_accuracy": 0.8,
        "semantic_macro_f1": 0.7,
        "fine_label_accuracy": 0.6,
        "unknown_rejection_f1": 0.5,
        "hallucination_rate": 0.1,
        "semantic_cold_seconds": 2.0,
        "sequential_peak_gib": 4.0,
    }
    comparison = _json(
        tmp_path / "comparison.json",
        {"pipelines": [metric_row, metric_row, metric_row]},
    )

    result = build_completion_gate(
        dataset_readiness=dataset,
        semantic_gt_status=semantic_gt,
        idsw_review_status=review,
        idsw_agreement=agreement,
        realtime_report=realtime,
        semantic_comparison=comparison,
        output_dir=tmp_path / "out",
    )

    assert result["complete"] is False
    realtime_check = next(
        row for row in result["checks"] if row["name"] == "physical_realtime"
    )
    assert realtime_check["passed"] is False


def test_completion_gate_uses_required_dataset_sources(tmp_path: Path) -> None:
    dataset = _json(
        tmp_path / "dataset.json",
        {
            "source_count": 3,
            "ready_count": 1,
            "required_source_count": 1,
            "required_ready_count": 1,
            "optional_source_count": 2,
            "optional_ready_count": 0,
        },
    )
    semantic_gt = _json(tmp_path / "semantic_gt.json", {"status": "ready"})
    review = _json(tmp_path / "review.json", {"status": "ready"})
    agreement = _json(tmp_path / "agreement.json", {"status": "agreed"})
    realtime = _json(
        tmp_path / "realtime.json",
        {
            "runs": [
                {
                    "name": f"{profile}_r{repeat}",
                    "source_kind": "rtsp",
                    "frames_processed": 150,
                    "p95_latency_ms": 20,
                    "processing_fps": 30,
                    "drop_rate": 0,
                }
                for profile in (
                    "bounded_tracking_only",
                    "bounded_semantic_deferred",
                    "no_drop_semantic_deferred",
                )
                for repeat in range(1, 4)
            ]
        },
    )
    metric_row = {
        "semantic_accuracy": 0.8,
        "semantic_macro_f1": 0.7,
        "fine_label_accuracy": 0.6,
        "unknown_rejection_f1": 0.5,
        "hallucination_rate": 0.1,
        "semantic_cold_seconds": 2.0,
        "sequential_peak_gib": 4.0,
    }
    comparison = _json(
        tmp_path / "comparison.json",
        {"pipelines": [metric_row, metric_row, metric_row]},
    )

    result = build_completion_gate(
        dataset_readiness=dataset,
        semantic_gt_status=semantic_gt,
        idsw_review_status=review,
        idsw_agreement=agreement,
        realtime_report=realtime,
        semantic_comparison=comparison,
        output_dir=tmp_path / "out",
    )

    dataset_check = next(
        row for row in result["checks"] if row["name"] == "official_multidomain_gt"
    )
    assert dataset_check["passed"] is True
    assert "1/1 required" in dataset_check["detail"]
