from __future__ import annotations

import json
from pathlib import Path

from football_tracking.adaptive_tracking.run_report import build_adaptive_run_report


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_adaptive_run_report_separates_coverage_from_gt_accuracy(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write(
        run / "discovery" / "scene_discovery.json",
        {"domain": {"name": "traffic"}, "objects": [{"canonical_name": "car"}]},
    )
    _write(run / "plan" / "detector_route.json", {"route_name": "coco_pretrained"})
    tracking = _write(
        tmp_path / "tracked.metadata.json",
        {
            "tracker": "ocsort",
            "frame_count": 10,
            "detection_count": 20,
            "unique_track_count": 3,
            "timing": {"end_to_end_fps": 30.0},
            "errors": [],
        },
    )
    _write(
        run / "fused_track_semantics.json",
        {"summary": {"accepted_count": 2, "coverage": 2 / 3}},
    )
    semantic = _write(
        tmp_path / "semantic.metadata.json",
        {
            "timing": {"render_fps": 100.0},
            "semantics_summary": {"track_coverage": 2 / 3},
            "video": {"rendered_frame_count": 10},
        },
    )

    result = build_adaptive_run_report(
        run_root=run,
        tracking_metadata=tracking,
        semantic_metadata=semantic,
        output_path=run / "adaptive_run_report.json",
    )

    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["scene"]["domain"] == "traffic"
    assert report["evaluation_scope"]["ground_truth_accuracy_available"] is False
    assert report["semantic_fusion"]["coverage"] == 2 / 3
