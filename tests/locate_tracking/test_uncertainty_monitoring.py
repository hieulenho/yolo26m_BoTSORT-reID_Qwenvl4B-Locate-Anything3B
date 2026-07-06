from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.monitoring.confidence_signals import (
    detect_confidence_signals,
)
from football_tracking.locate_tracking.monitoring.config import (
    load_uncertainty_pipeline_config,
)
from football_tracking.locate_tracking.monitoring.motion_signals import detect_motion_signals
from football_tracking.locate_tracking.monitoring.presence_signals import (
    detect_presence_signals,
)
from football_tracking.locate_tracking.monitoring.schemas import MonitoringConfig
from football_tracking.locate_tracking.monitoring.service import (
    analyze_and_plan_event_grounding,
)
from football_tracking.locate_tracking.monitoring.target_observer import (
    build_target_observation_timeline_from_paths,
)
from football_tracking.locate_tracking.semantic_memory.aggregator import build_semantic_memory
from football_tracking.locate_tracking.semantic_memory.schemas import SemanticMemoryConfig
from football_tracking.locate_tracking.semantic_memory.serialization import save_semantic_memory
from tests.locate_tracking.appearance_test_utils import tiny_video
from tests.locate_tracking.semantic_test_utils import resolved_frame


def _write_tracks(path: Path, *, confidence: bool = False, jump: bool = False) -> Path:
    rows: list[str] = []
    for frame in range(1, 11):
        if frame <= 4 or frame >= 8:
            x = 10 if not jump or frame < 8 else 90
            conf = "0.9" if confidence else "-1"
            rows.append(f"{frame},7,{x},10,10,10,{conf},1,1")
        rows.append(f"{frame},11,13,10,10,10,-1,1,1")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _semantic(path: Path) -> Path:
    memory = build_semantic_memory(
        query="player in red",
        frame_resolutions=(resolved_frame(1, 7), resolved_frame(4, 7)),
        config=SemanticMemoryConfig(min_usable_frames=1, min_support_frames=1),
    )
    return save_semantic_memory(memory, path, overwrite=True)


def _appearance(path: Path, *, score: float = 0.2) -> Path:
    path.write_text(
        json.dumps(
            {
                "query": "player in red",
                "source_video": "video.avi",
                "tracks_path": "tracks.txt",
                "semantic_memory_reference": "semantic_memory.json",
                "status": "resolved",
                "prototypes": [],
                "candidate_scores": [
                    {
                        "raw_track_id": 7,
                        "prototype_similarity": score,
                        "internal_consistency": 0.9,
                        "appearance_score": score,
                        "sample_count": 2,
                        "evidence_status": "weak",
                        "decision_reason": "test",
                    }
                ],
                "runtime_info": {"backend_name": "mock", "model_id": "mock"},
            }
        ),
        encoding="utf-8",
    )
    return path


def _fusion(path: Path, *, margin: float = 0.01) -> Path:
    top = 0.55
    path.write_text(
        json.dumps(
            {
                "query": "player in red",
                "status": "resolved",
                "selected_track_id": 7,
                "selected_track_ids": [7],
                "candidate_scores": [
                    {
                        "raw_track_id": 7,
                        "semantic_score": 0.8,
                        "appearance_score": 0.2,
                        "fused_score": top,
                        "appearance_status": "weak",
                    },
                    {
                        "raw_track_id": 11,
                        "semantic_score": 0.78,
                        "appearance_score": 0.2,
                        "fused_score": top - margin,
                        "appearance_status": "weak",
                    },
                ],
                "score_margin": margin,
                "decision_reason": "test",
                "semantic_memory_reference": "semantic_memory.json",
                "appearance_scores_reference": "appearance.json",
                "config": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_target_observer_keeps_unavailable_confidence_as_none(tmp_path: Path) -> None:
    tracks = _write_tracks(tmp_path / "tracks.txt", confidence=False)
    semantic = _semantic(tmp_path / "semantic_memory.json")
    timeline = build_target_observation_timeline_from_paths(
        tracks_path=tracks,
        semantic_memory_path=semantic,
        current_raw_track_id=7,
    )

    assert timeline.current_raw_track_id == 7
    assert timeline.frame(1).tracking_confidence is None  # type: ignore[union-attr]
    assert timeline.frame(5).target_present is False  # type: ignore[union-attr]

    signals = detect_confidence_signals(timeline, MonitoringConfig())
    assert signals[0].data_available is False
    assert signals[0].triggered is False


def test_presence_and_motion_signals_are_independent(tmp_path: Path) -> None:
    tracks = _write_tracks(tmp_path / "tracks.txt", confidence=True, jump=True)
    semantic = _semantic(tmp_path / "semantic_memory.json")
    timeline = build_target_observation_timeline_from_paths(
        tracks_path=tracks,
        semantic_memory_path=semantic,
        current_raw_track_id=7,
    )
    config = MonitoringConfig(presence_warning_absent_frames=2, motion_jump_threshold=0.2)

    presence = detect_presence_signals(timeline, config)
    motion = detect_motion_signals(timeline, config)

    assert any(signal.triggered for signal in presence)
    assert any(signal.triggered for signal in motion)


def test_uncertainty_service_writes_events_and_plan_without_recovered_id(tmp_path: Path) -> None:
    source_video = tiny_video(tmp_path / "video.avi", frame_count=10)
    tracks = _write_tracks(tmp_path / "tracks.txt", confidence=True, jump=False)
    semantic = _semantic(tmp_path / "semantic_memory.json")
    appearance = _appearance(tmp_path / "appearance.json", score=0.2)
    fusion = _fusion(tmp_path / "fusion.json", margin=0.01)
    config = load_uncertainty_pipeline_config(
        "configs/locate_tracking/uncertainty_monitoring.yaml",
        overrides={"output_dir": tmp_path / "out", "overwrite": True},
    )
    monitoring_config = MonitoringConfig(
        presence_warning_absent_frames=2,
        presence_critical_absent_frames=8,
        appearance_drift_threshold=0.35,
        semantic_margin_threshold=0.08,
        neighbor_distance_threshold=0.1,
        staleness_warning_frames=2,
    )

    run = analyze_and_plan_event_grounding(
        source_video=source_video,
        tracks_path=tracks,
        semantic_memory_path=semantic,
        appearance_result_path=appearance,
        fusion_result_path=fusion,
        output_dir=config.output_dir,
        monitoring_config=monitoring_config,
        scheduler_config=config.scheduler,
        current_raw_track_id=7,
        overwrite=True,
    )

    event_types = {event.event_type for event in run.events}
    assert "target_absent" in event_types
    assert "appearance_drift" in event_types
    assert run.paths["events_jsonl"].is_file()
    assert run.paths["grounding_plan_json"].is_file()
    assert run.paths["summary_md"].is_file()
    serialized = json.dumps(run.to_dict())
    assert "recovered_track_id" not in serialized
