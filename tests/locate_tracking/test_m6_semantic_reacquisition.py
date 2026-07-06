"""Milestone 6 semantic target reacquisition tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from football_tracking.locate_tracking.appearance.schemas import (
    AppearanceCandidateScore,
    AppearanceRuntimeInfo,
    AppearanceVerificationResult,
)
from football_tracking.locate_tracking.artifacts.mot_reader import read_mot_track_file
from football_tracking.locate_tracking.cli.init_semantic_target import run_init_semantic_target
from football_tracking.locate_tracking.cli.search_reacquisition_candidates import (
    run_search_reacquisition_candidates,
)
from football_tracking.locate_tracking.events.event_store import write_events_jsonl
from football_tracking.locate_tracking.events.schemas import UncertaintyEvent
from football_tracking.locate_tracking.grounding.schemas import (
    GroundedBox,
    GroundingRequest,
    GroundingResult,
    GroundingRuntimeInfo,
)
from football_tracking.locate_tracking.identity.schemas import IdentitySchemaError
from football_tracking.locate_tracking.identity.segment_store import (
    load_semantic_target,
    save_semantic_target,
)
from football_tracking.locate_tracking.identity.semantic_target import (
    create_initial_semantic_target,
)
from football_tracking.locate_tracking.identity.state_machine import (
    IdentityStateMachineError,
    validate_transition,
)
from football_tracking.locate_tracking.reacquisition.candidate_generator import (
    build_candidate_search_window,
    find_same_raw_id_resume,
    generate_reacquisition_candidates,
)
from football_tracking.locate_tracking.reacquisition.candidate_ranker import rank_candidates
from football_tracking.locate_tracking.reacquisition.decision_policy import (
    decide_reacquisition,
)
from football_tracking.locate_tracking.reacquisition.schemas import (
    CandidateSearchWindow,
    EvidenceScore,
    GateResult,
    ReacquisitionCandidate,
    ReacquisitionConfig,
)
from football_tracking.locate_tracking.reacquisition.service import (
    confirm_reacquisition_probation,
    run_reacquisition,
)
from football_tracking.locate_tracking.reacquisition.temporal_gate import temporal_gate


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_mot(path: Path, rows: list[tuple[int, int, float, float, float, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            f"{frame},{track},{left},{top},{width},{height},0.9"
            for frame, track, left, top, width, height in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_event(path: Path, *, event_id: str = "evt_loss") -> Path:
    event = UncertaintyEvent(
        event_id=event_id,
        event_type="target_absent",
        severity="high",
        frame_start=6,
        frame_end=8,
        trigger_frame=8,
        raw_track_id=7,
        signal_ids=("sig_absent",),
        score=0.9,
        evidence={"reason": "synthetic loss"},
    )
    return write_events_jsonl((event,), path, overwrite=True)


def _target(path: Path) -> Path:
    target = create_initial_semantic_target(
        query="the player in blue",
        raw_track_id=7,
        start_frame=1,
        semantic_target_id="target_player_blue",
    ).with_updates(last_confirmed_frame=5, last_update_frame=5)
    return save_semantic_target(target, path, overwrite=True)


def _config(**overrides: object) -> ReacquisitionConfig:
    base: dict[str, object] = {
        "pre_event_context_frames": 1,
        "post_event_context_frames": 10,
        "min_observations": 2,
        "duplicate_overlap_tolerance_frames": 1,
        "max_motion_distance_normalized": 0.60,
        "min_grounding_score": 0.10,
        "require_grounding_support": True,
        "min_final_score": 0.30,
        "ambiguity_margin": 0.05,
        "missing_evidence_policy": "ignore",
        "weights": {
            "grounding": 0.35,
            "appearance": 0.20,
            "motion": 0.20,
            "temporal": 0.15,
            "history": 0.10,
        },
        "probation_window_frames": 5,
        "probation_min_observations": 2,
        "auto_confirm": False,
    }
    base.update(overrides)
    return ReacquisitionConfig(**base)


def _candidate_rows() -> list[tuple[int, int, float, float, float, float]]:
    rows: list[tuple[int, int, float, float, float, float]] = []
    for frame in range(1, 6):
        rows.append((frame, 7, 10 + frame * 2, 10, 20, 30))
    for frame in range(9, 15):
        rows.append((frame, 42, 26 + frame, 10, 20, 30))
        rows.append((frame, 51, 260 + frame, 10, 20, 30))
    return rows


def _same_raw_rows() -> list[tuple[int, int, float, float, float, float]]:
    rows = [(frame, 7, 10 + frame * 2, 10, 20, 30) for frame in range(1, 6)]
    rows.extend((frame, 7, 25 + frame, 10, 20, 30) for frame in range(9, 13))
    rows.extend((frame, 42, 250 + frame, 10, 20, 30) for frame in range(9, 13))
    return rows


def _grounding_artifacts(root: Path) -> tuple[Path, Path]:
    result_path = root / "grounding" / "frame_000009_grounding.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = GroundingResult(
        request=GroundingRequest(
            image_path=root / "frames" / "frame_000009.jpg",
            query="the player in blue",
            backend="mock",
            model_id="mock-locate",
        ),
        image_width=640,
        image_height=480,
        boxes=(
            GroundedBox(
                label="player",
                bbox_xyxy=(35.0, 10.0, 55.0, 40.0),
                normalized_bbox=(35, 10, 55, 40),
                confidence=0.95,
                query="the player in blue",
            ),
        ),
        raw_response="mock",
        runtime_info=GroundingRuntimeInfo(backend="mock", model_id="mock-locate"),
    )
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    manifest_path = root / "grounding_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "executed_requests": [
                    {
                        "request_id": "ground_evt_loss",
                        "frames": [
                            {
                                "frame_index": 9,
                                "grounding_result_path": str(result_path),
                            }
                        ],
                    }
                ],
                "skipped_requests": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path, result_path


def _appearance_result(path: Path, tracks_path: Path) -> Path:
    result = AppearanceVerificationResult(
        query="the player in blue",
        source_video="video.mp4",
        tracks_path=str(tracks_path),
        semantic_memory_reference="semantic_memory.json",
        prototypes=(),
        candidate_scores=(
            AppearanceCandidateScore(
                raw_track_id=42,
                prototype_similarity=0.92,
                internal_consistency=0.88,
                appearance_score=0.90,
                sample_count=4,
                evidence_status="verified",
                decision_reason="mock strong match",
            ),
            AppearanceCandidateScore(
                raw_track_id=51,
                prototype_similarity=0.25,
                internal_consistency=0.80,
                appearance_score=0.30,
                sample_count=4,
                evidence_status="weak",
                decision_reason="mock weak match",
            ),
        ),
        runtime_info=AppearanceRuntimeInfo(backend_name="mock", model_id="mock-reid"),
        status="ok",
    )
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def test_identity_state_machine_and_segment_overlap() -> None:
    validate_transition("REACQUIRING", "ACTIVE")
    with pytest.raises(IdentityStateMachineError):
        validate_transition("ACTIVE", "PROBATION")

    target = create_initial_semantic_target(
        query="target",
        raw_track_id=7,
        start_frame=1,
        semantic_target_id="target_overlap",
    )
    overlapping = target.segments[0].with_updates(
        segment_id="overlap",
        raw_track_id=42,
        start_frame=2,
        end_frame=3,
    )
    with pytest.raises(IdentitySchemaError):
        target.with_updates(segments=target.segments + (overlapping,))


def test_candidate_search_window_and_same_raw_resume(tmp_path: Path) -> None:
    tracks = read_mot_track_file(_write_mot(tmp_path / "tracks.txt", _same_raw_rows()))
    event_path = _write_event(tmp_path / "events.jsonl")
    event = UncertaintyEvent.from_dict(json.loads(event_path.read_text().splitlines()[0]))
    window = build_candidate_search_window(
        event=event,
        last_confirmed_frame=5,
        total_frames=None,
        config=_config(require_grounding_support=False),
    )
    candidates = generate_reacquisition_candidates(
        observations=tracks.observations,
        search_window=window,
    )
    resumed = find_same_raw_id_resume(
        candidates=candidates,
        observations=tracks.observations,
        previous_raw_track_id=7,
        event_end_frame=8,
        min_observations=2,
    )
    assert resumed is not None
    assert resumed.raw_track_id == 7
    assert resumed.first_observed_frame == 9


def test_temporal_gate_rejects_candidate_present_during_previous_context(
    tmp_path: Path,
) -> None:
    rows = [
        (4, 42, 20, 10, 20, 30),
        (6, 42, 25, 10, 20, 30),
        (5, 7, 18, 10, 20, 30),
    ]
    observations = read_mot_track_file(_write_mot(tmp_path / "tracks.txt", rows)).observations
    candidate = ReacquisitionCandidate(
        raw_track_id=42,
        search_window=CandidateSearchWindow(
            start_frame=4,
            end_frame=12,
            last_confirmed_frame=5,
            event_start_frame=6,
            event_end_frame=8,
            pre_event_context_frames=1,
            post_event_context_frames=4,
            source_event_ids=("evt_loss",),
        ),
        first_observed_frame=4,
        last_observed_frame=6,
        observation_count=2,
    )
    gate = temporal_gate(
        candidate=candidate,
        previous_raw_track_id=7,
        all_observations=observations,
        config=_config(require_grounding_support=False),
    )
    assert not gate.passed
    assert gate.reason == "candidate_present_during_previous_target_context"


def test_reacquisition_no_commit_and_commit_preserve_read_only_inputs(
    tmp_path: Path,
) -> None:
    tracks = _write_mot(tmp_path / "tracks.txt", _candidate_rows())
    target = _target(tmp_path / "semantic_target.json")
    events = _write_event(tmp_path / "events.jsonl")
    manifest, _ = _grounding_artifacts(tmp_path)
    appearance = _appearance_result(tmp_path / "appearance_scores.json", tracks)
    mot_hash = _hash(tracks)
    appearance_hash = _hash(appearance)
    target_hash = _hash(target)

    no_commit = run_reacquisition(
        semantic_target_path=target,
        tracks_path=tracks,
        events_path=events,
        grounding_manifest_path=manifest,
        appearance_result_path=appearance,
        output_dir=tmp_path / "no_commit",
        config=_config(),
        commit=False,
        overwrite=True,
    )

    assert no_commit.decision.status == "provisional"
    assert no_commit.decision.selected_raw_track_id == 42
    assert _hash(tracks) == mot_hash
    assert _hash(appearance) == appearance_hash
    assert _hash(target) == target_hash
    assert "semantic_target_json" not in no_commit.paths

    committed = run_reacquisition(
        semantic_target_path=target,
        tracks_path=tracks,
        events_path=events,
        grounding_manifest_path=manifest,
        appearance_result_path=appearance,
        output_dir=tmp_path / "commit",
        config=_config(),
        commit=True,
        overwrite=True,
    )

    assert committed.decision.status == "provisional"
    assert committed.decision.selected_raw_track_id == 42
    assert (tmp_path / "commit" / "identity_transitions.jsonl").is_file()
    assert (tmp_path / "commit" / "semantic_target_timeline.json").is_file()
    updated = load_semantic_target(target)
    assert updated.state == "PROBATION"
    assert updated.current_raw_track_id == 42
    assert updated.segments[0].raw_track_id == 7
    assert updated.segments[0].end_frame == 5
    assert updated.segments[1].raw_track_id == 42
    assert updated.segments[1].status == "probation"
    assert _hash(tracks) == mot_hash
    assert _hash(appearance) == appearance_hash

    confirmed = confirm_reacquisition_probation(
        semantic_target_path=target,
        tracks_path=tracks,
        decision_path=tmp_path / "commit" / "reacquisition_result.json",
        output_dir=tmp_path / "confirm",
        config=_config(),
        overwrite=True,
    )
    assert confirmed.state == "ACTIVE"
    assert confirmed.current_raw_track_id == 42
    assert confirmed.segments[-1].status == "confirmed"
    assert (tmp_path / "confirm" / "semantic_target_timeline.json").is_file()


def test_same_raw_id_resume_is_decided_before_new_id_reacquisition(tmp_path: Path) -> None:
    tracks = _write_mot(tmp_path / "tracks.txt", _same_raw_rows())
    target = _target(tmp_path / "semantic_target.json")
    events = _write_event(tmp_path / "events.jsonl")

    run = run_reacquisition(
        semantic_target_path=target,
        tracks_path=tracks,
        events_path=events,
        output_dir=tmp_path / "same_raw",
        config=_config(require_grounding_support=False),
        commit=True,
        overwrite=True,
    )

    assert run.decision.status == "same_raw_id_resumed"
    assert run.decision.selected_raw_track_id == 7
    assert run.decision.selected_start_frame == 9
    updated = load_semantic_target(target)
    assert updated.state == "ACTIVE"
    assert updated.current_raw_track_id == 7
    assert len(updated.segments) == 1


def test_decision_policy_marks_close_scores_as_ambiguous() -> None:
    target = create_initial_semantic_target(
        query="target",
        raw_track_id=7,
        start_frame=1,
        semantic_target_id="target_ambiguous",
    ).with_updates(last_confirmed_frame=5)
    window = CandidateSearchWindow(
        start_frame=6,
        end_frame=20,
        last_confirmed_frame=5,
        event_start_frame=6,
        event_end_frame=8,
        pre_event_context_frames=0,
        post_event_context_frames=12,
        source_event_ids=("evt_loss",),
    )

    def candidate(track_id: int, appearance: float) -> ReacquisitionCandidate:
        return ReacquisitionCandidate(
            raw_track_id=track_id,
            search_window=window,
            first_observed_frame=9,
            last_observed_frame=14,
            observation_count=6,
            gate_results=(
                GateResult(
                    gate_name="temporal",
                    passed=True,
                    score=1.0,
                    threshold=None,
                    reason="test_pass",
                ),
            ),
            grounding_evidence=EvidenceScore("grounding", 0.80, True, "test"),
            appearance_evidence=EvidenceScore("appearance", appearance, True, "test"),
            motion_evidence=EvidenceScore("motion", 0.80, True, "test"),
            temporal_evidence=EvidenceScore("temporal", 0.80, True, "test"),
            history_evidence=EvidenceScore("history", 1.0, True, "test"),
        )

    config = _config(require_grounding_support=False, ambiguity_margin=0.20)
    ranked = rank_candidates((candidate(42, 0.90), candidate(51, 0.86)), config)
    decision = decide_reacquisition(
        target=target,
        ranked_candidates=ranked,
        all_candidates=ranked,
        config=config,
        event_ids=("evt_loss",),
    )
    assert decision.status == "ambiguous"
    assert decision.selected_raw_track_id is None


def test_m6_cli_helpers_create_real_artifacts(tmp_path: Path) -> None:
    tracks = _write_mot(tmp_path / "tracks.txt", _candidate_rows())
    events = _write_event(tmp_path / "events.jsonl")
    manifest, _ = _grounding_artifacts(tmp_path)
    appearance = _appearance_result(tmp_path / "appearance_scores.json", tracks)
    config_path = tmp_path / "reacquisition.yaml"
    config_path.write_text(
        """
reacquisition:
  pre_event_context_frames: 1
  post_event_context_frames: 10
  gates:
    min_observations: 2
    duplicate_overlap_tolerance_frames: 1
    max_motion_distance_normalized: 0.60
    min_grounding_score: 0.10
    require_grounding_support: true
  ranking:
    min_final_score: 0.30
    ambiguity_margin: 0.05
    missing_evidence_policy: ignore
    weights:
      grounding: 0.35
      appearance: 0.20
      motion: 0.20
      temporal: 0.15
      history: 0.10
  probation:
    window_frames: 5
    min_observations: 2
    auto_confirm: false
output:
  directory: unused
runtime:
  overwrite: true
""".strip(),
        encoding="utf-8",
    )
    target_path = tmp_path / "semantic_target.json"
    init = run_init_semantic_target(
        query="the player in blue",
        raw_track_id=7,
        start_frame=1,
        last_confirmed_frame=5,
        output=target_path,
        semantic_target_id="target_cli",
        overwrite=True,
    )

    result = run_search_reacquisition_candidates(
        config_path=config_path,
        semantic_target=target_path,
        tracks=tracks,
        events=events,
        output_dir=tmp_path / "cli_search",
        grounding_plan=None,
        grounding_manifest=manifest,
        appearance_result=appearance,
        event_id=None,
        overwrite=True,
    )

    assert init["status"] == "ok"
    assert result["status"] == "ok"
    assert result["decision"]["selected_raw_track_id"] == 42
    assert Path(result["paths"]["reacquisition_result_json"]).is_file()
