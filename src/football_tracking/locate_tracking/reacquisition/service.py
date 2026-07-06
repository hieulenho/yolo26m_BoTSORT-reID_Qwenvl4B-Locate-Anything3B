"""Offline semantic target reacquisition service."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from football_tracking.locate_tracking.appearance.schemas import AppearanceVerificationResult
from football_tracking.locate_tracking.artifacts.mot_reader import read_mot_track_file
from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.events.event_store import read_events_jsonl
from football_tracking.locate_tracking.events.schemas import UncertaintyEvent
from football_tracking.locate_tracking.grounding_scheduler.planner import load_grounding_plan
from football_tracking.locate_tracking.identity.schemas import (
    SemanticTarget,
    stable_artifact_id,
)
from football_tracking.locate_tracking.identity.segment_store import (
    load_semantic_target,
    save_semantic_target,
    save_semantic_target_timeline,
)
from football_tracking.locate_tracking.identity.service import (
    commit_provisional_reacquisition,
    commit_same_raw_resume,
    confirm_probation,
)
from football_tracking.locate_tracking.identity.state_machine import validate_transition
from football_tracking.locate_tracking.reacquisition.appearance_evidence import (
    appearance_evidence,
    load_appearance_result,
)
from football_tracking.locate_tracking.reacquisition.candidate_generator import (
    build_candidate_search_window,
    find_same_raw_id_resume,
    generate_reacquisition_candidates,
)
from football_tracking.locate_tracking.reacquisition.candidate_ranker import rank_candidates
from football_tracking.locate_tracking.reacquisition.decision_policy import decide_reacquisition
from football_tracking.locate_tracking.reacquisition.grounding_evidence import (
    grounding_evidence,
    grounding_result_paths_from_manifest,
)
from football_tracking.locate_tracking.reacquisition.history_evidence import (
    history_evidence,
    identity_conflict_gate,
)
from football_tracking.locate_tracking.reacquisition.motion_model import (
    motion_evidence,
    motion_gate,
)
from football_tracking.locate_tracking.reacquisition.probation import evaluate_probation
from football_tracking.locate_tracking.reacquisition.schemas import (
    ReacquisitionCandidate,
    ReacquisitionConfig,
    ReacquisitionDecision,
    ReacquisitionRun,
)
from football_tracking.locate_tracking.reacquisition.spatial_gate import (
    spatial_grounding_gate,
)
from football_tracking.locate_tracking.reacquisition.temporal_gate import (
    temporal_evidence,
    temporal_gate,
)
from football_tracking.locate_tracking.visualization.reacquisition_summary import (
    write_reacquisition_summary,
)


class ReacquisitionServiceError(RuntimeError):
    """Raised when semantic target reacquisition cannot complete."""


@dataclass(frozen=True)
class ReacquisitionInputs:
    semantic_target_path: Path
    tracks_path: Path
    events_path: Path
    grounding_plan_path: Path | None
    grounding_manifest_path: Path | None
    appearance_result_path: Path | None


def _sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest_grounding_paths(path: str | Path | None) -> tuple[Path, ...]:
    if path is None:
        return ()
    return grounding_result_paths_from_manifest(path)


def select_reacquisition_event(
    events: tuple[UncertaintyEvent, ...],
    *,
    event_id: str | None = None,
) -> UncertaintyEvent:
    if event_id is not None:
        for event in events:
            if event.event_id == event_id:
                return event
        raise ReacquisitionServiceError(f"Event id not found: {event_id}")
    preferred = [
        event for event in events if event.event_type in {"target_absent", "track_gap"}
    ]
    if not preferred:
        raise ReacquisitionServiceError("No target_absent or track_gap event available.")
    return sorted(preferred, key=lambda item: (item.frame_start, item.event_id))[0]


def _total_frame_hint(
    plan_path: str | Path | None,
    observations: tuple[MotTrackObservation, ...],
) -> int | None:
    if plan_path is not None and Path(plan_path).is_file():
        plan = load_grounding_plan(plan_path)
        frames = [
            frame
            for item in plan.items
            for frame in item.selected_frames
        ]
        if frames:
            return max(max(frames), max(row.frame_index for row in observations))
    return max((row.frame_index for row in observations), default=None)


def _mark_reacquiring(
    target: SemanticTarget,
    frame_index: int,
    event_ids: tuple[str, ...],
) -> SemanticTarget:
    state = target
    if state.state == "ACTIVE":
        validate_transition("ACTIVE", "UNCERTAIN")
        state = state.with_updates(state="UNCERTAIN", last_update_frame=frame_index)
    if state.state == "UNCERTAIN":
        validate_transition("UNCERTAIN", "LOST")
        state = state.with_updates(state="LOST", last_update_frame=frame_index)
    if state.state == "LOST":
        validate_transition("LOST", "REACQUIRING")
        state = state.with_updates(
            state="REACQUIRING",
            last_update_frame=frame_index,
            metadata={**state.metadata, "reacquiring_event_ids": list(event_ids)},
        )
    return state


def _score_candidates(
    *,
    candidates: tuple[ReacquisitionCandidate, ...],
    target: SemanticTarget,
    observations: tuple[MotTrackObservation, ...],
    grounding_paths: tuple[Path, ...],
    appearance_result: AppearanceVerificationResult | None,
    config: ReacquisitionConfig,
) -> tuple[ReacquisitionCandidate, ...]:
    scored: list[ReacquisitionCandidate] = []
    for candidate in candidates:
        temporal_gate_result = temporal_gate(
            candidate=candidate,
            previous_raw_track_id=target.current_raw_track_id,
            all_observations=observations,
            config=config,
        )
        motion = motion_evidence(
            candidate=candidate,
            previous_raw_track_id=target.current_raw_track_id,
            all_observations=observations,
            config=config,
        )
        motion_gate_result = motion_gate(motion, config)
        grounding = grounding_evidence(
            candidate=candidate,
            all_observations=observations,
            grounding_result_paths=grounding_paths,
        )
        grounding_gate_result = spatial_grounding_gate(grounding, config)
        appearance = appearance_evidence(
            candidate=candidate,
            appearance_result=appearance_result,
        )
        history = history_evidence(candidate=candidate, target=target)
        history_gate_result = identity_conflict_gate(history)
        gate_results = (
            temporal_gate_result,
            motion_gate_result,
            grounding_gate_result,
            history_gate_result,
        )
        rejections = tuple(gate.reason for gate in gate_results if not gate.passed)
        scored.append(
            candidate.with_updates(
                gate_results=gate_results,
                grounding_evidence=grounding,
                appearance_evidence=appearance,
                motion_evidence=motion,
                temporal_evidence=temporal_evidence(candidate),
                history_evidence=history,
                status="passed" if not rejections else "rejected",
                rejection_reasons=rejections,
            )
        )
    return tuple(scored)


def save_reacquisition_run(
    run: ReacquisitionRun,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise ReacquisitionServiceError(
            f"Reacquisition output exists and overwrite=false: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run.to_dict(), indent=2, default=str), encoding="utf-8")
    return output


def run_reacquisition(
    *,
    semantic_target_path: str | Path,
    tracks_path: str | Path,
    events_path: str | Path,
    output_dir: str | Path,
    config: ReacquisitionConfig,
    grounding_plan_path: str | Path | None = None,
    grounding_manifest_path: str | Path | None = None,
    appearance_result_path: str | Path | None = None,
    event_id: str | None = None,
    commit: bool = False,
    overwrite: bool = False,
) -> ReacquisitionRun:
    target_path = Path(semantic_target_path)
    target = load_semantic_target(target_path)
    observations = read_mot_track_file(tracks_path).observations
    events = read_events_jsonl(events_path)
    event = select_reacquisition_event(events, event_id=event_id)
    total_frames = _total_frame_hint(grounding_plan_path, observations)
    last_confirmed = target.last_confirmed_frame or event.frame_start
    search_window = build_candidate_search_window(
        event=event,
        last_confirmed_frame=last_confirmed,
        total_frames=total_frames,
        config=config,
    )
    candidates = generate_reacquisition_candidates(
        observations=observations,
        search_window=search_window,
    )
    same_resume = find_same_raw_id_resume(
        candidates=candidates,
        observations=observations,
        previous_raw_track_id=target.current_raw_track_id,
        event_end_frame=event.frame_end,
        min_observations=max(1, config.min_observations),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grounding_paths = _load_manifest_grounding_paths(grounding_manifest_path)
    appearance_result = load_appearance_result(appearance_result_path)
    input_hashes = {
        "mot_txt": _sha256(tracks_path),
        "semantic_target": _sha256(semantic_target_path),
        "events": _sha256(events_path),
        "grounding_plan": _sha256(grounding_plan_path),
        "grounding_manifest": _sha256(grounding_manifest_path),
        "appearance_result": _sha256(appearance_result_path),
    }
    if same_resume is not None:
        decision = ReacquisitionDecision(
            decision_id=stable_artifact_id(
                "decision",
                {
                    "target": target.semantic_target_id,
                    "event": event.event_id,
                    "status": "same_raw_id_resumed",
                },
            ),
            status="same_raw_id_resumed",
            semantic_target_id=target.semantic_target_id,
            previous_raw_track_id=target.current_raw_track_id,
            selected_raw_track_id=target.current_raw_track_id,
            selected_start_frame=same_resume.first_observed_frame,
            final_score=1.0,
            score_margin=None,
            reason="same_raw_id_resumed_after_gap",
            event_ids=(event.event_id,),
            candidate_count=len(candidates),
            ranked_candidates=(same_resume.with_updates(rank=1, final_score=1.0),),
        )
        all_candidates = candidates
    else:
        scored = _score_candidates(
            candidates=tuple(
                candidate
                for candidate in candidates
                if candidate.raw_track_id != target.current_raw_track_id
            ),
            target=target,
            observations=observations,
            grounding_paths=grounding_paths,
            appearance_result=appearance_result,
            config=config,
        )
        ranked = rank_candidates(scored, config)
        decision = decide_reacquisition(
            target=target,
            ranked_candidates=ranked,
            all_candidates=scored,
            config=config,
            event_ids=(event.event_id,),
        )
        all_candidates = scored
    run_path = output / "reacquisition_result.json"
    summary_path = output / "summary.md"
    paths: dict[str, Path] = {
        "reacquisition_result_json": run_path,
        "summary_md": summary_path,
    }
    run = ReacquisitionRun(
        semantic_target_id=target.semantic_target_id,
        search_window=search_window,
        decision=decision,
        candidates=all_candidates,
        paths=paths,
        input_hashes=input_hashes,
    )
    save_reacquisition_run(run, run_path, overwrite=overwrite)
    write_reacquisition_summary(
        run=run,
        output_path=summary_path,
        overwrite=overwrite,
    )
    if commit and decision.status == "same_raw_id_resumed":
        reacquiring_target = _mark_reacquiring(target, event.trigger_frame, (event.event_id,))
        committed = commit_same_raw_resume(
            target=reacquiring_target,
            resume_frame=decision.selected_start_frame or event.trigger_frame,
            event_ids=(event.event_id,),
            decision_id=decision.decision_id,
            target_path=target_path,
            transition_log_path=output / "identity_transitions.jsonl",
            evidence_reference=str(run_path),
        )
        paths["semantic_target_json"] = save_semantic_target(
            committed,
            output / "semantic_target.json",
            overwrite=True,
        )
        paths["semantic_target_timeline_json"] = save_semantic_target_timeline(
            committed,
            output / "semantic_target_timeline.json",
            overwrite=True,
        )
    elif commit and decision.status == "provisional" and decision.selected_raw_track_id is not None:
        reacquiring_target = _mark_reacquiring(target, event.trigger_frame, (event.event_id,))
        committed = commit_provisional_reacquisition(
            target=reacquiring_target,
            new_raw_track_id=decision.selected_raw_track_id,
            start_frame=decision.selected_start_frame or event.trigger_frame,
            previous_end_frame=max(1, event.frame_start - 1),
            confidence=decision.final_score or 0.0,
            event_ids=(event.event_id,),
            decision_id=decision.decision_id,
            target_path=target_path,
            transition_log_path=output / "identity_transitions.jsonl",
            evidence_reference=str(run_path),
        )
        paths["semantic_target_json"] = save_semantic_target(
            committed,
            output / "semantic_target.json",
            overwrite=True,
        )
        paths["semantic_target_timeline_json"] = save_semantic_target_timeline(
            committed,
            output / "semantic_target_timeline.json",
            overwrite=True,
        )
        if config.auto_confirm:
            selected = next(
                candidate
                for candidate in decision.ranked_candidates
                if candidate.raw_track_id == decision.selected_raw_track_id
            )
            passed, probation_info = evaluate_probation(
                candidate=selected,
                all_observations=observations,
                config=config,
            )
            if passed:
                confirmed = confirm_probation(
                    target=committed,
                    frame_index=int(probation_info["probation_end_frame"]),
                    event_ids=(event.event_id,),
                    decision_id=decision.decision_id,
                    target_path=target_path,
                    transition_log_path=output / "identity_transitions.jsonl",
                    evidence_reference=str(run_path),
                )
                paths["semantic_target_json"] = save_semantic_target(
                    confirmed,
                    output / "semantic_target.json",
                    overwrite=True,
                )
                paths["semantic_target_timeline_json"] = save_semantic_target_timeline(
                    confirmed,
                    output / "semantic_target_timeline.json",
                    overwrite=True,
                )
    final_run = ReacquisitionRun(
        semantic_target_id=run.semantic_target_id,
        search_window=run.search_window,
        decision=run.decision,
        candidates=run.candidates,
        paths=paths,
        input_hashes=input_hashes,
    )
    save_reacquisition_run(final_run, run_path, overwrite=True)
    write_reacquisition_summary(run=final_run, output_path=summary_path, overwrite=True)
    return final_run


def confirm_reacquisition_probation(
    *,
    semantic_target_path: str | Path,
    tracks_path: str | Path,
    decision_path: str | Path,
    output_dir: str | Path,
    config: ReacquisitionConfig,
    overwrite: bool = False,
) -> SemanticTarget:
    target = load_semantic_target(semantic_target_path)
    data = json.loads(Path(decision_path).read_text(encoding="utf-8"))
    decision = ReacquisitionDecision(
        decision_id=data["decision"]["decision_id"],
        status=data["decision"]["status"],
        semantic_target_id=data["decision"]["semantic_target_id"],
        previous_raw_track_id=data["decision"].get("previous_raw_track_id"),
        selected_raw_track_id=data["decision"].get("selected_raw_track_id"),
        selected_start_frame=data["decision"].get("selected_start_frame"),
        final_score=data["decision"].get("final_score"),
        score_margin=data["decision"].get("score_margin"),
        reason=data["decision"]["reason"],
        event_ids=tuple(data["decision"].get("event_ids", ())),
        candidate_count=int(data["decision"].get("candidate_count", 0)),
        ranked_candidates=tuple(
            ReacquisitionCandidate.from_dict(item)
            for item in data["decision"].get("ranked_candidates", ())
        ),
    )
    if decision.status not in {"provisional", "confirmed"}:
        raise ReacquisitionServiceError("Only provisional decisions can be confirmed.")
    selected = next(
        (
            candidate
            for candidate in decision.ranked_candidates
            if candidate.raw_track_id == decision.selected_raw_track_id
        ),
        None,
    )
    if selected is None:
        raise ReacquisitionServiceError("Selected candidate missing from decision artifact.")
    observations = read_mot_track_file(tracks_path).observations
    passed, probation_info = evaluate_probation(
        candidate=selected,
        all_observations=observations,
        config=config,
    )
    if not passed:
        raise ReacquisitionServiceError(str(probation_info["reason"]))
    output = Path(output_dir)
    updated = confirm_probation(
        target=target,
        frame_index=int(probation_info["probation_end_frame"]),
        event_ids=decision.event_ids,
        decision_id=decision.decision_id,
        target_path=semantic_target_path,
        transition_log_path=output / "identity_transitions.jsonl",
        evidence_reference=str(decision_path),
    )
    save_semantic_target(updated, output / "semantic_target.json", overwrite=overwrite)
    save_semantic_target_timeline(
        updated,
        output / "semantic_target_timeline.json",
        overwrite=overwrite,
    )
    return updated
