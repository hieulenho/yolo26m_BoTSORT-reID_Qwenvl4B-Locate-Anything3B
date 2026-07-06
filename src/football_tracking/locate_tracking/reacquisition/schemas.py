"""Schemas for semantic target reacquisition."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

CandidateStatus = Literal["generated", "rejected", "passed", "ranked", "winner"]
DecisionStatus = Literal[
    "same_raw_id_resumed",
    "provisional",
    "confirmed",
    "ambiguous",
    "rejected",
    "not_found",
]
MissingEvidencePolicy = Literal["ignore", "zero"]


class ReacquisitionSchemaError(ValueError):
    """Raised when reacquisition schemas receive invalid values."""


@dataclass(frozen=True)
class CandidateSearchWindow:
    start_frame: int
    end_frame: int
    last_confirmed_frame: int
    event_start_frame: int
    event_end_frame: int
    pre_event_context_frames: int
    post_event_context_frames: int
    source_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "start_frame",
            "end_frame",
            "last_confirmed_frame",
            "event_start_frame",
            "event_end_frame",
            "pre_event_context_frames",
            "post_event_context_frames",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        if self.start_frame < 1 or self.end_frame < self.start_frame:
            raise ReacquisitionSchemaError("Invalid candidate search window.")
        if self.event_end_frame < self.event_start_frame:
            raise ReacquisitionSchemaError("Invalid event frame range.")
        object.__setattr__(
            self,
            "source_event_ids",
            tuple(str(item) for item in self.source_event_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "last_confirmed_frame": self.last_confirmed_frame,
            "event_start_frame": self.event_start_frame,
            "event_end_frame": self.event_end_frame,
            "pre_event_context_frames": self.pre_event_context_frames,
            "post_event_context_frames": self.post_event_context_frames,
            "source_event_ids": list(self.source_event_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateSearchWindow:
        return cls(
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            last_confirmed_frame=int(data["last_confirmed_frame"]),
            event_start_frame=int(data["event_start_frame"]),
            event_end_frame=int(data["event_end_frame"]),
            pre_event_context_frames=int(data["pre_event_context_frames"]),
            post_event_context_frames=int(data["post_event_context_frames"]),
            source_event_ids=tuple(data.get("source_event_ids", ())),
        )


@dataclass(frozen=True)
class GateResult:
    gate_name: str
    passed: bool
    score: float | None
    threshold: float | None
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "score": self.score,
            "threshold": self.threshold,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateResult:
        return cls(
            gate_name=str(data["gate_name"]),
            passed=bool(data["passed"]),
            score=data.get("score"),
            threshold=data.get("threshold"),
            reason=str(data["reason"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class EvidenceScore:
    name: str
    score: float | None
    data_available: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is not None and not 0.0 <= float(self.score) <= 1.0:
            raise ReacquisitionSchemaError("evidence score must be None or in [0, 1].")
        object.__setattr__(self, "score", None if self.score is None else float(self.score))
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "data_available": self.data_available,
            "reason": self.reason,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceScore:
        return cls(
            name=str(data["name"]),
            score=data.get("score"),
            data_available=bool(data["data_available"]),
            reason=str(data["reason"]),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class ReacquisitionCandidate:
    raw_track_id: int
    search_window: CandidateSearchWindow
    first_observed_frame: int
    last_observed_frame: int
    observation_count: int
    gate_results: tuple[GateResult, ...] = ()
    grounding_evidence: EvidenceScore | None = None
    appearance_evidence: EvidenceScore | None = None
    motion_evidence: EvidenceScore | None = None
    temporal_evidence: EvidenceScore | None = None
    history_evidence: EvidenceScore | None = None
    component_scores: dict[str, float | None] = field(default_factory=dict)
    final_score: float | None = None
    rank: int | None = None
    status: CandidateStatus = "generated"
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.raw_track_id) < 1:
            raise ReacquisitionSchemaError("raw_track_id must be >= 1.")
        if int(self.first_observed_frame) < 1:
            raise ReacquisitionSchemaError("first_observed_frame must be >= 1.")
        if int(self.last_observed_frame) < int(self.first_observed_frame):
            raise ReacquisitionSchemaError("last_observed_frame must be >= first_observed_frame.")
        if int(self.observation_count) < 1:
            raise ReacquisitionSchemaError("observation_count must be >= 1.")
        object.__setattr__(self, "raw_track_id", int(self.raw_track_id))
        object.__setattr__(self, "first_observed_frame", int(self.first_observed_frame))
        object.__setattr__(self, "last_observed_frame", int(self.last_observed_frame))
        object.__setattr__(self, "observation_count", int(self.observation_count))
        object.__setattr__(self, "gate_results", tuple(self.gate_results))
        object.__setattr__(self, "component_scores", dict(self.component_scores))
        object.__setattr__(
            self,
            "rejection_reasons",
            tuple(str(item) for item in self.rejection_reasons),
        )

    @property
    def passed_gates(self) -> bool:
        return all(gate.passed for gate in self.gate_results)

    def with_updates(self, **changes: Any) -> ReacquisitionCandidate:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_track_id": self.raw_track_id,
            "search_window": self.search_window.to_dict(),
            "first_observed_frame": self.first_observed_frame,
            "last_observed_frame": self.last_observed_frame,
            "observation_count": self.observation_count,
            "gate_results": [item.to_dict() for item in self.gate_results],
            "grounding_evidence": self.grounding_evidence.to_dict()
            if self.grounding_evidence
            else None,
            "appearance_evidence": self.appearance_evidence.to_dict()
            if self.appearance_evidence
            else None,
            "motion_evidence": self.motion_evidence.to_dict() if self.motion_evidence else None,
            "temporal_evidence": self.temporal_evidence.to_dict()
            if self.temporal_evidence
            else None,
            "history_evidence": self.history_evidence.to_dict() if self.history_evidence else None,
            "component_scores": dict(self.component_scores),
            "final_score": self.final_score,
            "rank": self.rank,
            "status": self.status,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReacquisitionCandidate:
        return cls(
            raw_track_id=int(data["raw_track_id"]),
            search_window=CandidateSearchWindow.from_dict(data["search_window"]),
            first_observed_frame=int(data["first_observed_frame"]),
            last_observed_frame=int(data["last_observed_frame"]),
            observation_count=int(data["observation_count"]),
            gate_results=tuple(GateResult.from_dict(item) for item in data.get("gate_results", ())),
            grounding_evidence=EvidenceScore.from_dict(data["grounding_evidence"])
            if data.get("grounding_evidence")
            else None,
            appearance_evidence=EvidenceScore.from_dict(data["appearance_evidence"])
            if data.get("appearance_evidence")
            else None,
            motion_evidence=EvidenceScore.from_dict(data["motion_evidence"])
            if data.get("motion_evidence")
            else None,
            temporal_evidence=EvidenceScore.from_dict(data["temporal_evidence"])
            if data.get("temporal_evidence")
            else None,
            history_evidence=EvidenceScore.from_dict(data["history_evidence"])
            if data.get("history_evidence")
            else None,
            component_scores=dict(data.get("component_scores", {})),
            final_score=data.get("final_score"),
            rank=data.get("rank"),
            status=str(data.get("status", "generated")),  # type: ignore[arg-type]
            rejection_reasons=tuple(data.get("rejection_reasons", ())),
        )


@dataclass(frozen=True)
class ReacquisitionDecision:
    decision_id: str
    status: DecisionStatus
    semantic_target_id: str
    previous_raw_track_id: int | None
    selected_raw_track_id: int | None
    selected_start_frame: int | None
    final_score: float | None
    score_margin: float | None
    reason: str
    event_ids: tuple[str, ...]
    candidate_count: int
    ranked_candidates: tuple[ReacquisitionCandidate, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "status": self.status,
            "semantic_target_id": self.semantic_target_id,
            "previous_raw_track_id": self.previous_raw_track_id,
            "selected_raw_track_id": self.selected_raw_track_id,
            "selected_start_frame": self.selected_start_frame,
            "final_score": self.final_score,
            "score_margin": self.score_margin,
            "reason": self.reason,
            "event_ids": list(self.event_ids),
            "candidate_count": self.candidate_count,
            "ranked_candidates": [item.to_dict() for item in self.ranked_candidates],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReacquisitionRun:
    semantic_target_id: str
    search_window: CandidateSearchWindow
    decision: ReacquisitionDecision
    candidates: tuple[ReacquisitionCandidate, ...]
    paths: dict[str, Path]
    input_hashes: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_target_id": self.semantic_target_id,
            "search_window": self.search_window.to_dict(),
            "decision": self.decision.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "paths": {key: str(value) for key, value in self.paths.items()},
            "input_hashes": dict(self.input_hashes),
            "note": "semantic identity layer only; raw MOT IDs are not modified",
        }


@dataclass(frozen=True)
class ReacquisitionConfig:
    pre_event_context_frames: int = 20
    post_event_context_frames: int = 50
    min_observations: int = 2
    duplicate_overlap_tolerance_frames: int = 2
    max_motion_distance_normalized: float = 0.20
    min_grounding_score: float = 0.10
    require_grounding_support: bool = True
    min_final_score: float = 0.45
    ambiguity_margin: float = 0.08
    missing_evidence_policy: MissingEvidencePolicy = "ignore"
    probation_window_frames: int = 20
    probation_min_observations: int = 3
    auto_confirm: bool = False
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "grounding": 0.35,
            "appearance": 0.20,
            "motion": 0.20,
            "temporal": 0.15,
            "history": 0.10,
        }
    )

    def __post_init__(self) -> None:
        for field_name in (
            "pre_event_context_frames",
            "post_event_context_frames",
            "min_observations",
            "duplicate_overlap_tolerance_frames",
            "probation_window_frames",
            "probation_min_observations",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ReacquisitionSchemaError(f"{field_name} must be >= 0.")
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        for field_name in (
            "max_motion_distance_normalized",
            "min_grounding_score",
            "min_final_score",
            "ambiguity_margin",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ReacquisitionSchemaError(f"{field_name} must be in [0, 1].")
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "weights",
            {str(key): float(value) for key, value in self.weights.items()},
        )
