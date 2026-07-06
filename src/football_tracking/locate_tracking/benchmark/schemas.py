"""Typed schemas for the language-guided tracking benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

QueryMode = Literal["single_target", "multi_target"]
QueryCategory = Literal["appearance", "role", "team_or_group", "spatial", "compound"]
QueryDifficulty = Literal["easy", "medium", "hard"]
PredictionStatus = Literal["resolved", "ambiguous", "not_found", "insufficient_evidence"]

VALID_QUERY_MODES = {"single_target", "multi_target"}
VALID_QUERY_CATEGORIES = {"appearance", "role", "team_or_group", "spatial", "compound"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_PREDICTION_STATUSES = {
    "resolved",
    "ambiguous",
    "not_found",
    "insufficient_evidence",
}


class LanguageBenchmarkSchemaError(ValueError):
    """Raised when a language benchmark schema is invalid."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class GroundTruthIdentitySegment:
    gt_track_id: int
    start_frame: int
    end_frame: int
    visibility_notes: str | None = None

    def __post_init__(self) -> None:
        if int(self.gt_track_id) < 1:
            raise LanguageBenchmarkSchemaError("gt_track_id must be >= 1.")
        if int(self.start_frame) < 1 or int(self.end_frame) < int(self.start_frame):
            raise LanguageBenchmarkSchemaError("Invalid GT identity segment frame range.")
        object.__setattr__(self, "gt_track_id", int(self.gt_track_id))
        object.__setattr__(self, "start_frame", int(self.start_frame))
        object.__setattr__(self, "end_frame", int(self.end_frame))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gt_track_id": self.gt_track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "visibility_notes": self.visibility_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundTruthIdentitySegment:
        return cls(
            gt_track_id=int(data["gt_track_id"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            visibility_notes=data.get("visibility_notes"),
        )


@dataclass(frozen=True)
class GroundTruthLossEvent:
    event_id: str
    frame_start: int
    frame_end: int
    reason: str

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise LanguageBenchmarkSchemaError("loss event_id must not be empty.")
        if int(self.frame_start) < 1 or int(self.frame_end) < int(self.frame_start):
            raise LanguageBenchmarkSchemaError("Invalid loss event frame range.")
        object.__setattr__(self, "frame_start", int(self.frame_start))
        object.__setattr__(self, "frame_end", int(self.frame_end))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundTruthLossEvent:
        return cls(
            event_id=str(data["event_id"]),
            frame_start=int(data["frame_start"]),
            frame_end=int(data["frame_end"]),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class GroundTruthReacquisitionEvent:
    event_id: str
    loss_event_id: str | None
    target_lost_frame: int
    candidate_search_start: int
    candidate_search_end: int
    gt_reappearance_frame: int
    evaluation_start_frame: int
    evaluation_end_frame: int
    notes: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "target_lost_frame",
            "candidate_search_start",
            "candidate_search_end",
            "gt_reappearance_frame",
            "evaluation_start_frame",
            "evaluation_end_frame",
        ):
            if int(getattr(self, field_name)) < 1:
                raise LanguageBenchmarkSchemaError(f"{field_name} must be >= 1.")
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        if self.candidate_search_end < self.candidate_search_start:
            raise LanguageBenchmarkSchemaError("Invalid candidate search frame range.")
        if self.evaluation_end_frame < self.evaluation_start_frame:
            raise LanguageBenchmarkSchemaError("Invalid reacquisition evaluation range.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "loss_event_id": self.loss_event_id,
            "target_lost_frame": self.target_lost_frame,
            "candidate_search_start": self.candidate_search_start,
            "candidate_search_end": self.candidate_search_end,
            "gt_reappearance_frame": self.gt_reappearance_frame,
            "evaluation_start_frame": self.evaluation_start_frame,
            "evaluation_end_frame": self.evaluation_end_frame,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundTruthReacquisitionEvent:
        return cls(
            event_id=str(data["event_id"]),
            loss_event_id=data.get("loss_event_id"),
            target_lost_frame=int(data["target_lost_frame"]),
            candidate_search_start=int(data["candidate_search_start"]),
            candidate_search_end=int(data["candidate_search_end"]),
            gt_reappearance_frame=int(data["gt_reappearance_frame"]),
            evaluation_start_frame=int(data["evaluation_start_frame"]),
            evaluation_end_frame=int(data["evaluation_end_frame"]),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class LanguageQueryAnnotation:
    query_id: str
    query_text: str
    query_mode: QueryMode
    query_category: QueryCategory
    difficulty: QueryDifficulty
    evaluation_start_frame: int
    evaluation_end_frame: int
    target_gt_track_ids: tuple[int, ...]
    identity_segments: tuple[GroundTruthIdentitySegment, ...]
    loss_events: tuple[GroundTruthLossEvent, ...] = ()
    reacquisition_events: tuple[GroundTruthReacquisitionEvent, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.query_text.strip():
            raise LanguageBenchmarkSchemaError("query_id and query_text must not be empty.")
        if self.query_mode not in VALID_QUERY_MODES:
            raise LanguageBenchmarkSchemaError(f"Unsupported query_mode: {self.query_mode}")
        if self.query_category not in VALID_QUERY_CATEGORIES:
            raise LanguageBenchmarkSchemaError(
                f"Unsupported query_category: {self.query_category}"
            )
        if self.difficulty not in VALID_DIFFICULTIES:
            raise LanguageBenchmarkSchemaError(f"Unsupported difficulty: {self.difficulty}")
        if (
            int(self.evaluation_start_frame) < 1
            or int(self.evaluation_end_frame) < int(self.evaluation_start_frame)
        ):
            raise LanguageBenchmarkSchemaError("Invalid query evaluation range.")
        gt_ids = tuple(sorted({int(track_id) for track_id in self.target_gt_track_ids}))
        if not gt_ids or any(track_id < 1 for track_id in gt_ids):
            raise LanguageBenchmarkSchemaError("target_gt_track_ids must be positive.")
        object.__setattr__(self, "evaluation_start_frame", int(self.evaluation_start_frame))
        object.__setattr__(self, "evaluation_end_frame", int(self.evaluation_end_frame))
        object.__setattr__(self, "target_gt_track_ids", gt_ids)
        object.__setattr__(self, "identity_segments", tuple(self.identity_segments))
        object.__setattr__(self, "loss_events", tuple(self.loss_events))
        object.__setattr__(self, "reacquisition_events", tuple(self.reacquisition_events))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "query_mode": self.query_mode,
            "query_category": self.query_category,
            "difficulty": self.difficulty,
            "evaluation_start_frame": self.evaluation_start_frame,
            "evaluation_end_frame": self.evaluation_end_frame,
            "target_gt_track_ids": list(self.target_gt_track_ids),
            "identity_segments": [segment.to_dict() for segment in self.identity_segments],
            "loss_events": [event.to_dict() for event in self.loss_events],
            "reacquisition_events": [
                event.to_dict() for event in self.reacquisition_events
            ],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguageQueryAnnotation:
        return cls(
            query_id=str(data["query_id"]),
            query_text=str(data["query_text"]),
            query_mode=str(data["query_mode"]),  # type: ignore[arg-type]
            query_category=str(data["query_category"]),  # type: ignore[arg-type]
            difficulty=str(data.get("difficulty", "medium")),  # type: ignore[arg-type]
            evaluation_start_frame=int(data["evaluation_start_frame"]),
            evaluation_end_frame=int(data["evaluation_end_frame"]),
            target_gt_track_ids=tuple(data.get("target_gt_track_ids", ())),
            identity_segments=tuple(
                GroundTruthIdentitySegment.from_dict(item)
                for item in data.get("identity_segments", ())
            ),
            loss_events=tuple(
                GroundTruthLossEvent.from_dict(item) for item in data.get("loss_events", ())
            ),
            reacquisition_events=tuple(
                GroundTruthReacquisitionEvent.from_dict(item)
                for item in data.get("reacquisition_events", ())
            ),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class LanguageTrackingSequence:
    sequence_name: str
    split: str
    source_video: Path
    mot_ground_truth_path: Path
    frame_count: int
    fps: float | None
    queries: tuple[LanguageQueryAnnotation, ...]

    def __post_init__(self) -> None:
        if not self.sequence_name.strip():
            raise LanguageBenchmarkSchemaError("sequence_name must not be empty.")
        if int(self.frame_count) < 1:
            raise LanguageBenchmarkSchemaError("frame_count must be >= 1.")
        fps = None if self.fps is None else float(self.fps)
        if fps is not None and fps <= 0.0:
            raise LanguageBenchmarkSchemaError("fps must be positive when provided.")
        object.__setattr__(self, "source_video", Path(self.source_video))
        object.__setattr__(self, "mot_ground_truth_path", Path(self.mot_ground_truth_path))
        object.__setattr__(self, "frame_count", int(self.frame_count))
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "queries", tuple(self.queries))

    @property
    def query_count(self) -> int:
        return len(self.queries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_name": self.sequence_name,
            "split": self.split,
            "source_video": str(self.source_video),
            "mot_ground_truth_path": str(self.mot_ground_truth_path),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "queries": [query.to_dict() for query in self.queries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguageTrackingSequence:
        return cls(
            sequence_name=str(data["sequence_name"]),
            split=str(data.get("split", "dev")),
            source_video=Path(data["source_video"]),
            mot_ground_truth_path=Path(data["mot_ground_truth_path"]),
            frame_count=int(data["frame_count"]),
            fps=data.get("fps"),
            queries=tuple(
                LanguageQueryAnnotation.from_dict(item) for item in data.get("queries", ())
            ),
        )


@dataclass(frozen=True)
class LanguageTrackingBenchmarkManifest:
    benchmark_name: str
    benchmark_version: str
    dataset_name: str
    split: str
    annotation_policy: str
    sequences: tuple[LanguageTrackingSequence, ...]
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sequence_count(self) -> int:
        return len(self.sequences)

    @property
    def query_count(self) -> int:
        return sum(sequence.query_count for sequence in self.sequences)

    def with_updates(self, **changes: Any) -> LanguageTrackingBenchmarkManifest:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_count": self.sequence_count,
            "query_count": self.query_count,
            "annotation_policy": self.annotation_policy,
            "created_at": self.created_at,
            "sequences": [sequence.to_dict() for sequence in self.sequences],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguageTrackingBenchmarkManifest:
        return cls(
            benchmark_name=str(data["benchmark_name"]),
            benchmark_version=str(data.get("benchmark_version", "0.1.0")),
            dataset_name=str(data["dataset_name"]),
            split=str(data.get("split", "dev")),
            annotation_policy=str(data.get("annotation_policy", "")),
            created_at=str(data.get("created_at", utc_now_iso())),
            sequences=tuple(
                LanguageTrackingSequence.from_dict(item)
                for item in data.get("sequences", ())
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class LanguagePrediction:
    query_id: str
    sequence_name: str
    status: PredictionStatus
    semantic_target_path: Path | None
    tracks_path: Path | None
    reacquisition_result_path: Path | None = None
    grounding_call_count: int = 0
    runtime_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_PREDICTION_STATUSES:
            raise LanguageBenchmarkSchemaError(f"Unsupported prediction status: {self.status}")
        if int(self.grounding_call_count) < 0:
            raise LanguageBenchmarkSchemaError("grounding_call_count must be >= 0.")
        if self.runtime_seconds is not None and float(self.runtime_seconds) < 0.0:
            raise LanguageBenchmarkSchemaError("runtime_seconds must be >= 0 when provided.")
        object.__setattr__(self, "semantic_target_path", _optional_path(self.semantic_target_path))
        object.__setattr__(self, "tracks_path", _optional_path(self.tracks_path))
        object.__setattr__(
            self,
            "reacquisition_result_path",
            _optional_path(self.reacquisition_result_path),
        )
        object.__setattr__(self, "grounding_call_count", int(self.grounding_call_count))
        object.__setattr__(
            self,
            "runtime_seconds",
            None if self.runtime_seconds is None else float(self.runtime_seconds),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "sequence_name": self.sequence_name,
            "status": self.status,
            "semantic_target_path": _path_str(self.semantic_target_path),
            "tracks_path": _path_str(self.tracks_path),
            "reacquisition_result_path": _path_str(self.reacquisition_result_path),
            "grounding_call_count": self.grounding_call_count,
            "runtime_seconds": self.runtime_seconds,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguagePrediction:
        return cls(
            query_id=str(data["query_id"]),
            sequence_name=str(data["sequence_name"]),
            status=str(data.get("status", "resolved")),  # type: ignore[arg-type]
            semantic_target_path=_optional_path(data.get("semantic_target_path")),
            tracks_path=_optional_path(data.get("tracks_path")),
            reacquisition_result_path=_optional_path(data.get("reacquisition_result_path")),
            grounding_call_count=int(data.get("grounding_call_count", 0)),
            runtime_seconds=data.get("runtime_seconds"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class LanguagePredictionManifest:
    variant_id: str
    variant_name: str
    benchmark_name: str
    predictions: tuple[LanguagePrediction, ...]
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant_name": self.variant_name,
            "benchmark_name": self.benchmark_name,
            "created_at": self.created_at,
            "prediction_count": len(self.predictions),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguagePredictionManifest:
        return cls(
            variant_id=str(data["variant_id"]),
            variant_name=str(data.get("variant_name", data["variant_id"])),
            benchmark_name=str(data.get("benchmark_name", "")),
            created_at=str(data.get("created_at", utc_now_iso())),
            predictions=tuple(
                LanguagePrediction.from_dict(item) for item in data.get("predictions", ())
            ),
            metadata=dict(data.get("metadata", {})),
        )


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    return Path(value)


def _path_str(value: Path | None) -> str | None:
    return None if value is None else str(value)
