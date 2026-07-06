"""Schemas for team attribution and language-target benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

PipelineType = Literal[
    "qwen_team_labeling",
    "locate_qwen_language_retrieval",
    "custom",
]
PredictionStatus = Literal["resolved", "ambiguous", "not_found", "unknown"]

VALID_PIPELINE_TYPES = {
    "qwen_team_labeling",
    "locate_qwen_language_retrieval",
    "custom",
}
VALID_PREDICTION_STATUSES = {"resolved", "ambiguous", "not_found", "unknown"}


class TeamBenchmarkSchemaError(ValueError):
    """Raised when a team benchmark artifact is invalid."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TeamTrackAnnotation:
    track_id: int
    team_label: str
    start_frame: int
    end_frame: int
    role_label: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if int(self.track_id) < 1:
            raise TeamBenchmarkSchemaError("track_id must be >= 1.")
        if int(self.start_frame) < 1 or int(self.end_frame) < int(self.start_frame):
            raise TeamBenchmarkSchemaError("Invalid track annotation frame range.")
        if not self.team_label.strip():
            raise TeamBenchmarkSchemaError("team_label must not be empty.")
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "start_frame", int(self.start_frame))
        object.__setattr__(self, "end_frame", int(self.end_frame))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "team_label": self.team_label,
            "role_label": self.role_label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamTrackAnnotation:
        return cls(
            track_id=int(data["track_id"]),
            team_label=str(data["team_label"]),
            role_label=data.get("role_label"),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class TeamQueryAnnotation:
    query_id: str
    query_text: str
    expected_track_ids: tuple[int, ...]
    expected_team_label: str
    start_frame: int
    end_frame: int
    difficulty: str = "medium"
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.query_text.strip():
            raise TeamBenchmarkSchemaError("query_id and query_text must not be empty.")
        if not self.expected_team_label.strip():
            raise TeamBenchmarkSchemaError("expected_team_label must not be empty.")
        expected_track_ids = tuple(sorted({int(track_id) for track_id in self.expected_track_ids}))
        if not expected_track_ids or any(track_id < 1 for track_id in expected_track_ids):
            raise TeamBenchmarkSchemaError("expected_track_ids must contain positive ids.")
        if int(self.start_frame) < 1 or int(self.end_frame) < int(self.start_frame):
            raise TeamBenchmarkSchemaError("Invalid query frame range.")
        object.__setattr__(self, "expected_track_ids", expected_track_ids)
        object.__setattr__(self, "start_frame", int(self.start_frame))
        object.__setattr__(self, "end_frame", int(self.end_frame))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "expected_track_ids": list(self.expected_track_ids),
            "expected_team_label": self.expected_team_label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "difficulty": self.difficulty,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamQueryAnnotation:
        return cls(
            query_id=str(data["query_id"]),
            query_text=str(data["query_text"]),
            expected_track_ids=tuple(int(item) for item in data.get("expected_track_ids", ())),
            expected_team_label=str(data["expected_team_label"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            difficulty=str(data.get("difficulty", "medium")),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class TeamBenchmarkSequence:
    sequence_name: str
    source_video: Path
    frame_count: int
    split: str = "dev"
    tracks_path: Path | None = None
    mot_ground_truth_path: Path | None = None
    track_annotations: tuple[TeamTrackAnnotation, ...] = ()
    query_annotations: tuple[TeamQueryAnnotation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sequence_name.strip():
            raise TeamBenchmarkSchemaError("sequence_name must not be empty.")
        if int(self.frame_count) < 1:
            raise TeamBenchmarkSchemaError("frame_count must be >= 1.")
        object.__setattr__(self, "source_video", Path(self.source_video))
        object.__setattr__(self, "tracks_path", _optional_path(self.tracks_path))
        object.__setattr__(
            self,
            "mot_ground_truth_path",
            _optional_path(self.mot_ground_truth_path),
        )
        object.__setattr__(self, "frame_count", int(self.frame_count))
        object.__setattr__(self, "track_annotations", tuple(self.track_annotations))
        object.__setattr__(self, "query_annotations", tuple(self.query_annotations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_name": self.sequence_name,
            "split": self.split,
            "source_video": str(self.source_video),
            "tracks_path": _path_str(self.tracks_path),
            "mot_ground_truth_path": _path_str(self.mot_ground_truth_path),
            "frame_count": self.frame_count,
            "track_annotations": [item.to_dict() for item in self.track_annotations],
            "query_annotations": [item.to_dict() for item in self.query_annotations],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamBenchmarkSequence:
        return cls(
            sequence_name=str(data["sequence_name"]),
            split=str(data.get("split", "dev")),
            source_video=Path(data["source_video"]),
            tracks_path=_optional_path(data.get("tracks_path")),
            mot_ground_truth_path=_optional_path(data.get("mot_ground_truth_path")),
            frame_count=int(data["frame_count"]),
            track_annotations=tuple(
                TeamTrackAnnotation.from_dict(item)
                for item in data.get("track_annotations", ())
            ),
            query_annotations=tuple(
                TeamQueryAnnotation.from_dict(item)
                for item in data.get("query_annotations", ())
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class TeamBenchmarkManifest:
    benchmark_name: str
    benchmark_version: str
    dataset_name: str
    split: str
    annotation_policy: str
    sequences: tuple[TeamBenchmarkSequence, ...]
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sequence_count(self) -> int:
        return len(self.sequences)

    @property
    def annotated_track_count(self) -> int:
        return sum(len(sequence.track_annotations) for sequence in self.sequences)

    @property
    def query_count(self) -> int:
        return sum(len(sequence.query_annotations) for sequence in self.sequences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "annotation_policy": self.annotation_policy,
            "sequence_count": self.sequence_count,
            "annotated_track_count": self.annotated_track_count,
            "query_count": self.query_count,
            "created_at": self.created_at,
            "sequences": [sequence.to_dict() for sequence in self.sequences],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamBenchmarkManifest:
        return cls(
            benchmark_name=str(data["benchmark_name"]),
            benchmark_version=str(data.get("benchmark_version", "0.1.0")),
            dataset_name=str(data["dataset_name"]),
            split=str(data.get("split", "dev")),
            annotation_policy=str(data.get("annotation_policy", "")),
            sequences=tuple(
                TeamBenchmarkSequence.from_dict(item)
                for item in data.get("sequences", ())
            ),
            created_at=str(data.get("created_at", utc_now_iso())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class TrackTeamPrediction:
    sequence_name: str
    track_id: int
    status: PredictionStatus
    team_label: str | None = None
    role_label: str | None = None
    confidence: float | None = None
    evidence_frames: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_PREDICTION_STATUSES:
            raise TeamBenchmarkSchemaError(f"Unsupported status: {self.status}")
        if int(self.track_id) < 1:
            raise TeamBenchmarkSchemaError("track_id must be >= 1.")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise TeamBenchmarkSchemaError("confidence must be in [0, 1].")
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "confidence", _optional_float(self.confidence))
        object.__setattr__(
            self,
            "evidence_frames",
            tuple(sorted({int(frame) for frame in self.evidence_frames})),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_name": self.sequence_name,
            "track_id": self.track_id,
            "status": self.status,
            "team_label": self.team_label,
            "role_label": self.role_label,
            "confidence": self.confidence,
            "evidence_frames": list(self.evidence_frames),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackTeamPrediction:
        return cls(
            sequence_name=str(data["sequence_name"]),
            track_id=int(data["track_id"]),
            status=str(data.get("status", "resolved")),  # type: ignore[arg-type]
            team_label=data.get("team_label"),
            role_label=data.get("role_label"),
            confidence=data.get("confidence"),
            evidence_frames=tuple(int(frame) for frame in data.get("evidence_frames", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class QueryTargetPrediction:
    sequence_name: str
    query_id: str
    status: PredictionStatus
    selected_track_ids: tuple[int, ...] = ()
    team_label: str | None = None
    confidence: float | None = None
    support_ratio: float | None = None
    grounding_call_count: int = 0
    runtime_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_PREDICTION_STATUSES:
            raise TeamBenchmarkSchemaError(f"Unsupported status: {self.status}")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise TeamBenchmarkSchemaError("confidence must be in [0, 1].")
        if self.support_ratio is not None and not 0.0 <= float(self.support_ratio) <= 1.0:
            raise TeamBenchmarkSchemaError("support_ratio must be in [0, 1].")
        if int(self.grounding_call_count) < 0:
            raise TeamBenchmarkSchemaError("grounding_call_count must be >= 0.")
        if self.runtime_seconds is not None and float(self.runtime_seconds) < 0.0:
            raise TeamBenchmarkSchemaError("runtime_seconds must be >= 0.")
        object.__setattr__(
            self,
            "selected_track_ids",
            tuple(sorted({int(track_id) for track_id in self.selected_track_ids})),
        )
        object.__setattr__(self, "confidence", _optional_float(self.confidence))
        object.__setattr__(self, "support_ratio", _optional_float(self.support_ratio))
        object.__setattr__(self, "grounding_call_count", int(self.grounding_call_count))
        object.__setattr__(
            self,
            "runtime_seconds",
            None if self.runtime_seconds is None else float(self.runtime_seconds),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_name": self.sequence_name,
            "query_id": self.query_id,
            "status": self.status,
            "selected_track_ids": list(self.selected_track_ids),
            "team_label": self.team_label,
            "confidence": self.confidence,
            "support_ratio": self.support_ratio,
            "grounding_call_count": self.grounding_call_count,
            "runtime_seconds": self.runtime_seconds,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryTargetPrediction:
        return cls(
            sequence_name=str(data["sequence_name"]),
            query_id=str(data["query_id"]),
            status=str(data.get("status", "resolved")),  # type: ignore[arg-type]
            selected_track_ids=tuple(int(item) for item in data.get("selected_track_ids", ())),
            team_label=data.get("team_label"),
            confidence=data.get("confidence"),
            support_ratio=data.get("support_ratio"),
            grounding_call_count=int(data.get("grounding_call_count", 0)),
            runtime_seconds=data.get("runtime_seconds"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class TeamPredictionManifest:
    variant_id: str
    variant_name: str
    benchmark_name: str
    pipeline_type: PipelineType
    track_predictions: tuple[TrackTeamPrediction, ...] = ()
    query_predictions: tuple[QueryTargetPrediction, ...] = ()
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pipeline_type not in VALID_PIPELINE_TYPES:
            raise TeamBenchmarkSchemaError(f"Unsupported pipeline_type: {self.pipeline_type}")
        object.__setattr__(self, "track_predictions", tuple(self.track_predictions))
        object.__setattr__(self, "query_predictions", tuple(self.query_predictions))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant_name": self.variant_name,
            "benchmark_name": self.benchmark_name,
            "pipeline_type": self.pipeline_type,
            "created_at": self.created_at,
            "track_prediction_count": len(self.track_predictions),
            "query_prediction_count": len(self.query_predictions),
            "track_predictions": [item.to_dict() for item in self.track_predictions],
            "query_predictions": [item.to_dict() for item in self.query_predictions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamPredictionManifest:
        return cls(
            variant_id=str(data["variant_id"]),
            variant_name=str(data.get("variant_name", data["variant_id"])),
            benchmark_name=str(data.get("benchmark_name", "")),
            pipeline_type=str(data.get("pipeline_type", "custom")),  # type: ignore[arg-type]
            created_at=str(data.get("created_at", utc_now_iso())),
            track_predictions=tuple(
                TrackTeamPrediction.from_dict(item)
                for item in data.get("track_predictions", ())
            ),
            query_predictions=tuple(
                QueryTargetPrediction.from_dict(item)
                for item in data.get("query_predictions", ())
            ),
            metadata=dict(data.get("metadata", {})),
        )


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    return Path(value)


def _path_str(value: Path | None) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: float | int | str | None) -> float | None:
    return None if value is None else float(value)
