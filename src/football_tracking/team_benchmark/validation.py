"""Validation for team benchmark artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from football_tracking.team_benchmark.manifest import (
    load_team_benchmark_manifest,
    load_team_prediction_manifest,
)

IssueSeverity = Literal["ERROR", "WARNING"]


@dataclass(frozen=True)
class TeamBenchmarkIssue:
    severity: IssueSeverity
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


@dataclass(frozen=True)
class TeamBenchmarkValidationReport:
    issues: tuple[TeamBenchmarkIssue, ...]
    sequence_count: int
    annotated_track_count: int
    query_count: int

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "WARNING")

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "errors": self.error_count,
                "warnings": self.warning_count,
                "sequence_count": self.sequence_count,
                "annotated_track_count": self.annotated_track_count,
                "query_count": self.query_count,
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output


def validate_team_benchmark_manifest(
    manifest_path: str | Path,
) -> TeamBenchmarkValidationReport:
    manifest = load_team_benchmark_manifest(manifest_path)
    issues: list[TeamBenchmarkIssue] = []
    if not manifest.sequences:
        issues.append(_error("empty_manifest", "Benchmark has no sequences."))
    for sequence in manifest.sequences:
        _validate_file(sequence.source_video, "missing_source_video", issues)
        if sequence.tracks_path is not None:
            _validate_file(sequence.tracks_path, "missing_tracks", issues)
        if sequence.mot_ground_truth_path is not None:
            _validate_file(sequence.mot_ground_truth_path, "missing_mot_ground_truth", issues)
        if not sequence.track_annotations:
            issues.append(
                _warning(
                    "no_track_team_annotations",
                    f"Sequence has no track team annotations: {sequence.sequence_name}",
                )
            )
        if not sequence.query_annotations:
            issues.append(
                _warning(
                    "no_language_queries",
                    f"Sequence has no language query annotations: {sequence.sequence_name}",
                )
            )
        seen_tracks: set[int] = set()
        for annotation in sequence.track_annotations:
            if annotation.track_id in seen_tracks:
                issues.append(
                    _error(
                        "duplicate_track_annotation",
                        (
                            "Duplicate team annotation for "
                            f"{sequence.sequence_name}/track {annotation.track_id}."
                        ),
                    )
                )
            seen_tracks.add(annotation.track_id)
            if annotation.end_frame > sequence.frame_count:
                issues.append(
                    _error(
                        "track_range_exceeds_sequence",
                        (
                            f"Track annotation exceeds frame_count in {sequence.sequence_name}: "
                            f"track {annotation.track_id}."
                        ),
                    )
                )
        seen_queries: set[str] = set()
        for query in sequence.query_annotations:
            if query.query_id in seen_queries:
                issues.append(
                    _error(
                        "duplicate_query_annotation",
                        f"Duplicate query annotation: {sequence.sequence_name}/{query.query_id}.",
                    )
                )
            seen_queries.add(query.query_id)
            if query.end_frame > sequence.frame_count:
                issues.append(
                    _error(
                        "query_range_exceeds_sequence",
                        (
                            "Query range exceeds frame_count: "
                            f"{sequence.sequence_name}/{query.query_id}."
                        ),
                    )
                )
    return TeamBenchmarkValidationReport(
        issues=tuple(issues),
        sequence_count=manifest.sequence_count,
        annotated_track_count=manifest.annotated_track_count,
        query_count=manifest.query_count,
    )


def validate_team_prediction_manifest(
    prediction_manifest_path: str | Path,
) -> TeamBenchmarkValidationReport:
    predictions = load_team_prediction_manifest(prediction_manifest_path)
    issues: list[TeamBenchmarkIssue] = []
    seen_tracks: set[tuple[str, int]] = set()
    for prediction in predictions.track_predictions:
        key = (prediction.sequence_name, prediction.track_id)
        if key in seen_tracks:
            issues.append(
                _error(
                    "duplicate_track_prediction",
                    (
                        "Duplicate track prediction: "
                        f"{prediction.sequence_name}/{prediction.track_id}."
                    ),
                )
            )
        seen_tracks.add(key)
    seen_queries: set[tuple[str, str]] = set()
    for prediction in predictions.query_predictions:
        key = (prediction.sequence_name, prediction.query_id)
        if key in seen_queries:
            issues.append(
                _error(
                    "duplicate_query_prediction",
                    (
                        "Duplicate query prediction: "
                        f"{prediction.sequence_name}/{prediction.query_id}."
                    ),
                )
            )
        seen_queries.add(key)
    return TeamBenchmarkValidationReport(
        issues=tuple(issues),
        sequence_count=0,
        annotated_track_count=len(predictions.track_predictions),
        query_count=len(predictions.query_predictions),
    )


def _validate_file(
    path: Path,
    code: str,
    issues: list[TeamBenchmarkIssue],
) -> None:
    if not path.is_file():
        issues.append(_error(code, f"Required file does not exist: {path}", str(path)))


def _error(code: str, message: str, path: str | None = None) -> TeamBenchmarkIssue:
    return TeamBenchmarkIssue("ERROR", code, message, path)


def _warning(code: str, message: str, path: str | None = None) -> TeamBenchmarkIssue:
    return TeamBenchmarkIssue("WARNING", code, message, path)
