"""Validation for language benchmark manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.benchmark.manifest import load_benchmark_manifest
from football_tracking.locate_tracking.benchmark.schemas import (
    LanguageQueryAnnotation,
    LanguageTrackingBenchmarkManifest,
    LanguageTrackingSequence,
)


class LanguageBenchmarkValidationError(RuntimeError):
    """Raised when a benchmark manifest is invalid in fail-fast mode."""


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
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
class BenchmarkValidationReport:
    manifest_path: Path
    sequence_count: int
    query_count: int
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

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
            "manifest_path": str(self.manifest_path),
            "summary": {
                "errors": self.error_count,
                "warnings": self.warning_count,
                "sequence_count": self.sequence_count,
                "query_count": self.query_count,
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_benchmark_manifest(
    manifest_path: str | Path,
    *,
    fail_fast: bool = False,
) -> BenchmarkValidationReport:
    path = Path(manifest_path)
    issues: list[ValidationIssue] = []
    try:
        manifest = load_benchmark_manifest(path)
    except Exception as exc:  # noqa: BLE001
        report = BenchmarkValidationReport(
            manifest_path=path,
            sequence_count=0,
            query_count=0,
            issues=(
                ValidationIssue(
                    severity="ERROR",
                    code="manifest_load_failed",
                    message=str(exc),
                    path=str(path),
                ),
            ),
        )
        if fail_fast:
            raise LanguageBenchmarkValidationError(str(exc)) from exc
        return report

    _validate_manifest(manifest, issues)
    report = BenchmarkValidationReport(
        manifest_path=path,
        sequence_count=manifest.sequence_count,
        query_count=manifest.query_count,
        issues=tuple(issues),
    )
    if fail_fast and report.has_errors:
        raise LanguageBenchmarkValidationError("Language benchmark validation failed.")
    return report


def _validate_manifest(
    manifest: LanguageTrackingBenchmarkManifest,
    issues: list[ValidationIssue],
) -> None:
    if not manifest.sequences:
        issues.append(_error("empty_manifest", "Benchmark manifest has no sequences."))
    seen_sequences: set[str] = set()
    seen_queries: set[str] = set()
    for sequence in manifest.sequences:
        _validate_sequence(sequence, seen_sequences, seen_queries, issues)


def _validate_sequence(
    sequence: LanguageTrackingSequence,
    seen_sequences: set[str],
    seen_queries: set[str],
    issues: list[ValidationIssue],
) -> None:
    if sequence.sequence_name in seen_sequences:
        issues.append(_error("duplicate_sequence", f"Duplicate sequence: {sequence.sequence_name}"))
    seen_sequences.add(sequence.sequence_name)
    _require_file(sequence.source_video, "missing_source_reference", issues)
    _require_file(sequence.mot_ground_truth_path, "missing_gt_mot", issues)
    if not sequence.queries:
        issues.append(_warning("sequence_has_no_queries", sequence.sequence_name))
    for query in sequence.queries:
        _validate_query(query, sequence, seen_queries, issues)


def _validate_query(
    query: LanguageQueryAnnotation,
    sequence: LanguageTrackingSequence,
    seen_queries: set[str],
    issues: list[ValidationIssue],
) -> None:
    if query.query_id in seen_queries:
        issues.append(_error("duplicate_query_id", f"Duplicate query_id: {query.query_id}"))
    seen_queries.add(query.query_id)
    if query.evaluation_end_frame > sequence.frame_count:
        issues.append(
            _error(
                "query_range_exceeds_sequence",
                f"{query.query_id} ends after sequence frame_count.",
            )
        )
    if not query.identity_segments:
        issues.append(_error("query_has_no_identity_segments", query.query_id))
    previous_end = 0
    for segment in sorted(query.identity_segments, key=lambda item: item.start_frame):
        if segment.gt_track_id not in query.target_gt_track_ids:
            issues.append(_error("segment_gt_id_not_in_target_set", query.query_id))
        if segment.start_frame <= previous_end:
            issues.append(_error("identity_segment_chronology", query.query_id))
        previous_end = segment.end_frame
    loss_ids = {event.event_id for event in query.loss_events}
    for event in query.reacquisition_events:
        if event.loss_event_id is not None and event.loss_event_id not in loss_ids:
            issues.append(_error("unknown_loss_event_reference", event.event_id))
        if event.evaluation_end_frame > sequence.frame_count:
            issues.append(_error("reacquisition_range_exceeds_sequence", event.event_id))


def _require_file(path: Path, code: str, issues: list[ValidationIssue]) -> None:
    if not path.is_file():
        issues.append(_error(code, f"Required file does not exist: {path}", path=str(path)))


def _error(code: str, message: str, path: str | None = None) -> ValidationIssue:
    return ValidationIssue("ERROR", code, message, path)


def _warning(code: str, message: str, path: str | None = None) -> ValidationIssue:
    return ValidationIssue("WARNING", code, message, path)
