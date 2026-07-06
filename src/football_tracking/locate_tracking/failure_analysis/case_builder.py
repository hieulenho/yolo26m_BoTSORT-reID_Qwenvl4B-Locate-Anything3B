"""Build failure cases from per-query metric rows."""

from __future__ import annotations

from football_tracking.locate_tracking.failure_analysis.classifier import classify_failure
from football_tracking.locate_tracking.failure_analysis.evidence_collector import (
    collect_failure_evidence,
)
from football_tracking.locate_tracking.failure_analysis.schemas import FailureCase


def build_failure_cases(rows: list[dict]) -> tuple[FailureCase, ...]:
    cases: list[FailureCase] = []
    for row in rows:
        case = classify_failure(row)
        if case is None:
            continue
        cases.append(
            FailureCase(
                sequence_name=case.sequence_name,
                query_id=case.query_id,
                category=case.category,
                reason=case.reason,
                severity=case.severity,
                evidence={**collect_failure_evidence(row), **case.evidence},
            )
        )
    return tuple(cases)
