"""Severity helpers for event filtering."""

from __future__ import annotations

from football_tracking.locate_tracking.monitoring.schemas import Severity
from football_tracking.locate_tracking.monitoring.signal_utils import SEVERITY_RANK


def severity_at_least(severity: Severity, minimum: str) -> bool:
    if minimum not in SEVERITY_RANK:
        raise ValueError(f"Unknown severity: {minimum}")
    return SEVERITY_RANK[severity] >= SEVERITY_RANK[minimum]  # type: ignore[index]


def max_severity(left: Severity, right: Severity) -> Severity:
    return left if SEVERITY_RANK[left] >= SEVERITY_RANK[right] else right
