"""Schemas for deterministic language tracking failure cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FailureCase:
    sequence_name: str
    query_id: str
    category: str
    reason: str
    severity: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_name": self.sequence_name,
            "query_id": self.query_id,
            "category": self.category,
            "reason": self.reason,
            "severity": self.severity,
            "evidence": dict(self.evidence),
        }
