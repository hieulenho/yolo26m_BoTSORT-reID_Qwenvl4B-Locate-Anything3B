"""Spatial plausibility gate for grounding-backed reacquisition."""

from __future__ import annotations

from football_tracking.locate_tracking.reacquisition.grounding_evidence import grounding_gate
from football_tracking.locate_tracking.reacquisition.schemas import (
    EvidenceScore,
    GateResult,
    ReacquisitionConfig,
)


def spatial_grounding_gate(
    evidence: EvidenceScore,
    config: ReacquisitionConfig,
) -> GateResult:
    base = grounding_gate(evidence, config)
    return GateResult(
        gate_name="spatial_grounding",
        passed=base.passed,
        score=base.score,
        threshold=base.threshold,
        reason=base.reason,
        metadata=dict(base.metadata),
    )
