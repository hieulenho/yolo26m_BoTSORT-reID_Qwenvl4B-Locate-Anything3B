"""Appearance evidence from frozen M4 appearance artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.appearance.schemas import AppearanceVerificationResult
from football_tracking.locate_tracking.reacquisition.schemas import (
    EvidenceScore,
    ReacquisitionCandidate,
)


def load_appearance_result(path: str | Path | None) -> AppearanceVerificationResult | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    return AppearanceVerificationResult.from_dict(json.loads(resolved.read_text(encoding="utf-8")))


def appearance_evidence(
    *,
    candidate: ReacquisitionCandidate,
    appearance_result: AppearanceVerificationResult | None,
) -> EvidenceScore:
    if appearance_result is None:
        return EvidenceScore(
            name="appearance",
            score=None,
            data_available=False,
            reason="appearance_artifact_unavailable",
        )
    for item in appearance_result.candidate_scores:
        if item.raw_track_id == candidate.raw_track_id:
            return EvidenceScore(
                name="appearance",
                score=item.appearance_score,
                data_available=item.appearance_score is not None,
                reason=item.evidence_status,
                details={
                    "sample_count": item.sample_count,
                    "prototype_similarity": item.prototype_similarity,
                    "internal_consistency": item.internal_consistency,
                    "prototype_mutated": False,
                },
            )
    return EvidenceScore(
        name="appearance",
        score=None,
        data_available=False,
        reason="candidate_not_in_frozen_appearance_artifact",
        details={"prototype_mutated": False},
    )
