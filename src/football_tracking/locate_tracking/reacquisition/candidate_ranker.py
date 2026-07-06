"""Deterministic multi-source candidate ranking."""

from __future__ import annotations

from football_tracking.locate_tracking.reacquisition.schemas import (
    EvidenceScore,
    ReacquisitionCandidate,
    ReacquisitionConfig,
)


def _score_map(candidate: ReacquisitionCandidate) -> dict[str, EvidenceScore | None]:
    return {
        "grounding": candidate.grounding_evidence,
        "appearance": candidate.appearance_evidence,
        "motion": candidate.motion_evidence,
        "temporal": candidate.temporal_evidence,
        "history": candidate.history_evidence,
    }


def final_candidate_score(
    candidate: ReacquisitionCandidate,
    config: ReacquisitionConfig,
) -> tuple[float | None, dict[str, float | None], dict[str, float]]:
    weighted = 0.0
    total_weight = 0.0
    components: dict[str, float | None] = {}
    effective_weights: dict[str, float] = {}
    for name, evidence in _score_map(candidate).items():
        weight = config.weights.get(name, 0.0)
        if weight <= 0.0:
            continue
        score = evidence.score if evidence is not None else None
        components[name] = score
        if score is None:
            if config.missing_evidence_policy == "zero":
                total_weight += weight
            continue
        weighted += weight * score
        total_weight += weight
    if total_weight <= 0.0:
        return None, components, effective_weights
    for name, evidence in _score_map(candidate).items():
        if name not in config.weights:
            continue
        if evidence is not None and evidence.score is not None:
            effective_weights[name] = config.weights[name] / total_weight
        elif config.missing_evidence_policy == "zero":
            effective_weights[name] = config.weights[name] / total_weight
    return weighted / total_weight, components, effective_weights


def rank_candidates(
    candidates: tuple[ReacquisitionCandidate, ...],
    config: ReacquisitionConfig,
) -> tuple[ReacquisitionCandidate, ...]:
    passed = [candidate for candidate in candidates if candidate.passed_gates]
    ranked: list[ReacquisitionCandidate] = []
    for candidate in passed:
        score, components, effective_weights = final_candidate_score(candidate, config)
        ranked.append(
            candidate.with_updates(
                final_score=score,
                component_scores=components,
                status="ranked",
                rejection_reasons=(),
                history_evidence=candidate.history_evidence,
            )
        )
    ranked.sort(
        key=lambda item: (
            -(item.final_score if item.final_score is not None else -1.0),
            item.first_observed_frame,
            item.raw_track_id,
        )
    )
    return tuple(
        candidate.with_updates(rank=index + 1, status="winner" if index == 0 else "ranked")
        for index, candidate in enumerate(ranked)
    )
