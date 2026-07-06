"""Aggregate frame-level evidence into semantic track memories."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from football_tracking.locate_tracking.association.schemas import FrameQueryResolution
from football_tracking.locate_tracking.semantic_memory.evidence import (
    evidence_from_frame_resolution,
)
from football_tracking.locate_tracking.semantic_memory.schemas import (
    CandidateSemanticMemory,
    SemanticEvidence,
    SemanticMemory,
    SemanticMemoryConfig,
)


def _as_dict(resolution: FrameQueryResolution | Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(resolution, "to_dict"):
        return resolution.to_dict()  # type: ignore[no-any-return]
    return dict(resolution)


def _has_usable_grounding(data: Mapping[str, Any]) -> bool:
    return int(data.get("grounded_box_count", 0)) > 0


def _has_resolved_association(data: Mapping[str, Any]) -> bool:
    return any(
        dict(association).get("status") == "resolved"
        for association in data.get("associations", [])
    )


def _is_ambiguous_only(data: Mapping[str, Any]) -> bool:
    associations = [dict(item) for item in data.get("associations", [])]
    return (
        bool(associations)
        and any(item.get("status") == "ambiguous" for item in associations)
        and not any(item.get("status") == "resolved" for item in associations)
    )


def _majority_rank(memory: CandidateSemanticMemory) -> tuple[int, float, float, int]:
    return (
        -memory.support_count,
        -memory.mean_score,
        -memory.best_score,
        memory.raw_track_id,
    )


def _weighted_rank(memory: CandidateSemanticMemory) -> tuple[float, int, float, int]:
    return (
        -memory.aggregate_score,
        -memory.support_count,
        -memory.mean_score,
        memory.raw_track_id,
    )


def _rank_memories(
    memories: tuple[CandidateSemanticMemory, ...],
    config: SemanticMemoryConfig,
) -> tuple[CandidateSemanticMemory, ...]:
    key = _majority_rank if config.aggregation_strategy == "majority_support" else _weighted_rank
    ranked = sorted(memories, key=key)
    output: list[CandidateSemanticMemory] = []
    for rank, memory in enumerate(ranked, 1):
        output.append(
            memory.with_decision_metadata(
                {
                    "rank": rank,
                    "ranking_strategy": config.aggregation_strategy,
                    "majority_tie_breaker": [
                        "support_count_desc",
                        "mean_association_score_desc",
                        "best_association_score_desc",
                        "raw_track_id_asc",
                    ],
                }
                if config.aggregation_strategy == "majority_support"
                else {"rank": rank, "ranking_strategy": config.aggregation_strategy}
            )
        )
    return tuple(output)


def build_semantic_memory(
    *,
    query: str,
    frame_resolutions: tuple[FrameQueryResolution | Mapping[str, Any], ...],
    config: SemanticMemoryConfig | None = None,
    sampled_frames: tuple[int, ...] | None = None,
    runtime_info: dict[str, Any] | None = None,
) -> SemanticMemory:
    cfg = config or SemanticMemoryConfig()
    resolution_dicts = tuple(_as_dict(item) for item in frame_resolutions)
    sampled = (
        tuple(sorted(set(int(item) for item in sampled_frames)))
        if sampled_frames is not None
        else tuple(sorted({int(item["frame_index"]) for item in resolution_dicts}))
    )
    usable_count = sum(1 for item in resolution_dicts if _has_usable_grounding(item))
    resolved_count = sum(1 for item in resolution_dicts if _has_resolved_association(item))
    ambiguous_count = sum(1 for item in resolution_dicts if _is_ambiguous_only(item))
    not_found_count = sum(
        1 for item in resolution_dicts if str(item.get("overall_status")) == "not_found"
    )
    grouped: dict[int, list[SemanticEvidence]] = defaultdict(list)
    for resolution in resolution_dicts:
        for evidence in evidence_from_frame_resolution(resolution, cfg):
            grouped[evidence.raw_track_id].append(evidence)
    memories = tuple(
        CandidateSemanticMemory.from_evidence(
            raw_track_id=track_id,
            evidence_history=tuple(history),
            sampled_frames=sampled,
            usable_grounding_frame_count=usable_count,
            config=cfg,
        )
        for track_id, history in sorted(grouped.items())
    )
    ranked_memories = _rank_memories(memories, cfg)
    return SemanticMemory(
        query=query,
        query_mode=cfg.query_mode,
        planned_frame_count=len(sampled),
        processed_frame_count=len(resolution_dicts),
        usable_grounding_frame_count=usable_count,
        resolved_frame_count=resolved_count,
        ambiguous_frame_count=ambiguous_count,
        not_found_frame_count=not_found_count,
        sampled_frames=sampled,
        candidate_memories=ranked_memories,
        aggregation_config=cfg.to_dict(),
        runtime_info=dict(runtime_info or {}),
    )
