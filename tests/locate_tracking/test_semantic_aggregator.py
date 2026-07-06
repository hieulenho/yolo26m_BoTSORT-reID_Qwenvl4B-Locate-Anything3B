from __future__ import annotations

from football_tracking.locate_tracking.semantic_memory.aggregator import (
    build_semantic_memory,
)
from football_tracking.locate_tracking.semantic_memory.schemas import SemanticMemoryConfig
from tests.locate_tracking.semantic_test_utils import resolved_frame


def test_weighted_aggregator_components_and_no_nan() -> None:
    config = SemanticMemoryConfig(min_usable_frames=1, min_support_frames=1)
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(resolved_frame(1, 7, 0.8),),
        config=config,
    )
    candidate = memory.candidate_memories[0]

    assert 0.0 <= candidate.aggregate_score <= 1.0
    assert candidate.mean_score == 0.8
    assert candidate.top_k_mean_score == 0.8
    assert candidate.decision_metadata["aggregate_components"]["effective_weights"]


def test_top_k_larger_than_history_is_valid() -> None:
    config = SemanticMemoryConfig(top_k_quality=5, min_usable_frames=1, min_support_frames=1)
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(resolved_frame(1, 7, 0.6), resolved_frame(2, 7, 0.8)),
        config=config,
    )

    assert memory.candidate_memories[0].top_k_mean_score == 0.7


def test_exact_tie_is_deterministic_by_track_id() -> None:
    config = SemanticMemoryConfig(min_usable_frames=2, min_support_frames=1, winner_margin=0.0)
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(resolved_frame(1, 7, 0.8), resolved_frame(2, 3, 0.8)),
        config=config,
    )

    assert [item.raw_track_id for item in memory.sorted_candidates()] == [3, 7]


def test_majority_support_baseline_uses_support_count_first() -> None:
    config = SemanticMemoryConfig(
        aggregation_strategy="majority_support",
        min_usable_frames=3,
        min_support_frames=1,
        min_support_ratio=0.0,
    )
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(
            resolved_frame(1, 7, 0.6),
            resolved_frame(2, 7, 0.6),
            resolved_frame(3, 3, 0.95),
        ),
        config=config,
    )

    assert memory.sorted_candidates()[0].raw_track_id == 7
