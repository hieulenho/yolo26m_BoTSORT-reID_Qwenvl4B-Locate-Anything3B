from __future__ import annotations

from football_tracking.locate_tracking.semantic_memory.aggregator import (
    build_semantic_memory,
)
from football_tracking.locate_tracking.semantic_memory.decision_policy import (
    decide_final_resolution,
)
from football_tracking.locate_tracking.semantic_memory.schemas import SemanticMemoryConfig
from tests.locate_tracking.semantic_test_utils import (
    ambiguous_frame,
    frame_resolution,
    resolved_frame,
)


def test_clear_winner_resolved() -> None:
    config = SemanticMemoryConfig(min_usable_frames=2, min_support_frames=2)
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(resolved_frame(1, 7), resolved_frame(2, 7)),
        config=config,
    )

    assert decide_final_resolution(memory, config).status == "resolved"


def test_winner_below_min_support_frames_is_insufficient() -> None:
    config = SemanticMemoryConfig(min_usable_frames=1, min_support_frames=2)
    memory = build_semantic_memory(
        query="player", frame_resolutions=(resolved_frame(1, 7),), config=config
    )

    assert decide_final_resolution(memory, config).status == "insufficient_evidence"


def test_winner_below_min_aggregate_is_not_found() -> None:
    config = SemanticMemoryConfig(
        min_usable_frames=1,
        min_support_frames=1,
        min_aggregate_score=0.99,
    )
    memory = build_semantic_memory(
        query="player", frame_resolutions=(resolved_frame(1, 7, 0.4),), config=config
    )

    assert decide_final_resolution(memory, config).status == "not_found"


def test_top_two_within_margin_is_ambiguous() -> None:
    config = SemanticMemoryConfig(
        min_usable_frames=2,
        min_support_frames=1,
        min_support_ratio=0.0,
        min_aggregate_score=0.0,
        winner_margin=0.5,
    )
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(resolved_frame(1, 7, 0.8), resolved_frame(2, 3, 0.8)),
        config=config,
    )

    assert decide_final_resolution(memory, config).status == "ambiguous"


def test_no_candidates_is_not_found() -> None:
    config = SemanticMemoryConfig(min_usable_frames=0, min_support_frames=0)
    memory = build_semantic_memory(
        query="player", frame_resolutions=(frame_resolution(1, ()),), config=config
    )

    assert decide_final_resolution(memory, config).status == "not_found"


def test_no_usable_frames_is_insufficient() -> None:
    config = SemanticMemoryConfig(min_usable_frames=1, min_support_frames=0)
    memory = build_semantic_memory(
        query="player", frame_resolutions=(frame_resolution(1, ()),), config=config
    )

    assert decide_final_resolution(memory, config).status == "insufficient_evidence"


def test_multi_target_selects_all_tracks_passing_thresholds() -> None:
    config = SemanticMemoryConfig(
        query_mode="multi_target",
        min_usable_frames=3,
        min_support_frames=2,
        min_support_ratio=0.3,
        min_aggregate_score=0.3,
    )
    memory = build_semantic_memory(
        query="players",
        frame_resolutions=(
            resolved_frame(1, 3, 0.8),
            resolved_frame(2, 3, 0.8),
            resolved_frame(3, 7, 0.8),
            resolved_frame(4, 7, 0.8),
            resolved_frame(5, 11, 0.8),
            ambiguous_frame(6, 11, 12),
        ),
        config=config,
    )

    assert decide_final_resolution(memory, config).selected_track_ids == (3, 7)
