from __future__ import annotations

import pytest

from football_tracking.locate_tracking.semantic_memory.aggregator import (
    build_semantic_memory,
)
from football_tracking.locate_tracking.semantic_memory.candidate_memory import (
    SemanticMemoryBuilder,
    SemanticMemoryBuilderError,
)
from football_tracking.locate_tracking.semantic_memory.decision_policy import (
    decide_final_resolution,
)
from football_tracking.locate_tracking.semantic_memory.schemas import SemanticMemoryConfig
from tests.locate_tracking.semantic_test_utils import ambiguous_frame, resolved_frame


def test_stable_winner_memory_resolves() -> None:
    config = SemanticMemoryConfig(min_usable_frames=2, min_support_frames=2)
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(resolved_frame(1, 7), resolved_frame(2, 7), resolved_frame(3, 7)),
        config=config,
    )
    final = decide_final_resolution(memory, config)

    assert final.status == "resolved"
    assert final.selected_track_id == 7
    assert memory.candidate_memories[0].support_count == 3


def test_one_false_frame_still_keeps_stable_winner() -> None:
    config = SemanticMemoryConfig(
        min_usable_frames=3,
        min_support_frames=2,
        min_support_ratio=0.5,
        winner_margin=0.05,
    )
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(resolved_frame(1, 7), resolved_frame(2, 7), resolved_frame(3, 3, 0.6)),
        config=config,
    )
    final = decide_final_resolution(memory, config)

    assert final.status == "resolved"
    assert final.selected_track_id == 7


def test_repeated_ambiguity_is_not_forced_to_resolved() -> None:
    config = SemanticMemoryConfig(min_usable_frames=2, min_support_frames=1)
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(ambiguous_frame(1), ambiguous_frame(2)),
        config=config,
    )
    final = decide_final_resolution(memory, config)

    assert final.status in {"insufficient_evidence", "not_found"}
    assert final.selected_track_id is None


def test_too_little_positive_evidence_is_insufficient() -> None:
    config = SemanticMemoryConfig(min_usable_frames=2, min_support_frames=2)
    memory = build_semantic_memory(
        query="player", frame_resolutions=(resolved_frame(1, 7),), config=config
    )
    final = decide_final_resolution(memory, config)

    assert final.status == "insufficient_evidence"


def test_duplicate_frame_update_is_rejected() -> None:
    builder = SemanticMemoryBuilder(query="player")
    builder.add_frame_resolution(resolved_frame(1, 7))

    with pytest.raises(SemanticMemoryBuilderError):
        builder.add_frame_resolution(resolved_frame(1, 7))


def test_incremental_builder_matches_batch_memory() -> None:
    config = SemanticMemoryConfig(min_usable_frames=2, min_support_frames=2)
    frames = (resolved_frame(1, 7), resolved_frame(2, 7))
    builder = SemanticMemoryBuilder(query="player", config=config)
    builder.extend(frames)

    incremental = builder.build()
    batch = build_semantic_memory(query="player", frame_resolutions=frames, config=config)

    assert incremental.to_dict() == batch.to_dict()
