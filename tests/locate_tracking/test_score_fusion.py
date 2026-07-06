from __future__ import annotations

from dataclasses import replace

from football_tracking.locate_tracking.appearance.schemas import AppearanceCandidateScore
from football_tracking.locate_tracking.fusion.decision_policy import decide_fused_resolution
from football_tracking.locate_tracking.fusion.schemas import FusionConfig
from football_tracking.locate_tracking.fusion.score_fusion import fuse_scores
from football_tracking.locate_tracking.semantic_memory.aggregator import (
    build_semantic_memory,
)
from football_tracking.locate_tracking.semantic_memory.schemas import SemanticMemoryConfig
from tests.locate_tracking.semantic_test_utils import resolved_frame


def _appearance(track_id: int, score: float | None) -> AppearanceCandidateScore:
    return AppearanceCandidateScore(
        raw_track_id=track_id,
        prototype_similarity=score,
        internal_consistency=score,
        appearance_score=score,
        sample_count=2 if score is not None else 0,
        evidence_status="verified" if score is not None else "unavailable",
        decision_reason="test",
    )


def _memory():
    return build_semantic_memory(
        query="player",
        frame_resolutions=(resolved_frame(1, 7, 0.9), resolved_frame(2, 11, 0.5)),
        config=SemanticMemoryConfig(
            min_usable_frames=1,
            min_support_frames=0,
            min_support_ratio=0,
            min_aggregate_score=0,
            winner_margin=0,
        ),
    )


def test_fusion_semantic_and_appearance_strong_stable_winner() -> None:
    memory = _memory()
    config = FusionConfig(min_fused_score=0.0, winner_margin=0.0)

    scores = fuse_scores(
        semantic_memory=memory,
        appearance_scores=(_appearance(7, 0.95), _appearance(11, 0.4)),
        config=config,
    )
    result = decide_fused_resolution(
        query="player",
        fused_scores=scores,
        semantic_memory_reference="semantic.json",
        appearance_scores_reference="appearance.json",
        config=config,
    )

    assert result.status == "resolved"
    assert result.selected_track_id == 7


def test_fusion_appearance_can_disambiguate_semantic_tie() -> None:
    memory = _memory()
    tied = tuple(replace(candidate, aggregate_score=0.5) for candidate in memory.candidate_memories)
    memory = replace(memory, candidate_memories=tied)
    config = FusionConfig(min_fused_score=0.0, winner_margin=0.01)

    scores = fuse_scores(
        semantic_memory=memory,
        appearance_scores=(_appearance(7, 0.95), _appearance(11, 0.5)),
        config=config,
    )
    result = decide_fused_resolution(
        query="player",
        fused_scores=scores,
        semantic_memory_reference="semantic.json",
        appearance_scores_reference="appearance.json",
        config=config,
    )

    assert result.status == "resolved"
    assert result.selected_track_id == 7


def test_fusion_missing_appearance_falls_back_to_semantic_or_penalty() -> None:
    memory = _memory()
    semantic_only = FusionConfig(missing_appearance_policy="semantic_only")
    penalized = FusionConfig(missing_appearance_policy="penalize", missing_appearance_penalty=0.2)

    keep = fuse_scores(semantic_memory=memory, appearance_scores=(), config=semantic_only)[0]
    penalty = fuse_scores(semantic_memory=memory, appearance_scores=(), config=penalized)[0]

    assert keep.fused_score == keep.semantic_score
    assert penalty.fused_score < penalty.semantic_score


def test_fusion_top_scores_inside_margin_are_ambiguous() -> None:
    memory = _memory()
    tied = tuple(replace(candidate, aggregate_score=0.5) for candidate in memory.candidate_memories)
    memory = replace(memory, candidate_memories=tied)
    config = FusionConfig(min_fused_score=0.0, winner_margin=0.5)

    scores = fuse_scores(semantic_memory=memory, appearance_scores=(), config=config)
    result = decide_fused_resolution(
        query="player",
        fused_scores=scores,
        semantic_memory_reference="semantic.json",
        appearance_scores_reference="appearance.json",
        config=config,
    )

    assert result.status == "ambiguous"
