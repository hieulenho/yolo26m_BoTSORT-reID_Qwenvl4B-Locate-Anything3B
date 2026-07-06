"""Fusion of semantic memory and appearance verification evidence."""

from football_tracking.locate_tracking.fusion.decision_policy import decide_fused_resolution
from football_tracking.locate_tracking.fusion.schemas import (
    FusedCandidateScore,
    FusionConfig,
    FusionResult,
)
from football_tracking.locate_tracking.fusion.score_fusion import fuse_scores

__all__ = [
    "FusedCandidateScore",
    "FusionConfig",
    "FusionResult",
    "decide_fused_resolution",
    "fuse_scores",
]
