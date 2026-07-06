"""Creation helpers for semantic target identity artifacts."""

from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.identity.schemas import (
    SemanticIdentitySegment,
    SemanticTarget,
    stable_artifact_id,
    stable_identity_id,
)
from football_tracking.locate_tracking.semantic_memory.serialization import load_semantic_memory


def create_initial_semantic_target(
    *,
    query: str,
    raw_track_id: int,
    start_frame: int,
    end_frame: int | None = None,
    semantic_memory_path: str | Path | None = None,
    appearance_reference_path: str | Path | None = None,
    semantic_target_id: str | None = None,
) -> SemanticTarget:
    query_mode = "single_target"
    if semantic_memory_path is not None:
        memory = load_semantic_memory(semantic_memory_path)
        query = memory.query
        query_mode = memory.query_mode
    target_id = semantic_target_id or stable_identity_id(query)
    segment = SemanticIdentitySegment(
        segment_id=stable_artifact_id(
            "segment",
            {
                "semantic_target_id": target_id,
                "raw_track_id": raw_track_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
            },
        ),
        semantic_target_id=target_id,
        raw_track_id=raw_track_id,
        start_frame=start_frame,
        end_frame=end_frame,
        source="initial_semantic_resolution",
        confidence=1.0,
        status="confirmed",
        evidence_summary={
            "semantic_memory": str(semantic_memory_path) if semantic_memory_path else None,
        },
    )
    return SemanticTarget(
        semantic_target_id=target_id,
        query=query,
        query_mode=query_mode,
        state="ACTIVE",
        current_raw_track_id=int(raw_track_id),
        segments=(segment,),
        reference_semantic_memory=str(semantic_memory_path) if semantic_memory_path else None,
        reference_appearance_prototype=(
            str(appearance_reference_path) if appearance_reference_path else None
        ),
        last_confirmed_frame=end_frame or start_frame,
        last_update_frame=end_frame or start_frame,
        metadata={"created_by": "create_initial_semantic_target"},
    )
