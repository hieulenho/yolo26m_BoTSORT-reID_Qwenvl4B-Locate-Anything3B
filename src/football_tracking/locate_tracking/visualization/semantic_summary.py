"""Simple text summary writer for semantic memory artifacts."""

from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.semantic_memory.schemas import (
    FinalLanguageTrackResolution,
    SemanticMemory,
)


def write_semantic_summary(
    semantic_memory: SemanticMemory,
    final_resolution: FinalLanguageTrackResolution,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Semantic Track Resolution: {final_resolution.status}",
        "",
        f"- Query: {semantic_memory.query}",
        f"- Mode: {semantic_memory.query_mode}",
        f"- Selected track ids: {list(final_resolution.selected_track_ids)}",
        f"- Reason: {final_resolution.decision_reason}",
        f"- Usable grounding frames: {semantic_memory.usable_grounding_frame_count}",
        "",
        "## Candidates",
    ]
    for candidate in semantic_memory.sorted_candidates():
        lines.append(
            f"- Track {candidate.raw_track_id}: aggregate={candidate.aggregate_score:.3f}, "
            f"support={candidate.support_count}, ratio={candidate.support_ratio:.3f}, "
            f"mean_score={candidate.mean_score:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
