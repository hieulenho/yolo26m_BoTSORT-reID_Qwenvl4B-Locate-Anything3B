"""Text summary writer for appearance verification artifacts."""

from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.appearance.schemas import AppearanceVerificationResult
from football_tracking.locate_tracking.fusion.schemas import FusionResult


def write_appearance_summary(
    appearance_result: AppearanceVerificationResult,
    fusion_result: FusionResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Appearance Verification: {fusion_result.status}",
        "",
        f"- Query: {appearance_result.query}",
        f"- Selected track ids: {list(fusion_result.selected_track_ids)}",
        f"- Decision: {fusion_result.decision_reason}",
        f"- Backend: {appearance_result.runtime_info.backend_name}",
        f"- Model: {appearance_result.runtime_info.model_id}",
        f"- Crops: {appearance_result.runtime_info.crop_count}",
        f"- Cache hits: {appearance_result.runtime_info.cache_hits}",
        f"- Cache misses: {appearance_result.runtime_info.cache_misses}",
        "",
        "## Candidates",
    ]
    for candidate in fusion_result.candidate_scores:
        lines.append(
            f"- Track {candidate.raw_track_id}: fused={candidate.fused_score:.3f}, "
            f"semantic={candidate.semantic_score:.3f}, "
            f"appearance={candidate.appearance_score}, status={candidate.appearance_status}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
