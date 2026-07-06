"""Convert single-frame association results into semantic evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from football_tracking.locate_tracking.association.schemas import FrameQueryResolution
from football_tracking.locate_tracking.semantic_memory.schemas import (
    SemanticEvidence,
    SemanticMemoryConfig,
)


def _as_dict(resolution: FrameQueryResolution | Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(resolution, "to_dict"):
        return resolution.to_dict()  # type: ignore[no-any-return]
    return dict(resolution)


def _candidate_score(candidate: Mapping[str, Any]) -> float | None:
    value = candidate.get("final_score")
    return None if value is None else float(value)


def _candidate_metrics(candidate: Mapping[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics")
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def evidence_from_frame_resolution(
    resolution: FrameQueryResolution | Mapping[str, Any],
    config: SemanticMemoryConfig | None = None,
) -> tuple[SemanticEvidence, ...]:
    """Return candidate-level evidence without turning ambiguity into a vote."""

    cfg = config or SemanticMemoryConfig()
    data = _as_dict(resolution)
    grounding = dict(data.get("grounding", {}))
    frame_index = int(data["frame_index"])
    query = str(data["query"])
    cache_hit = bool(grounding.get("cache_hit", False))
    evidence: list[SemanticEvidence] = []
    for association in data.get("associations", []):
        assoc = dict(association)
        status = str(assoc.get("status", "not_found"))
        grounded_box_index = int(assoc.get("grounded_box_index", 0))
        grounded_label = str(assoc.get("label", assoc.get("grounded_label", query)))
        selected_track_id = assoc.get("selected_track_id")
        for candidate_data in assoc.get("candidates", []):
            candidate = dict(candidate_data)
            track_id = int(candidate["track_id"])
            selected = status == "resolved" and selected_track_id == track_id
            passed_gate = bool(candidate.get("passed_gate", False))
            if selected:
                weight = cfg.resolved_selected_weight
                reason = "resolved_selected_candidate"
            elif status == "ambiguous" and passed_gate:
                weight = cfg.ambiguous_candidate_weight
                reason = "ambiguous_candidate_preserved_without_positive_vote"
            elif passed_gate:
                weight = cfg.weak_candidate_weight
                reason = "non_selected_passed_candidate"
            else:
                weight = 0.0
                reason = str(candidate.get("gate_reason", "candidate_failed_gate"))
            metrics = _candidate_metrics(candidate)
            evidence.append(
                SemanticEvidence(
                    query=query,
                    frame_index=frame_index,
                    grounded_box_index=grounded_box_index,
                    grounded_label=grounded_label,
                    raw_track_id=track_id,
                    single_frame_status=status,
                    association_score=_candidate_score(candidate),
                    iou=metrics.get("iou"),
                    track_coverage=metrics.get("track_coverage"),
                    center_similarity=metrics.get("center_similarity"),
                    candidate_rank=candidate.get("rank"),
                    passed_gate=passed_gate,
                    selected_in_frame=selected,
                    grounding_cache_hit=cache_hit,
                    evidence_weight=weight,
                    evidence_reason=reason,
                )
            )
    return tuple(evidence)
