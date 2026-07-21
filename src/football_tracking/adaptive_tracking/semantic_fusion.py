"""Temporal/source fusion and unknown rejection for track semantics."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from football_tracking.adaptive_tracking.ontology import VocabularyRegistry
from football_tracking.paths import get_project_root


class SemanticFusionError(RuntimeError):
    """Raised when semantic evidence cannot be parsed or fused."""


@dataclass(frozen=True)
class TrackSemanticEvidence:
    track_id: int
    class_label: str
    confidence: float
    source: str
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence_frames: tuple[int, ...] = ()
    evidence: str = ""
    fine_label: str = "unknown"
    fine_confidence: float = 0.0
    taxonomy_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.track_id) <= 0:
            raise SemanticFusionError("track_id must be positive.")
        label = str(self.class_label).strip().lower().replace("_", " ")
        if not label:
            label = "unknown"
        confidence = float(self.confidence)
        if not math.isfinite(confidence):
            confidence = 0.0
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "class_label", label)
        object.__setattr__(self, "confidence", min(max(confidence, 0.0), 1.0))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "attributes", dict(self.attributes))
        fine_label = str(self.fine_label).strip().lower().replace("_", " ") or "unknown"
        fine_confidence = float(self.fine_confidence)
        if not math.isfinite(fine_confidence):
            fine_confidence = 0.0
        object.__setattr__(self, "fine_label", fine_label)
        object.__setattr__(
            self,
            "fine_confidence",
            min(max(fine_confidence, 0.0), 1.0),
        )
        object.__setattr__(
            self,
            "taxonomy_path",
            tuple(
                value
                for item in self.taxonomy_path
                if (value := str(item).strip().lower().replace("_", " "))
            ),
        )
        object.__setattr__(
            self,
            "evidence_frames",
            tuple(int(item) for item in self.evidence_frames),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_qwen_answer(data: dict[str, Any] | str) -> list[TrackSemanticEvidence]:
    """Parse either a raw Qwen answer or the persisted vlm_answer.json wrapper."""
    if isinstance(data, dict):
        raw = data.get("answer", data)
    else:
        raw = data
    if isinstance(raw, dict):
        parsed = raw
    else:
        match = re.search(r"\{.*\}", str(raw), re.DOTALL)
        if not match:
            raise SemanticFusionError("No JSON object found in Qwen semantic answer.")
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise SemanticFusionError(f"Invalid Qwen semantic JSON: {exc}") from exc
    rows = parsed.get("track_predictions", [])
    if not isinstance(rows, list):
        raise SemanticFusionError("track_predictions must be a list.")
    evidence: list[TrackSemanticEvidence] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("track_id") is None:
            continue
        if row.get("accepted_for_fusion") is False:
            continue
        observations = row.get("observations")
        if isinstance(observations, list) and observations:
            observation_count = 0
            for observation in observations:
                if not isinstance(observation, dict):
                    continue
                if observation.get("accepted_for_fusion") is False:
                    continue
                evidence.append(
                    _qwen_evidence_from_row(
                        track_id=int(row["track_id"]),
                        row=observation,
                        fallback=row,
                    )
                )
                observation_count += 1
            if observation_count:
                continue
        evidence.append(
            _qwen_evidence_from_row(
                track_id=int(row["track_id"]),
                row=row,
                fallback=None,
            )
        )
    return evidence


def _qwen_evidence_from_row(
    *,
    track_id: int,
    row: dict[str, Any],
    fallback: dict[str, Any] | None,
) -> TrackSemanticEvidence:
    parent = fallback or {}
    label = row.get("class_label", parent.get("class_label"))
    if label is None:
        role = str(row.get("role_label", parent.get("role_label", "unknown")))
        team = str(row.get("team_label", parent.get("team_label", "unknown")))
        label = role if team == "unknown" else f"{team} {role}"
    attributes = dict(parent.get("attributes", {}))
    attributes.update(dict(row.get("attributes", {})))
    frame_value = row.get("frame_index")
    evidence_frames = (
        (int(frame_value),)
        if frame_value is not None
        else tuple(row.get("evidence_frames", parent.get("evidence_frames", ())))
    )
    return TrackSemanticEvidence(
        track_id=track_id,
        class_label=str(label),
        confidence=float(row.get("confidence", parent.get("confidence", 0.0))),
        source="qwen",
        attributes=attributes,
        evidence_frames=evidence_frames,
        evidence=str(row.get("evidence", parent.get("evidence", ""))),
        fine_label=str(
            row.get(
                "fine_label",
                row.get(
                    "fine_grained_label",
                    parent.get(
                        "fine_label",
                        parent.get("fine_grained_label", "unknown"),
                    ),
                ),
            )
        ),
        fine_confidence=float(
            row.get("fine_confidence", parent.get("fine_confidence", 0.0))
        ),
        taxonomy_path=tuple(row.get("taxonomy_path", parent.get("taxonomy_path", ()))),
    )


def parse_locate_evidence(data: dict[str, Any]) -> list[TrackSemanticEvidence]:
    rows = data.get("associations", [])
    evidence: list[TrackSemanticEvidence] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("track_id") is None:
            continue
        if row.get("accepted_for_fusion") is False:
            continue
        evidence.append(
            TrackSemanticEvidence(
                track_id=int(row["track_id"]),
                class_label=str(row.get("class_label", "unknown")),
                confidence=float(row.get("confidence", 0.0)),
                source="locateanything",
                evidence_frames=(int(row["frame_index"]),),
                evidence=f"grounding IoU={float(row.get('iou', 0.0)):.3f}",
            )
        )
    return evidence


def fuse_track_semantics(
    evidence: list[TrackSemanticEvidence],
    *,
    unknown_threshold: float = 0.45,
    minimum_margin: float = 0.10,
    temporal_half_life_frames: float = 900.0,
    minimum_temporal_stability: float = 0.60,
    fine_unknown_threshold: float = 0.85,
    fine_minimum_margin: float = 0.15,
    fine_minimum_temporal_stability: float = 0.67,
    source_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fuse repeated observations, then reject weak or ambiguous labels."""
    if temporal_half_life_frames <= 0:
        raise SemanticFusionError("temporal_half_life_frames must be positive.")
    if not 0.0 <= minimum_temporal_stability <= 1.0:
        raise SemanticFusionError("minimum_temporal_stability must be in [0, 1].")
    if not 0.0 <= fine_minimum_temporal_stability <= 1.0:
        raise SemanticFusionError("fine_minimum_temporal_stability must be in [0, 1].")
    weights = {"qwen": 1.0, "locateanything": 0.9, **(source_weights or {})}
    grouped: dict[int, list[TrackSemanticEvidence]] = defaultdict(list)
    for row in evidence:
        grouped[row.track_id].append(row)
    tracks: list[dict[str, Any]] = []
    for track_id, rows in sorted(grouped.items()):
        observed_frames = [
            int(frame)
            for row in rows
            for frame in row.evidence_frames
        ]
        latest_frame = max(observed_frames) if observed_frames else None
        scores: dict[str, float] = defaultdict(float)
        label_weight_totals: dict[str, float] = defaultdict(float)
        total = 0.0
        for row in rows:
            source_weight = float(weights.get(row.source, 1.0))
            row_frame = max(row.evidence_frames) if row.evidence_frames else latest_frame
            age = (
                max(latest_frame - row_frame, 0)
                if latest_frame is not None and row_frame is not None
                else 0
            )
            temporal_weight = 0.5 ** (age / temporal_half_life_frames)
            effective_weight = source_weight * temporal_weight
            weighted = row.confidence * effective_weight
            scores[row.class_label] += weighted
            label_weight_totals[row.class_label] += effective_weight
            total += weighted
        ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_label, best_score = ranking[0]
        second_score = ranking[1][1] if len(ranking) > 1 else 0.0
        consensus = best_score / total if total > 0 else 0.0
        absolute_confidence = (
            best_score / label_weight_totals[best_label]
            if label_weight_totals[best_label] > 0
            else 0.0
        )
        fused_confidence = absolute_confidence * consensus
        margin = (best_score - second_score) / total if total > 0 else 0.0
        temporal_rows = [row for row in rows if row.evidence_frames]
        evidence_by_frame: dict[int, list[TrackSemanticEvidence]] = defaultdict(list)
        for row in temporal_rows:
            evidence_by_frame[max(row.evidence_frames)].append(row)
        frame_winners: list[str] = []
        temporal_label_scores: dict[str, float] = defaultdict(float)
        for frame in sorted(evidence_by_frame):
            frame_scores: dict[str, float] = defaultdict(float)
            for row in evidence_by_frame[frame]:
                weighted_confidence = row.confidence * float(
                    weights.get(row.source, 1.0)
                )
                frame_scores[row.class_label] += weighted_confidence
                temporal_label_scores[row.class_label] += weighted_confidence
            frame_winners.append(
                min(frame_scores, key=lambda label: (-frame_scores[label], label))
            )
        transition_count = sum(
            previous != current
            for previous, current in zip(
                frame_winners,
                frame_winners[1:],
                strict=False,
            )
        )
        temporal_stability = (
            temporal_label_scores.get(best_label, 0.0)
            / max(sum(temporal_label_scores.values()), 1e-9)
            if evidence_by_frame
            else consensus
        )
        temporal_requirement_met = (
            len(evidence_by_frame) < 2
            or temporal_stability >= minimum_temporal_stability
        )
        accepted = (
            best_label != "unknown"
            and fused_confidence >= unknown_threshold
            and margin >= minimum_margin
            and temporal_requirement_met
        )
        accepted_rows = [row for row in rows if row.class_label == best_label]
        attributes = _fuse_attributes(accepted_rows) if accepted else {}
        fine = _fuse_fine_labels(
            accepted_rows if accepted else [],
            weights=weights,
            latest_frame=latest_frame,
            temporal_half_life_frames=temporal_half_life_frames,
            unknown_threshold=fine_unknown_threshold,
            minimum_margin=fine_minimum_margin,
            minimum_temporal_stability=fine_minimum_temporal_stability,
        )
        fine_accepted = bool(accepted and fine["accepted"])
        fine_label = str(fine["label"]) if fine_accepted else "unknown"
        display_label = (
            f"{best_label} > {fine_label}"
            if accepted and fine_accepted and fine_label != best_label
            else best_label
            if accepted
            else "unknown"
        )
        tracks.append(
            {
                "track_id": track_id,
                "class_label": best_label if accepted else "unknown",
                "confidence": round(fused_confidence, 6),
                "absolute_confidence": round(absolute_confidence, 6),
                "consensus": round(consensus, 6),
                "margin": round(margin, 6),
                "temporal_stability": round(temporal_stability, 6),
                "temporal_observation_count": len(evidence_by_frame),
                "temporal_span_frames": (
                    max(observed_frames) - min(observed_frames) if observed_frames else 0
                ),
                "latest_evidence_frame": latest_frame,
                "label_transition_count": transition_count,
                "accepted": accepted,
                "unknown_reason": (
                    None
                    if accepted
                    else "low_confidence_or_conflicting_semantic_evidence"
                ),
                "attributes": attributes,
                "fine_label": fine_label,
                "fine_confidence": round(float(fine["confidence"]), 6),
                "fine_consensus": round(float(fine["consensus"]), 6),
                "fine_margin": round(float(fine["margin"]), 6),
                "fine_temporal_stability": round(
                    float(fine["temporal_stability"]), 6
                ),
                "fine_accepted": fine_accepted,
                "fine_unknown_reason": (
                    None
                    if fine_accepted
                    else "base_class_rejected"
                    if not accepted
                    else str(fine["unknown_reason"])
                ),
                "fine_label_scores": dict(fine["label_scores"]),
                "taxonomy_path": (
                    _select_taxonomy_path(accepted_rows, fine_label)
                    if fine_accepted
                    else [best_label] if accepted else []
                ),
                "display_label": display_label,
                "sources": sorted({row.source for row in rows}),
                "evidence_count": len(rows),
                "label_scores": {
                    label: round(score / total, 6) if total > 0 else 0.0
                    for label, score in ranking
                },
            }
        )
    accepted_count = sum(bool(row["accepted"]) for row in tracks)
    fine_accepted_count = sum(bool(row["fine_accepted"]) for row in tracks)
    return {
        "schema_version": "2.0",
        "policy": {
            "unknown_threshold": unknown_threshold,
            "minimum_margin": minimum_margin,
            "temporal_half_life_frames": temporal_half_life_frames,
            "minimum_temporal_stability": minimum_temporal_stability,
            "fine_unknown_threshold": fine_unknown_threshold,
            "fine_minimum_margin": fine_minimum_margin,
            "fine_minimum_temporal_stability": fine_minimum_temporal_stability,
            "source_weights": weights,
        },
        "summary": {
            "track_count": len(tracks),
            "accepted_count": accepted_count,
            "unknown_count": len(tracks) - accepted_count,
            "coverage": accepted_count / len(tracks) if tracks else 0.0,
            "fine_accepted_count": fine_accepted_count,
            "fine_unknown_count": len(tracks) - fine_accepted_count,
            "fine_coverage": fine_accepted_count / len(tracks) if tracks else 0.0,
        },
        "tracks": tracks,
    }


def normalize_semantic_evidence(
    evidence: list[TrackSemanticEvidence],
    registry: VocabularyRegistry,
) -> list[TrackSemanticEvidence]:
    normalized: list[TrackSemanticEvidence] = []
    for row in evidence:
        if row.class_label == "unknown":
            normalized.append(row)
            continue
        entry, _attributes = registry.resolve(row.class_label)
        if entry is None or entry.canonical_name == row.class_label:
            normalized.append(row)
            continue
        attributes = dict(row.attributes)
        attributes.setdefault("specific_class", row.class_label)
        normalized.append(
            TrackSemanticEvidence(
                track_id=row.track_id,
                class_label=entry.canonical_name,
                confidence=row.confidence,
                source=row.source,
                attributes=attributes,
                evidence_frames=row.evidence_frames,
                evidence=row.evidence,
                fine_label=(
                    row.fine_label
                    if row.fine_label != "unknown"
                    else row.class_label
                ),
                fine_confidence=(
                    row.fine_confidence
                    if row.fine_label != "unknown"
                    else row.confidence
                ),
                taxonomy_path=row.taxonomy_path,
            )
        )
    return normalized


def fuse_semantic_files(
    *,
    qwen_answer: str | Path | None,
    locate_result: str | Path | None,
    output_path: str | Path,
    unknown_threshold: float = 0.45,
    minimum_margin: float = 0.10,
    fine_unknown_threshold: float = 0.85,
    fine_minimum_margin: float = 0.15,
    registry_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    memory_context_id: str | None = None,
    max_memory_observations_per_track: int = 32,
    overwrite: bool = False,
) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Semantic fusion output exists: {path}")
    evidence: list[TrackSemanticEvidence] = []
    if qwen_answer is not None and Path(qwen_answer).is_file():
        qwen_data = json.loads(Path(qwen_answer).read_text(encoding="utf-8"))
        evidence.extend(parse_qwen_answer(qwen_data))
    if locate_result is not None and Path(locate_result).is_file():
        locate_data = json.loads(Path(locate_result).read_text(encoding="utf-8"))
        evidence.extend(parse_locate_evidence(locate_data))
    resolved_registry: Path | None = None
    if registry_path is not None:
        candidate = Path(registry_path)
        resolved_registry = (
            candidate.resolve()
            if candidate.is_absolute()
            else (get_project_root() / candidate).resolve()
        )
        evidence = normalize_semantic_evidence(
            evidence,
            VocabularyRegistry.load(resolved_registry),
        )
    resolved_memory: Path | None = None
    if memory_path is not None:
        from football_tracking.adaptive_tracking.temporal_memory import (
            TemporalSemanticMemory,
        )

        resolved_memory = Path(memory_path)
        memory = TemporalSemanticMemory.load(
            resolved_memory,
            context_id=memory_context_id,
        )
        memory.merge(
            evidence,
            max_observations_per_track=max_memory_observations_per_track,
        )
        memory.save(resolved_memory)
        evidence = list(memory.observations)
    result = fuse_track_semantics(
        evidence,
        unknown_threshold=unknown_threshold,
        minimum_margin=minimum_margin,
        fine_unknown_threshold=fine_unknown_threshold,
        fine_minimum_margin=fine_minimum_margin,
    )
    result["policy"]["ontology_registry"] = (
        str(resolved_registry) if resolved_registry is not None else None
    )
    result["policy"]["temporal_memory"] = (
        str(resolved_memory.resolve()) if resolved_memory is not None else None
    )
    result["policy"]["memory_context_id"] = memory_context_id
    result["policy"]["max_memory_observations_per_track"] = (
        max_memory_observations_per_track
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
    return result


def _fuse_attributes(rows: list[TrackSemanticEvidence]) -> dict[str, Any]:
    values: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    original: dict[tuple[str, str], Any] = {}
    for row in rows:
        for key, value in row.attributes.items():
            normalized = json.dumps(value, sort_keys=True, ensure_ascii=False)
            values[str(key)][normalized] += row.confidence
            original[(str(key), normalized)] = value
    return {
        key: original[(key, max(scores, key=scores.get))]
        for key, scores in values.items()
    }


def _fuse_fine_labels(
    rows: list[TrackSemanticEvidence],
    *,
    weights: dict[str, float],
    latest_frame: int | None,
    temporal_half_life_frames: float,
    unknown_threshold: float,
    minimum_margin: float,
    minimum_temporal_stability: float,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.fine_label not in {"", "unknown", row.class_label}
        and row.fine_confidence > 0.0
    ]
    empty = {
        "label": "unknown",
        "confidence": 0.0,
        "consensus": 0.0,
        "margin": 0.0,
        "temporal_stability": 0.0,
        "accepted": False,
        "unknown_reason": "no_visually_supported_fine_label",
        "label_scores": {},
    }
    if not candidates:
        return empty

    scores: dict[str, float] = defaultdict(float)
    weight_totals: dict[str, float] = defaultdict(float)
    total = 0.0
    by_frame: dict[int, list[TrackSemanticEvidence]] = defaultdict(list)
    for row in candidates:
        row_frame = max(row.evidence_frames) if row.evidence_frames else latest_frame
        age = (
            max(latest_frame - row_frame, 0)
            if latest_frame is not None and row_frame is not None
            else 0
        )
        effective_weight = float(weights.get(row.source, 1.0)) * (
            0.5 ** (age / temporal_half_life_frames)
        )
        weighted = row.fine_confidence * effective_weight
        scores[row.fine_label] += weighted
        weight_totals[row.fine_label] += effective_weight
        total += weighted
        if row_frame is not None:
            by_frame[int(row_frame)].append(row)

    ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    label, score = ranking[0]
    second = ranking[1][1] if len(ranking) > 1 else 0.0
    consensus = score / total if total > 0 else 0.0
    absolute = score / weight_totals[label] if weight_totals[label] > 0 else 0.0
    confidence = absolute * consensus
    margin = (score - second) / total if total > 0 else 0.0
    temporal_scores: dict[str, float] = defaultdict(float)
    for frame_rows in by_frame.values():
        for row in frame_rows:
            temporal_scores[row.fine_label] += row.fine_confidence * float(
                weights.get(row.source, 1.0)
            )
    temporal_stability = (
        temporal_scores.get(label, 0.0) / max(sum(temporal_scores.values()), 1e-9)
        if by_frame
        else consensus
    )
    temporal_met = len(by_frame) < 2 or temporal_stability >= minimum_temporal_stability
    accepted = (
        confidence >= unknown_threshold
        and margin >= minimum_margin
        and temporal_met
    )
    return {
        "label": label,
        "confidence": confidence,
        "consensus": consensus,
        "margin": margin,
        "temporal_stability": temporal_stability,
        "accepted": accepted,
        "unknown_reason": (
            None
            if accepted
            else "low_confidence_or_conflicting_fine_grained_evidence"
        ),
        "label_scores": {
            name: round(value / total, 6) if total > 0 else 0.0
            for name, value in ranking
        },
    }


def _select_taxonomy_path(
    rows: list[TrackSemanticEvidence],
    fine_label: str,
) -> list[str]:
    matching = [
        row
        for row in rows
        if row.fine_label == fine_label and row.taxonomy_path
    ]
    if matching:
        selected = max(matching, key=lambda row: row.fine_confidence)
        path = list(dict.fromkeys(selected.taxonomy_path))
        if path[-1] != fine_label:
            path.append(fine_label)
        return path
    base = rows[0].class_label if rows else "unknown"
    return [base, fine_label] if base != fine_label else [base]
