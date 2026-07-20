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
        label = row.get("class_label")
        if label is None:
            role = str(row.get("role_label", "unknown"))
            team = str(row.get("team_label", "unknown"))
            label = role if team == "unknown" else f"{team} {role}"
        evidence.append(
            TrackSemanticEvidence(
                track_id=int(row["track_id"]),
                class_label=str(label),
                confidence=float(row.get("confidence", 0.0)),
                source="qwen",
                attributes=dict(row.get("attributes", {})),
                evidence_frames=tuple(row.get("evidence_frames", ())),
                evidence=str(row.get("evidence", "")),
            )
        )
    return evidence


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
    source_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fuse repeated observations, then reject weak or ambiguous labels."""
    weights = {"qwen": 1.0, "locateanything": 0.9, **(source_weights or {})}
    grouped: dict[int, list[TrackSemanticEvidence]] = defaultdict(list)
    for row in evidence:
        grouped[row.track_id].append(row)
    tracks: list[dict[str, Any]] = []
    for track_id, rows in sorted(grouped.items()):
        scores: dict[str, float] = defaultdict(float)
        label_weight_totals: dict[str, float] = defaultdict(float)
        total = 0.0
        for row in rows:
            source_weight = float(weights.get(row.source, 1.0))
            weighted = row.confidence * source_weight
            scores[row.class_label] += weighted
            label_weight_totals[row.class_label] += source_weight
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
        accepted = (
            best_label != "unknown"
            and fused_confidence >= unknown_threshold
            and margin >= minimum_margin
        )
        accepted_rows = [row for row in rows if row.class_label == best_label]
        attributes = _fuse_attributes(accepted_rows) if accepted else {}
        tracks.append(
            {
                "track_id": track_id,
                "class_label": best_label if accepted else "unknown",
                "confidence": round(fused_confidence, 6),
                "absolute_confidence": round(absolute_confidence, 6),
                "consensus": round(consensus, 6),
                "margin": round(margin, 6),
                "accepted": accepted,
                "unknown_reason": (
                    None
                    if accepted
                    else "low_confidence_or_conflicting_semantic_evidence"
                ),
                "attributes": attributes,
                "sources": sorted({row.source for row in rows}),
                "evidence_count": len(rows),
                "label_scores": {
                    label: round(score / total, 6) if total > 0 else 0.0
                    for label, score in ranking
                },
            }
        )
    accepted_count = sum(bool(row["accepted"]) for row in tracks)
    return {
        "schema_version": "1.0",
        "policy": {
            "unknown_threshold": unknown_threshold,
            "minimum_margin": minimum_margin,
            "source_weights": weights,
        },
        "summary": {
            "track_count": len(tracks),
            "accepted_count": accepted_count,
            "unknown_count": len(tracks) - accepted_count,
            "coverage": accepted_count / len(tracks) if tracks else 0.0,
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
    registry_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
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
    result = fuse_track_semantics(
        evidence,
        unknown_threshold=unknown_threshold,
        minimum_margin=minimum_margin,
    )
    result["policy"]["ontology_registry"] = (
        str(resolved_registry) if resolved_registry is not None else None
    )
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Semantic fusion output exists: {path}")
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
