"""Typed contracts shared by adaptive scene discovery and detector routing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


class AdaptiveSchemaError(ValueError):
    """Raised when an adaptive-pipeline record is invalid."""


def _confidence(value: Any, *, default: float = 0.5) -> float:
    aliases = {"low": 0.3, "medium": 0.6, "high": 0.9}
    if isinstance(value, str) and value.strip().lower() in aliases:
        return aliases[value.strip().lower()]
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if not math.isfinite(result):
        result = default
    return min(max(result, 0.0), 1.0)


@dataclass(frozen=True)
class DiscoveredObject:
    """One normalized object concept produced by scene discovery."""

    canonical_name: str
    display_name: str
    action: str
    confidence: float
    aliases: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    coco_id: int | None = None
    open_vocabulary: bool = False
    source_names: tuple[str, ...] = ()
    fine_grained_candidates: tuple[str, ...] = ()
    semantic_facets: tuple[str, ...] = ()
    taxonomy_hint: str = ""

    def __post_init__(self) -> None:
        canonical = str(self.canonical_name).strip().lower()
        if not canonical:
            raise AdaptiveSchemaError("canonical_name must not be empty.")
        action = str(self.action).strip().lower()
        if action not in {"track", "detect", "context"}:
            raise AdaptiveSchemaError(f"Unsupported object action: {self.action}")
        coco_id = self.coco_id
        if coco_id is not None and not 0 <= int(coco_id) < 80:
            raise AdaptiveSchemaError("coco_id must be None or an integer in [0, 79].")
        object.__setattr__(self, "canonical_name", canonical)
        object.__setattr__(self, "display_name", str(self.display_name).strip() or canonical)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "aliases", tuple(str(item) for item in self.aliases))
        object.__setattr__(self, "attributes", tuple(str(item) for item in self.attributes))
        object.__setattr__(self, "source_names", tuple(str(item) for item in self.source_names))
        fine_candidates = tuple(
            dict.fromkeys(
                value
                for item in self.fine_grained_candidates
                if (value := str(item).strip().lower().replace("_", " "))
                and value != canonical
            )
        )[:8]
        semantic_facets = tuple(
            dict.fromkeys(
                value
                for item in self.semantic_facets
                if (value := str(item).strip().lower().replace(" ", "_"))
            )
        )[:8]
        object.__setattr__(self, "fine_grained_candidates", fine_candidates)
        object.__setattr__(self, "semantic_facets", semantic_facets)
        object.__setattr__(
            self,
            "taxonomy_hint",
            str(self.taxonomy_hint).strip().lower().replace(" ", "_"),
        )
        object.__setattr__(self, "coco_id", int(coco_id) if coco_id is not None else None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoveredObject:
        return cls(
            canonical_name=str(data["canonical_name"]),
            display_name=str(data.get("display_name", data["canonical_name"])),
            action=str(data.get("action", "track")),
            confidence=_confidence(data.get("confidence")),
            aliases=tuple(data.get("aliases", ())),
            attributes=tuple(data.get("attributes", ())),
            coco_id=data.get("coco_id"),
            open_vocabulary=bool(data.get("open_vocabulary", False)),
            source_names=tuple(data.get("source_names", ())),
            fine_grained_candidates=tuple(data.get("fine_grained_candidates", ())),
            semantic_facets=tuple(data.get("semantic_facets", ())),
            taxonomy_hint=str(data.get("taxonomy_hint", "")),
        )


@dataclass(frozen=True)
class SceneDiscovery:
    """Normalized, cacheable description of a video's domain and vocabulary."""

    source_video: str
    domain: str
    domain_confidence: float
    description: str
    objects: tuple[DiscoveredObject, ...]
    background_regions: tuple[str, ...] = ()
    keyframes: tuple[dict[str, Any], ...] = ()
    shot_starts: tuple[int, ...] = ()
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    prompt_version: str = "dynamic-v3-hierarchical"
    raw_response: str = ""
    created_at: str = ""
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        domain = str(self.domain).strip().lower().replace(" ", "_") or "unknown"
        object.__setattr__(self, "source_video", str(self.source_video))
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "domain_confidence", _confidence(self.domain_confidence))
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(
            self,
            "background_regions",
            tuple(str(item) for item in self.background_regions),
        )
        object.__setattr__(self, "keyframes", tuple(dict(item) for item in self.keyframes))
        shot_starts = tuple(sorted({int(value) for value in self.shot_starts}))
        if any(value < 1 for value in shot_starts):
            raise ValueError("shot_starts must contain positive frame indices.")
        object.__setattr__(self, "shot_starts", shot_starts)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def detector_objects(self) -> tuple[DiscoveredObject, ...]:
        return tuple(item for item in self.objects if item.action in {"track", "detect"})

    @property
    def tracking_objects(self) -> tuple[DiscoveredObject, ...]:
        return tuple(item for item in self.objects if item.action == "track")

    def to_dict(self, *, include_legacy: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": "2.2",
            "source_video": self.source_video,
            "domain": {
                "name": self.domain,
                "confidence": self.domain_confidence,
                "description": self.description,
            },
            "objects": [item.to_dict() for item in self.objects],
            "background_regions": list(self.background_regions),
            "keyframes": [dict(item) for item in self.keyframes],
            "shot_starts": list(self.shot_starts),
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "raw_response": self.raw_response,
            "created_at": self.created_at,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }
        if include_legacy:
            coco_objects = [item for item in self.detector_objects if item.coco_id is not None]
            payload.update(
                {
                    "context": self.description,
                    "context_short": self.domain,
                    "objects_found": [item.canonical_name for item in self.detector_objects],
                    "coco_class_ids": [int(item.coco_id) for item in coco_objects],
                    "coco_class_names": {
                        str(item.coco_id): item.canonical_name for item in coco_objects
                    },
                    "confidence": (
                        "high"
                        if self.domain_confidence >= 0.75
                        else "medium"
                        if self.domain_confidence >= 0.45
                        else "low"
                    ),
                    "frames_sampled": len(self.keyframes),
                }
            )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneDiscovery:
        domain_data = data.get("domain", {})
        if not isinstance(domain_data, dict):
            domain_data = {"name": domain_data}
        objects_data = data.get("objects", [])
        if not objects_data and data.get("objects_found"):
            names_by_id = {
                int(key): str(value)
                for key, value in (data.get("coco_class_names", {}) or {}).items()
            }
            ids = [int(value) for value in data.get("coco_class_ids", [])]
            objects_data = [
                {
                    "canonical_name": names_by_id.get(class_id, f"class_{class_id}"),
                    "action": "track",
                    "confidence": data.get("confidence", 0.5),
                    "coco_id": class_id,
                    "open_vocabulary": False,
                }
                for class_id in ids
            ]
        return cls(
            source_video=str(data.get("source_video", "")),
            domain=str(domain_data.get("name", data.get("context_short", "unknown"))),
            domain_confidence=_confidence(
                domain_data.get("confidence", data.get("confidence", 0.5))
            ),
            description=str(domain_data.get("description", data.get("context", ""))),
            objects=tuple(DiscoveredObject.from_dict(item) for item in objects_data),
            background_regions=tuple(data.get("background_regions", ())),
            keyframes=tuple(data.get("keyframes", ())),
            shot_starts=tuple(data.get("shot_starts", ())),
            model_id=str(data.get("model_id", "Qwen/Qwen3-VL-4B-Instruct")),
            prompt_version=str(data.get("prompt_version", "legacy")),
            raw_response=str(data.get("raw_response", "")),
            created_at=str(data.get("created_at", "")),
            warnings=tuple(data.get("warnings", ())),
            metadata=dict(data.get("metadata", {})),
        )
