"""Open vocabulary normalization with optional mappings to a small registry."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_tracking.adaptive_tracking.schemas import DiscoveredObject

COCO80_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
COCO_ID_BY_NAME = {name: index for index, name in enumerate(COCO80_CLASSES)}


class VocabularyRegistryError(RuntimeError):
    """Raised when a vocabulary registry is malformed."""


def normalize_phrase(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


@dataclass(frozen=True)
class RegistryEntry:
    canonical_name: str
    aliases: tuple[str, ...]
    coco_id: int | None
    default_action: str


class VocabularyRegistry:
    """Registry augments VLM classes without rejecting unseen concepts."""

    def __init__(self, entries: tuple[RegistryEntry, ...]) -> None:
        self.entries = entries
        aliases: dict[str, RegistryEntry] = {}
        for entry in entries:
            for alias in (entry.canonical_name, *entry.aliases):
                normalized = normalize_phrase(alias)
                if normalized:
                    previous = aliases.get(normalized)
                    if (
                        previous is not None
                        and previous.canonical_name != entry.canonical_name
                    ):
                        raise VocabularyRegistryError(
                            f"Alias '{normalized}' maps to both "
                            f"'{previous.canonical_name}' and '{entry.canonical_name}'."
                        )
                    aliases[normalized] = entry
        self._aliases = aliases

    @classmethod
    def load(cls, path: str | Path) -> VocabularyRegistry:
        registry_path = Path(path)
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        rows = raw.get("classes", [])
        if not isinstance(rows, list):
            raise VocabularyRegistryError("Registry classes must be a list.")
        entries: list[RegistryEntry] = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("name"):
                raise VocabularyRegistryError("Each registry class needs a name.")
            name = normalize_phrase(str(row["name"]))
            coco_name = normalize_phrase(str(row.get("coco_class", "")))
            coco_id = COCO_ID_BY_NAME.get(coco_name)
            if coco_name and coco_id is None:
                raise VocabularyRegistryError(
                    f"Unknown COCO class '{coco_name}' for registry class '{name}'."
                )
            default_action = str(row.get("default_action", "track")).strip().lower()
            if default_action not in {"track", "detect", "context"}:
                raise VocabularyRegistryError(
                    f"Invalid default_action '{default_action}' for registry class '{name}'."
                )
            entries.append(
                RegistryEntry(
                    canonical_name=name,
                    aliases=tuple(str(item) for item in row.get("aliases", ())),
                    coco_id=coco_id,
                    default_action=default_action,
                )
            )
        return cls(tuple(entries))

    def resolve(self, raw_name: str) -> tuple[RegistryEntry | None, tuple[str, ...]]:
        normalized = normalize_phrase(raw_name)
        if normalized in self._aliases:
            return self._aliases[normalized], ()
        matches = [
            (alias, entry)
            for alias, entry in self._aliases.items()
            if re.search(rf"\b{re.escape(alias)}\b", normalized)
        ]
        if not matches:
            return None, ()
        alias, entry = max(matches, key=lambda item: len(item[0]))
        attributes = normalize_phrase(normalized.replace(alias, " ")).split()
        return entry, tuple(attributes)


def normalize_objects(
    raw_objects: list[dict[str, Any]],
    *,
    registry: VocabularyRegistry,
    max_classes: int = 24,
    minimum_confidence: float = 0.15,
) -> tuple[DiscoveredObject, ...]:
    """Normalize, merge, prioritize, and retain open-vocabulary classes."""
    if max_classes <= 0:
        raise VocabularyRegistryError("max_classes must be positive.")
    merged: dict[str, DiscoveredObject] = {}
    for row in raw_objects:
        raw_name = str(
            row.get("canonical_name")
            or row.get("base_class")
            or row.get("name")
            or row.get("class")
            or ""
        ).strip()
        if not raw_name:
            continue
        try:
            confidence = float(row.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(max(confidence, 0.0), 1.0)
        if confidence < minimum_confidence:
            continue
        entry, inferred_attributes = registry.resolve(raw_name)
        normalized_name = normalize_phrase(raw_name)
        canonical = entry.canonical_name if entry is not None else normalized_name
        if not canonical:
            continue
        action = str(row.get("action", row.get("role", ""))).strip().lower()
        action_aliases = {
            "tracking": "track",
            "trackable": "track",
            "detection": "detect",
            "detect_only": "detect",
            "background": "context",
            "region": "context",
        }
        action = action_aliases.get(action, action)
        if action not in {"track", "detect", "context"}:
            action = entry.default_action if entry is not None else "track"
        attributes = tuple(
            dict.fromkeys(
                [
                    *[str(item) for item in row.get("attributes", ())],
                    *inferred_attributes,
                ]
            )
        )
        coco_id = entry.coco_id if entry is not None else COCO_ID_BY_NAME.get(canonical)
        candidate = DiscoveredObject(
            canonical_name=canonical,
            display_name=str(row.get("display_name", canonical)),
            action=action,
            confidence=confidence,
            aliases=entry.aliases if entry is not None else (),
            attributes=attributes,
            coco_id=coco_id,
            open_vocabulary=coco_id is None,
            source_names=(raw_name,),
            fine_grained_candidates=tuple(
                str(item) for item in row.get("fine_grained_candidates", ())
            ),
            semantic_facets=tuple(str(item) for item in row.get("semantic_facets", ())),
            taxonomy_hint=str(row.get("taxonomy_hint", "")),
        )
        existing = merged.get(canonical)
        if existing is None:
            merged[canonical] = candidate
            continue
        action_rank = {"context": 0, "detect": 1, "track": 2}
        merged[canonical] = DiscoveredObject(
            canonical_name=canonical,
            display_name=existing.display_name,
            action=max((existing.action, candidate.action), key=action_rank.get),
            confidence=max(existing.confidence, candidate.confidence),
            aliases=tuple(dict.fromkeys((*existing.aliases, *candidate.aliases))),
            attributes=tuple(dict.fromkeys((*existing.attributes, *candidate.attributes))),
            coco_id=existing.coco_id if existing.coco_id is not None else candidate.coco_id,
            open_vocabulary=existing.open_vocabulary and candidate.open_vocabulary,
            source_names=tuple(
                dict.fromkeys((*existing.source_names, *candidate.source_names))
            ),
            fine_grained_candidates=tuple(
                dict.fromkeys(
                    (
                        *existing.fine_grained_candidates,
                        *candidate.fine_grained_candidates,
                    )
                )
            ),
            semantic_facets=tuple(
                dict.fromkeys((*existing.semantic_facets, *candidate.semantic_facets))
            ),
            taxonomy_hint=existing.taxonomy_hint or candidate.taxonomy_hint,
        )
    action_rank = {"track": 0, "detect": 1, "context": 2}
    ordered = sorted(
        merged.values(),
        key=lambda item: (action_rank[item.action], -item.confidence, item.canonical_name),
    )
    return tuple(ordered[:max_classes])
