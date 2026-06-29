"""Class mapping for the single-class football player MVP."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from football_tracking.data.schemas import ObjectAnnotation

MappingStatus = Literal["mapped", "ignored", "unknown"]
LOGGER = logging.getLogger(__name__)


class ClassMappingError(RuntimeError):
    """Raised when class mapping configuration is invalid."""


@dataclass(frozen=True)
class ClassMappingResult:
    source_class: str
    normalized_class: str
    status: MappingStatus
    target_class: str | None
    target_class_id: int | None
    message: str


@dataclass(frozen=True)
class ClassMapping:
    target_classes: dict[str, int]
    source_mapping: dict[str, str]
    ignored_classes: set[str]
    unknown_class_policy: str = "warn_and_skip"

    def map_class(self, source_class: str) -> ClassMappingResult:
        normalized = normalize_class_name(source_class)
        if normalized in self.source_mapping:
            target_class = self.source_mapping[normalized]
            target_class_id = self.target_classes[target_class]
            return ClassMappingResult(
                source_class=source_class,
                normalized_class=normalized,
                status="mapped",
                target_class=target_class,
                target_class_id=target_class_id,
                message=f"Mapped {source_class!r} to {target_class!r}.",
            )

        if normalized in self.ignored_classes:
            return ClassMappingResult(
                source_class=source_class,
                normalized_class=normalized,
                status="ignored",
                target_class=None,
                target_class_id=None,
                message=f"Ignored source class {source_class!r}.",
            )

        message = f"Unknown source class {source_class!r}; policy={self.unknown_class_policy}."
        LOGGER.warning(message)
        return ClassMappingResult(
            source_class=source_class,
            normalized_class=normalized,
            status="unknown",
            target_class=None,
            target_class_id=None,
            message=message,
        )


def normalize_class_name(class_name: str) -> str:
    cleaned = class_name.strip().lower()
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def load_class_mapping(path: str | Path) -> ClassMapping:
    mapping_path = Path(path)
    if not mapping_path.is_file():
        raise ClassMappingError(f"Class mapping file does not exist: {mapping_path}")

    raw = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ClassMappingError("Class mapping root must be a mapping.")

    target_classes = raw.get("target_classes")
    source_mapping = raw.get("source_mapping")
    ignored_classes = raw.get("ignored_classes", [])
    if not isinstance(target_classes, dict) or not target_classes:
        raise ClassMappingError("target_classes must be a non-empty mapping.")
    if not isinstance(source_mapping, dict):
        raise ClassMappingError("source_mapping must be a mapping.")
    if not isinstance(ignored_classes, list):
        raise ClassMappingError("ignored_classes must be a list.")

    normalized_targets = {
        normalize_class_name(str(name)): int(class_id) for name, class_id in target_classes.items()
    }
    normalized_sources = {
        normalize_class_name(str(source)): normalize_class_name(str(target))
        for source, target in source_mapping.items()
    }
    normalized_ignored = {normalize_class_name(str(name)) for name in ignored_classes}

    missing_targets = sorted(
        target for target in normalized_sources.values() if target not in normalized_targets
    )
    if missing_targets:
        raise ClassMappingError(
            f"source_mapping references unknown target classes: {missing_targets}"
        )

    return ClassMapping(
        target_classes=normalized_targets,
        source_mapping=normalized_sources,
        ignored_classes=normalized_ignored,
        unknown_class_policy=str(raw.get("unknown_class_policy", "warn_and_skip")),
    )


def apply_mapping_to_object(
    annotation: ObjectAnnotation,
    class_mapping: ClassMapping,
) -> ObjectAnnotation:
    result = class_mapping.map_class(annotation.source_class)
    metadata = dict(annotation.metadata)
    metadata["class_mapping_status"] = result.status
    metadata["normalized_source_class"] = result.normalized_class
    if result.status == "unknown":
        metadata["unknown_class"] = True
    return ObjectAnnotation(
        frame_index=annotation.frame_index,
        track_id=annotation.track_id,
        source_class=annotation.source_class,
        target_class=result.target_class,
        target_class_id=result.target_class_id,
        bbox_xyxy=annotation.bbox_xyxy,
        confidence=annotation.confidence,
        visibility=annotation.visibility,
        is_ignored=result.status != "mapped",
        metadata=metadata,
    )
