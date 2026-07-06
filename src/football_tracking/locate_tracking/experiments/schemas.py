"""Schemas for language tracking ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LanguageAblationVariant:
    variant_id: str
    name: str
    prediction_manifest: Path
    description: str = ""
    config_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "prediction_manifest": str(self.prediction_manifest),
            "description": self.description,
            "config_path": str(self.config_path) if self.config_path else None,
        }


@dataclass(frozen=True)
class LanguageAblationConfig:
    benchmark_manifest: Path
    variants: tuple[LanguageAblationVariant, ...]
    iou_threshold: float = 0.5
    output_dir: Path = Path("outputs/locate_tracking/benchmark/ablation")
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_manifest": str(self.benchmark_manifest),
            "iou_threshold": self.iou_threshold,
            "output_dir": str(self.output_dir),
            "variants": [variant.to_dict() for variant in self.variants],
            "metadata": dict(self.metadata),
        }
