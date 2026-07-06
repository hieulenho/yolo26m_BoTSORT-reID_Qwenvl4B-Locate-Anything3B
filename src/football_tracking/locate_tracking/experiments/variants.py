"""Load language ablation variant manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from football_tracking.locate_tracking.experiments.schemas import (
    LanguageAblationConfig,
    LanguageAblationVariant,
)
from football_tracking.paths import get_project_root, resolve_project_path


class LanguageAblationConfigError(RuntimeError):
    """Raised when an ablation config is invalid."""


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else resolve_project_path(value, root)


def load_language_ablation_config(path: str | Path) -> LanguageAblationConfig:
    root = get_project_root()
    resolved = _resolve(path, root)
    if not resolved.is_file():
        raise LanguageAblationConfigError(f"Ablation config does not exist: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise LanguageAblationConfigError("Ablation config root must be a mapping.")
    variants_raw = raw.get("variants", [])
    if not isinstance(variants_raw, list) or not variants_raw:
        raise LanguageAblationConfigError("variants must be a non-empty list.")
    variants = tuple(_variant(item, root) for item in variants_raw)
    output = raw.get("output", {}) if isinstance(raw.get("output", {}), dict) else {}
    benchmark = raw.get("benchmark", {}) if isinstance(raw.get("benchmark", {}), dict) else {}
    return LanguageAblationConfig(
        benchmark_manifest=_resolve(benchmark.get("manifest"), root),
        iou_threshold=float(benchmark.get("iou_threshold", 0.5)),
        variants=variants,
        output_dir=_resolve(
            output.get("directory", "outputs/locate_tracking/benchmark/ablation"),
            root,
        ),
        metadata=dict(raw.get("metadata", {})),
    )


def _variant(data: Any, root: Path) -> LanguageAblationVariant:
    if not isinstance(data, dict):
        raise LanguageAblationConfigError("Each variant must be a mapping.")
    return LanguageAblationVariant(
        variant_id=str(data["variant_id"]),
        name=str(data.get("name", data["variant_id"])),
        description=str(data.get("description", "")),
        prediction_manifest=_resolve(data["prediction_manifest"], root),
        config_path=_resolve(data["config_path"], root) if data.get("config_path") else None,
    )
