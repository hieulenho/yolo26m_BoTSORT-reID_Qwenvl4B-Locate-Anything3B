"""Configuration loading for appearance verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.locate_tracking.appearance.crop_quality import CropQualityConfig
from football_tracking.locate_tracking.appearance.crop_selection import (
    RepresentativeCropSelectionConfig,
)
from football_tracking.locate_tracking.appearance.schemas import PrototypeAggregationStrategy
from football_tracking.locate_tracking.appearance.verifier import AppearanceVerifierConfig
from football_tracking.locate_tracking.fusion.schemas import FusionConfig
from football_tracking.paths import get_project_root, resolve_project_path


class AppearanceConfigError(RuntimeError):
    """Raised when appearance verification configuration is invalid."""


@dataclass(frozen=True)
class AppearanceVerificationConfig:
    project_root: Path
    config_path: Path
    backend_name: str
    model_id: str
    device: str | None
    batch_size: int
    normalize: bool
    cache_enabled: bool
    cache_directory: Path
    crop_quality: CropQualityConfig
    crop_selection: RepresentativeCropSelectionConfig
    prototype_strategy: PrototypeAggregationStrategy
    verifier: AppearanceVerifierConfig
    fusion: FusionConfig
    output_dir: Path
    save_crops: bool
    include_vectors_in_json: bool
    overwrite: bool
    log_level: str


def _mapping(value: Any, section: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if value is None and default is not None:
        return dict(default)
    if not isinstance(value, dict):
        raise AppearanceConfigError(f"{section} must be a mapping.")
    return value


def _resolve_path(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise AppearanceConfigError(f"{section} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def load_appearance_verification_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> AppearanceVerificationConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path)
    if not resolved.is_file():
        raise AppearanceConfigError(f"Appearance config does not exist: {resolved}")
    root = _mapping(yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}, "root", default={})
    appearance = _mapping(root.get("appearance"), "appearance", default={})
    cache = _mapping(root.get("cache"), "cache", default={})
    quality = _mapping(root.get("crop_quality"), "crop_quality", default={})
    selection = _mapping(root.get("crop_selection"), "crop_selection", default={})
    prototype = _mapping(root.get("prototype"), "prototype", default={})
    verifier = _mapping(root.get("verification"), "verification", default={})
    fusion = _mapping(root.get("fusion"), "fusion", default={})
    output = _mapping(root.get("output"), "output", default={})
    runtime = _mapping(root.get("runtime"), "runtime", default={})
    config = AppearanceVerificationConfig(
        project_root=project_root,
        config_path=resolved,
        backend_name=str(appearance.get("backend", "mock")),
        model_id=str(appearance.get("model_id", "yolo26n-cls.pt")),
        device=appearance.get("device", "cuda"),
        batch_size=int(appearance.get("batch_size", 8)),
        normalize=bool(appearance.get("normalize", True)),
        cache_enabled=bool(cache.get("enabled", True)),
        cache_directory=_resolve_path(
            cache.get("directory", "outputs/locate_tracking/cache/appearance"),
            project_root,
            "cache.directory",
        ),
        crop_quality=CropQualityConfig(
            min_width=int(quality.get("min_width", 8)),
            min_height=int(quality.get("min_height", 8)),
            min_area=float(quality.get("min_area", 64.0)),
            min_visible_fraction=float(quality.get("min_visible_fraction", 0.30)),
            min_sharpness=quality.get("min_sharpness"),
        ),
        crop_selection=RepresentativeCropSelectionConfig(
            max_samples_per_track=int(selection.get("max_samples_per_track", 4)),
            min_frame_gap=int(selection.get("min_frame_gap", 5)),
            require_quality_gate=bool(selection.get("require_quality_gate", True)),
        ),
        prototype_strategy=str(prototype.get("aggregation_strategy", "mean")),  # type: ignore[arg-type]
        verifier=AppearanceVerifierConfig(
            min_samples_for_consistency=int(verifier.get("min_samples_for_consistency", 2)),
            min_verified_score=float(verifier.get("min_verified_score", 0.70)),
            weak_score_threshold=float(verifier.get("weak_score_threshold", 0.55)),
        ),
        fusion=FusionConfig(
            semantic_weight=float(fusion.get("semantic_weight", 0.65)),
            appearance_weight=float(fusion.get("appearance_weight", 0.35)),
            missing_appearance_policy=str(fusion.get("missing_appearance_policy", "semantic_only")),  # type: ignore[arg-type]
            missing_appearance_penalty=float(fusion.get("missing_appearance_penalty", 0.15)),
            min_fused_score=float(fusion.get("min_fused_score", 0.35)),
            winner_margin=float(fusion.get("winner_margin", 0.05)),
            query_mode=str(fusion.get("query_mode", "single_target")),  # type: ignore[arg-type]
        ),
        output_dir=_resolve_path(
            output.get("directory", "outputs/locate_tracking/appearance"),
            project_root,
            "output.directory",
        ),
        save_crops=bool(output.get("save_crops", True)),
        include_vectors_in_json=bool(output.get("include_vectors_in_json", True)),
        overwrite=bool(runtime.get("overwrite", False)),
        log_level=str(runtime.get("log_level", "INFO")),
    )
    if overrides:
        config = _apply_overrides(config, overrides)
    if config.backend_name not in {"mock", "ultralytics"}:
        raise AppearanceConfigError(f"Unsupported appearance backend: {config.backend_name}")
    if config.batch_size < 1:
        raise AppearanceConfigError("appearance.batch_size must be >= 1.")
    return config


def _apply_overrides(
    config: AppearanceVerificationConfig,
    overrides: dict[str, Any],
) -> AppearanceVerificationConfig:
    changes: dict[str, Any] = {}
    for key in ("backend_name", "model_id", "device"):
        if overrides.get(key) is not None:
            changes[key] = overrides[key]
    for key in ("batch_size", "normalize", "overwrite", "save_crops"):
        if overrides.get(key) is not None:
            changes[key] = (
                bool(overrides[key])
                if key in {"normalize", "overwrite", "save_crops"}
                else int(overrides[key])
            )
    if overrides.get("output_dir") is not None:
        changes["output_dir"] = _resolve_path(
            overrides["output_dir"], config.project_root, "--output-dir"
        )
    return replace(config, **changes) if changes else config
