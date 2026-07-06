"""Configuration loading for single-frame association."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.locate_tracking.association.schemas import AssociationConfig
from football_tracking.paths import get_project_root, resolve_project_path


class AssociationConfigError(RuntimeError):
    """Raised when frame association configuration is invalid."""


@dataclass(frozen=True)
class FrameAssociationConfig:
    project_root: Path
    config_path: Path
    association: AssociationConfig
    output_dir: Path
    overwrite: bool
    log_level: str


def _mapping(value: Any, section: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if value is None and default is not None:
        return dict(default)
    if not isinstance(value, dict):
        raise AssociationConfigError(f"{section} must be a mapping.")
    return value


def _resolve_path(
    value: Any,
    project_root: Path,
    section: str,
    *,
    required: bool = True,
) -> Path | None:
    if value is None and not required:
        return None
    if not isinstance(value, str | Path) or not str(value).strip():
        raise AssociationConfigError(f"{section} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def load_frame_association_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> FrameAssociationConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved_config = path.resolve() if path.is_absolute() else resolve_project_path(path)
    if not resolved_config.is_file():
        raise AssociationConfigError(f"Association config does not exist: {resolved_config}")
    raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, "association config root", default={})
    association = _mapping(root.get("association"), "association", default={})
    score = _mapping(association.get("score"), "association.score", default={})
    decision = _mapping(association.get("decision"), "association.decision", default={})
    geometry = _mapping(root.get("geometry"), "geometry", default={})
    output = _mapping(root.get("output"), "output", default={})
    runtime = _mapping(root.get("runtime"), "runtime", default={})
    config = FrameAssociationConfig(
        project_root=project_root,
        config_path=resolved_config,
        association=AssociationConfig(
            min_iou=float(association.get("min_iou", 0.10)),
            min_track_coverage=float(association.get("min_track_coverage", 0.50)),
            iou_weight=float(score.get("iou_weight", 0.70)),
            track_coverage_weight=float(score.get("track_coverage_weight", 0.20)),
            center_similarity_weight=float(score.get("center_similarity_weight", 0.10)),
            min_score=float(decision.get("min_score", 0.20)),
            ambiguity_margin=float(decision.get("ambiguity_margin", 0.05)),
            top_k=int(decision.get("top_k", 5)),
            clip_tracks_to_frame=bool(geometry.get("clip_tracks_to_frame", True)),
            reject_fully_outside_tracks=bool(geometry.get("reject_fully_outside_tracks", True)),
            save_candidates=bool(output.get("save_candidates", True)),
            save_overlay=bool(output.get("save_overlay", True)),
        ),
        output_dir=_resolve_path(
            output.get("directory", "outputs/locate_tracking/queries"),
            project_root,
            "output.directory",
        ),
        overwrite=bool(runtime.get("overwrite", False)),
        log_level=str(runtime.get("log_level", "INFO")),
    )
    if overrides:
        config = _apply_overrides(config, overrides)
    return config


def _apply_overrides(
    config: FrameAssociationConfig,
    overrides: dict[str, Any],
) -> FrameAssociationConfig:
    changes: dict[str, Any] = {}
    if overrides.get("output_dir") is not None:
        changes["output_dir"] = _resolve_path(
            overrides["output_dir"],
            config.project_root,
            "--output-dir",
        )
    if overrides.get("overwrite") is not None:
        changes["overwrite"] = bool(overrides["overwrite"])
    return replace(config, **changes) if changes else config
