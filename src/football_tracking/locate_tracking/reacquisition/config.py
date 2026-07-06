"""Configuration loader for semantic target reacquisition."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.locate_tracking.reacquisition.schemas import ReacquisitionConfig
from football_tracking.paths import get_project_root, resolve_project_path


class ReacquisitionConfigError(RuntimeError):
    """Raised when reacquisition configuration cannot be loaded."""


@dataclass(frozen=True)
class ReacquisitionPipelineConfig:
    project_root: Path
    config_path: Path
    reacquisition: ReacquisitionConfig
    output_dir: Path
    overwrite: bool = False


def _mapping(value: Any, section: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if value is None and default is not None:
        return dict(default)
    if not isinstance(value, dict):
        raise ReacquisitionConfigError(f"{section} must be a mapping.")
    return dict(value)


def _resolve_path(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise ReacquisitionConfigError(f"{section} must be a non-empty path.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def load_reacquisition_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> ReacquisitionPipelineConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved_config = path.resolve() if path.is_absolute() else resolve_project_path(path)
    if not resolved_config.is_file():
        raise ReacquisitionConfigError(f"Reacquisition config does not exist: {resolved_config}")
    raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, "reacquisition config root")
    reacq = _mapping(root.get("reacquisition"), "reacquisition", default={})
    gates = _mapping(reacq.get("gates"), "reacquisition.gates", default={})
    ranking = _mapping(reacq.get("ranking"), "reacquisition.ranking", default={})
    probation = _mapping(reacq.get("probation"), "reacquisition.probation", default={})
    output = _mapping(root.get("output"), "output", default={})
    runtime = _mapping(root.get("runtime"), "runtime", default={})
    config = ReacquisitionPipelineConfig(
        project_root=project_root,
        config_path=resolved_config,
        reacquisition=ReacquisitionConfig(
            pre_event_context_frames=int(reacq.get("pre_event_context_frames", 20)),
            post_event_context_frames=int(reacq.get("post_event_context_frames", 50)),
            min_observations=int(gates.get("min_observations", 2)),
            duplicate_overlap_tolerance_frames=int(
                gates.get("duplicate_overlap_tolerance_frames", 2)
            ),
            max_motion_distance_normalized=float(
                gates.get("max_motion_distance_normalized", 0.20)
            ),
            min_grounding_score=float(gates.get("min_grounding_score", 0.10)),
            require_grounding_support=bool(gates.get("require_grounding_support", True)),
            min_final_score=float(ranking.get("min_final_score", 0.45)),
            ambiguity_margin=float(ranking.get("ambiguity_margin", 0.08)),
            missing_evidence_policy=str(
                ranking.get("missing_evidence_policy", "ignore")
            ),  # type: ignore[arg-type]
            weights=dict(ranking.get("weights", {})) or ReacquisitionConfig().weights,
            probation_window_frames=int(probation.get("window_frames", 20)),
            probation_min_observations=int(probation.get("min_observations", 3)),
            auto_confirm=bool(probation.get("auto_confirm", False)),
        ),
        output_dir=_resolve_path(
            output.get("directory", "outputs/locate_tracking/reacquisition"),
            project_root,
            "output.directory",
        ),
        overwrite=bool(runtime.get("overwrite", False)),
    )
    if overrides:
        changes: dict[str, Any] = {}
        if overrides.get("output_dir") is not None:
            changes["output_dir"] = _resolve_path(
                overrides["output_dir"],
                project_root,
                "output_dir",
            )
        if overrides.get("overwrite") is not None:
            changes["overwrite"] = bool(overrides["overwrite"])
        config = replace(config, **changes) if changes else config
    return config
