"""Configuration loading for multi-frame semantic language-track resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.locate_tracking.sampling.explicit_selector import (
    parse_explicit_frames,
)
from football_tracking.locate_tracking.semantic_memory.schemas import (
    SemanticMemoryConfig,
)
from football_tracking.paths import get_project_root, resolve_project_path


class SemanticMemoryConfigError(RuntimeError):
    """Raised when semantic memory configuration is invalid."""


@dataclass(frozen=True)
class SemanticMemoryPipelineConfig:
    project_root: Path
    config_path: Path
    sampling: dict[str, Any]
    semantic_memory: SemanticMemoryConfig
    output_dir: Path
    overwrite: bool
    log_level: str


def _mapping(value: Any, section: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if value is None and default is not None:
        return dict(default)
    if not isinstance(value, dict):
        raise SemanticMemoryConfigError(f"{section} must be a mapping.")
    return value


def _resolve_path(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise SemanticMemoryConfigError(f"{section} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def _semantic_config(
    root: dict[str, Any], overrides: dict[str, Any] | None
) -> SemanticMemoryConfig:
    memory = _mapping(root.get("semantic_memory"), "semantic_memory", default={})
    aggregation = _mapping(memory.get("aggregation"), "semantic_memory.aggregation", default={})
    weights = _mapping(
        aggregation.get("weights"), "semantic_memory.aggregation.weights", default={}
    )
    evidence_weights = _mapping(
        memory.get("evidence_weights"), "semantic_memory.evidence_weights", default={}
    )
    decision = _mapping(memory.get("decision"), "semantic_memory.decision", default={})
    values = {
        "query_mode": memory.get("query_mode", "single_target"),
        "aggregation_strategy": aggregation.get("strategy", "weighted"),
        "quality_score_mode": aggregation.get("quality_score_mode", "top_k_mean"),
        "top_k_quality": aggregation.get("top_k_quality", 3),
        "support_weight": weights.get("support_weight", 0.50),
        "quality_weight": weights.get("quality_weight", 0.30),
        "consistency_weight": weights.get("consistency_weight", 0.20),
        "resolved_selected_weight": evidence_weights.get("resolved_selected_weight", 1.0),
        "ambiguous_candidate_weight": evidence_weights.get("ambiguous_candidate_weight", 0.25),
        "weak_candidate_weight": evidence_weights.get("weak_candidate_weight", 0.10),
        "min_usable_frames": decision.get("min_usable_frames", 2),
        "min_support_frames": decision.get("min_support_frames", 2),
        "min_support_ratio": decision.get("min_support_ratio", 0.40),
        "min_aggregate_score": decision.get("min_aggregate_score", 0.35),
        "winner_margin": decision.get("winner_margin", 0.08),
    }
    if overrides:
        for key in (
            "query_mode",
            "aggregation_strategy",
            "min_usable_frames",
            "min_support_frames",
            "min_support_ratio",
            "min_aggregate_score",
            "winner_margin",
        ):
            if overrides.get(key) is not None:
                values[key] = overrides[key]
    return SemanticMemoryConfig(**values)


def load_semantic_memory_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> SemanticMemoryPipelineConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path)
    if not resolved.is_file():
        raise SemanticMemoryConfigError(f"Semantic memory config does not exist: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, "semantic memory config root", default={})
    sampling = _mapping(root.get("sampling"), "sampling", default={})
    output = _mapping(root.get("output"), "output", default={})
    runtime = _mapping(root.get("runtime"), "runtime", default={})
    config = SemanticMemoryPipelineConfig(
        project_root=project_root,
        config_path=resolved,
        sampling={
            "mode": str(sampling.get("mode", "uniform")),
            "max_frames": int(sampling.get("max_frames", 5)),
            "start_frame": int(sampling.get("start_frame", 1)),
            "end_frame": sampling.get("end_frame"),
            "explicit_frames": parse_explicit_frames(sampling.get("explicit_frames")),
        },
        semantic_memory=_semantic_config(root, overrides),
        output_dir=_resolve_path(
            output.get("directory", "outputs/locate_tracking/semantic_memory"),
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
    config: SemanticMemoryPipelineConfig,
    overrides: dict[str, Any],
) -> SemanticMemoryPipelineConfig:
    changes: dict[str, Any] = {}
    sampling = dict(config.sampling)
    for key in ("mode", "max_frames", "start_frame", "end_frame"):
        if overrides.get(key) is not None:
            sampling[key] = overrides[key]
    if overrides.get("explicit_frames") is not None:
        sampling["explicit_frames"] = parse_explicit_frames(overrides["explicit_frames"])
    if sampling != config.sampling:
        changes["sampling"] = sampling
    if overrides.get("output_dir") is not None:
        changes["output_dir"] = _resolve_path(
            overrides["output_dir"],
            config.project_root,
            "--output-dir",
        )
    if overrides.get("overwrite") is not None:
        changes["overwrite"] = bool(overrides["overwrite"])
    return replace(config, **changes) if changes else config
