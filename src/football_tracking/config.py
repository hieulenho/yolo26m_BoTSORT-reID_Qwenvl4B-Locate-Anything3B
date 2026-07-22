"""YAML configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_tracking.paths import get_project_root, resolve_project_path


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


REQUIRED_SECTIONS = ("project", "paths", "runtime")


@dataclass(frozen=True)
class AppConfig:
    """Resolved application configuration."""

    project_root: Path
    config_path: Path
    project: dict[str, Any]
    paths: dict[str, Path]
    runtime: dict[str, Any]


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a mapping.")
    return value


def _resolve_config_path(config_path: str | Path | None, project_root: Path) -> Path:
    if config_path is None:
        return resolve_project_path("configs/project.yaml", project_root=project_root)

    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return resolve_project_path(path, project_root=project_root)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc

    if loaded is None:
        raise ConfigError(f"Config file is empty: {path}")
    return _require_mapping(loaded, "Config root")


def _validate_sections(raw_config: dict[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in raw_config]
    if missing:
        raise ConfigError(f"Config is missing required section(s): {', '.join(missing)}")


def _resolve_paths(paths_config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key, value in paths_config.items():
        if not isinstance(value, str):
            raise ConfigError(f"paths.{key} must be a relative path string.")

        raw_path = Path(value)
        if raw_path.is_absolute():
            raise ConfigError(f"paths.{key} must be relative to the project root: {value}")

        resolved[key] = resolve_project_path(raw_path, project_root=project_root)
    return resolved


def load_config(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> AppConfig:
    """Load and validate the project YAML config."""

    root = Path(project_root).resolve() if project_root is not None else get_project_root()
    path = _resolve_config_path(config_path, root)
    raw_config = _load_yaml(path)
    _validate_sections(raw_config)

    project = _require_mapping(raw_config["project"], "project")
    paths = _resolve_paths(_require_mapping(raw_config["paths"], "paths"), root)
    runtime = _require_mapping(raw_config["runtime"], "runtime")

    return AppConfig(
        project_root=root,
        config_path=path,
        project=dict(project),
        paths=paths,
        runtime=dict(runtime),
    )
