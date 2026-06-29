"""Project path helpers."""

from __future__ import annotations

import os
from pathlib import Path


class ProjectPathError(RuntimeError):
    """Raised when a project path cannot be resolved safely."""


PROJECT_ROOT_ENV_VAR = "FOOTBALL_TRACKING_ROOT"
PROJECT_MARKER = "pyproject.toml"
OUTPUT_DIRECTORIES = (
    "outputs",
    "outputs/detections",
    "outputs/tracks",
    "outputs/videos",
    "outputs/metrics",
    "outputs/figures",
    "outputs/logs",
)


def _candidate_directories(start_path: Path) -> tuple[Path, ...]:
    start = start_path.resolve()
    directory = start if start.is_dir() else start.parent
    return (directory, *directory.parents)


def _contains_project_marker(path: Path) -> bool:
    marker = path / PROJECT_MARKER
    return marker.is_file()


def get_project_root(start_path: str | Path | None = None) -> Path:
    """Return the repository root without hard-coding a machine-specific path."""

    env_value = os.environ.get(PROJECT_ROOT_ENV_VAR)
    if env_value:
        env_root = Path(env_value).expanduser().resolve()
        if not env_root.is_dir():
            raise ProjectPathError(
                f"{PROJECT_ROOT_ENV_VAR} points to a directory that does not exist: {env_root}"
            )
        if not _contains_project_marker(env_root):
            raise ProjectPathError(
                f"{PROJECT_ROOT_ENV_VAR} does not contain {PROJECT_MARKER}: {env_root}"
            )
        return env_root

    starts = [Path(start_path)] if start_path is not None else [Path(__file__), Path.cwd()]
    for start in starts:
        for candidate in _candidate_directories(start):
            if _contains_project_marker(candidate):
                return candidate

    raise ProjectPathError(
        f"Could not find project root. Set {PROJECT_ROOT_ENV_VAR} or run from the repository."
    )


def _ensure_within_root(path: Path, project_root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ProjectPathError(f"Resolved path escapes project root: {resolved_path}")
    return resolved_path


def resolve_project_path(
    relative_path: str | Path,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve a path inside the project root and reject path traversal."""

    root = Path(project_root).resolve() if project_root is not None else get_project_root()
    raw_path = Path(relative_path)
    target = raw_path if raw_path.is_absolute() else root / raw_path
    return _ensure_within_root(target, root)


def ensure_output_directories(project_root: str | Path | None = None) -> list[Path]:
    """Create the standard output directories inside the project root."""

    root = Path(project_root).resolve() if project_root is not None else get_project_root()
    created: list[Path] = []

    for directory in OUTPUT_DIRECTORIES:
        path = resolve_project_path(directory, project_root=root)
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)

    return created
