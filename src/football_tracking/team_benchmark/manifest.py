"""Load and save team benchmark manifests."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.team_benchmark.schemas import (
    TeamBenchmarkManifest,
    TeamPredictionManifest,
)


class TeamBenchmarkManifestError(RuntimeError):
    """Raised when a team benchmark manifest cannot be loaded."""


def load_team_benchmark_manifest(path: str | Path) -> TeamBenchmarkManifest:
    resolved = _resolve_existing(path)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        return _resolve_manifest_paths(TeamBenchmarkManifest.from_dict(raw))
    except Exception as exc:  # noqa: BLE001
        raise TeamBenchmarkManifestError(
            f"Failed to load team benchmark manifest: {resolved}: {exc}"
        ) from exc


def load_team_prediction_manifest(path: str | Path) -> TeamPredictionManifest:
    resolved = _resolve_existing(path)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        return TeamPredictionManifest.from_dict(raw)
    except Exception as exc:  # noqa: BLE001
        raise TeamBenchmarkManifestError(
            f"Failed to load team prediction manifest: {resolved}: {exc}"
        ) from exc


def save_team_benchmark_manifest(manifest: TeamBenchmarkManifest, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return output


def save_team_prediction_manifest(manifest: TeamPredictionManifest, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return output


def _resolve_existing(path: str | Path) -> Path:
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else resolve_project_path(candidate)
    if not resolved.is_file():
        raise TeamBenchmarkManifestError(f"Manifest does not exist: {resolved}")
    return resolved


def _resolve_manifest_paths(manifest: TeamBenchmarkManifest) -> TeamBenchmarkManifest:
    # Rehydrate through dicts so paths are resolved without relying on dataclass mutation.
    root = get_project_root()
    data = manifest.to_dict()
    for sequence in data["sequences"]:
        for key in ("source_video", "tracks_path", "mot_ground_truth_path"):
            value = sequence.get(key)
            if value:
                path = Path(value)
                sequence[key] = str(path if path.is_absolute() else (root / path).resolve())
    return TeamBenchmarkManifest.from_dict(data)
