"""Checkpoint resolution and metadata helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class CheckpointError(RuntimeError):
    """Raised when checkpoint management fails."""


def compute_file_hash(path: Path) -> str:
    if not path.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path) -> Path:
    if not path.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {path}")
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise CheckpointError(f"Unsupported checkpoint extension: {path}")
    return path


def resolve_best_checkpoint(run_dir: Path) -> Path:
    return validate_checkpoint(run_dir / "weights" / "best.pt")


def resolve_last_checkpoint(run_dir: Path) -> Path:
    return validate_checkpoint(run_dir / "weights" / "last.pt")


def read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    validate_checkpoint(path)
    return {
        "path": str(path),
        "sha256": compute_file_hash(path),
        "size_bytes": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
    }


def _runtime_versions() -> dict[str, Any]:
    payload: dict[str, Any] = {"python_version": platform.python_version()}
    try:
        import torch  # type: ignore[import-not-found]

        payload["torch_version"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # noqa: BLE001
        payload["torch_version"] = None
        payload["cuda_available"] = False
        payload["gpu"] = None
        payload["torch_error"] = str(exc)
    try:
        import ultralytics  # type: ignore[import-not-found]

        payload["ultralytics_version"] = ultralytics.__version__
    except Exception as exc:  # noqa: BLE001
        payload["ultralytics_version"] = None
        payload["ultralytics_error"] = str(exc)
    return payload


def copy_final_checkpoints(
    run_dir: Path,
    models_dir: Path,
    run_name: str,
    dataset_manifest: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Path]:
    best = resolve_best_checkpoint(run_dir)
    last = resolve_last_checkpoint(run_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    best_destination = models_dir / f"{run_name}_best.pt"
    last_destination = models_dir / f"{run_name}_last.pt"
    shutil.copy2(best, best_destination)
    shutil.copy2(last, last_destination)

    metadata = {
        "source_checkpoint": str(best),
        "sha256": compute_file_hash(best_destination),
        "training_run": str(run_dir),
        "dataset_manifest_hash": compute_file_hash(dataset_manifest)
        if dataset_manifest and dataset_manifest.is_file()
        else None,
        "config_hash": compute_file_hash(config_path)
        if config_path and config_path.is_file()
        else None,
        "epoch": None,
        "date": datetime.now(UTC).isoformat(),
        **_runtime_versions(),
    }
    metadata_path = models_dir / f"{run_name}_best.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "best": best_destination,
        "last": last_destination,
        "metadata": metadata_path,
    }
