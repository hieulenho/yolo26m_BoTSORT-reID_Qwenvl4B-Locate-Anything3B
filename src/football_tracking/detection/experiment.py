"""Experiment metadata for detector training."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def environment_metadata(project_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_root": str(project_root),
        "git_commit": _git_value(["rev-parse", "HEAD"], project_root),
        "git_dirty": bool(_git_value(["status", "--porcelain"], project_root)),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.platform(),
    }
    try:
        import torch  # type: ignore[import-not-found]

        payload["torch_version"] = torch.__version__
        payload["cuda_version"] = getattr(torch.version, "cuda", None)
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        payload["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as exc:  # noqa: BLE001
        payload["torch_error"] = str(exc)
        payload["cuda_available"] = False
        payload["gpu_name"] = None
        payload["gpu_count"] = 0
    try:
        import ultralytics  # type: ignore[import-not-found]

        payload["ultralytics_version"] = ultralytics.__version__
    except Exception as exc:  # noqa: BLE001
        payload["ultralytics_error"] = str(exc)
        payload["ultralytics_version"] = None
    return payload


@dataclass
class ExperimentManifest:
    experiment_name: str
    run_dir: Path
    project_root: Path
    status: str = "initialized"
    start_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    end_time: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            **environment_metadata(self.project_root),
            "warnings": self.warnings,
            "errors": self.errors,
            **self.payload,
        }

    def write(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "experiment_manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def finish(self, status: str, errors: list[str] | None = None) -> Path:
        self.status = status
        self.end_time = datetime.now(UTC).isoformat()
        if errors:
            self.errors.extend(errors)
        return self.write()
