"""Detector checkpoint resolution for tracking runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from football_tracking.detection.detector import KNOWN_ULTRALYTICS_CHECKPOINTS
from football_tracking.paths import resolve_project_path

CheckpointType = Literal["fine_tuned", "smoke", "pretrained_coco", "unknown"]


class CheckpointResolutionError(RuntimeError):
    """Raised when a detector checkpoint cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedCheckpoint:
    checkpoint: str | Path
    checkpoint_type: CheckpointType
    checkpoint_hash: str | None
    source: str
    fallback_used: bool
    smoke_only: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "checkpoint_type": self.checkpoint_type,
            "checkpoint_hash": self.checkpoint_hash,
            "source": self.source,
            "fallback_used": self.fallback_used,
            "smoke_only": self.smoke_only,
            "warnings": self.warnings,
        }


def compute_checkpoint_hash(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify_checkpoint_type(checkpoint: str | Path) -> CheckpointType:
    text = str(checkpoint).replace("\\", "/").lower()
    name = Path(text).name
    if name in KNOWN_ULTRALYTICS_CHECKPOINTS:
        return "pretrained_coco"
    if "smoke" in text:
        return "smoke"
    if Path(str(checkpoint)).is_file():
        return "fine_tuned"
    return "unknown"


def _resolve_candidate(value: str | Path, project_root: Path) -> str | Path:
    text = str(value)
    if text in KNOWN_ULTRALYTICS_CHECKPOINTS:
        return text
    path = Path(text)
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def validate_detector_checkpoint(
    checkpoint: str | Path,
    allow_pretrained: bool = False,
) -> None:
    if str(checkpoint) in KNOWN_ULTRALYTICS_CHECKPOINTS:
        if allow_pretrained:
            return
        raise CheckpointResolutionError(
            f"Pretrained checkpoint requires fallback permission: {checkpoint}"
        )
    if not Path(checkpoint).is_file():
        raise CheckpointResolutionError(f"Detector checkpoint does not exist: {checkpoint}")


def _existing_local(value: str | Path | None, project_root: Path) -> Path | None:
    if not value:
        return None
    candidate = _resolve_candidate(value, project_root)
    return candidate if isinstance(candidate, Path) and candidate.is_file() else None


def resolve_detector_checkpoint(
    model_config: dict[str, Any],
    project_root: Path,
    explicit_checkpoint: str | Path | None = None,
) -> ResolvedCheckpoint:
    warnings: list[str] = []
    allow_pretrained = bool(
        model_config.get("allow_pretrained_fallback")
        or model_config.get("allow_fallback_checkpoint")
    )
    allow_smoke = bool(model_config.get("allow_smoke_checkpoint", False))

    if explicit_checkpoint is not None:
        checkpoint = _resolve_candidate(explicit_checkpoint, project_root)
        checkpoint_type = identify_checkpoint_type(checkpoint)
        if checkpoint_type == "pretrained_coco":
            validate_detector_checkpoint(checkpoint, allow_pretrained=True)
        else:
            validate_detector_checkpoint(checkpoint, allow_pretrained=False)
        if checkpoint_type == "smoke" and not allow_smoke:
            raise CheckpointResolutionError(
                f"Smoke checkpoint was provided but allow_smoke_checkpoint=false: {checkpoint}"
            )
        return ResolvedCheckpoint(
            checkpoint=checkpoint,
            checkpoint_type=checkpoint_type,
            checkpoint_hash=compute_checkpoint_hash(checkpoint),
            source="cli",
            fallback_used=False,
            smoke_only=checkpoint_type == "smoke",
            warnings=warnings,
        )

    configured = model_config.get("checkpoint")
    configured_path = _existing_local(configured, project_root)
    if configured_path is not None:
        checkpoint_type = identify_checkpoint_type(configured_path)
        if checkpoint_type == "smoke" and not allow_smoke:
            raise CheckpointResolutionError(
                "Configured checkpoint is smoke but allow_smoke_checkpoint=false: "
                f"{configured_path}"
            )
        return ResolvedCheckpoint(
            checkpoint=configured_path,
            checkpoint_type=checkpoint_type,
            checkpoint_hash=compute_checkpoint_hash(configured_path),
            source="config.checkpoint",
            fallback_used=False,
            smoke_only=checkpoint_type == "smoke",
            warnings=warnings,
        )

    for index, alternative in enumerate(model_config.get("alternative_checkpoints", []) or []):
        alternative_path = _existing_local(alternative, project_root)
        if alternative_path is None:
            continue
        checkpoint_type = identify_checkpoint_type(alternative_path)
        if checkpoint_type == "smoke" and not allow_smoke:
            continue
        warnings.append(
            "Using alternative checkpoint because configured one is missing: "
            f"{alternative_path}"
        )
        return ResolvedCheckpoint(
            checkpoint=alternative_path,
            checkpoint_type=checkpoint_type,
            checkpoint_hash=compute_checkpoint_hash(alternative_path),
            source=f"config.alternative_checkpoints[{index}]",
            fallback_used=True,
            smoke_only=checkpoint_type == "smoke",
            warnings=warnings,
        )

    standard_candidates = (
        Path("models/detector/yolov8m_players_best.pt"),
        Path("models/detector/yolov8m_sportsmot_best.pt"),
    )
    for index, standard in enumerate(standard_candidates):
        standard_path = _existing_local(standard, project_root)
        if standard_path is None:
            continue
        warnings.append(f"Using standard checkpoint candidate: {standard_path}")
        return ResolvedCheckpoint(
            checkpoint=standard_path,
            checkpoint_type=identify_checkpoint_type(standard_path),
            checkpoint_hash=compute_checkpoint_hash(standard_path),
            source=f"standard[{index}]",
            fallback_used=True,
            smoke_only=False,
            warnings=warnings,
        )

    if allow_smoke:
        for index, smoke in enumerate(
            (
                Path("models/detector/yolov8m_sportsmot_smoke_best.pt"),
                Path("models/detector/yolov8m_smoke_best.pt"),
            )
        ):
            smoke_path = _existing_local(smoke, project_root)
            if smoke_path is None:
                continue
            warnings.append(f"Using smoke checkpoint for smoke-only tracking: {smoke_path}")
            return ResolvedCheckpoint(
                checkpoint=smoke_path,
                checkpoint_type="smoke",
                checkpoint_hash=compute_checkpoint_hash(smoke_path),
                source=f"smoke[{index}]",
                fallback_used=True,
                smoke_only=True,
                warnings=warnings,
            )

    fallback = model_config.get("fallback_checkpoint")
    if allow_pretrained and fallback:
        checkpoint = _resolve_candidate(fallback, project_root)
        checkpoint_type = identify_checkpoint_type(checkpoint)
        validate_detector_checkpoint(
            checkpoint,
            allow_pretrained=checkpoint_type == "pretrained_coco",
        )
        warnings.append(f"Using pretrained fallback for plumbing only: {checkpoint}")
        return ResolvedCheckpoint(
            checkpoint=checkpoint,
            checkpoint_type=checkpoint_type,
            checkpoint_hash=compute_checkpoint_hash(checkpoint),
            source="config.fallback_checkpoint",
            fallback_used=True,
            smoke_only=checkpoint_type == "smoke",
            warnings=warnings,
        )

    missing = configured or "models/detector/yolov8m_players_best.pt"
    raise CheckpointResolutionError(
        "No usable detector checkpoint found. "
        f"Configured checkpoint is missing and fallback is disabled: {missing}"
    )
