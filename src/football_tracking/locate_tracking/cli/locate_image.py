"""Standalone locate-image CLI implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.locate_tracking.grounding.backend import MockGroundingBackend
from football_tracking.locate_tracking.grounding.cache import GroundingCache
from football_tracking.locate_tracking.grounding.locate_anything_backend import (
    DEFAULT_LOCATEANYTHING_MODEL_ID,
    DEFAULT_LOCATEANYTHING_PROMPT_TEMPLATE,
    LocateAnythingBackend,
    LocateAnythingBackendError,
)
from football_tracking.locate_tracking.grounding.service import (
    GroundingService,
    GroundingServiceError,
)
from football_tracking.paths import get_project_root, resolve_project_path


class LocateImageError(RuntimeError):
    """Raised when the standalone locate-image command cannot run."""


@dataclass(frozen=True)
class LocateImageConfig:
    project_root: Path
    config_path: Path
    backend_name: str
    model_id: str
    device: str
    torch_dtype: str
    max_new_tokens: int
    prompt_template: str
    cache_enabled: bool
    cache_directory: Path
    output_directory: Path
    overwrite: bool
    log_level: str
    mock_response: str


def _mapping(value: Any, section: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if value is None and default is not None:
        return dict(default)
    if not isinstance(value, dict):
        raise LocateImageError(f"{section} must be a mapping.")
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
        raise LocateImageError(f"{section} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def load_locate_image_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> LocateImageConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved_config = path.resolve() if path.is_absolute() else resolve_project_path(path)
    if not resolved_config.is_file():
        raise LocateImageError(f"Locate image config does not exist: {resolved_config}")
    raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, "locate image config root")
    backend = _mapping(root.get("backend"), "backend")
    inference = _mapping(root.get("inference"), "inference", default={})
    cache = _mapping(root.get("cache"), "cache", default={})
    output = _mapping(root.get("output"), "output", default={})
    runtime = _mapping(root.get("runtime"), "runtime", default={})
    config = LocateImageConfig(
        project_root=project_root,
        config_path=resolved_config,
        backend_name=str(backend.get("name", "locate_anything")),
        model_id=str(backend.get("model_id", DEFAULT_LOCATEANYTHING_MODEL_ID)),
        device=str(backend.get("device", "cuda")),
        torch_dtype=str(backend.get("torch_dtype", "bfloat16")),
        max_new_tokens=int(inference.get("max_new_tokens", backend.get("max_new_tokens", 4096))),
        prompt_template=str(
            inference.get(
                "prompt_template",
                backend.get(
                    "prompt_template",
                    DEFAULT_LOCATEANYTHING_PROMPT_TEMPLATE,
                ),
            )
        ),
        cache_enabled=bool(cache.get("enabled", True)),
        cache_directory=_resolve_path(
            cache.get("directory", "outputs/locate_tracking/cache"),
            project_root,
            "cache.directory",
        ),
        output_directory=_resolve_path(
            output.get("directory", "outputs/locate_tracking/grounding"),
            project_root,
            "output.directory",
        ),
        overwrite=bool(runtime.get("overwrite", False)),
        log_level=str(runtime.get("log_level", "INFO")),
        mock_response=str(backend.get("mock_response", "<box>none</box>")),
    )
    if overrides:
        config = _apply_overrides(config, overrides)
    _validate_config(config)
    return config


def _apply_overrides(
    config: LocateImageConfig,
    overrides: dict[str, Any],
) -> LocateImageConfig:
    changes: dict[str, Any] = {}
    for field_name in ("backend_name", "model_id", "device", "torch_dtype"):
        if overrides.get(field_name) is not None:
            changes[field_name] = str(overrides[field_name])
    if overrides.get("max_new_tokens") is not None:
        changes["max_new_tokens"] = int(overrides["max_new_tokens"])
    if overrides.get("overwrite") is not None:
        changes["overwrite"] = bool(overrides["overwrite"])
    return replace(config, **changes) if changes else config


def _validate_config(config: LocateImageConfig) -> None:
    if config.backend_name not in {"locate_anything", "mock"}:
        raise LocateImageError(f"Unsupported grounding backend: {config.backend_name}")
    if not config.model_id:
        raise LocateImageError("backend.model_id must not be empty.")
    if config.max_new_tokens <= 0:
        raise LocateImageError("inference.max_new_tokens must be positive.")


def _default_output_path(output_directory: Path, image_path: Path) -> Path:
    return output_directory / f"{image_path.stem}.grounding.json"


def _build_backend(config: LocateImageConfig) -> Any:
    if config.backend_name == "mock":
        return MockGroundingBackend(default_response=config.mock_response, model_id=config.model_id)
    return LocateAnythingBackend(
        model_id=config.model_id,
        device=config.device,
        torch_dtype=config.torch_dtype,
        max_new_tokens=config.max_new_tokens,
        prompt_template=config.prompt_template,
    )


def run_locate_image(
    config_path: str | Path,
    *,
    image: str | Path,
    query: str,
    output: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_locate_image_config(config_path, overrides=overrides)
    image_path = _resolve_path(image, config.project_root, "--image")
    assert image_path is not None
    output_path = (
        _resolve_path(output, config.project_root, "--output")
        if output is not None
        else _default_output_path(config.output_directory, image_path)
    )
    assert output_path is not None
    if dry_run:
        return {
            "dry_run": True,
            "backend": {
                "name": config.backend_name,
                "model_id": config.model_id,
            },
            "image": str(image_path),
            "query": query,
            "output": str(output_path),
            "cache_directory": str(config.cache_directory),
            "action": "validated config and planned output; model was not loaded",
        }
    cache = GroundingCache(
        config.cache_directory,
        enabled=config.cache_enabled,
        overwrite=config.overwrite,
    )
    service = GroundingService(
        backend=_build_backend(config),
        cache=cache,
        overwrite=config.overwrite,
    )
    try:
        result = service.ground_image(
            image_path=image_path,
            query=query,
            output_path=output_path,
            overwrite=config.overwrite,
        )
    except (GroundingServiceError, LocateAnythingBackendError) as exc:
        raise LocateImageError(str(exc)) from exc
    return {
        "dry_run": False,
        "result": result.to_dict(),
        "paths": {
            "output": str(output_path),
            "cache_directory": str(config.cache_directory),
        },
    }
