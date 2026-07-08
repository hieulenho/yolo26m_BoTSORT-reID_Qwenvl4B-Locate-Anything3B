"""Configuration loading for VLM analysis on tracking outputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.vlm.quantization import normalize_quantization


class VlmConfigError(RuntimeError):
    """Raised when VLM analysis configuration is invalid."""


DEFAULT_QWEN4B_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


@dataclass(frozen=True)
class VlmTrackingConfig:
    project_root: Path
    config_path: Path
    source_video: Path
    tracked_video: Path | None
    tracks_path: Path
    metadata_path: Path | None
    output_dir: Path
    keyframes_dir: Path
    crops_dir: Path
    keyframe_interval_seconds: float
    max_keyframes: int
    max_tracks: int
    max_crops_per_track: int
    crop_padding: float
    task_prompt: str
    model_id: str
    device: str
    torch_dtype: str
    quantization: str
    max_new_tokens: int
    temperature: float
    do_sample: bool
    run_model: bool
    overwrite: bool


def _mapping(value: Any, section: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if value is None and default is not None:
        return dict(default)
    if not isinstance(value, dict):
        raise VlmConfigError(f"{section} must be a mapping.")
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
        raise VlmConfigError(f"{section} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def _resolve_child_dir(value: Any, output_dir: Path, section: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise VlmConfigError(f"{section} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else output_dir / path


def load_vlm_tracking_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> VlmTrackingConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved_config = (
        path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    )
    if not resolved_config.is_file():
        raise VlmConfigError(f"VLM config does not exist: {resolved_config}")
    raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, "VLM config root")
    input_cfg = _mapping(root.get("input"), "input")
    output_cfg = _mapping(root.get("output"), "output")
    sampling_cfg = _mapping(root.get("sampling"), "sampling", default={})
    model_cfg = _mapping(root.get("model"), "model", default={})
    prompt_cfg = _mapping(root.get("prompt"), "prompt", default={})
    runtime_cfg = _mapping(root.get("runtime"), "runtime", default={})
    task_prompt = _load_task_prompt(prompt_cfg, project_root)

    output_dir = _resolve_path(
        output_cfg.get("dir", "outputs/vlm/qwen4b/tracking"),
        project_root,
        "output.dir",
    )
    assert output_dir is not None
    config = VlmTrackingConfig(
        project_root=project_root,
        config_path=resolved_config,
        source_video=_resolve_path(
            input_cfg.get("source_video"),
            project_root,
            "input.source_video",
        ),
        tracked_video=_resolve_path(
            input_cfg.get("tracked_video"),
            project_root,
            "input.tracked_video",
            required=False,
        ),
        tracks_path=_resolve_path(input_cfg.get("tracks"), project_root, "input.tracks"),
        metadata_path=_resolve_path(
            input_cfg.get("metadata"),
            project_root,
            "input.metadata",
            required=False,
        ),
        output_dir=output_dir,
        keyframes_dir=_resolve_child_dir(
            output_cfg.get("keyframes_dir", "keyframes"),
            output_dir,
            "output.keyframes_dir",
        ),
        crops_dir=_resolve_child_dir(
            output_cfg.get("crops_dir", "crops"),
            output_dir,
            "output.crops_dir",
        ),
        keyframe_interval_seconds=float(sampling_cfg.get("keyframe_interval_seconds", 1.0)),
        max_keyframes=int(sampling_cfg.get("max_keyframes", 12)),
        max_tracks=int(sampling_cfg.get("max_tracks", 40)),
        max_crops_per_track=int(sampling_cfg.get("max_crops_per_track", 3)),
        crop_padding=float(sampling_cfg.get("crop_padding", 0.12)),
        task_prompt=task_prompt,
        model_id=str(model_cfg.get("model_id", DEFAULT_QWEN4B_MODEL_ID)),
        device=str(model_cfg.get("device", "auto")),
        torch_dtype=str(model_cfg.get("torch_dtype", "auto")),
        quantization=normalize_quantization(str(model_cfg.get("quantization", "none"))),
        max_new_tokens=int(model_cfg.get("max_new_tokens", 1024)),
        temperature=float(model_cfg.get("temperature", 0.1)),
        do_sample=bool(model_cfg.get("do_sample", False)),
        run_model=bool(model_cfg.get("run_model", False)),
        overwrite=bool(runtime_cfg.get("overwrite", False)),
    )
    if overrides:
        config = _apply_overrides(config, overrides)
    _validate_config(config)
    return config


def _apply_overrides(
    config: VlmTrackingConfig,
    overrides: dict[str, Any],
) -> VlmTrackingConfig:
    changes: dict[str, Any] = {}
    path_fields = {
        "source_video": "source_video",
        "tracked_video": "tracked_video",
        "tracks": "tracks_path",
        "metadata": "metadata_path",
        "output_dir": "output_dir",
    }
    for override_key, field_name in path_fields.items():
        if overrides.get(override_key) is not None:
            changes[field_name] = _resolve_path(
                overrides[override_key],
                config.project_root,
                f"--{override_key.replace('_', '-')}",
                required=field_name not in {"tracked_video", "metadata_path"},
            )
    if "output_dir" in changes:
        output_dir = changes["output_dir"]
        changes["keyframes_dir"] = output_dir / "keyframes"
        changes["crops_dir"] = output_dir / "crops"
    scalar_fields = {
        "keyframe_interval_seconds",
        "max_keyframes",
        "max_tracks",
        "max_crops_per_track",
        "crop_padding",
        "model_id",
        "device",
        "torch_dtype",
        "quantization",
        "max_new_tokens",
        "temperature",
        "do_sample",
        "run_model",
        "overwrite",
    }
    for field_name in scalar_fields:
        if overrides.get(field_name) is not None:
            changes[field_name] = overrides[field_name]
    if overrides.get("task_prompt") is not None:
        changes["task_prompt"] = overrides["task_prompt"]
    if "quantization" in changes:
        changes["quantization"] = normalize_quantization(str(changes["quantization"]))
    return replace(config, **changes) if changes else config


def _load_task_prompt(prompt_cfg: dict[str, Any], project_root: Path) -> str:
    task_file = prompt_cfg.get("task_file")
    if task_file:
        path = _resolve_path(task_file, project_root, "prompt.task_file")
        assert path is not None
        if not path.is_file():
            raise VlmConfigError(f"prompt.task_file does not exist: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise VlmConfigError(f"prompt.task_file is empty: {path}")
        return text
    return str(prompt_cfg.get("task", _default_task_prompt()))


def _validate_config(config: VlmTrackingConfig) -> None:
    if config.keyframe_interval_seconds <= 0:
        raise VlmConfigError("sampling.keyframe_interval_seconds must be positive.")
    for field_name in ("max_keyframes", "max_tracks", "max_crops_per_track"):
        if int(getattr(config, field_name)) <= 0:
            raise VlmConfigError(f"sampling.{field_name} must be positive.")
    if not 0.0 <= config.crop_padding <= 1.0:
        raise VlmConfigError("sampling.crop_padding must be in [0, 1].")
    if config.max_new_tokens <= 0:
        raise VlmConfigError("model.max_new_tokens must be positive.")
    if config.temperature < 0:
        raise VlmConfigError("model.temperature must be non-negative.")
    if not config.model_id:
        raise VlmConfigError("model.model_id must not be empty.")


def _default_task_prompt() -> str:
    return (
        "Phan tich video da duoc detect va tracking. Hay dua ra tom tat ngan gon, "
        "cac track_id dang chu y, hanh vi/chuyen dong noi bat, va cac diem can kiem tra "
        "neu ID co dau hieu bi nhay hoac bi mat."
    )
