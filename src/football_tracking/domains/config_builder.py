"""Build runnable configs from reusable domain profiles."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_tracking.paths import get_project_root, resolve_project_path


class DomainConfigError(RuntimeError):
    """Raised when a domain profile or generated config is invalid."""


@dataclass(frozen=True)
class DomainProfile:
    project_root: Path
    profile_path: Path
    name: str
    namespace: str
    description: str
    model: dict[str, Any]
    detector: dict[str, Any]
    tracker: dict[str, Any]
    render: dict[str, Any]
    output: dict[str, Any]
    dataset: dict[str, Any] | None

    @property
    def default_tracker_name(self) -> str:
        return str(self.tracker.get("default_name", "botsort_reid")).lower()

    def tracker_config_for_preset(self, preset: str | None = None) -> Path:
        preset_name = preset or str(self.tracker.get("default_preset", "balanced"))
        presets = _mapping(self.tracker.get("presets", {}), "tracker.presets")
        value = presets.get(preset_name, self.tracker.get("default_config"))
        if value is None:
            raise DomainConfigError(
                f"Tracker preset is not configured for domain '{self.name}': {preset_name}"
            )
        path = _resolve_existing_path(value, self.project_root, f"tracker.presets.{preset_name}")
        return path

    def seqmap_for_split(self, split: str) -> Path | None:
        if self.dataset is None:
            return None
        seqmaps = self.dataset.get("seqmaps", {}) or {}
        if not isinstance(seqmaps, dict):
            raise DomainConfigError("dataset.seqmaps must be a mapping.")
        value = seqmaps.get(split, self.dataset.get("seqmap"))
        if value is None:
            return None
        return _resolve_path(value, self.project_root, f"dataset.seqmaps.{split}")

    def split(self, override: str | None = None) -> str:
        if override:
            return override
        if self.dataset is None:
            return "val"
        return str(self.dataset.get("default_split", self.dataset.get("split", "val")))


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DomainConfigError(f"{section} must be a mapping.")
    return value


def _resolve_path(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise DomainConfigError(f"{section} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)


def _resolve_existing_path(value: Any, project_root: Path, section: str) -> Path:
    path = _resolve_path(value, project_root, section)
    if not path.is_file():
        raise DomainConfigError(f"{section} does not exist: {path}")
    return path


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise DomainConfigError("domain.name must contain at least one alphanumeric character.")
    return slug


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _optional_relative(path: Path | None, project_root: Path) -> str | None:
    return _relative(path, project_root) if path is not None else None


def _clean_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_clean_none(item) for item in value]
    return value


def _dump_yaml(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise DomainConfigError(f"Generated config exists and overwrite=false: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_clean_none(payload), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def load_domain_profile(profile_path: str | Path) -> DomainProfile:
    project_root = get_project_root()
    path = Path(profile_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    if not resolved.is_file():
        raise DomainConfigError(f"Domain profile does not exist: {resolved}")
    raw = _mapping(yaml.safe_load(resolved.read_text(encoding="utf-8")), "domain profile root")
    domain = _mapping(raw.get("domain"), "domain")
    name = str(domain.get("name", "")).strip()
    namespace = _slug(str(domain.get("namespace", name)))
    model = dict(_mapping(raw.get("model"), "model"))
    detector = dict(_mapping(raw.get("detector"), "detector"))
    tracker = dict(_mapping(raw.get("tracker"), "tracker"))
    render = dict(raw.get("render", {}) or {})
    output = dict(raw.get("output", {}) or {})
    dataset = raw.get("dataset")
    profile = DomainProfile(
        project_root=project_root,
        profile_path=resolved,
        name=name,
        namespace=namespace,
        description=str(domain.get("description", "")),
        model=model,
        detector=detector,
        tracker=tracker,
        render=render,
        output=output,
        dataset=dict(_mapping(dataset, "dataset")) if dataset is not None else None,
    )
    _validate_profile(profile)
    return profile


def _validate_profile(profile: DomainProfile) -> None:
    _slug(profile.name)
    if not profile.model.get("checkpoint"):
        raise DomainConfigError("model.checkpoint is required.")
    if not profile.detector.get("target_class_name"):
        raise DomainConfigError("detector.target_class_name is required.")
    for key in ("conf", "iou"):
        value = float(profile.detector.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise DomainConfigError(f"detector.{key} must be in [0, 1].")
    if int(profile.detector.get("imgsz", 0)) <= 0:
        raise DomainConfigError("detector.imgsz must be positive.")
    if int(profile.detector.get("max_det", 0)) <= 0:
        raise DomainConfigError("detector.max_det must be positive.")
    profile.tracker_config_for_preset()
    if profile.dataset is None:
        return
    mot_root = _resolve_path(
        profile.dataset.get("mot_root"),
        profile.project_root,
        "dataset.mot_root",
    )
    if not mot_root.is_dir():
        raise DomainConfigError(f"dataset.mot_root does not exist: {mot_root}")
    seqmap = profile.seqmap_for_split(profile.split())
    if seqmap is not None and not seqmap.is_file():
        raise DomainConfigError(f"dataset seqmap does not exist: {seqmap}")


def build_domain_configs(
    profile_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    preset: str | None = None,
    split: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    profile = load_domain_profile(profile_path)
    project_root = profile.project_root
    generated_dir = (
        _resolve_path(output_dir, project_root, "output_dir")
        if output_dir is not None
        else resolve_project_path(Path("configs/generated") / profile.namespace, project_root)
    )
    tracker_config = profile.tracker_config_for_preset(preset)
    split_name = profile.split(split)
    seqmap = profile.seqmap_for_split(split_name)
    configs = _domain_config_payloads(profile, generated_dir, tracker_config, split_name, seqmap)
    planned_paths = {name: path for name, (path, _payload) in configs.items()}
    if not dry_run:
        for path, payload in configs.values():
            _dump_yaml(path, payload, overwrite=overwrite)
        _write_manifest(
            profile,
            generated_dir,
            planned_paths,
            tracker_config,
            split_name,
            overwrite,
        )
    commands = _domain_commands(profile, planned_paths, split_name)
    return {
        "dry_run": dry_run,
        "domain": profile.name,
        "namespace": profile.namespace,
        "preset": preset or profile.tracker.get("default_preset", "balanced"),
        "split": split_name,
        "tracker_config": _relative(tracker_config, project_root),
        "generated_dir": _relative(generated_dir, project_root),
        "configs": {name: _relative(path, project_root) for name, path in planned_paths.items()},
        "commands": commands,
        "note": (
            "Generated configs are wrappers around the existing CLI; "
            "the old configs still work."
        ),
    }


def _domain_config_payloads(
    profile: DomainProfile,
    generated_dir: Path,
    tracker_config: Path,
    split: str,
    seqmap: Path | None,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    project_root = profile.project_root
    common_model = copy.deepcopy(profile.model)
    detector = copy.deepcopy(profile.detector)
    tracker = {
        "name": profile.default_tracker_name,
        "config": _relative(tracker_config, project_root),
    }
    namespace = profile.namespace
    video_config = {
        "model": common_model,
        "detector": detector,
        "tracker": tracker,
        "source": {"path": f"data/samples/{namespace}.mp4", "type": "video"},
        "output": {
            "video": f"outputs/videos/{namespace}/tracked.mp4",
            "mot": f"outputs/tracks/{namespace}/tracked.txt",
            "metadata": f"outputs/tracks/{namespace}/tracked.metadata.json",
            "render_video": True,
            "save_mot": True,
        },
        "render": _render_defaults(profile),
        "runtime": _runtime_defaults(profile),
    }
    payloads: dict[str, tuple[Path, dict[str, Any]]] = {
        "track_video": (generated_dir / "track_video.yaml", video_config)
    }
    if profile.dataset is None:
        return payloads

    mot_root = _resolve_path(profile.dataset.get("mot_root"), project_root, "dataset.mot_root")
    dataset = {
        "mot_root": _relative(mot_root, project_root),
        "split": split,
        "seqmap": _optional_relative(seqmap, project_root),
    }
    cache_root = f"outputs/detections/cache/{namespace}"
    tracking_dataset = {
        "model": common_model,
        "detector": detector,
        "tracker": tracker,
        "dataset": dataset,
        "output": {
            "tracks_dir": f"outputs/tracks/{namespace}/{profile.default_tracker_name}",
            "videos_dir": f"outputs/videos/{namespace}/{profile.default_tracker_name}",
            "metrics_dir": f"outputs/metrics/{namespace}",
            "render_video": False,
            "save_mot": True,
        },
        "render": _render_defaults(profile, enabled=False),
        "runtime": _runtime_defaults(profile),
    }
    detection_cache = {
        "model": common_model,
        "dataset": {
            "name": profile.name,
            "mot_root": dataset["mot_root"],
            "split": split,
            "seqmap": dataset["seqmap"],
        },
        "inference": {
            "imgsz": detector["imgsz"],
            "conf_floor": profile.detector.get("conf_floor", 0.001),
            "iou": detector["iou"],
            "max_det": detector["max_det"],
            "device": detector.get("device", "auto"),
            "half": detector.get("half", False),
            "batch": profile.detector.get("batch", 1),
            "class_ids": detector.get("class_ids"),
            "target_class_id": detector.get("target_class_id", 0),
            "target_class_name": detector["target_class_name"],
            "preserve_source_classes": detector.get("preserve_source_classes", False),
            "source_class_names": detector.get("source_class_names", {}),
        },
        "cache": {
            "root": cache_root,
            "format": "jsonl",
            "save_npz": False,
            "include_empty_frames": True,
            "overwrite": False,
            "validate_after_write": True,
        },
        "runtime": _runtime_defaults(profile, warmup_iterations=3),
    }
    compare = {
        "experiment": {
            "name": f"{namespace}_{profile.default_tracker_name}",
            "seed": 42,
            "split": split,
        },
        "dataset": dataset,
        "detections": {
            "cache_config": _relative(generated_dir / "detection_cache.yaml", project_root),
            "cache_root": cache_root,
            "confidence_threshold": detector.get("conf", 0.10),
        },
        "trackers": [
            {
                "name": profile.default_tracker_name,
                "config": _relative(tracker_config, project_root),
            }
        ],
        "evaluation": {
            "trackeval_config": "configs/trackeval.yaml",
            "metrics": ["HOTA", "CLEAR", "Identity"],
            "allow_partial_sequences": False,
        },
        "benchmark": {
            "render_video": False,
            "measure_tracker_only": True,
            "measure_end_to_end_from_cache": True,
            "warmup_sequences": 0,
        },
        "output": {
            "root": f"outputs/experiments/{namespace}_{profile.default_tracker_name}",
            "tracks_root": f"outputs/tracks/{namespace}_comparison",
            "metrics_root": (
                f"outputs/metrics/experiments/{namespace}_{profile.default_tracker_name}"
            ),
            "figures_root": (
                f"outputs/figures/experiments/{namespace}_{profile.default_tracker_name}"
            ),
            "videos_root": f"outputs/videos/comparison/{namespace}_{profile.default_tracker_name}",
        },
        "runtime": _runtime_defaults(profile),
    }
    payloads.update(
        {
            "track_dataset": (generated_dir / "track_dataset.yaml", tracking_dataset),
            "detection_cache": (generated_dir / "detection_cache.yaml", detection_cache),
            "compare_trackers": (generated_dir / "compare_trackers.yaml", compare),
        }
    )
    return payloads


def _runtime_defaults(
    profile: DomainProfile,
    *,
    warmup_iterations: int | None = None,
) -> dict[str, Any]:
    runtime = {
        "max_sequences": None,
        "max_frames_per_sequence": None,
        "overwrite": False,
        "show_window": False,
        "fail_fast": True,
        "smoke_only": False,
        "log_level": "INFO",
    }
    runtime.update(profile.output.get("runtime", {}) or {})
    if warmup_iterations is not None:
        runtime["warmup_iterations"] = warmup_iterations
    return runtime


def _render_defaults(profile: DomainProfile, *, enabled: bool | None = None) -> dict[str, Any]:
    render = {
        "enabled": True,
        "show_confidence": False,
        "show_class": False,
        "show_track_id": True,
        "show_trajectory": True,
        "trajectory_length": 20,
        "line_thickness": 2,
        "font_scale": 0.55,
        "show_fps": True,
    }
    render.update(profile.render)
    if enabled is not None:
        render["enabled"] = enabled
    return render


def _write_manifest(
    profile: DomainProfile,
    generated_dir: Path,
    paths: dict[str, Path],
    tracker_config: Path,
    split: str,
    overwrite: bool,
) -> None:
    manifest_path = generated_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise DomainConfigError(f"Generated manifest exists and overwrite=false: {manifest_path}")
    payload = {
        "domain": profile.name,
        "namespace": profile.namespace,
        "profile": _relative(profile.profile_path, profile.project_root),
        "tracker_config": _relative(tracker_config, profile.project_root),
        "split": split,
        "configs": {name: _relative(path, profile.project_root) for name, path in paths.items()},
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _domain_commands(
    profile: DomainProfile,
    paths: dict[str, Path],
    split: str,
) -> dict[str, str]:
    py = r".\.venv\Scripts\python.exe -m football_tracking.cli"
    commands = {
        "track_video": (
            f"{py} track-video --config {_relative(paths['track_video'], profile.project_root)} "
            "--source F:\\videos\\1.mp4 --output-video F:\\videos\\1_tracking.mp4 --overwrite"
        )
    }
    if "detection_cache" in paths:
        commands["cache_detections"] = (
            f"{py} cache-detections --config "
            f"{_relative(paths['detection_cache'], profile.project_root)} --overwrite"
        )
        commands["compare_trackers"] = (
            f"$env:FOOTBALL_TRACKING_PROGRESS=\"1\"; {py} compare-trackers --config "
            f"{_relative(paths['compare_trackers'], profile.project_root)} --overwrite --debug"
        )
        commands["evaluate_tracking"] = (
            f"{py} evaluate-tracking --config "
            f"{_relative(paths['compare_trackers'], profile.project_root)} --split {split}"
        )
    return commands
