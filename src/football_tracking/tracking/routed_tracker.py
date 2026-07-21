"""Motion-regime tracker routing with global IDs and stable class labels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.tracking.schemas import TrackerDetection, TrackOutput


class RoutedTrackerConfigError(RuntimeError):
    """Raised when an adaptive tracker routing config is invalid."""


@dataclass(frozen=True)
class TrackerRouteSpec:
    route_name: str
    tracker_name: str
    config_path: Path
    class_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_name": self.route_name,
            "tracker_name": self.tracker_name,
            "config_path": str(self.config_path),
            "class_ids": list(self.class_ids),
        }


@dataclass(frozen=True)
class RoutedTrackerRuntimeConfig:
    default: TrackerRouteSpec
    routes: tuple[TrackerRouteSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker_type": "adaptive_routed",
            "default": self.default.to_dict(),
            "routes": [route.to_dict() for route in self.routes],
        }


def _resolve_nested_config(value: Any, project_root: Path, section: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise RoutedTrackerConfigError(f"{section}.config must be a path string.")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    if not resolved.is_file():
        raise RoutedTrackerConfigError(
            f"{section}.config does not exist: {resolved}"
        )
    return resolved


def _parse_spec(
    value: Any,
    *,
    project_root: Path,
    section: str,
    default_route_name: str | None = None,
) -> TrackerRouteSpec:
    if not isinstance(value, dict):
        raise RoutedTrackerConfigError(f"{section} must be a mapping.")
    tracker_name = str(value.get("name", "")).strip().lower()
    if not tracker_name:
        raise RoutedTrackerConfigError(f"{section}.name must not be empty.")
    if tracker_name in {"adaptive_routed", "routed", "adaptive-routed"}:
        raise RoutedTrackerConfigError("Nested adaptive tracker routing is not supported.")
    route_name = str(
        value.get("route_name", default_route_name or tracker_name)
    ).strip()
    if not route_name:
        raise RoutedTrackerConfigError(f"{section}.route_name must not be empty.")
    raw_class_ids = value.get("class_ids", [])
    if not isinstance(raw_class_ids, list | tuple):
        raise RoutedTrackerConfigError(f"{section}.class_ids must be a list.")
    class_ids = tuple(sorted({int(class_id) for class_id in raw_class_ids}))
    if any(class_id < 0 for class_id in class_ids):
        raise RoutedTrackerConfigError(f"{section}.class_ids must be non-negative.")
    return TrackerRouteSpec(
        route_name=route_name,
        tracker_name=tracker_name,
        config_path=_resolve_nested_config(value.get("config"), project_root, section),
        class_ids=class_ids,
    )


def load_routed_tracker_config(
    config_path: str | Path,
) -> RoutedTrackerRuntimeConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    if not resolved.is_file():
        raise RoutedTrackerConfigError(f"Tracker routing config does not exist: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("tracker"), dict):
        raise RoutedTrackerConfigError("Tracker routing config requires a tracker mapping.")
    tracker = raw["tracker"]
    default = _parse_spec(
        tracker.get("default"),
        project_root=project_root,
        section="tracker.default",
        default_route_name="default",
    )
    raw_routes = tracker.get("routes", [])
    if not isinstance(raw_routes, list):
        raise RoutedTrackerConfigError("tracker.routes must be a list.")
    routes = tuple(
        _parse_spec(
            item,
            project_root=project_root,
            section=f"tracker.routes[{index}]",
        )
        for index, item in enumerate(raw_routes)
    )
    route_names = [route.route_name for route in routes]
    if len(route_names) != len(set(route_names)):
        raise RoutedTrackerConfigError("tracker.routes route_name values must be unique.")
    if default.route_name in route_names:
        raise RoutedTrackerConfigError(
            "tracker.default.route_name must not be reused by tracker.routes."
        )
    assigned: dict[int, str] = {}
    for route in routes:
        for class_id in route.class_ids:
            previous = assigned.get(class_id)
            if previous is not None:
                raise RoutedTrackerConfigError(
                    f"Class {class_id} is assigned to both {previous} and {route.route_name}."
                )
            assigned[class_id] = route.route_name
    return RoutedTrackerRuntimeConfig(default=default, routes=routes)


class RoutedTrackerAdapter:
    """Delegate class groups to trackers while exposing one global ID namespace."""

    _CLASS_SWITCH_MIN_OBSERVATIONS = 6
    _CLASS_SWITCH_SCORE_RATIO = 1.60
    _CLASS_SWITCH_SCORE_MARGIN = 1.25

    def __init__(
        self,
        config: RoutedTrackerRuntimeConfig,
        *,
        device: str = "auto",
        tracker_factory: Callable[[str, Path, str], Any] | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self._tracker_factory = tracker_factory
        self._adapters: dict[str, Any] = {}
        self._id_map: dict[tuple[str, int], int] = {}
        self._next_id = 1
        self._class_scores: dict[int, dict[int, float]] = {}
        self._class_names: dict[int, dict[int, str]] = {}
        self._stable_class: dict[int, int] = {}
        self._last_raw_class: dict[int, int] = {}
        self._class_observations: dict[int, int] = {}
        self._raw_class_switches = 0
        self._stable_class_switches = 0
        self._suppressed_class_switches = 0
        self._scene_reset_count = 0
        self._spec_by_name = {
            route.route_name: route for route in (config.default, *config.routes)
        }
        self._route_by_class = {
            class_id: route.route_name
            for route in config.routes
            for class_id in route.class_ids
        }

    def _create_delegate(self, spec: TrackerRouteSpec) -> Any:
        if self._tracker_factory is not None:
            return self._tracker_factory(
                spec.tracker_name,
                spec.config_path,
                self.device,
            )
        from football_tracking.tracking.tracker_factory import create_tracker

        return create_tracker(spec.tracker_name, spec.config_path, device=self.device)

    def _delegate(self, route_name: str) -> Any:
        delegate = self._adapters.get(route_name)
        if delegate is None:
            delegate = self._create_delegate(self._spec_by_name[route_name])
            delegate.reset()
            self._adapters[route_name] = delegate
        return delegate

    def reset(self) -> None:
        for delegate in self._adapters.values():
            delegate.reset()
        self._id_map.clear()
        self._next_id = 1
        self._reset_class_memory()
        self._scene_reset_count = 0

    def reset_scene(self) -> None:
        """Reset motion state at a hard cut while preserving globally unique IDs."""
        for delegate in self._adapters.values():
            if hasattr(delegate, "reset_scene"):
                delegate.reset_scene()
            else:
                delegate.reset()
        self._id_map.clear()
        self._reset_class_memory(reset_counters=False)
        self._scene_reset_count += 1

    def close(self) -> None:
        for delegate in self._adapters.values():
            delegate.close()
        self._adapters.clear()
        self._id_map.clear()
        self._next_id = 1
        self._reset_class_memory()

    def _reset_class_memory(self, *, reset_counters: bool = True) -> None:
        self._class_scores.clear()
        self._class_names.clear()
        self._stable_class.clear()
        self._last_raw_class.clear()
        self._class_observations.clear()
        if reset_counters:
            self._raw_class_switches = 0
            self._stable_class_switches = 0
            self._suppressed_class_switches = 0

    def get_runtime_config(self) -> dict[str, Any]:
        payload = self.config.to_dict()
        payload["active_routes"] = sorted(self._adapters)
        return payload

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "class_stabilization_enabled": True,
            "tracked_identity_count": len(self._class_observations),
            "raw_class_switches": self._raw_class_switches,
            "stable_class_switches": self._stable_class_switches,
            "suppressed_class_switches": self._suppressed_class_switches,
            "suppressed_class_mismatch_frames": self._suppressed_class_switches,
            "scene_reset_count": self._scene_reset_count,
        }

    def _stabilize_class(self, global_id: int, output: TrackOutput) -> TrackOutput:
        raw_class = output.class_id
        previous_raw = self._last_raw_class.get(global_id)
        if previous_raw is not None and previous_raw != raw_class:
            self._raw_class_switches += 1
        self._last_raw_class[global_id] = raw_class

        scores = self._class_scores.setdefault(global_id, {})
        for class_id in tuple(scores):
            scores[class_id] *= 0.97
            if scores[class_id] < 1e-6:
                del scores[class_id]
        weight = max(float(output.confidence or 0.0), 0.25)
        scores[raw_class] = scores.get(raw_class, 0.0) + weight
        self._class_names.setdefault(global_id, {})[raw_class] = output.class_name
        observations = self._class_observations.get(global_id, 0) + 1
        self._class_observations[global_id] = observations

        incumbent = self._stable_class.setdefault(global_id, raw_class)
        challenger = max(scores, key=lambda class_id: (scores[class_id], -class_id))
        incumbent_score = scores.get(incumbent, 0.0)
        challenger_score = scores[challenger]
        stable_switched = False
        if challenger != incumbent:
            enough_evidence = (
                observations >= self._CLASS_SWITCH_MIN_OBSERVATIONS
                and challenger_score
                >= incumbent_score * self._CLASS_SWITCH_SCORE_RATIO
                and challenger_score - incumbent_score
                >= self._CLASS_SWITCH_SCORE_MARGIN
            )
            if enough_evidence:
                self._stable_class[global_id] = challenger
                incumbent = challenger
                self._stable_class_switches += 1
                stable_switched = True
        if raw_class != incumbent and not stable_switched:
            self._suppressed_class_switches += 1

        total_score = sum(scores.values())
        stable_confidence = scores.get(incumbent, 0.0) / max(total_score, 1e-9)
        class_name = self._class_names[global_id].get(incumbent, output.class_name)
        return replace(
            output,
            class_id=incumbent,
            class_name=class_name,
            metadata={
                **output.metadata,
                "raw_class_id": raw_class,
                "raw_class_name": output.class_name,
                "stable_class_id": incumbent,
                "stable_class_name": class_name,
                "stable_class_confidence": round(stable_confidence, 6),
                "class_observations": observations,
            },
        )

    def _global_track_id(self, route_name: str, delegate_id: int) -> int:
        key = (route_name, delegate_id)
        if key not in self._id_map:
            self._id_map[key] = self._next_id
            self._next_id += 1
        return self._id_map[key]

    def update(
        self,
        frame_index: int,
        sequence_name: str,
        detections: list[TrackerDetection],
        frame: Any | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[TrackOutput]:
        grouped: dict[str, list[TrackerDetection]] = {}
        for detection in detections:
            route_name = self._route_by_class.get(
                detection.class_id,
                self.config.default.route_name,
            )
            grouped.setdefault(route_name, []).append(detection)

        route_names = sorted(set(self._adapters) | set(grouped))
        outputs: list[TrackOutput] = []
        for route_name in route_names:
            delegate = self._delegate(route_name)
            delegated_outputs = delegate.update(
                frame_index=frame_index,
                sequence_name=sequence_name,
                detections=grouped.get(route_name, []),
                frame=frame,
                image_width=image_width,
                image_height=image_height,
            )
            for output in delegated_outputs:
                global_id = self._global_track_id(route_name, output.track_id)
                stabilized = self._stabilize_class(global_id, output)
                outputs.append(
                    replace(
                        stabilized,
                        track_id=global_id,
                        metadata={
                            **stabilized.metadata,
                            "routed_tracker": True,
                            "tracker_route": route_name,
                            "delegate_track_id": output.track_id,
                        },
                    )
                )
        return sorted(outputs, key=lambda item: item.track_id)
