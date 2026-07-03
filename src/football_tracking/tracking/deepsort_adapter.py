"""Adapter between project detections and deep-sort-realtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_tracking.data.bbox import clip_xyxy_to_image, is_valid_bbox
from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.tracking.lifecycle import should_emit_track, track_lifecycle_state
from football_tracking.tracking.schemas import TrackerDetection, TrackOutput


class DeepSortConfigError(RuntimeError):
    """Raised when DeepSORT config is invalid."""


@dataclass(frozen=True)
class DeepSortRuntimeConfig:
    max_age: int
    n_init: int
    max_iou_distance: float
    max_cosine_distance: float
    nn_budget: int | None
    embedder: str
    half: bool
    bgr: bool
    embedder_gpu: bool
    polygon: bool
    today: str | None
    only_position: bool
    use_appearance: bool
    confirmed_only: bool
    require_recent_update: bool
    max_time_since_update_for_output: int
    use_original_detection_box: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeepSortConfigError(f"{section} must be a mapping.")
    return value


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def load_deepsort_config(config_path: str | Path, device: str = "auto") -> DeepSortRuntimeConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = _mapping(raw, "DeepSORT config root")
    tracker = _mapping(root.get("tracker"), "tracker")
    association = _mapping(root.get("association", {}), "association")
    output = _mapping(root.get("output", {}), "output")

    embedder_gpu = bool(tracker.get("embedder_gpu", True))
    half = bool(tracker.get("half", True))
    device_text = str(device).lower()
    cuda_available = _cuda_available()
    if device_text == "cpu" or (device_text == "auto" and not cuda_available) or not cuda_available:
        embedder_gpu = False
        half = False

    config = DeepSortRuntimeConfig(
        max_age=int(tracker.get("max_age", 30)),
        n_init=int(tracker.get("n_init", 3)),
        max_iou_distance=float(tracker.get("max_iou_distance", 0.7)),
        max_cosine_distance=float(tracker.get("max_cosine_distance", 0.3)),
        nn_budget=(None if tracker.get("nn_budget") is None else int(tracker.get("nn_budget"))),
        embedder=str(tracker.get("embedder", "mobilenet")),
        half=half,
        bgr=bool(tracker.get("bgr", True)),
        embedder_gpu=embedder_gpu,
        polygon=bool(tracker.get("polygon", False)),
        today=tracker.get("today"),
        only_position=bool(association.get("only_position", False)),
        use_appearance=bool(association.get("use_appearance", True)),
        confirmed_only=bool(output.get("confirmed_only", True)),
        require_recent_update=bool(output.get("require_recent_update", True)),
        max_time_since_update_for_output=int(output.get("max_time_since_update_for_output", 1)),
        use_original_detection_box=bool(output.get("use_original_detection_box", True)),
    )
    validate_deepsort_config(config)
    return config


def validate_deepsort_config(config: DeepSortRuntimeConfig) -> None:
    if config.max_age < 1:
        raise DeepSortConfigError("tracker.max_age must be >= 1.")
    if config.n_init < 1:
        raise DeepSortConfigError("tracker.n_init must be >= 1.")
    if not 0.0 <= config.max_iou_distance <= 1.0:
        raise DeepSortConfigError("tracker.max_iou_distance must be in [0, 1].")
    if config.max_cosine_distance < 0.0:
        raise DeepSortConfigError("tracker.max_cosine_distance must be >= 0.")
    if config.nn_budget is not None and config.nn_budget <= 0:
        raise DeepSortConfigError("tracker.nn_budget must be null or positive.")
    if config.max_time_since_update_for_output < 0:
        raise DeepSortConfigError("output.max_time_since_update_for_output must be >= 0.")


def detections_to_deepsort_input(
    detections: list[TrackerDetection],
) -> list[tuple[list[float], float, str]]:
    raw: list[tuple[list[float], float, str]] = []
    for detection in detections:
        box = detection.bbox_ltwh
        if box.width <= 0 or box.height <= 0:
            continue
        if not 0.0 <= detection.confidence <= 1.0:
            continue
        raw.append(
            (
                [float(box.x), float(box.y), float(box.width), float(box.height)],
                float(detection.confidence),
                detection.class_name,
            )
        )
    return raw


class DeepSortTrackerAdapter:
    """Small wrapper around deep-sort-realtime with deterministic output filtering."""

    def __init__(
        self,
        config: DeepSortRuntimeConfig,
        tracker_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.tracker_factory = tracker_factory
        self.tracker: Any | None = None
        self.initialization_count = 0

    def initialize(self) -> Any:
        if self.tracker is not None:
            return self.tracker
        if self.tracker_factory is None:
            from deep_sort_realtime.deepsort_tracker import (
                DeepSort,  # type: ignore[import-not-found]
            )

            self.tracker_factory = DeepSort
        self.tracker = self.tracker_factory(
            max_iou_distance=self.config.max_iou_distance,
            max_age=self.config.max_age,
            n_init=self.config.n_init,
            max_cosine_distance=self.config.max_cosine_distance,
            nn_budget=self.config.nn_budget,
            gating_only_position=self.config.only_position,
            embedder=(self.config.embedder if self.config.use_appearance else None),
            half=self.config.half,
            bgr=self.config.bgr,
            embedder_gpu=self.config.embedder_gpu,
            polygon=self.config.polygon,
            today=self.config.today,
        )
        self.initialization_count += 1
        return self.tracker

    def reset(self) -> None:
        self.close()
        self.initialize()

    def close(self) -> None:
        self.tracker = None

    def get_runtime_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def update(
        self,
        frame_index: int,
        sequence_name: str,
        detections: list[TrackerDetection],
        frame: Any,
        image_width: int,
        image_height: int,
    ) -> list[TrackOutput]:
        tracker = self.initialize()
        raw_detections = detections_to_deepsort_input(detections)
        raw_tracks = tracker.update_tracks(raw_detections, frame=frame)
        outputs: list[TrackOutput] = []
        for raw_track in raw_tracks:
            output = self._to_track_output(
                raw_track,
                frame_index,
                sequence_name,
                image_width,
                image_height,
            )
            if output is not None:
                outputs.append(output)
        return sorted(outputs, key=lambda item: item.track_id)

    def _to_track_output(
        self,
        raw_track: Any,
        frame_index: int,
        sequence_name: str,
        image_width: int,
        image_height: int,
    ) -> TrackOutput | None:
        time_since_update = int(getattr(raw_track, "time_since_update", 0))
        state = track_lifecycle_state(raw_track)
        if not should_emit_track(
            state,
            self.config.confirmed_only,
            self.config.require_recent_update,
            time_since_update,
            self.config.max_time_since_update_for_output,
        ):
            return None
        bbox, bbox_source = self._track_bbox(raw_track)
        if bbox is None:
            return None
        clipped = clip_xyxy_to_image(bbox, image_width, image_height)
        if not is_valid_bbox(clipped):
            return None
        track_id = int(raw_track.track_id)
        confidence = _maybe_call(raw_track, "get_det_conf")
        if confidence is not None:
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                confidence = None
        class_name = _maybe_call(raw_track, "get_det_class") or "player"
        hits = getattr(raw_track, "hits", None)
        return TrackOutput.from_xyxy(
            frame_index=frame_index,
            sequence_name=sequence_name,
            track_id=track_id,
            bbox_xyxy=clipped,
            confidence=confidence,
            class_id=0,
            class_name=str(class_name) if class_name else "player",
            confirmed=state == "confirmed",
            time_since_update=time_since_update,
            hits=int(hits) if hits is not None else None,
            metadata={"bbox_source": bbox_source, "lifecycle_state": state},
        )

    def _track_bbox(self, raw_track: Any) -> tuple[BoundingBoxXYXY | None, str]:
        if self.config.use_original_detection_box and hasattr(raw_track, "to_ltrb"):
            try:
                values = raw_track.to_ltrb(orig=True, orig_strict=True)
                if values is not None:
                    return _box_from_values(values), "original_detection"
            except TypeError:
                pass
        for method_name in ("to_ltrb", "to_tlbr"):
            if not hasattr(raw_track, method_name):
                continue
            values = getattr(raw_track, method_name)()
            if values is not None:
                return _box_from_values(values), "kalman_prediction"
        return None, "unavailable"


def _maybe_call(obj: Any, method_name: str) -> Any:
    if not hasattr(obj, method_name):
        return None
    return getattr(obj, method_name)()


def _box_from_values(values: Any) -> BoundingBoxXYXY | None:
    try:
        x1, y1, x2, y2 = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    return BoundingBoxXYXY(x1, y1, x2, y2)
