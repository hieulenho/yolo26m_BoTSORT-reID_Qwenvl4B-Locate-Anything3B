"""SORT tracker adapter compatible with the DeepSORT adapter interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from football_tracking.data.bbox import clip_xyxy_to_image, is_valid_bbox
from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.tracking.assignment import associate_detections_to_tracks
from football_tracking.tracking.schemas import TrackerDetection, TrackOutput
from football_tracking.tracking.sort_kalman import SortKalmanFilter


class SortConfigError(RuntimeError):
    """Raised when SORT config is invalid."""


@dataclass(frozen=True)
class SortRuntimeConfig:
    max_age: int
    min_hits: int
    iou_threshold: float
    confirmed_only: bool
    require_recent_update: bool
    max_time_since_update_for_output: int
    output_predicted_tracks_without_detection: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SortConfigError(f"{section} must be a mapping.")
    return value


def load_sort_config(config_path: str | Path) -> SortRuntimeConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = _mapping(raw, "SORT config root")
    tracker = _mapping(root.get("tracker"), "tracker")
    output = _mapping(root.get("output", {}), "output")
    config = SortRuntimeConfig(
        max_age=int(tracker.get("max_age", 30)),
        min_hits=int(tracker.get("min_hits", 3)),
        iou_threshold=float(tracker.get("iou_threshold", 0.3)),
        confirmed_only=bool(output.get("confirmed_only", True)),
        require_recent_update=bool(output.get("require_recent_update", True)),
        max_time_since_update_for_output=int(output.get("max_time_since_update_for_output", 1)),
        output_predicted_tracks_without_detection=bool(
            output.get("output_predicted_tracks_without_detection", False)
        ),
    )
    validate_sort_config(config)
    return config


def validate_sort_config(config: SortRuntimeConfig) -> None:
    if config.max_age < 1:
        raise SortConfigError("tracker.max_age must be >= 1.")
    if config.min_hits < 1:
        raise SortConfigError("tracker.min_hits must be >= 1.")
    if not 0.0 <= config.iou_threshold <= 1.0:
        raise SortConfigError("tracker.iou_threshold must be in [0, 1].")
    if config.max_time_since_update_for_output < 0:
        raise SortConfigError("output.max_time_since_update_for_output must be >= 0.")


@dataclass
class _SortTrack:
    track_id: int
    kalman: SortKalmanFilter
    hits: int
    age: int
    time_since_update: int
    last_confidence: float | None
    class_id: int
    class_name: str
    bbox_source: str

    @classmethod
    def from_detection(cls, track_id: int, detection: TrackerDetection) -> _SortTrack:
        return cls(
            track_id=track_id,
            kalman=SortKalmanFilter.from_bbox(detection.bbox_xyxy),
            hits=1,
            age=1,
            time_since_update=0,
            last_confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            bbox_source="matched_detection",
        )

    def predict(self) -> BoundingBoxXYXY:
        box = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        self.bbox_source = "kalman_prediction"
        return box

    def update(self, detection: TrackerDetection) -> None:
        self.kalman.update(detection.bbox_xyxy)
        self.hits += 1
        self.time_since_update = 0
        self.last_confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.bbox_source = "matched_detection"

    def is_confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits


class SortTrackerAdapter:
    """Clean-room SORT implementation with Kalman + IoU + Hungarian assignment."""

    def __init__(self, config: SortRuntimeConfig) -> None:
        self.config = config
        self.tracks: list[_SortTrack] = []
        self.next_track_id = 1
        self.initialization_count = 0

    def initialize(self) -> SortTrackerAdapter:
        self.initialization_count += 1
        return self

    def reset(self) -> None:
        self.tracks = []
        self.next_track_id = 1
        self.initialize()

    def close(self) -> None:
        self.tracks = []

    def get_runtime_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def update(
        self,
        frame_index: int,
        sequence_name: str,
        detections: list[TrackerDetection],
        frame: Any | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[TrackOutput]:
        del frame
        predicted_boxes: list[BoundingBoxXYXY] = []
        valid_tracks: list[_SortTrack] = []
        for track in self.tracks:
            try:
                predicted_boxes.append(track.predict())
                valid_tracks.append(track)
            except Exception:  # noqa: BLE001
                continue
        self.tracks = valid_tracks

        detection_boxes = [detection.bbox_xyxy for detection in detections]
        association = associate_detections_to_tracks(
            predicted_boxes,
            detection_boxes,
            self.config.iou_threshold,
        )
        for track_index, detection_index in association.matched_pairs:
            self.tracks[track_index].update(detections[detection_index])

        for detection_index in association.unmatched_detection_indices:
            self.tracks.append(
                _SortTrack.from_detection(self.next_track_id, detections[detection_index])
            )
            self.next_track_id += 1

        self.tracks = [
            track
            for track in self.tracks
            if track.time_since_update <= self.config.max_age
        ]
        outputs = [
            output
            for track in self.tracks
            if (output := self._to_track_output(
                track,
                frame_index,
                sequence_name,
                image_width,
                image_height,
            ))
            is not None
        ]
        return sorted(outputs, key=lambda item: item.track_id)

    def _to_track_output(
        self,
        track: _SortTrack,
        frame_index: int,
        sequence_name: str,
        image_width: int | None,
        image_height: int | None,
    ) -> TrackOutput | None:
        confirmed = track.is_confirmed(self.config.min_hits)
        if self.config.confirmed_only and not confirmed:
            return None
        if (
            self.config.require_recent_update
            and track.time_since_update > self.config.max_time_since_update_for_output
        ):
            return None
        if (
            not self.config.output_predicted_tracks_without_detection
            and track.time_since_update > 0
        ):
            return None
        try:
            box = track.kalman.bbox()
        except Exception:  # noqa: BLE001
            return None
        if image_width is not None and image_height is not None:
            box = clip_xyxy_to_image(box, image_width, image_height)
        if not is_valid_bbox(box):
            return None
        confidence = track.last_confidence if track.time_since_update == 0 else None
        return TrackOutput.from_xyxy(
            frame_index=frame_index,
            sequence_name=sequence_name,
            track_id=track.track_id,
            bbox_xyxy=box,
            confidence=confidence,
            class_id=track.class_id,
            class_name=track.class_name,
            confirmed=confirmed,
            time_since_update=track.time_since_update,
            hits=track.hits,
            metadata={
                "bbox_source": track.bbox_source,
                "lifecycle_state": "confirmed" if confirmed else "tentative",
                "tracker": "sort",
            },
        )
