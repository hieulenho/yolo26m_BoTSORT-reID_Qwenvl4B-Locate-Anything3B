"""Adapters for Ultralytics tracker implementations such as BoT-SORT and ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from football_tracking.data.bbox import clip_xyxy_to_image, is_valid_bbox
from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.tracking.schemas import TrackerDetection, TrackOutput


class UltralyticsTrackerConfigError(RuntimeError):
    """Raised when an Ultralytics tracker config is invalid."""


@dataclass(frozen=True)
class UltralyticsTrackerRuntimeConfig:
    tracker_type: str
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    track_buffer: int
    match_thresh: float
    fuse_score: bool
    gmc_method: str
    proximity_thresh: float
    appearance_thresh: float
    with_reid: bool
    model: str
    output_confirmed_only: bool
    require_recent_update: bool
    max_time_since_update_for_output: int
    min_hits_for_output: int
    compact_ids: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def to_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            tracker_type=self.tracker_type,
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            fuse_score=self.fuse_score,
            gmc_method=self.gmc_method,
            proximity_thresh=self.proximity_thresh,
            appearance_thresh=self.appearance_thresh,
            with_reid=self.with_reid,
            model=self.model,
        )


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UltralyticsTrackerConfigError(f"{section} must be a mapping.")
    return value


def load_ultralytics_tracker_config(
    config_path: str | Path,
    default_tracker_type: str = "botsort",
) -> UltralyticsTrackerRuntimeConfig:
    project_root = get_project_root()
    path = Path(config_path)
    resolved = path.resolve() if path.is_absolute() else resolve_project_path(path, project_root)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    root = _mapping(raw, "Ultralytics tracker config root")
    tracker = _mapping(root.get("tracker"), "tracker")
    output = _mapping(root.get("output", {}), "output")
    tracker_type = str(
        tracker.get("tracker_type", tracker.get("type", default_tracker_type))
    ).lower()
    if tracker_type == "botsort_reid":
        tracker_type = "botsort"
    with_reid = bool(tracker.get("with_reid", tracker.get("use_reid", False)))
    reid_model = str(tracker.get("model", "auto"))
    if with_reid and reid_model == "auto":
        # The Ultralytics "auto" ReID path expects predictor-side feature hooks.
        # This adapter calls tracker.update directly, so use a real embedder model.
        reid_model = "yolo26n-cls.pt"
    config = UltralyticsTrackerRuntimeConfig(
        tracker_type=tracker_type,
        track_high_thresh=float(tracker.get("track_high_thresh", 0.25)),
        track_low_thresh=float(tracker.get("track_low_thresh", 0.1)),
        new_track_thresh=float(tracker.get("new_track_thresh", 0.25)),
        track_buffer=int(tracker.get("track_buffer", tracker.get("max_age", 30))),
        match_thresh=float(tracker.get("match_thresh", 0.8)),
        fuse_score=bool(tracker.get("fuse_score", True)),
        gmc_method=str(tracker.get("gmc_method", "sparseOptFlow")),
        proximity_thresh=float(tracker.get("proximity_thresh", 0.5)),
        appearance_thresh=float(tracker.get("appearance_thresh", 0.25)),
        with_reid=with_reid,
        model=reid_model,
        output_confirmed_only=bool(output.get("confirmed_only", True)),
        require_recent_update=bool(output.get("require_recent_update", True)),
        max_time_since_update_for_output=int(
            output.get("max_time_since_update_for_output", 0)
        ),
        min_hits_for_output=int(output.get("min_hits_for_output", 1)),
        compact_ids=bool(output.get("compact_ids", False)),
    )
    validate_ultralytics_tracker_config(config)
    return config


def validate_ultralytics_tracker_config(config: UltralyticsTrackerRuntimeConfig) -> None:
    if config.tracker_type not in {"botsort", "bytetrack"}:
        raise UltralyticsTrackerConfigError(
            f"Unsupported Ultralytics tracker_type: {config.tracker_type}"
        )
    for field_name in (
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "match_thresh",
        "proximity_thresh",
        "appearance_thresh",
    ):
        value = float(getattr(config, field_name))
        if not 0.0 <= value <= 1.0:
            raise UltralyticsTrackerConfigError(f"{field_name} must be in [0, 1].")
    if config.track_buffer < 1:
        raise UltralyticsTrackerConfigError("track_buffer must be >= 1.")
    if config.max_time_since_update_for_output < 0:
        raise UltralyticsTrackerConfigError(
            "max_time_since_update_for_output must be >= 0."
        )
    if config.min_hits_for_output < 1:
        raise UltralyticsTrackerConfigError("min_hits_for_output must be >= 1.")


class _TrackerResults:
    """Minimal results-like object accepted by Ultralytics trackers."""

    def __init__(self, detections: list[TrackerDetection]) -> None:
        self._detections = list(detections)
        self.xyxy = np.asarray(
            [
                [
                    det.bbox_xyxy.x1,
                    det.bbox_xyxy.y1,
                    det.bbox_xyxy.x2,
                    det.bbox_xyxy.y2,
                ]
                for det in self._detections
            ],
            dtype=np.float32,
        ).reshape((-1, 4))
        self.xywh = np.asarray(
            [
                [
                    det.bbox_ltwh.x + det.bbox_ltwh.width / 2.0,
                    det.bbox_ltwh.y + det.bbox_ltwh.height / 2.0,
                    det.bbox_ltwh.width,
                    det.bbox_ltwh.height,
                ]
                for det in self._detections
            ],
            dtype=np.float32,
        ).reshape((-1, 4))
        self.conf = np.asarray([det.confidence for det in self._detections], dtype=np.float32)
        self.cls = np.asarray([det.class_id for det in self._detections], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._detections)

    def __getitem__(self, item: Any) -> _TrackerResults:
        if isinstance(item, np.ndarray) and item.dtype == bool:
            selected = [
                det
                for det, keep in zip(self._detections, item.tolist(), strict=True)
                if keep
            ]
        elif isinstance(item, list | tuple):
            selected = [self._detections[int(index)] for index in item]
        elif isinstance(item, slice):
            selected = self._detections[item]
        else:
            selected = [self._detections[int(item)]]
        return _TrackerResults(selected)


class UltralyticsTrackerAdapter:
    """Project tracker adapter around Ultralytics BoT-SORT/ByteTrack classes."""

    def __init__(
        self,
        config: UltralyticsTrackerRuntimeConfig,
        tracker_factory: Any | None = None,
    ) -> None:
        self.config = config
        self.tracker_factory = tracker_factory
        self.tracker: Any | None = None
        self.initialization_count = 0
        self._output_id_map: dict[int, int] = {}
        self._next_output_id = 1

    def initialize(self) -> Any:
        if self.tracker is not None:
            return self.tracker
        if self.tracker_factory is None:
            try:
                if self.config.tracker_type == "botsort":
                    from ultralytics.trackers.bot_sort import (  # type: ignore[import-not-found]
                        BOTSORT,
                    )

                    self.tracker_factory = BOTSORT
                else:
                    from ultralytics.trackers.byte_tracker import (  # type: ignore[import-not-found]
                        BYTETracker,
                    )

                    self.tracker_factory = BYTETracker
            except Exception as exc:  # noqa: BLE001
                raise UltralyticsTrackerConfigError(
                    "Could not import Ultralytics tracker dependencies. "
                    "Install the tracker requirements used by Ultralytics, then retry."
                ) from exc
        self.tracker = self.tracker_factory(self.config.to_namespace())
        self.initialization_count += 1
        return self.tracker

    def reset(self) -> None:
        tracker = self.initialize()
        if hasattr(tracker, "reset"):
            tracker.reset()
        self._output_id_map.clear()
        self._next_output_id = 1

    def close(self) -> None:
        self.tracker = None

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
        tracker = self.initialize()
        results = _TrackerResults(detections)
        raw_tracks = tracker.update(results, frame)
        track_info = self._track_info_by_id(tracker)
        outputs = [
            output
            for row in np.asarray(raw_tracks).tolist()
            if (
                output := self._to_track_output(
                    row,
                    frame_index,
                    sequence_name,
                    detections,
                    image_width,
                    image_height,
                    track_info.get(int(row[4])) if len(row) >= 5 else None,
                )
            )
            is not None
        ]
        return sorted(outputs, key=lambda item: item.track_id)

    @staticmethod
    def _track_info_by_id(tracker: Any) -> dict[int, Any]:
        tracks = getattr(tracker, "tracked_stracks", None)
        if not tracks:
            return {}
        return {
            int(track.track_id): track
            for track in tracks
            if getattr(track, "track_id", None) is not None
        }

    def _visible_track_id(self, raw_track_id: int) -> int:
        if not self.config.compact_ids:
            return raw_track_id
        if raw_track_id not in self._output_id_map:
            self._output_id_map[raw_track_id] = self._next_output_id
            self._next_output_id += 1
        return self._output_id_map[raw_track_id]

    def _should_emit_track(self, track: Any | None) -> bool:
        if track is None:
            return True
        if self.config.output_confirmed_only and not bool(
            getattr(track, "is_activated", True)
        ):
            return False
        if self.config.require_recent_update:
            tracker_frame = int(getattr(self.tracker, "frame_id", 0) or 0)
            track_frame = int(getattr(track, "frame_id", tracker_frame) or tracker_frame)
            if (
                tracker_frame - track_frame
                > self.config.max_time_since_update_for_output
            ):
                return False
        hits = int(getattr(track, "tracklet_len", 0) or 0) + 1
        return hits >= self.config.min_hits_for_output

    def _to_track_output(
        self,
        row: list[float],
        frame_index: int,
        sequence_name: str,
        detections: list[TrackerDetection],
        image_width: int | None,
        image_height: int | None,
        track: Any | None,
    ) -> TrackOutput | None:
        if len(row) < 7:
            return None
        x1, y1, x2, y2 = [float(value) for value in row[:4]]
        raw_track_id = int(row[4])
        if not self._should_emit_track(track):
            return None
        track_id = self._visible_track_id(raw_track_id)
        confidence = float(row[5]) if 0.0 <= float(row[5]) <= 1.0 else None
        class_id = int(row[6])
        class_name = f"class_{class_id}"
        if len(row) >= 8:
            detection_index = int(row[-1])
            if 0 <= detection_index < len(detections):
                class_name = detections[detection_index].class_name
        box = BoundingBoxXYXY(x1, y1, x2, y2)
        if image_width is not None and image_height is not None:
            box = clip_xyxy_to_image(box, image_width, image_height)
        if track_id <= 0 or not is_valid_bbox(box):
            return None
        return TrackOutput.from_xyxy(
            frame_index=frame_index,
            sequence_name=sequence_name,
            track_id=track_id,
            bbox_xyxy=box,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            confirmed=True,
            time_since_update=0,
            metadata={
                "tracker": self.config.tracker_type,
                "bbox_source": "ultralytics_tracker",
                "with_reid": self.config.with_reid,
                "raw_track_id": raw_track_id,
            },
        )
