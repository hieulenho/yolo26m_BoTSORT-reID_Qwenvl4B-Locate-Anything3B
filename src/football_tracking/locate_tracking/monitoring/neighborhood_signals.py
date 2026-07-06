"""Neighbor ambiguity signals from nearby raw tracks."""

from __future__ import annotations

import math

from football_tracking.locate_tracking.artifacts.mot_reader import read_mot_track_file
from football_tracking.locate_tracking.artifacts.track_index import FrameTrackIndex
from football_tracking.locate_tracking.association.geometry import (
    bbox_center,
    bbox_iou,
    center_distance_px,
)
from football_tracking.locate_tracking.monitoring.schemas import (
    MonitoringConfig,
    TargetObservationTimeline,
    UncertaintySignal,
)
from football_tracking.locate_tracking.monitoring.signal_utils import make_signal


def _diagonal(timeline: TargetObservationTimeline) -> float:
    width = timeline.metadata.get("frame_width")
    height = timeline.metadata.get("frame_height")
    if width and height:
        return math.hypot(float(width), float(height))
    return max(
        1.0,
        math.hypot(
            max(
                (item.bbox_xyxy[2] for item in timeline.observations if item.bbox_xyxy),
                default=1.0,
            ),
            max(
                (item.bbox_xyxy[3] for item in timeline.observations if item.bbox_xyxy),
                default=1.0,
            ),
        ),
    )


def detect_neighbor_ambiguity_signals(
    timeline: TargetObservationTimeline,
    config: MonitoringConfig,
) -> tuple[UncertaintySignal, ...]:
    tracks_path = timeline.metadata.get("tracks_path")
    if not tracks_path:
        return ()
    mot_file = read_mot_track_file(str(tracks_path))
    index = FrameTrackIndex.from_observations(mot_file.observations)
    diagonal = _diagonal(timeline)
    signals: list[UncertaintySignal] = []
    for observation in timeline.observations:
        if not observation.target_present or observation.bbox_xyxy is None:
            continue
        close: list[dict[str, float | int]] = []
        for row in index.get_frame(observation.frame_index):
            if row.track_id == timeline.current_raw_track_id:
                continue
            distance = center_distance_px(observation.bbox_xyxy, row.bbox_xyxy) / diagonal
            iou = bbox_iou(observation.bbox_xyxy, row.bbox_xyxy)
            if (
                distance <= config.neighbor_distance_threshold
                or iou >= config.neighbor_iou_threshold
            ):
                close.append(
                    {
                        "track_id": row.track_id,
                        "distance_normalized": distance,
                        "iou": iou,
                    }
                )
        if len(close) >= config.neighbor_count_threshold > 0:
            min_distance = min(float(item["distance_normalized"]) for item in close)
            max_iou = max(float(item["iou"]) for item in close)
            signals.append(
                make_signal(
                    signal_type="NEIGHBOR_AMBIGUITY",
                    frame_start=observation.frame_index,
                    frame_end=observation.frame_index,
                    frame_index=observation.frame_index,
                    raw_track_id=timeline.current_raw_track_id,
                    score=float(len(close)),
                    severity="warning",
                    threshold=float(config.neighbor_count_threshold),
                    triggered=True,
                    evidence={
                        "neighbor_count": len(close),
                        "min_distance_normalized": min_distance,
                        "max_iou": max_iou,
                        "neighbors": close,
                        "target_center": bbox_center(observation.bbox_xyxy),
                    },
                )
            )
    return tuple(signals)
