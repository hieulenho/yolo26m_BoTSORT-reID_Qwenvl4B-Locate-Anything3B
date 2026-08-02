"""Lightweight temporal appearance hints for realtime semantic rendering."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from football_tracking.tracking.schemas import TrackOutput

_VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "van",
    "sedan",
    "suv",
    "hatchback",
    "pickup",
    "minivan",
}
_COLOR_NAMES = {
    "black",
    "white",
    "gray",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "brown",
}


@dataclass(frozen=True)
class _ColorObservation:
    frame_index: int
    color: str
    confidence: float


class FastVisualAttributeStore:
    """Estimate conservative vehicle colors and fuse them over time."""

    def __init__(
        self,
        *,
        sample_interval_frames: int = 5,
        minimum_observations: int = 3,
        history_frames: int = 90,
        retention_frames: int = 900,
        minimum_consensus: float = 0.62,
        switch_consensus: float = 0.76,
    ) -> None:
        if sample_interval_frames < 1:
            raise ValueError("sample_interval_frames must be positive.")
        if minimum_observations < 1:
            raise ValueError("minimum_observations must be positive.")
        if history_frames < 1 or retention_frames < history_frames:
            raise ValueError(
                "retention_frames must be greater than or equal to history_frames."
            )
        if not 0.0 <= minimum_consensus <= 1.0:
            raise ValueError("minimum_consensus must be in [0, 1].")
        if not minimum_consensus <= switch_consensus <= 1.0:
            raise ValueError(
                "switch_consensus must be in [minimum_consensus, 1]."
            )
        self.sample_interval_frames = int(sample_interval_frames)
        self.minimum_observations = int(minimum_observations)
        self.history_frames = int(history_frames)
        self.retention_frames = int(retention_frames)
        self.minimum_consensus = float(minimum_consensus)
        self.switch_consensus = float(switch_consensus)
        self._history: dict[int, deque[_ColorObservation]] = defaultdict(deque)
        self._stable: dict[int, tuple[str, float, int]] = {}
        self.sampled_track_boxes = 0
        self.accepted_colors = 0
        self.color_switches = 0

    def reset(self) -> None:
        self._history.clear()
        self._stable.clear()

    def update(
        self,
        frame: np.ndarray,
        tracks: list[TrackOutput],
        frame_index: int,
    ) -> list[TrackOutput]:
        if (frame_index - 1) % self.sample_interval_frames == 0:
            for track in tracks:
                if not _is_vehicle_track(track):
                    continue
                estimate = _estimate_vehicle_color(frame, track)
                self.sampled_track_boxes += 1
                if estimate is None:
                    continue
                color, confidence = estimate
                self._history[int(track.track_id)].append(
                    _ColorObservation(
                        frame_index=frame_index,
                        color=color,
                        confidence=confidence,
                    )
                )
        self._refresh(frame_index)
        return [self._decorate(track, frame_index) for track in tracks]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "sample_interval_frames": self.sample_interval_frames,
            "minimum_observations": self.minimum_observations,
            "history_frames": self.history_frames,
            "retention_frames": self.retention_frames,
            "switch_consensus": self.switch_consensus,
            "active_track_count": len(self._history),
            "stable_track_count": len(self._stable),
            "sampled_track_boxes": self.sampled_track_boxes,
            "accepted_colors": self.accepted_colors,
            "color_switches": self.color_switches,
        }

    def _refresh(self, frame_index: int) -> None:
        minimum_frame = frame_index - self.history_frames + 1
        stale_tracks: list[int] = []
        for track_id, history in self._history.items():
            while history and history[0].frame_index < minimum_frame:
                history.popleft()
            if not history:
                stale_tracks.append(track_id)
                continue
            scores: Counter[str] = Counter()
            counts: Counter[str] = Counter()
            latest: dict[str, int] = {}
            for row in history:
                scores[row.color] += max(row.confidence, 0.01)
                counts[row.color] += 1
                latest[row.color] = max(latest.get(row.color, 0), row.frame_index)
            candidate = max(
                scores,
                key=lambda color: (scores[color], counts[color], color),
            )
            total_score = sum(scores.values())
            consensus = scores[candidate] / max(total_score, 1e-9)
            if (
                counts[candidate] < self.minimum_observations
                or consensus < self.minimum_consensus
            ):
                continue
            confidence = min(scores[candidate] / counts[candidate], 1.0)
            next_value = (candidate, confidence, latest[candidate])
            previous = self._stable.get(track_id)
            if (
                previous is not None
                and previous[0] != candidate
                and consensus < self.switch_consensus
            ):
                continue
            if previous is None:
                self.accepted_colors += 1
            elif previous[0] != candidate:
                self.color_switches += 1
            self._stable[track_id] = next_value
        for track_id in stale_tracks:
            self._history.pop(track_id, None)
        expired = [
            track_id
            for track_id, (_color, _confidence, last_frame) in self._stable.items()
            if frame_index - last_frame > self.retention_frames
        ]
        for track_id in expired:
            self._stable.pop(track_id, None)
            self._history.pop(track_id, None)

    def _decorate(self, track: TrackOutput, frame_index: int) -> TrackOutput:
        stable = self._stable.get(int(track.track_id))
        if stable is None:
            return track
        color, confidence, last_frame = stable
        if frame_index - last_frame > self.retention_frames:
            return track
        original_name = str(track.class_name)
        label_tokens = original_name.casefold().split()
        display_name = (
            original_name
            if color in label_tokens
            else f"{color} {original_name}"
        )
        return replace(
            track,
            class_name=display_name,
            metadata={
                **track.metadata,
                "base_detector_class_id": int(
                    track.metadata.get("base_detector_class_id", track.class_id)
                ),
                "base_detector_class_name": str(
                    track.metadata.get(
                        "base_detector_class_name",
                        original_name,
                    )
                ),
                "fast_visual_color": color,
                "fast_visual_color_confidence": confidence,
                "fast_visual_color_last_frame": last_frame,
                "fast_visual_color_source": "temporal_hsv",
            },
        )


def _is_vehicle_track(track: TrackOutput) -> bool:
    base_name = str(
        track.metadata.get("base_detector_class_name", track.class_name)
    )
    names = {
        base_name.strip().casefold(),
        str(track.class_name).strip().casefold(),
        str(track.metadata.get("fast_semantic_label", "")).strip().casefold(),
    }
    return bool(names & _VEHICLE_CLASSES)


def _estimate_vehicle_color(
    frame: np.ndarray,
    track: TrackOutput,
) -> tuple[str, float] | None:
    height, width = frame.shape[:2]
    box = track.bbox_xyxy
    x1 = max(0, min(width - 1, int(round(box.x1))))
    y1 = max(0, min(height - 1, int(round(box.y1))))
    x2 = max(x1 + 1, min(width, int(round(box.x2))))
    y2 = max(y1 + 1, min(height, int(round(box.y2))))
    box_width = x2 - x1
    box_height = y2 - y1
    roi_x1 = x1 + int(round(box_width * 0.16))
    roi_x2 = x1 + int(round(box_width * 0.84))
    roi_y1 = y1 + int(round(box_height * 0.34))
    roi_y2 = y1 + int(round(box_height * 0.86))
    crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    labels = np.full(hue.shape, "", dtype=object)
    usable = value >= 25
    achromatic = usable & (saturation < 48)
    labels[achromatic & (value < 68)] = "black"
    labels[achromatic & (value >= 68) & (value < 132)] = "gray"
    labels[achromatic & (value >= 132) & (value < 205)] = "silver"
    labels[achromatic & (value >= 205)] = "white"
    chromatic = usable & (saturation >= 48)
    labels[chromatic & ((hue < 10) | (hue >= 170))] = "red"
    labels[chromatic & (hue >= 10) & (hue < 22) & (value >= 145)] = "orange"
    labels[chromatic & (hue >= 10) & (hue < 25) & (value < 145)] = "brown"
    labels[chromatic & (hue >= 22) & (hue < 38)] = "yellow"
    labels[chromatic & (hue >= 38) & (hue < 85)] = "green"
    labels[chromatic & (hue >= 85) & (hue < 140)] = "blue"
    values = [str(item) for item in labels.ravel() if str(item) in _COLOR_NAMES]
    if len(values) < 80:
        return None
    counts = Counter(values)
    color, count = counts.most_common(1)[0]
    confidence = count / len(values)
    if confidence < 0.40:
        return None
    return color, min(max(confidence, 0.0), 1.0)
