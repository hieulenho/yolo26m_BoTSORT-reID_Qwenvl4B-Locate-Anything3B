"""Short trajectory history for track visualization."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

from football_tracking.tracking.schemas import TrackOutput


@dataclass
class TrajectoryStore:
    trajectory_length: int = 30
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.trajectory_length < 1:
            raise ValueError("trajectory_length must be >= 1.")
        self._points: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.trajectory_length)
        )

    def reset(self) -> None:
        self._points.clear()

    def update(self, tracks: Iterable[TrackOutput]) -> None:
        if not self.enabled:
            return
        for track in tracks:
            center_x = (track.bbox_xyxy.x1 + track.bbox_xyxy.x2) / 2.0
            center_y = (track.bbox_xyxy.y1 + track.bbox_xyxy.y2) / 2.0
            self._points[track.track_id].append((center_x, center_y))

    def delete(self, track_id: int) -> None:
        self._points.pop(track_id, None)

    def get(self, track_id: int) -> list[tuple[float, float]]:
        return list(self._points.get(track_id, ()))

    def as_dict(self) -> dict[int, list[tuple[float, float]]]:
        return {track_id: list(points) for track_id, points in self._points.items()}
