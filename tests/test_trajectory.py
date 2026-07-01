from __future__ import annotations

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.schemas import TrackOutput
from football_tracking.tracking.trajectory import TrajectoryStore


def _track(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackOutput:
    return TrackOutput.from_xyxy(
        frame_index=1,
        sequence_name="seq",
        track_id=track_id,
        bbox_xyxy=BoundingBoxXYXY(x1, y1, x2, y2),
        confidence=0.9,
    )


def test_trajectory_limits_history_and_resets() -> None:
    store = TrajectoryStore(trajectory_length=2)
    store.update([_track(1, 0, 0, 10, 10)])
    store.update([_track(1, 10, 10, 20, 20)])
    store.update([_track(1, 20, 20, 30, 30)])

    assert store.get(1) == [(15, 15), (25, 25)]

    store.delete(1)
    assert store.get(1) == []

    store.update([_track(2, 0, 0, 2, 2)])
    store.reset()
    assert store.as_dict() == {}
