"""Frame-based index for read-only MOT observations."""

from __future__ import annotations

from dataclasses import dataclass

from football_tracking.locate_tracking.artifacts.mot_schemas import (
    MotArtifactError,
    MotTrackObservation,
)


@dataclass(frozen=True)
class FrameTrackIndex:
    _by_frame: dict[int, tuple[MotTrackObservation, ...]]
    observation_count: int
    unique_track_ids: tuple[int, ...]

    @classmethod
    def from_observations(
        cls,
        observations: list[MotTrackObservation] | tuple[MotTrackObservation, ...],
    ) -> FrameTrackIndex:
        by_frame: dict[int, list[MotTrackObservation]] = {}
        seen: set[tuple[int, int]] = set()
        track_ids: set[int] = set()
        for observation in tuple(observations):
            key = (observation.frame_index, observation.track_id)
            if key in seen:
                raise MotArtifactError(f"Duplicate frame-track pair: {key}")
            seen.add(key)
            track_ids.add(observation.track_id)
            by_frame.setdefault(observation.frame_index, []).append(observation)
        frozen = {
            frame_index: tuple(sorted(rows, key=lambda row: row.track_id))
            for frame_index, rows in by_frame.items()
        }
        return cls(
            _by_frame=frozen,
            observation_count=len(seen),
            unique_track_ids=tuple(sorted(track_ids)),
        )

    def get_frame(self, frame_index: int) -> tuple[MotTrackObservation, ...]:
        if frame_index < 1:
            raise MotArtifactError("frame_index must be >= 1.")
        return self._by_frame.get(frame_index, ())

    @property
    def frame_count_with_tracks(self) -> int:
        return len(self._by_frame)

    @property
    def available_frame_range(self) -> tuple[int, int] | None:
        if not self._by_frame:
            return None
        frames = sorted(self._by_frame)
        return frames[0], frames[-1]
