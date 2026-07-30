"""Temporal stabilization for detector classes attached to persistent tracks."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Any

from football_tracking.tracking.schemas import TrackOutput


@dataclass(frozen=True)
class _ClassObservation:
    frame_index: int
    class_id: int
    class_name: str
    confidence: float


class TrackClassStabilizer:
    """Suppress short detector-class flips without changing tracker identities."""

    def __init__(self, *, history_frames: int = 30, switch_margin: float = 0.20) -> None:
        if history_frames < 2:
            raise ValueError("history_frames must be >= 2.")
        if not 0.0 <= switch_margin <= 1.0:
            raise ValueError("switch_margin must be in [0, 1].")
        self.history_frames = int(history_frames)
        self.switch_margin = float(switch_margin)
        self._history: dict[int, deque[_ClassObservation]] = defaultdict(deque)
        self._stable: dict[int, tuple[int, str]] = {}
        self._last_seen: dict[int, int] = {}
        self.suppressed_switches = 0
        self.accepted_switches = 0

    def reset(self) -> None:
        self._history.clear()
        self._stable.clear()
        self._last_seen.clear()

    def update(self, tracks: list[TrackOutput], frame_index: int) -> list[TrackOutput]:
        output = [self._update_track(track, frame_index) for track in tracks]
        self._prune(frame_index)
        return output

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "history_frames": self.history_frames,
            "switch_margin": self.switch_margin,
            "active_track_count": len(self._history),
            "suppressed_switches": self.suppressed_switches,
            "accepted_switches": self.accepted_switches,
        }

    def _update_track(self, track: TrackOutput, frame_index: int) -> TrackOutput:
        track_id = int(track.track_id)
        history = self._history[track_id]
        minimum_frame = frame_index - self.history_frames + 1
        while history and history[0].frame_index < minimum_frame:
            history.popleft()
        history.append(
            _ClassObservation(
                frame_index=frame_index,
                class_id=int(track.class_id),
                class_name=str(track.class_name),
                confidence=float(track.confidence if track.confidence is not None else 0.5),
            )
        )
        self._last_seen[track_id] = frame_index
        scores: dict[tuple[int, str], float] = defaultdict(float)
        for item in history:
            age = max(frame_index - item.frame_index, 0)
            recency = 0.97**age
            scores[(item.class_id, item.class_name)] += max(item.confidence, 0.05) * recency
        candidate = max(scores, key=lambda label: (scores[label], label[0], label[1]))
        stable = self._stable.get(track_id)
        if stable is None:
            stable = candidate
            self._stable[track_id] = stable
        elif candidate != stable:
            stable_score = scores.get(stable, 0.0)
            candidate_score = scores[candidate]
            required = stable_score * (1.0 + self.switch_margin)
            if stable_score <= 0.0 or candidate_score >= required:
                stable = candidate
                self._stable[track_id] = stable
                self.accepted_switches += 1
            else:
                self.suppressed_switches += 1
        if stable == (track.class_id, track.class_name):
            return track
        return replace(
            track,
            class_id=stable[0],
            class_name=stable[1],
            metadata={
                **track.metadata,
                "raw_detector_class_id": track.class_id,
                "raw_detector_class_name": track.class_name,
                "class_stabilized": True,
            },
        )

    def _prune(self, frame_index: int) -> None:
        retention = self.history_frames * 10
        stale = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if frame_index - last_seen > retention
        ]
        for track_id in stale:
            self._history.pop(track_id, None)
            self._stable.pop(track_id, None)
            self._last_seen.pop(track_id, None)
