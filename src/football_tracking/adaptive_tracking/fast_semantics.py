"""Associate sparse open-vocabulary detections with persistent tracker IDs."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Any

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.schemas import TrackerDetection, TrackOutput


@dataclass(frozen=True)
class _ProposalObservation:
    frame_index: int
    class_id: int
    class_name: str
    confidence: float
    association_score: float


class FastSemanticProposalStore:
    """Keep sparse YOLOE labels attached to the matching base-detector track."""

    def __init__(
        self,
        *,
        minimum_semantic_class_id: int = 1000,
        minimum_association_score: float = 0.35,
        minimum_observations: int = 2,
        history_frames: int = 90,
        retention_frames: int = 180,
        minimum_consensus: float = 0.55,
    ) -> None:
        if minimum_semantic_class_id < 1:
            raise ValueError("minimum_semantic_class_id must be positive.")
        if not 0.0 <= minimum_association_score <= 1.0:
            raise ValueError("minimum_association_score must be in [0, 1].")
        if minimum_observations < 1:
            raise ValueError("minimum_observations must be positive.")
        if history_frames < 1 or retention_frames < history_frames:
            raise ValueError(
                "retention_frames must be greater than or equal to history_frames."
            )
        if not 0.0 <= minimum_consensus <= 1.0:
            raise ValueError("minimum_consensus must be in [0, 1].")
        self.minimum_semantic_class_id = int(minimum_semantic_class_id)
        self.minimum_association_score = float(minimum_association_score)
        self.minimum_observations = int(minimum_observations)
        self.history_frames = int(history_frames)
        self.retention_frames = int(retention_frames)
        self.minimum_consensus = float(minimum_consensus)
        self._history: dict[int, deque[_ProposalObservation]] = defaultdict(deque)
        self._stable: dict[int, tuple[int, str, float, int]] = {}
        self.matched_proposals = 0
        self.rejected_proposals = 0
        self.accepted_labels = 0
        self.label_switches = 0

    def reset(self) -> None:
        self._history.clear()
        self._stable.clear()

    def update(
        self,
        tracks: list[TrackOutput],
        proposals: list[TrackerDetection],
        frame_index: int,
    ) -> list[TrackOutput]:
        semantic_proposals = [
            row
            for row in proposals
            if int(row.class_id) >= self.minimum_semantic_class_id
        ]
        matches = _greedy_matches(
            tracks,
            semantic_proposals,
            minimum_score=self.minimum_association_score,
        )
        for track_index, proposal_index, score in matches:
            track = tracks[track_index]
            proposal = semantic_proposals[proposal_index]
            self._history[int(track.track_id)].append(
                _ProposalObservation(
                    frame_index=frame_index,
                    class_id=int(proposal.class_id),
                    class_name=str(proposal.class_name),
                    confidence=float(proposal.confidence),
                    association_score=score,
                )
            )
            self.matched_proposals += 1
        self.rejected_proposals += max(
            len(semantic_proposals)
            - len({proposal_index for _track_index, proposal_index, _score in matches}),
            0,
        )
        self._refresh(frame_index)
        return [self._decorate(track, frame_index) for track in tracks]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "minimum_semantic_class_id": self.minimum_semantic_class_id,
            "minimum_association_score": self.minimum_association_score,
            "minimum_observations": self.minimum_observations,
            "history_frames": self.history_frames,
            "retention_frames": self.retention_frames,
            "active_track_count": len(self._history),
            "stable_track_count": len(self._stable),
            "matched_proposals": self.matched_proposals,
            "rejected_proposals": self.rejected_proposals,
            "accepted_labels": self.accepted_labels,
            "label_switches": self.label_switches,
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
            scores: dict[tuple[int, str], float] = defaultdict(float)
            counts: dict[tuple[int, str], int] = defaultdict(int)
            latest: dict[tuple[int, str], int] = {}
            for row in history:
                key = (row.class_id, row.class_name)
                scores[key] += max(row.confidence, 0.01) * row.association_score
                counts[key] += 1
                latest[key] = max(latest.get(key, 0), row.frame_index)
            candidate = max(scores, key=lambda key: (scores[key], counts[key], key))
            total = sum(scores.values())
            consensus = scores[candidate] / total if total > 0.0 else 0.0
            if (
                counts[candidate] < self.minimum_observations
                or consensus < self.minimum_consensus
            ):
                continue
            confidence = min(
                scores[candidate] / max(counts[candidate], 1),
                1.0,
            )
            previous = self._stable.get(track_id)
            next_value = (
                candidate[0],
                candidate[1],
                confidence,
                latest[candidate],
            )
            if previous is None:
                self.accepted_labels += 1
            elif previous[:2] != next_value[:2]:
                self.label_switches += 1
            self._stable[track_id] = next_value
        for track_id in stale_tracks:
            self._history.pop(track_id, None)
        expired = [
            track_id
            for track_id, (_class_id, _name, _confidence, last_frame) in self._stable.items()
            if frame_index - last_frame > self.retention_frames
        ]
        for track_id in expired:
            self._stable.pop(track_id, None)
            self._history.pop(track_id, None)

    def _decorate(self, track: TrackOutput, frame_index: int) -> TrackOutput:
        semantic = self._stable.get(int(track.track_id))
        if semantic is None:
            return track
        class_id, class_name, confidence, last_frame = semantic
        if frame_index - last_frame > self.retention_frames:
            return track
        return replace(
            track,
            class_id=class_id,
            class_name=class_name,
            metadata={
                **track.metadata,
                "base_detector_class_id": int(track.class_id),
                "base_detector_class_name": str(track.class_name),
                "fast_semantic_label": class_name,
                "fast_semantic_confidence": confidence,
                "fast_semantic_last_frame": last_frame,
                "fast_semantic_source": "supplemental_detector",
            },
        )


def _greedy_matches(
    tracks: list[TrackOutput],
    proposals: list[TrackerDetection],
    *,
    minimum_score: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for proposal_index, proposal in enumerate(proposals):
            score = _overlap_score(track.bbox_xyxy, proposal.bbox_xyxy)
            if score >= minimum_score:
                candidates.append((score, track_index, proposal_index))
    candidates.sort(reverse=True)
    used_tracks: set[int] = set()
    used_proposals: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, track_index, proposal_index in candidates:
        if track_index in used_tracks or proposal_index in used_proposals:
            continue
        used_tracks.add(track_index)
        used_proposals.add(proposal_index)
        matches.append((track_index, proposal_index, score))
    return matches


def _overlap_score(first: BoundingBoxXYXY, second: BoundingBoxXYXY) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    first_area = max((first.x2 - first.x1) * (first.y2 - first.y1), 1e-9)
    second_area = max((second.x2 - second.x1) * (second.y2 - second.y1), 1e-9)
    union = first_area + second_area - intersection
    iou = intersection / max(union, 1e-9)
    containment = intersection / max(min(first_area, second_area), 1e-9)
    return min(max(max(iou, containment), 0.0), 1.0)
