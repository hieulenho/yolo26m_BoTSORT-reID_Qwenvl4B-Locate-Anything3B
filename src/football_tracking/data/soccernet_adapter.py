"""SoccerNet-style adapter with explicit schema checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_tracking.data.adapters import DatasetAdapter, DatasetAdapterError
from football_tracking.data.discover import DatasetDiscoveryError, discover_sequence_candidates
from football_tracking.data.schemas import (
    BoundingBoxXYXY,
    FrameAnnotation,
    ObjectAnnotation,
    SequenceCandidate,
    SequenceInfo,
)

EXPECTED_JSON_SCHEMA = {
    "sequence": {"name", "fps", "width", "height", "frame_count"},
    "frames[]": {"frame_index", "image", "objects"},
    "objects[]": {"track_id", "class", "bbox"},
}


class SoccerNetAdapter(DatasetAdapter):
    """Load a conservative SoccerNet tracking annotation export.

    The real SoccerNet release can be arranged in different ways. This adapter
    intentionally accepts only a clear JSON schema and raises a descriptive
    error for anything else, so future work can add exact production schemas
    without silent guessing.
    """

    def can_handle(self, path: Path) -> bool:
        try:
            return bool(discover_sequence_candidates(path))
        except DatasetDiscoveryError:
            return False

    def discover_sequences(self, path: Path) -> list[SequenceCandidate]:
        return discover_sequence_candidates(path)

    def load_sequence(self, path: Path) -> SequenceInfo:
        candidate = self._candidate_for_path(path)
        payload = self._read_annotation_json(candidate.annotations_path)
        sequence_payload = self._require_mapping(payload.get("sequence"), "sequence")
        self._require_keys(
            sequence_payload,
            EXPECTED_JSON_SCHEMA["sequence"],
            candidate.annotations_path,
            "sequence",
        )

        frames = self._parse_frames(candidate, payload)
        width = int(sequence_payload["width"])
        height = int(sequence_payload["height"])
        frame_count = int(sequence_payload["frame_count"])
        metadata = dict(candidate.metadata)
        metadata["video_path"] = str(candidate.video_path) if candidate.video_path else None
        metadata["annotation_schema"] = "explicit_json_v1"

        return SequenceInfo(
            name=str(sequence_payload.get("name") or candidate.name),
            source_path=candidate.source_path,
            frames_dir=candidate.frames_dir,
            video_path=candidate.video_path,
            annotations_path=candidate.annotations_path,
            fps=float(sequence_payload["fps"]),
            width=width,
            height=height,
            frame_count=frame_count,
            annotations=frames,
            metadata=metadata,
        )

    def load_annotations(self, path: Path) -> list[FrameAnnotation]:
        return self.load_sequence(path).annotations

    def _candidate_for_path(self, path: Path) -> SequenceCandidate:
        if not path.is_dir():
            raise DatasetAdapterError(f"Sequence path is not a directory: {path}")
        candidates = discover_sequence_candidates(path.parent)
        for candidate in candidates:
            if candidate.source_path.resolve() == path.resolve():
                return candidate
        raise DatasetAdapterError(f"Could not discover sequence path: {path}")

    def _read_annotation_json(self, path: Path) -> dict[str, Any]:
        if path.suffix.lower() != ".json":
            raise DatasetAdapterError(
                f"Unsupported annotation file for SoccerNetAdapter: {path}. "
                f"Expected schema: {EXPECTED_JSON_SCHEMA}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetAdapterError(f"Annotation JSON is invalid: {path}: {exc}") from exc
        return self._require_mapping(payload, "annotation root")

    @staticmethod
    def _require_mapping(value: Any, context: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise DatasetAdapterError(f"{context} must be a JSON object.")
        return value

    @staticmethod
    def _require_keys(
        value: dict[str, Any],
        expected: set[str],
        path: Path,
        context: str,
    ) -> None:
        missing = sorted(expected - set(value))
        if missing:
            keys = sorted(str(key) for key in value)
            raise DatasetAdapterError(
                f"Annotation file {path} is missing {context} field(s): {missing}. "
                f"Actual keys: {keys}. Expected schema: {EXPECTED_JSON_SCHEMA}"
            )

    def _parse_frames(
        self,
        candidate: SequenceCandidate,
        payload: dict[str, Any],
    ) -> list[FrameAnnotation]:
        sequence_payload = self._require_mapping(payload["sequence"], "sequence")
        frame_payloads = payload.get("frames")
        if not isinstance(frame_payloads, list):
            raise DatasetAdapterError(
                f"Annotation file {candidate.annotations_path} must contain a frames list."
            )

        sequence_name = str(sequence_payload.get("name") or candidate.name)
        sequence_width = int(sequence_payload["width"])
        sequence_height = int(sequence_payload["height"])
        frames: list[FrameAnnotation] = []
        for frame_payload in frame_payloads:
            frame = self._require_mapping(frame_payload, "frame")
            self._require_keys(
                frame,
                EXPECTED_JSON_SCHEMA["frames[]"],
                candidate.annotations_path,
                "frame",
            )
            image_path = candidate.source_path / str(frame["image"])
            objects = [
                self._parse_object(sequence_name, int(frame["frame_index"]), obj)
                for obj in frame.get("objects", [])
            ]
            frames.append(
                FrameAnnotation(
                    sequence_name=sequence_name,
                    frame_index=int(frame["frame_index"]),
                    image_path=image_path,
                    width=int(frame.get("width", sequence_width)),
                    height=int(frame.get("height", sequence_height)),
                    objects=objects,
                )
            )
        return sorted(frames, key=lambda item: item.frame_index)

    def _parse_object(
        self,
        sequence_name: str,
        frame_index: int,
        payload: Any,
    ) -> ObjectAnnotation:
        obj = self._require_mapping(payload, "object")
        self._require_keys(obj, EXPECTED_JSON_SCHEMA["objects[]"], Path("<object>"), "object")
        bbox = obj["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise DatasetAdapterError(
                f"Object in {sequence_name} frame {frame_index} must have bbox [x1, y1, x2, y2]."
            )
        return ObjectAnnotation(
            frame_index=frame_index,
            track_id=obj["track_id"],
            source_class=str(obj["class"]),
            target_class=None,
            target_class_id=None,
            bbox_xyxy=BoundingBoxXYXY(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
            ),
            confidence=float(obj.get("confidence", 1.0)),
            visibility=float(obj.get("visibility", 1.0)),
            is_ignored=False,
            metadata={"raw": obj},
        )
