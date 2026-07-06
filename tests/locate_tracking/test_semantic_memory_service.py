from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from football_tracking.locate_tracking.association.matcher import GroundingTrackMatcher
from football_tracking.locate_tracking.association.service import FrameTrackQueryService
from football_tracking.locate_tracking.grounding.backend import MockGroundingBackend
from football_tracking.locate_tracking.grounding.cache import GroundingCache
from football_tracking.locate_tracking.grounding.service import GroundingService
from football_tracking.locate_tracking.sampling.schemas import FrameSamplingRequest
from football_tracking.locate_tracking.semantic_memory.schemas import SemanticMemoryConfig
from football_tracking.locate_tracking.semantic_memory.service import SemanticMemoryService


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _video(path: Path, frame_count: int = 3) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (40, 40),
    )
    assert writer.isOpened()
    for value in range(frame_count):
        writer.write(np.full((40, 40, 3), value * 40, dtype=np.uint8))
    writer.release()
    return path


def _tracks(path: Path, frame_count: int = 3) -> Path:
    rows = [f"{frame},7,4,4,12,12,-1,1,1" for frame in range(1, frame_count + 1)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _service(tmp_path: Path, backend: MockGroundingBackend) -> SemanticMemoryService:
    grounding_service = GroundingService(
        backend=backend,
        cache=GroundingCache(tmp_path / "cache"),
        overwrite=True,
    )
    frame_service = FrameTrackQueryService(
        matcher=GroundingTrackMatcher(),
        grounding_service=grounding_service,
        overwrite=True,
    )
    return SemanticMemoryService(
        config=SemanticMemoryConfig(min_usable_frames=2, min_support_frames=2),
        frame_query_service=frame_service,
        overwrite=True,
    )


def test_semantic_memory_service_end_to_end_with_mock_backend(tmp_path: Path) -> None:
    backend = MockGroundingBackend({"player": "<ref>player</ref><box><100><100><400><400></box>"})
    video = _video(tmp_path / "video.avi")
    tracks = _tracks(tmp_path / "tracks.txt")
    before = _hash(tracks)

    session = _service(tmp_path, backend).resolve_language_track(
        source_video=video,
        tracks_path=tracks,
        query="player",
        sampling_request=FrameSamplingRequest(total_frames=3, explicit_frames=(1, 2, 3)),
        output_dir=tmp_path / "semantic",
    )

    assert session.final_resolution.status == "resolved"
    assert session.final_resolution.selected_track_id == 7
    assert (tmp_path / "semantic" / "semantic_memory.json").is_file()
    assert (tmp_path / "semantic" / "final_resolution.json").is_file()
    assert before == _hash(tracks)


def test_semantic_memory_service_reuses_grounding_cache(tmp_path: Path) -> None:
    backend = MockGroundingBackend({"player": "<ref>player</ref><box><100><100><400><400></box>"})
    video = _video(tmp_path / "video.avi")
    tracks = _tracks(tmp_path / "tracks.txt")
    service = _service(tmp_path, backend)
    request = FrameSamplingRequest(total_frames=3, explicit_frames=(1, 2, 3))

    service.resolve_language_track(
        source_video=video,
        tracks_path=tracks,
        query="player",
        sampling_request=request,
        output_dir=tmp_path / "run1",
    )
    first_call_count = backend.call_count
    service.resolve_language_track(
        source_video=video,
        tracks_path=tracks,
        query="player",
        sampling_request=request,
        output_dir=tmp_path / "run2",
    )

    assert first_call_count == 3
    assert backend.call_count == 3
