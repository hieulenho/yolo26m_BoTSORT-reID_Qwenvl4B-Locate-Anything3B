from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from football_tracking.locate_tracking.artifacts.mot_reader import read_mot_track_file
from football_tracking.locate_tracking.association.matcher import GroundingTrackMatcher
from football_tracking.locate_tracking.association.service import FrameTrackQueryService
from football_tracking.locate_tracking.grounding.backend import MockGroundingBackend
from football_tracking.locate_tracking.grounding.cache import GroundingCache
from football_tracking.locate_tracking.grounding.service import GroundingService


def _video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 32),
    )
    assert writer.isOpened()
    for value in (40, 140):
        writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
    writer.release()
    return path


def _tracks(path: Path) -> Path:
    path.write_text("1,7,3,3,10,10,-1,1,1\n2,9,3,3,10,10,-1,1,1\n", encoding="utf-8")
    return path


def test_frame_query_service_end_to_end_with_mock_backend(tmp_path: Path) -> None:
    backend = MockGroundingBackend({"player": "<ref>player</ref><box><90><90><410><410></box>"})
    grounding_service = GroundingService(
        backend=backend,
        cache=GroundingCache(tmp_path / "cache"),
        overwrite=True,
    )
    service = FrameTrackQueryService(
        matcher=GroundingTrackMatcher(),
        grounding_service=grounding_service,
        overwrite=True,
    )

    result = service.query_track_frame(
        source_video=_video(tmp_path / "video.avi"),
        tracks_path=_tracks(tmp_path / "tracks.txt"),
        frame_index=1,
        query="player",
        output_dir=tmp_path / "query",
    )

    assert result.overall_status == "resolved"
    assert result.resolved_track_ids == (7,)
    assert (tmp_path / "query" / "frame_000001.jpg").is_file()
    assert (tmp_path / "query" / "grounding.json").is_file()
    assert (tmp_path / "query" / "association.json").is_file()


def test_frame_query_service_reuses_grounding_cache(tmp_path: Path) -> None:
    backend = MockGroundingBackend({"player": "<ref>player</ref><box><90><90><410><410></box>"})
    grounding_service = GroundingService(
        backend=backend,
        cache=GroundingCache(tmp_path / "cache"),
        overwrite=True,
    )
    service = FrameTrackQueryService(
        matcher=GroundingTrackMatcher(),
        grounding_service=grounding_service,
        overwrite=True,
    )
    video = _video(tmp_path / "video.avi")
    tracks = _tracks(tmp_path / "tracks.txt")

    first = service.query_track_frame(
        source_video=video,
        tracks_path=tracks,
        frame_index=1,
        query="player",
        output_dir=tmp_path / "query1",
    )
    second = service.query_track_frame(
        source_video=video,
        tracks_path=tracks,
        frame_index=1,
        query="player",
        output_dir=tmp_path / "query2",
    )

    assert first.grounding["cache_hit"] is False
    assert second.grounding["cache_hit"] is True
    assert backend.call_count == 1


def test_frame_query_service_match_existing_grounding(tmp_path: Path) -> None:
    backend = MockGroundingBackend({"player": "<ref>player</ref><box><90><90><410><410></box>"})
    grounding_service = GroundingService(backend=backend, cache=None, overwrite=True)
    grounding_path = tmp_path / "grounding.json"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), image)
    grounding_service.ground_image(
        image_path=image_path,
        query="player",
        output_path=grounding_path,
    )

    service = FrameTrackQueryService(matcher=GroundingTrackMatcher(), overwrite=True)
    result = service.match_existing_grounding(
        grounding_result_path=grounding_path,
        tracks_path=_tracks(tmp_path / "tracks.txt"),
        frame_index=1,
        frame_width=32,
        frame_height=32,
        output_path=tmp_path / "association.json",
    )

    assert result.overall_status == "resolved"
    assert result.resolved_track_ids == (7,)
    assert read_mot_track_file(tmp_path / "tracks.txt").observation_count == 2
