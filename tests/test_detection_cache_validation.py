from __future__ import annotations

from datetime import UTC, datetime

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.cache_schema import (
    SCHEMA_VERSION,
    CachedDetection,
    CachedFrameDetections,
    DetectionCacheMetadata,
)
from football_tracking.detection.cache_validation import validate_detection_cache_sequence
from football_tracking.detection.cache_writer import DetectionCacheWriter


def _write_cache(tmp_path, frames, confidence_floor: float = 0.001):
    writer = DetectionCacheWriter(tmp_path / "cache", overwrite=True)
    file_hash = writer.write_frames(frames)
    writer.write_metadata(
        DetectionCacheMetadata(
            schema_version=SCHEMA_VERSION,
            dataset="fixture",
            split="val",
            sequence_name="seq",
            expected_frame_count=2,
            cached_frame_count=len(frames),
            checkpoint="model.pt",
            checkpoint_type="fine_tuned",
            checkpoint_hash="abc",
            detector_name="YOLOv8m",
            image_size=64,
            confidence_floor=confidence_floor,
            nms_iou=0.7,
            max_det=100,
            class_filter=[0],
            device="cpu",
            python_version="3.12",
            torch_version=None,
            ultralytics_version=None,
            created_at=datetime.now(UTC).isoformat(),
            source_sequence_path="seq",
            complete=len(frames) == 2,
            partial=len(frames) != 2,
            max_frame=max(frame.frame_index for frame in frames),
            file_hash=file_hash,
        )
    )
    return writer.cache_dir


def test_cache_validation_accepts_empty_frames(tmp_path) -> None:
    cache_dir = _write_cache(
        tmp_path,
        [
            CachedFrameDetections("seq", 1, "1.jpg", 100, 80, []),
            CachedFrameDetections(
                "seq",
                2,
                "2.jpg",
                100,
                80,
                [CachedDetection(BoundingBoxXYXY(1, 2, 11, 22), 0.9)],
            ),
        ],
    )
    report = validate_detection_cache_sequence(cache_dir, expected_frame_count=2)
    assert report.error_count == 0


def test_cache_validation_detects_missing_frame_and_high_floor(tmp_path) -> None:
    cache_dir = _write_cache(
        tmp_path,
        [CachedFrameDetections("seq", 2, "2.jpg", 100, 80, [])],
        confidence_floor=0.25,
    )
    report = validate_detection_cache_sequence(
        cache_dir,
        expected_frame_count=2,
        confidence_threshold=0.1,
    )
    codes = {issue.code for issue in report.issues}
    assert "missing_frame" in codes
    assert "confidence_floor_too_high" in codes
