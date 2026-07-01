from __future__ import annotations

from datetime import UTC, datetime

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.cache_schema import (
    SCHEMA_VERSION,
    CachedDetection,
    CachedFrameDetections,
    DetectionCacheMetadata,
)
from football_tracking.detection.cache_writer import DetectionCacheWriter
from football_tracking.tracking.cached_detection_source import CachedDetectionSource


def test_cached_detection_source_filters_tracker_inputs(tmp_path) -> None:
    writer = DetectionCacheWriter(tmp_path / "cache", overwrite=True)
    frames = [
        CachedFrameDetections(
            "seq",
            1,
            "1.jpg",
            100,
            80,
            [
                CachedDetection(BoundingBoxXYXY(1, 2, 11, 22), 0.9),
                CachedDetection(BoundingBoxXYXY(20, 2, 30, 22), 0.1),
            ],
        )
    ]
    file_hash = writer.write_frames(frames)
    writer.write_metadata(
        DetectionCacheMetadata(
            schema_version=SCHEMA_VERSION,
            dataset="fixture",
            split="val",
            sequence_name="seq",
            expected_frame_count=1,
            cached_frame_count=1,
            checkpoint="model.pt",
            checkpoint_type="fine_tuned",
            checkpoint_hash="abc",
            detector_name="YOLOv8m",
            image_size=64,
            confidence_floor=0.001,
            nms_iou=0.7,
            max_det=100,
            class_filter=[0],
            device="cpu",
            python_version="3.12",
            torch_version=None,
            ultralytics_version=None,
            created_at=datetime.now(UTC).isoformat(),
            source_sequence_path="seq",
            complete=True,
            partial=False,
            max_frame=1,
            file_hash=file_hash,
        )
    )

    source = CachedDetectionSource(writer.cache_dir, confidence_threshold=0.5)
    detections = source.tracker_detections(1)

    assert len(detections) == 1
    assert detections[0].confidence == 0.9
    assert detections[0].metadata["detection_source"] == "cache"
