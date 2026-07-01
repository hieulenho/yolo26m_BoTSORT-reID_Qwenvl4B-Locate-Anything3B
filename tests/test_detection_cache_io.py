from __future__ import annotations

from datetime import UTC, datetime

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.detection.cache_reader import DetectionCacheReader
from football_tracking.detection.cache_schema import (
    SCHEMA_VERSION,
    CachedDetection,
    CachedFrameDetections,
    DetectionCacheMetadata,
)
from football_tracking.detection.cache_writer import DetectionCacheWriter
from football_tracking.detection.serialization import file_sha256


def _metadata(file_hash: str | None) -> DetectionCacheMetadata:
    return DetectionCacheMetadata(
        schema_version=SCHEMA_VERSION,
        dataset="fixture",
        split="val",
        sequence_name="seq",
        expected_frame_count=2,
        cached_frame_count=2,
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
        max_frame=2,
        file_hash=file_hash,
    )


def test_detection_cache_writer_and_reader_roundtrip(tmp_path) -> None:
    cache_dir = tmp_path / "cache" / "val" / "seq"
    frames = [
        CachedFrameDetections("seq", 1, "1.jpg", 100, 80, []),
        CachedFrameDetections(
            "seq",
            2,
            "2.jpg",
            100,
            80,
            [CachedDetection(BoundingBoxXYXY(1, 2, 11, 22), 0.9)],
        ),
    ]
    writer = DetectionCacheWriter(cache_dir, overwrite=True)
    written_hash = writer.write_frames(frames)
    writer.write_metadata(_metadata(written_hash))

    reader = DetectionCacheReader(cache_dir)
    assert reader.read_metadata().file_hash == file_sha256(writer.detections_path)
    assert len(reader.read_frames()) == 2
    assert reader.frame(1).detections == []
    assert reader.frame(2).detections[0].confidence == 0.9
