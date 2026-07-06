from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from football_tracking.locate_tracking.appearance.config import (
    load_appearance_verification_config,
)
from football_tracking.locate_tracking.appearance.schemas import (
    AppearanceEmbedding,
    CropQualityMetrics,
    CropReference,
    TrackEmbeddingSample,
)
from football_tracking.locate_tracking.semantic_memory.aggregator import (
    build_semantic_memory,
)
from football_tracking.locate_tracking.semantic_memory.schemas import SemanticMemoryConfig
from football_tracking.locate_tracking.semantic_memory.serialization import (
    save_semantic_memory,
)
from tests.locate_tracking.semantic_test_utils import ambiguous_frame, resolved_frame


def tiny_video(path: Path, frame_count: int = 6) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 64),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :] = (20 + index * 10, 30, 40)
        frame[5:25, 5:25] = (0, 255, 0)
        frame[30:50, 30:50] = (255, 0, 0)
        writer.write(frame)
    writer.release()
    return path


def tiny_tracks(path: Path, frame_count: int = 6) -> Path:
    rows: list[str] = []
    for frame in range(1, frame_count + 1):
        rows.append(f"{frame},7,5,5,20,20,-1,1,1")
        rows.append(f"{frame},11,30,30,20,20,-1,1,1")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def crop_reference(track_id: int = 7, frame_index: int = 1) -> CropReference:
    quality = CropQualityMetrics(
        width=20,
        height=20,
        area=400,
        aspect_ratio=1.0,
        visible_fraction=1.0,
        sharpness_score=None,
        brightness_mean=50.0,
        passed_quality_gate=True,
        rejection_reasons=(),
        quality_score=1.0,
    )
    return CropReference(
        raw_track_id=track_id,
        frame_index=frame_index,
        source_video="video.avi",
        raw_bbox_xyxy=(0, 0, 20, 20),
        clipped_bbox_xyxy=(0, 0, 20, 20),
        crop_width=20,
        crop_height=20,
        quality_metrics=quality,
    )


def embedding_sample(
    vector: tuple[float, ...],
    *,
    track_id: int = 7,
    frame_index: int = 1,
    quality_weight: float = 1.0,
) -> TrackEmbeddingSample:
    return TrackEmbeddingSample(
        crop_reference=crop_reference(track_id, frame_index),
        embedding=AppearanceEmbedding(
            backend="mock",
            model_id="mock-appearance",
            dimension=len(vector),
            vector=vector,
            normalized=True,
            source_track_id=track_id,
            source_frame_index=frame_index,
            metadata={},
        ),
        quality_weight=quality_weight,
    )


def semantic_memory_fixture(path: Path) -> Path:
    config = SemanticMemoryConfig(
        min_usable_frames=1,
        min_support_frames=0,
        min_support_ratio=0.0,
        min_aggregate_score=0.0,
        winner_margin=0.0,
    )
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(ambiguous_frame(1, 7, 11), ambiguous_frame(6, 7, 11)),
        config=config,
    )
    return save_semantic_memory(memory, path, overwrite=True)


def one_track_semantic_memory_fixture(path: Path) -> Path:
    config = SemanticMemoryConfig(min_usable_frames=1, min_support_frames=1)
    memory = build_semantic_memory(
        query="player",
        frame_resolutions=(resolved_frame(1, 7), resolved_frame(6, 7)),
        config=config,
    )
    return save_semantic_memory(memory, path, overwrite=True)


def appearance_config(tmp_path: Path):
    config = load_appearance_verification_config(
        "configs/locate_tracking/appearance_verification.yaml",
        overrides={"output_dir": tmp_path / "appearance", "overwrite": True},
    )
    return replace(
        config,
        cache_directory=tmp_path / "cache",
    )
