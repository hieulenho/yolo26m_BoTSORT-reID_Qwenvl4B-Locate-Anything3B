from __future__ import annotations

from pathlib import Path

import numpy as np

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.deepsort_adapter import DeepSortRuntimeConfig
from football_tracking.tracking.pipeline import partition_tracking_detections, run_tracking
from football_tracking.tracking.schemas import TrackerDetection, TrackOutput


class FakeDetector:
    def __init__(self) -> None:
        self.load_count = 0
        self.predict_count = 0

    def load_model(self) -> None:
        self.load_count += 1

    def predict_frame(self, *_args, **_kwargs):
        self.predict_count += 1
        return {"xyxy": [[5, 6, 25, 36]], "conf": [0.9], "cls": [0]}


class FakeAdapter:
    reset_count = 0

    def __init__(self, _config: DeepSortRuntimeConfig) -> None:
        self.updates = 0

    def reset(self) -> None:
        self.reset_count += 1

    def update(self, frame_index, sequence_name, detections, frame, image_width, image_height):
        self.updates += 1
        if not detections:
            return []
        detection = detections[0]
        return [
            TrackOutput.from_xyxy(
                frame_index=frame_index,
                sequence_name=sequence_name,
                track_id=1,
                bbox_xyxy=detection.bbox_xyxy,
                confidence=detection.confidence,
                metadata={"bbox_source": "fake"},
            )
        ]


def test_detect_only_classes_are_not_sent_to_tracker() -> None:
    detections = [
        TrackerDetection.from_xyxy(
            frame_index=1,
            sequence_name="scene",
            bbox_xyxy=BoundingBoxXYXY(1, 1, 10, 20),
            confidence=0.9,
            class_id=0,
            class_name="person",
        ),
        TrackerDetection.from_xyxy(
            frame_index=1,
            sequence_name="scene",
            bbox_xyxy=BoundingBoxXYXY(20, 1, 30, 20),
            confidence=0.8,
            class_id=9,
            class_name="traffic light",
        ),
    ]

    tracker_inputs, detection_only = partition_tracking_detections(detections, (0,))

    assert [item.class_name for item in tracker_inputs] == ["person"]
    assert [item.class_name for item in detection_only] == ["traffic light"]


def _write_sequence(root: Path) -> Path:
    import cv2  # type: ignore[import-not-found]

    sequence_dir = root / "data" / "mot" / "sportsmot_football" / "val" / "seq"
    img_dir = sequence_dir / "img1"
    img_dir.mkdir(parents=True)
    for frame in (1, 2):
        cv2.imwrite(str(img_dir / f"{frame:06d}.jpg"), np.zeros((40, 50, 3), dtype=np.uint8))
    (sequence_dir / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                "name=seq",
                "imDir=img1",
                "frameRate=25",
                "seqLength=2",
                "imWidth=50",
                "imHeight=40",
                "imExt=.jpg",
            ]
        ),
        encoding="utf-8",
    )
    seqmap = root / "data" / "mot" / "sportsmot_football" / "seqmaps" / "val.txt"
    seqmap.parent.mkdir(parents=True)
    seqmap.write_text("name\nseq\n", encoding="utf-8")
    return sequence_dir


def _write_video(path: Path) -> Path:
    import cv2  # type: ignore[import-not-found]

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (50, 40),
    )
    try:
        for _ in range(2):
            writer.write(np.zeros((40, 50, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def _write_deepsort_config(path: Path) -> None:
    path.write_text(
        """
tracker:
  name: deepsort
  max_age: 30
  n_init: 1
  max_iou_distance: 0.7
  max_cosine_distance: 0.3
  nn_budget: 10
  embedder: mobilenet
  half: false
  bgr: true
  embedder_gpu: false
  polygon: false
  today: null
association:
  only_position: false
  use_appearance: true
output:
  confirmed_only: true
  require_recent_update: true
  max_time_since_update_for_output: 1
  use_original_detection_box: true
runtime:
  log_level: INFO
""".strip(),
        encoding="utf-8",
    )


def _write_tracking_config(root: Path, checkpoint: Path, deepsort_config: Path) -> Path:
    config = root / "track.yaml"
    config.write_text(
        f"""
model:
  checkpoint: {checkpoint.as_posix()}
  allow_smoke_checkpoint: false
  allow_pretrained_fallback: false
detector:
  imgsz: 64
  conf: 0.1
  iou: 0.7
  max_det: 10
  device: cpu
  half: false
  class_ids: [0]
tracker:
  config: {deepsort_config.as_posix()}
dataset:
  mot_root: {(root / "data" / "mot" / "sportsmot_football").as_posix()}
  split: val
  seqmap: {(root / "data" / "mot" / "sportsmot_football" / "seqmaps" / "val.txt").as_posix()}
output:
  tracks_dir: {(root / "outputs" / "tracks" / "deepsort").as_posix()}
  videos_dir: {(root / "outputs" / "videos" / "deepsort").as_posix()}
  metrics_dir: {(root / "outputs" / "metrics").as_posix()}
  render_video: false
  save_mot: true
render:
  enabled: false
runtime:
  max_sequences: 1
  max_frames_per_sequence: 2
  overwrite: true
  smoke_only: false
""".strip(),
        encoding="utf-8",
    )
    return config


def test_tracking_pipeline_writes_mot_and_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _write_sequence(tmp_path)
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(b"weights")
    deepsort_config = tmp_path / "deepsort.yaml"
    _write_deepsort_config(deepsort_config)
    config = _write_tracking_config(tmp_path, checkpoint, deepsort_config)
    detector = FakeDetector()

    result = run_tracking(
        config,
        detector=detector,
        tracker_adapter_factory=lambda config: FakeAdapter(config),
    )

    sequence = result["summary"]["sequences"][0]
    mot_path = Path(sequence["output_mot"])
    metadata_path = mot_path.with_suffix(".metadata.json")

    assert detector.load_count == 1
    assert detector.predict_count == 2
    assert mot_path.is_file()
    assert metadata_path.is_file()
    assert result["summary"]["validation"]["summary"]["errors"] == 0
    assert result["summary"]["detection_count"] == 2
    assert result["summary"]["unique_predicted_track_count"] == 1


def test_tracking_pipeline_source_override_outputs_next_to_video(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _write_sequence(tmp_path)
    source = _write_video(tmp_path / "videos" / "clip.mp4")
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(b"weights")
    deepsort_config = tmp_path / "deepsort.yaml"
    _write_deepsort_config(deepsort_config)
    config = _write_tracking_config(tmp_path, checkpoint, deepsort_config)

    result = run_tracking(
        config,
        overrides={"source": source, "render": True, "overwrite": True, "max_frames": 2},
        detector=FakeDetector(),
        tracker_adapter_factory=lambda config: FakeAdapter(config),
    )

    sequence = result["summary"]["sequences"][0]
    assert Path(sequence["output_video"]) == source.with_name("clip_tracked.mp4")
    assert Path(sequence["output_mot"]) == source.with_name("clip_tracked.txt")
    assert source.with_name("clip_tracked.mp4").is_file()
    assert source.with_name("clip_tracked.txt").is_file()
    assert source.with_name("clip_tracked.metadata.json").is_file()


def test_tracking_pipeline_allows_custom_output_video_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _write_sequence(tmp_path)
    source = _write_video(tmp_path / "videos" / "clip.mp4")
    output_video = tmp_path / "videos" / "my_custom_tracking.mp4"
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(b"weights")
    deepsort_config = tmp_path / "deepsort.yaml"
    _write_deepsort_config(deepsort_config)
    config = _write_tracking_config(tmp_path, checkpoint, deepsort_config)

    result = run_tracking(
        config,
        overrides={
            "source": source,
            "output_video": output_video,
            "render": True,
            "overwrite": True,
            "max_frames": 2,
        },
        detector=FakeDetector(),
        tracker_adapter_factory=lambda config: FakeAdapter(config),
    )

    sequence = result["summary"]["sequences"][0]
    assert Path(sequence["output_video"]) == output_video
    assert Path(sequence["output_mot"]) == output_video.with_suffix(".txt")
    assert output_video.is_file()
    assert output_video.with_suffix(".txt").is_file()
    assert output_video.with_name("my_custom_tracking.metadata.json").is_file()


def test_tracker_output_uses_project_bbox_model() -> None:
    track = TrackOutput.from_xyxy(1, "seq", 1, BoundingBoxXYXY(0, 0, 10, 20), 0.9)
    assert track.bbox_ltwh.width == 10
