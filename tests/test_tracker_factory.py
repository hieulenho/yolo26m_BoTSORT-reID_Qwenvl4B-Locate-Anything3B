from __future__ import annotations

from pathlib import Path

import pytest

from football_tracking.data.schemas import BoundingBoxXYXY
from football_tracking.tracking.routed_tracker import (
    RoutedTrackerAdapter,
    RoutedTrackerConfigError,
    load_routed_tracker_config,
)
from football_tracking.tracking.schemas import TrackerDetection, TrackOutput
from football_tracking.tracking.sort_adapter import SortTrackerAdapter
from football_tracking.tracking.tracker_factory import TrackerFactoryError, create_tracker
from football_tracking.tracking.ultralytics_adapter import UltralyticsTrackerAdapter


def test_tracker_factory_creates_sort(tmp_path) -> None:
    config = tmp_path / "sort.yaml"
    config.write_text(
        """
tracker:
  name: sort
  max_age: 5
  min_hits: 1
  iou_threshold: 0.3
output:
  confirmed_only: true
  require_recent_update: true
  max_time_since_update_for_output: 1
  output_predicted_tracks_without_detection: false
""".strip(),
        encoding="utf-8",
    )

    tracker = create_tracker("sort", config)

    assert isinstance(tracker, SortTrackerAdapter)


def test_tracker_factory_rejects_unknown_tracker(tmp_path) -> None:
    with pytest.raises(TrackerFactoryError):
        create_tracker("unknown", tmp_path / "missing.yaml")


def test_tracker_factory_creates_botsort_without_importing_runtime(tmp_path) -> None:
    config = tmp_path / "botsort.yaml"
    config.write_text(
        """
tracker:
  name: botsort_reid
  tracker_type: botsort
  track_high_thresh: 0.25
  track_low_thresh: 0.1
  new_track_thresh: 0.25
  track_buffer: 30
  match_thresh: 0.8
  fuse_score: true
  gmc_method: sparseOptFlow
  proximity_thresh: 0.5
  appearance_thresh: 0.25
  with_reid: true
  model: auto
output:
  confirmed_only: true
  require_recent_update: true
  max_time_since_update_for_output: 0
""".strip(),
        encoding="utf-8",
    )

    tracker = create_tracker("botsort_reid", config)

    assert isinstance(tracker, UltralyticsTrackerAdapter)
    assert tracker.get_runtime_config()["tracker_type"] == "botsort"
    assert tracker.get_runtime_config()["model"] == "yolo26n-cls.pt"


@pytest.mark.parametrize("tracker_name", ["fasttrack", "tracktrack"])
def test_tracker_factory_creates_modern_ultralytics_trackers(
    tmp_path, tracker_name: str
) -> None:
    config = tmp_path / f"{tracker_name}.yaml"
    config.write_text(
        f"""
tracker:
  name: {tracker_name}
  tracker_type: {tracker_name}
  track_high_thresh: 0.3
  track_low_thresh: 0.1
  new_track_thresh: 0.35
  track_buffer: 30
  match_thresh: 0.8
  with_reid: false
  model: auto
output:
  confirmed_only: true
  require_recent_update: true
  max_time_since_update_for_output: 0
""".strip(),
        encoding="utf-8",
    )

    tracker = create_tracker(tracker_name, config)

    assert isinstance(tracker, UltralyticsTrackerAdapter)
    assert tracker.get_runtime_config()["tracker_type"] == tracker_name


def test_tracker_factory_loads_adaptive_routing_without_runtime_imports(
    tmp_path: Path,
) -> None:
    delegate = tmp_path / "delegate.yaml"
    delegate.write_text(
        "tracker:\n  name: ocsort\n  tracker_type: ocsort\n",
        encoding="utf-8",
    )
    config = tmp_path / "routed.yaml"
    config.write_text(
        f"""
tracker:
  name: adaptive_routed
  default:
    name: ocsort
    config: {delegate.as_posix()}
  routes:
    - route_name: people
      class_ids: [0]
      name: ocsort
      config: {delegate.as_posix()}
""".strip(),
        encoding="utf-8",
    )

    tracker = create_tracker("adaptive_routed", config)

    assert isinstance(tracker, RoutedTrackerAdapter)
    assert tracker.get_runtime_config()["routes"][0]["class_ids"] == [0]


def test_routed_tracker_rejects_default_route_name_collision(tmp_path: Path) -> None:
    delegate = tmp_path / "delegate.yaml"
    delegate.write_text("tracker: {}", encoding="utf-8")
    config = tmp_path / "routed.yaml"
    config.write_text(
        f"""
tracker:
  default:
    route_name: shared
    name: ocsort
    config: {delegate.as_posix()}
  routes:
    - route_name: shared
      class_ids: [0]
      name: ocsort
      config: {delegate.as_posix()}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(RoutedTrackerConfigError, match="must not be reused"):
        load_routed_tracker_config(config)


def test_routed_tracker_maintains_global_ids_across_delegates(tmp_path: Path) -> None:
    delegate_config = tmp_path / "delegate.yaml"
    delegate_config.write_text("tracker: {}", encoding="utf-8")
    config_path = tmp_path / "routed.yaml"
    config_path.write_text(
        f"""
tracker:
  name: adaptive_routed
  default:
    name: ocsort
    config: {delegate_config.as_posix()}
  routes:
    - route_name: people
      class_ids: [0]
      name: ocsort
      config: {delegate_config.as_posix()}
    - route_name: vehicles
      class_ids: [2]
      name: bytetrack
      config: {delegate_config.as_posix()}
""".strip(),
        encoding="utf-8",
    )

    class FakeDelegate:
        def __init__(self, tracker_name: str) -> None:
            self.tracker_name = tracker_name

        def reset(self) -> None:
            return None

        def close(self) -> None:
            return None

        def update(self, **kwargs):
            detections = kwargs["detections"]
            return [
                TrackOutput.from_xyxy(
                    frame_index=kwargs["frame_index"],
                    sequence_name=kwargs["sequence_name"],
                    track_id=1,
                    bbox_xyxy=detection.bbox_xyxy,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    metadata={"tracker": self.tracker_name},
                )
                for detection in detections
            ]

    tracker = RoutedTrackerAdapter(
        load_routed_tracker_config(config_path),
        tracker_factory=lambda name, _config, _device: FakeDelegate(name),
    )
    detections = [
        TrackerDetection.from_xyxy(
            frame_index=1,
            sequence_name="sequence",
            bbox_xyxy=BoundingBoxXYXY(0, 0, 10, 20),
            confidence=0.9,
            class_id=0,
            class_name="person",
        ),
        TrackerDetection.from_xyxy(
            frame_index=1,
            sequence_name="sequence",
            bbox_xyxy=BoundingBoxXYXY(20, 0, 40, 20),
            confidence=0.8,
            class_id=2,
            class_name="car",
        ),
    ]

    outputs = tracker.update(
        frame_index=1,
        sequence_name="sequence",
        detections=detections,
    )

    assert [output.track_id for output in outputs] == [1, 2]
    assert {output.metadata["tracker_route"] for output in outputs} == {
        "people",
        "vehicles",
    }


def test_routed_tracker_stabilizes_transient_class_changes(tmp_path: Path) -> None:
    delegate_config = tmp_path / "delegate.yaml"
    delegate_config.write_text("tracker: {}", encoding="utf-8")
    config_path = tmp_path / "routed.yaml"
    config_path.write_text(
        f"""
tracker:
  default:
    name: ocsort
    config: {delegate_config.as_posix()}
  routes: []
""".strip(),
        encoding="utf-8",
    )

    class StableIdentityDelegate:
        def reset(self) -> None:
            return None

        def close(self) -> None:
            return None

        def update(self, **kwargs):
            detection = kwargs["detections"][0]
            return [
                TrackOutput.from_xyxy(
                    frame_index=kwargs["frame_index"],
                    sequence_name=kwargs["sequence_name"],
                    track_id=7,
                    bbox_xyxy=detection.bbox_xyxy,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                )
            ]

    tracker = RoutedTrackerAdapter(
        load_routed_tracker_config(config_path),
        tracker_factory=lambda _name, _config, _device: StableIdentityDelegate(),
    )
    observed: list[TrackOutput] = []
    labels = [(2, "car"), (2, "car"), (7, "truck"), (2, "car")]
    for frame_index, (class_id, class_name) in enumerate(labels, start=1):
        detection = TrackerDetection.from_xyxy(
            frame_index=frame_index,
            sequence_name="traffic",
            bbox_xyxy=BoundingBoxXYXY(frame_index, 0, frame_index + 20, 20),
            confidence=0.9,
            class_id=class_id,
            class_name=class_name,
        )
        observed.extend(
            tracker.update(
                frame_index=frame_index,
                sequence_name="traffic",
                detections=[detection],
            )
        )

    assert {row.track_id for row in observed} == {1}
    assert [row.class_name for row in observed] == ["car", "car", "car", "car"]
    assert observed[2].metadata["raw_class_name"] == "truck"
    diagnostics = tracker.get_diagnostics()
    assert diagnostics["raw_class_switches"] == 2
    assert diagnostics["stable_class_switches"] == 0
    assert diagnostics["suppressed_class_switches"] >= 1
    assert diagnostics["suppressed_class_mismatch_frames"] >= 1


def test_routed_tracker_scene_reset_preserves_global_id_uniqueness(
    tmp_path: Path,
) -> None:
    delegate_config = tmp_path / "delegate.yaml"
    delegate_config.write_text("tracker: {}", encoding="utf-8")
    config_path = tmp_path / "routed.yaml"
    config_path.write_text(
        f"""
tracker:
  default:
    name: ocsort
    config: {delegate_config.as_posix()}
  routes: []
""".strip(),
        encoding="utf-8",
    )

    class RestartingDelegate:
        def reset(self) -> None:
            return None

        def close(self) -> None:
            return None

        def update(self, **kwargs):
            detection = kwargs["detections"][0]
            return [
                TrackOutput.from_xyxy(
                    frame_index=kwargs["frame_index"],
                    sequence_name=kwargs["sequence_name"],
                    track_id=1,
                    bbox_xyxy=detection.bbox_xyxy,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                )
            ]

    tracker = RoutedTrackerAdapter(
        load_routed_tracker_config(config_path),
        tracker_factory=lambda _name, _config, _device: RestartingDelegate(),
    )
    detection = TrackerDetection.from_xyxy(
        frame_index=1,
        sequence_name="cuts",
        bbox_xyxy=BoundingBoxXYXY(0, 0, 20, 20),
        confidence=0.9,
        class_id=0,
        class_name="person",
    )
    before = tracker.update(1, "cuts", [detection])
    tracker.reset_scene()
    after = tracker.update(2, "cuts", [detection])

    assert before[0].track_id == 1
    assert after[0].track_id == 2
    assert tracker.get_diagnostics()["scene_reset_count"] == 1


def test_routed_tracker_can_correct_a_persistently_wrong_initial_class(
    tmp_path: Path,
) -> None:
    delegate_config = tmp_path / "delegate.yaml"
    delegate_config.write_text("tracker: {}", encoding="utf-8")
    config_path = tmp_path / "routed.yaml"
    config_path.write_text(
        f"""
tracker:
  default:
    name: ocsort
    config: {delegate_config.as_posix()}
  routes: []
""".strip(),
        encoding="utf-8",
    )

    class StableIdentityDelegate:
        def reset(self) -> None:
            return None

        def close(self) -> None:
            return None

        def update(self, **kwargs):
            detection = kwargs["detections"][0]
            return [
                TrackOutput.from_xyxy(
                    frame_index=kwargs["frame_index"],
                    sequence_name=kwargs["sequence_name"],
                    track_id=5,
                    bbox_xyxy=detection.bbox_xyxy,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                )
            ]

    tracker = RoutedTrackerAdapter(
        load_routed_tracker_config(config_path),
        tracker_factory=lambda _name, _config, _device: StableIdentityDelegate(),
    )
    labels = [(2, "car"), *([(7, "truck")] * 12)]
    outputs = []
    for frame_index, (class_id, class_name) in enumerate(labels, start=1):
        detection = TrackerDetection.from_xyxy(
            frame_index=frame_index,
            sequence_name="traffic",
            bbox_xyxy=BoundingBoxXYXY(0, 0, 20, 20),
            confidence=0.9,
            class_id=class_id,
            class_name=class_name,
        )
        outputs.extend(
            tracker.update(
                frame_index=frame_index,
                sequence_name="traffic",
                detections=[detection],
            )
        )

    assert outputs[0].class_name == "car"
    assert outputs[-1].class_name == "truck"
    assert tracker.get_diagnostics()["stable_class_switches"] == 1
