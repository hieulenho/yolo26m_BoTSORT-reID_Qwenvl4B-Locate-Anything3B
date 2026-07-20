from __future__ import annotations

import pytest

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
