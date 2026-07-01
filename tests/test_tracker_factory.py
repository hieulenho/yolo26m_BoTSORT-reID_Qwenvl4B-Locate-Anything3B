from __future__ import annotations

import pytest

from football_tracking.tracking.sort_adapter import SortTrackerAdapter
from football_tracking.tracking.tracker_factory import TrackerFactoryError, create_tracker


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
