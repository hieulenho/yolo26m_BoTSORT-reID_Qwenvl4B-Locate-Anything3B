from __future__ import annotations

from football_tracking.tracking.lifecycle import should_emit_track, track_lifecycle_state


class FakeTrack:
    time_since_update = 0

    def __init__(self, state: str) -> None:
        self.state = state

    def is_deleted(self) -> bool:
        return self.state == "deleted"

    def is_tentative(self) -> bool:
        return self.state == "tentative"

    def is_confirmed(self) -> bool:
        return self.state == "confirmed"


def test_lifecycle_maps_confirmed_and_lost() -> None:
    assert track_lifecycle_state(FakeTrack("confirmed")) == "confirmed"
    stale = FakeTrack("confirmed")
    stale.time_since_update = 2
    assert track_lifecycle_state(stale) == "lost"


def test_should_emit_default_only_recent_confirmed_tracks() -> None:
    assert should_emit_track("confirmed", True, True, 0, 1)
    assert not should_emit_track("tentative", True, True, 0, 1)
    assert not should_emit_track("lost", True, True, 2, 1)
