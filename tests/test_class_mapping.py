from pathlib import Path

from football_tracking.data.class_mapping import load_class_mapping, normalize_class_name


def _mapping():
    return load_class_mapping(Path("configs/class_mapping.yaml"))


def test_player_is_kept() -> None:
    result = _mapping().map_class("player")

    assert result.status == "mapped"
    assert result.target_class == "player"
    assert result.target_class_id == 0


def test_goalkeeper_maps_to_player() -> None:
    result = _mapping().map_class("Goalkeeper Team Right")

    assert result.status == "mapped"
    assert result.target_class == "player"


def test_referee_is_ignored() -> None:
    result = _mapping().map_class("assistant-referee")

    assert result.status == "ignored"


def test_unknown_class_warn_and_skip() -> None:
    result = _mapping().map_class("camera operator")

    assert result.status == "unknown"
    assert result.target_class_id is None


def test_normalize_class_name() -> None:
    assert normalize_class_name(" Player Team-Left ") == "player_team_left"
