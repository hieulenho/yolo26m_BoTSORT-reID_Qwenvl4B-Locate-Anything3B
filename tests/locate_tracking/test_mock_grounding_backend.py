from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.grounding.backend import MockGroundingBackend


def test_mock_backend_returns_predefined_response() -> None:
    backend = MockGroundingBackend({"goalkeeper": "<box><1><2><3><4></box>"})

    response = backend.ground(Path("frame.jpg"), "goalkeeper")

    assert backend.name == "mock"
    assert backend.model_id == "mock-grounding"
    assert response.raw_response == "<box><1><2><3><4></box>"
    assert backend.call_count == 1


def test_mock_backend_returns_default_response() -> None:
    backend = MockGroundingBackend(default_response="<box>none</box>")

    response = backend.ground(Path("frame.jpg"), "unknown")

    assert response.raw_response == "<box>none</box>"

