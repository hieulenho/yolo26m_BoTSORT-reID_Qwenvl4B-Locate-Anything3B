from __future__ import annotations

from pathlib import Path

import pytest

from football_tracking.locate_tracking.grounding.schemas import (
    GroundedBox,
    GroundingRequest,
    GroundingResult,
    GroundingRuntimeInfo,
    GroundingSchemaError,
)


def _request() -> GroundingRequest:
    return GroundingRequest(
        image_path=Path("frame.jpg"),
        query="goalkeeper wearing green",
        backend="mock",
        model_id="mock-grounding",
    )


def _box(confidence: float | None = None) -> GroundedBox:
    return GroundedBox(
        label="goalkeeper",
        bbox_xyxy=(10.0, 20.0, 50.0, 90.0),
        normalized_bbox=(100, 200, 500, 900),
        confidence=confidence,
        query="goalkeeper wearing green",
    )


def test_valid_request() -> None:
    request = _request()

    assert request.query == "goalkeeper wearing green"
    assert request.backend == "mock"


def test_invalid_image_dimensions() -> None:
    with pytest.raises(GroundingSchemaError):
        GroundingResult(
            request=_request(),
            image_width=0,
            image_height=100,
            boxes=(),
            raw_response="<box>none</box>",
            runtime_info=GroundingRuntimeInfo(backend="mock", model_id="mock"),
        )


def test_invalid_xyxy_box() -> None:
    with pytest.raises(GroundingSchemaError):
        GroundedBox(
            label="player",
            bbox_xyxy=(50.0, 20.0, 10.0, 90.0),
            normalized_bbox=(500, 200, 100, 900),
            confidence=None,
            query="player",
        )


def test_optional_confidence() -> None:
    assert _box(confidence=None).confidence is None
    assert _box(confidence=0.7).confidence == 0.7


def test_serialization_round_trip() -> None:
    result = GroundingResult(
        request=_request(),
        image_width=100,
        image_height=120,
        boxes=(_box(),),
        raw_response="<ref>goalkeeper</ref><box><100><200><500><900></box>",
        runtime_info=GroundingRuntimeInfo(
            backend="mock",
            model_id="mock-grounding",
            cache_status="miss",
        ),
    )

    restored = GroundingResult.from_dict(result.to_dict())

    assert restored == result

