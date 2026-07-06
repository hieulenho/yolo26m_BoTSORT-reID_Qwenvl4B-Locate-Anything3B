from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from football_tracking.locate_tracking.grounding.backend import MockGroundingBackend
from football_tracking.locate_tracking.grounding.cache import GroundingCache
from football_tracking.locate_tracking.grounding.service import GroundingService


def _image(path: Path) -> Path:
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_service_request_backend_result(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "frame.jpg")
    backend = MockGroundingBackend(
        {"goalkeeper": "<ref>goalkeeper</ref><box><100><200><500><800></box>"}
    )
    service = GroundingService(backend=backend, cache=None, overwrite=True)

    result = service.ground_image(image_path=image_path, query="goalkeeper")

    assert backend.call_count == 1
    assert len(result.boxes) == 1
    assert result.boxes[0].bbox_xyxy == (10.0, 10.0, 50.0, 40.0)
    assert result.cache_hit is False


def test_service_cache_hit_on_second_identical_call(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "frame.jpg")
    backend = MockGroundingBackend(
        {"goalkeeper": "<ref>goalkeeper</ref><box><100><200><500><800></box>"}
    )
    cache = GroundingCache(tmp_path / "cache")
    service = GroundingService(backend=backend, cache=cache, overwrite=True)

    first = service.ground_image(image_path=image_path, query="goalkeeper")
    second = service.ground_image(image_path=image_path, query="goalkeeper")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert backend.call_count == 1


def test_service_writes_output_json(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "frame.jpg")
    output = tmp_path / "grounding.json"
    backend = MockGroundingBackend({"player": "<box>none</box>"})
    service = GroundingService(backend=backend, cache=None, overwrite=True)

    service.ground_image(image_path=image_path, query="player", output_path=output)

    assert output.is_file()
