from __future__ import annotations

from pathlib import Path

from football_tracking.locate_tracking.grounding.cache import GroundingCache
from football_tracking.locate_tracking.grounding.schemas import (
    GroundingRequest,
    GroundingResult,
    GroundingRuntimeInfo,
)


def _write_file(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _request(image_path: Path, query: str = "player", model_id: str = "mock"):
    return GroundingRequest(
        image_path=image_path,
        query=query,
        backend="mock",
        model_id=model_id,
        inference_config={"temperature": 0},
    )


def _result(request: GroundingRequest) -> GroundingResult:
    return GroundingResult(
        request=request,
        image_width=10,
        image_height=10,
        boxes=(),
        raw_response="<box>none</box>",
        runtime_info=GroundingRuntimeInfo(
            backend=request.backend,
            model_id=request.model_id,
            cache_status="miss",
        ),
    )


def test_identical_request_has_identical_cache_key(tmp_path: Path) -> None:
    image = _write_file(tmp_path / "frame.jpg", b"same")
    cache = GroundingCache(tmp_path / "cache")

    assert cache.cache_key(_request(image)) == cache.cache_key(_request(image))


def test_different_query_has_different_key(tmp_path: Path) -> None:
    image = _write_file(tmp_path / "frame.jpg", b"same")
    cache = GroundingCache(tmp_path / "cache")

    assert cache.cache_key(_request(image, "player")) != cache.cache_key(
        _request(image, "goalkeeper")
    )


def test_different_image_content_has_different_key(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _write_file(tmp_path / "a" / "frame.jpg", b"one")
    second = _write_file(tmp_path / "b" / "frame.jpg", b"two")
    cache = GroundingCache(tmp_path / "cache")

    assert cache.cache_key(_request(first)) != cache.cache_key(_request(second))


def test_different_model_has_different_key(tmp_path: Path) -> None:
    image = _write_file(tmp_path / "frame.jpg", b"same")
    cache = GroundingCache(tmp_path / "cache")

    assert cache.cache_key(_request(image, model_id="a")) != cache.cache_key(
        _request(image, model_id="b")
    )


def test_cache_hit_and_miss(tmp_path: Path) -> None:
    image = _write_file(tmp_path / "frame.jpg", b"same")
    request = _request(image)
    cache = GroundingCache(tmp_path / "cache")

    assert not cache.get(request).cache_hit
    cache.set(_result(request))
    lookup = cache.get(request)

    assert lookup.cache_hit
    assert lookup.result is not None
    assert lookup.result.cache_hit


def test_corrupt_cache_file_handling(tmp_path: Path) -> None:
    image = _write_file(tmp_path / "frame.jpg", b"same")
    request = _request(image)
    cache = GroundingCache(tmp_path / "cache")
    path = cache.cache_path(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")

    lookup = cache.get(request)

    assert not lookup.cache_hit
    assert lookup.error is not None
