from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from football_tracking.vlm.qwen_runner import (
    _job_fingerprint,
    _read_cached_batch,
    _safe_batch_id,
    _write_cached_batch,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_id="model",
        device="cuda",
        torch_dtype="auto",
        quantization="8bit",
        max_new_tokens=192,
        image_min_pixels=32768,
        image_max_pixels=196608,
        temperature=0.0,
        do_sample=False,
    )


def test_qwen_batch_checkpoint_requires_matching_input(tmp_path: Path) -> None:
    image = tmp_path / "crop.jpg"
    image.write_bytes(b"pixels")
    fingerprint = _job_fingerprint(
        _config(),  # type: ignore[arg-type]
        prompt="classify",
        image_paths=[image],
        image_labels=["track 1"],
        batch_id="batch_001",
    )
    cache = tmp_path / "batch_001.json"
    result = {"status": "ok", "answer": "{}"}

    _write_cached_batch(cache, fingerprint, result)

    assert _read_cached_batch(cache, fingerprint) == result
    assert _read_cached_batch(cache, "different") is None


def test_qwen_batch_checkpoint_id_is_filesystem_safe() -> None:
    assert _safe_batch_id("batch 1/2") == "batch_1_2"
