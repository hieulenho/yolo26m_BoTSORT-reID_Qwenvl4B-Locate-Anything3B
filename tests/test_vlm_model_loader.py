from __future__ import annotations

import os
from pathlib import Path

from football_tracking.vlm.model_loader import (
    _cached_snapshot_is_complete,
    enable_cached_offline_mode,
)


def _snapshot(root: Path, model_id: str) -> Path:
    path = root / f"models--{model_id.replace('/', '--')}" / "snapshots" / "revision"
    path.mkdir(parents=True)
    return path


def test_complete_cached_snapshot_is_detected(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "Qwen/test-model")
    for name in (
        "config.json",
        "model-00001.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
    ):
        (snapshot / name).write_text("cached", encoding="utf-8")

    assert _cached_snapshot_is_complete("Qwen/test-model", hub_root=tmp_path)


def test_incomplete_cached_snapshot_is_rejected(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, "Qwen/test-model")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    assert not _cached_snapshot_is_complete("Qwen/test-model", hub_root=tmp_path)


def test_network_override_prevents_automatic_offline_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FOOTBALL_TRACKING_ALLOW_HF_NETWORK", "1")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    assert not enable_cached_offline_mode("Qwen/Qwen3-VL-4B-Instruct")
    assert "HF_HUB_OFFLINE" not in os.environ
