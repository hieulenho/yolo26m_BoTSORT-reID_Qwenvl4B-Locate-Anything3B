from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from football_tracking.vlm import qwen_runner


def test_qwen_batch_session_loads_once_for_multiple_calls(monkeypatch) -> None:
    calls = {"load": 0, "release": 0}
    model = object()
    processor = object()

    def fake_load(_config):
        calls["load"] += 1
        return model, processor

    def fake_release(loaded_model, loaded_processor):
        assert loaded_model is model
        assert loaded_processor is processor
        calls["release"] += 1

    def fake_run(_config, **kwargs):
        return {
            "batch_id": kwargs["batch_id"],
            "image_count": len(kwargs["image_paths"]),
            "answer": "{}",
        }

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        types.SimpleNamespace(process_vision_info=lambda _messages: ([], [])),
    )
    monkeypatch.setattr(qwen_runner, "load_qwen_model", fake_load)
    monkeypatch.setattr(qwen_runner, "release_model_memory", fake_release)
    monkeypatch.setattr(qwen_runner, "_run_loaded_qwen", fake_run)
    config = SimpleNamespace(
        model_id="Qwen/test",
        device="cpu",
        torch_dtype="float32",
        quantization="none",
    )
    job = {"batch_id": "event", "prompt": "classify", "image_paths": []}

    with qwen_runner.QwenVlmBatchSession(config) as session:
        first = session.run(config, [job])
        second = session.run(config, [job])

    assert calls == {"load": 1, "release": 1}
    assert first["timing"]["session_call_index"] == 1
    assert second["timing"]["session_call_index"] == 2
    assert second["timing"]["model_load_seconds"] == 0.0
