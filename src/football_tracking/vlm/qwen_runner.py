"""Optional Qwen-VL execution for prepared tracking context."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from football_tracking.vlm.config import VlmTrackingConfig
from football_tracking.vlm.model_loader import (
    VlmModelLoadError,
    first_model_device,
    load_qwen_model,
    release_model_memory,
)
from football_tracking.vlm.quantization import normalize_quantization


class QwenRunnerError(RuntimeError):
    """Raised when local Qwen inference cannot run."""


class QwenVlmBatchSession:
    """Keep one Qwen model loaded while processing several bounded job batches."""

    def __init__(self, config: VlmTrackingConfig) -> None:
        self.config = config
        self.model: Any | None = None
        self.processor: Any | None = None
        self.process_vision_info: Any | None = None
        self.model_load_seconds = 0.0
        self.call_count = 0

    def __enter__(self) -> QwenVlmBatchSession:
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
        except ImportError as exc:
            raise QwenRunnerError(
                "Missing Qwen VLM dependencies. Install them with: "
                "pip install -r requirements/vlm.txt"
            ) from exc

        load_started = time.perf_counter()
        try:
            self.model, self.processor = load_qwen_model(self.config)
        except VlmModelLoadError as exc:
            raise QwenRunnerError(str(exc)) from exc
        self.process_vision_info = process_vision_info
        self.model_load_seconds = time.perf_counter() - load_started
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.model is None and self.processor is None:
            return
        release_model_memory(self.model, self.processor)
        self.model = None
        self.processor = None
        self.process_vision_info = None

    def run(
        self,
        config: VlmTrackingConfig,
        jobs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not jobs:
            raise QwenRunnerError("At least one Qwen inference batch is required.")
        if self.model is None or self.processor is None or self.process_vision_info is None:
            raise QwenRunnerError("Qwen session is not open.")
        if _model_signature(config) != _model_signature(self.config):
            raise QwenRunnerError("Qwen session configuration changed while the model was loaded.")

        batches: list[dict[str, Any]] = []
        inference_started = time.perf_counter()
        _reset_peak_cuda_memory()
        for index, job in enumerate(jobs, start=1):
            prompt = str(job.get("prompt", ""))
            image_paths = [Path(path) for path in job.get("image_paths", [])]
            image_labels = [str(label) for label in job.get("image_labels", [])]
            if not prompt.strip():
                raise QwenRunnerError(f"Qwen batch {index} has an empty prompt.")
            if image_labels and len(image_labels) != len(image_paths):
                raise QwenRunnerError(
                    f"Qwen batch {index} image_labels must match image_paths."
                )
            batches.append(
                _run_loaded_qwen(
                    config,
                    model=self.model,
                    processor=self.processor,
                    process_vision_info=self.process_vision_info,
                    prompt=prompt,
                    image_paths=image_paths,
                    image_labels=image_labels,
                    batch_id=str(job.get("batch_id") or f"batch_{index:03d}"),
                )
            )
        self.call_count += 1
        return {
            "status": "ok",
            "model_id": config.model_id,
            "quantization": normalize_quantization(config.quantization),
            "torch_dtype": config.torch_dtype,
            "batch_count": len(batches),
            "image_count": sum(int(row["image_count"]) for row in batches),
            "timing": {
                "model_load_seconds": self.model_load_seconds if self.call_count == 1 else 0.0,
                "inference_seconds": time.perf_counter() - inference_started,
                "session_call_index": self.call_count,
            },
            "cuda_memory": _peak_cuda_memory(),
            "batches": batches,
        }


def run_qwen_vlm(
    config: VlmTrackingConfig,
    prompt: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    result = run_qwen_vlm_batches(
        config,
        [{"batch_id": "batch_001", "prompt": prompt, "image_paths": image_paths}],
    )
    return result["batches"][0]


def run_qwen_vlm_batches(
    config: VlmTrackingConfig,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Load Qwen once and execute several bounded image batches."""
    with QwenVlmBatchSession(config) as session:
        return session.run(config, jobs)


def _model_signature(config: VlmTrackingConfig) -> tuple[str, str, str, str]:
    return (
        config.model_id,
        config.device,
        config.torch_dtype,
        normalize_quantization(config.quantization),
    )


def _run_loaded_qwen(
    config: VlmTrackingConfig,
    *,
    model: Any,
    processor: Any,
    process_vision_info: Any,
    prompt: str,
    image_paths: list[Path],
    image_labels: list[str],
    batch_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    content = _build_user_content(
        prompt,
        image_paths,
        image_labels,
        image_min_pixels=config.image_min_pixels,
        image_max_pixels=config.image_max_pixels,
    )
    messages = [{"role": "user", "content": content}]

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            do_resize=False,
            return_tensors="pt",
        )
        device = first_model_device(model)
        inputs = inputs.to(device)
        generate_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        }
        if config.do_sample:
            generate_kwargs["temperature"] = config.temperature
        import torch  # type: ignore[import-not-found]

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **generate_kwargs)
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        answer = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    except Exception as exc:  # noqa: BLE001
        raise QwenRunnerError(
            f"Qwen generation failed. Root error: {type(exc).__name__}: {exc}"
        ) from exc
    del inputs, generated_ids, trimmed_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "ok",
        "batch_id": batch_id,
        "model_id": config.model_id,
        "quantization": normalize_quantization(config.quantization),
        "torch_dtype": config.torch_dtype,
        "image_count": len(image_paths),
        "image_labels": image_labels,
        "inference_seconds": time.perf_counter() - started,
        "answer": answer,
    }


def _build_user_content(
    prompt: str,
    image_paths: list[Path],
    image_labels: list[str],
    *,
    image_min_pixels: int = 64 * 32 * 32,
    image_max_pixels: int = 512 * 32 * 32,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, path in enumerate(image_paths):
        label = image_labels[index] if image_labels else f"Input image {index + 1}."
        content.extend(
            [
                {"type": "text", "text": label},
                {
                    "type": "image",
                    "image": str(path.resolve()),
                    "min_pixels": image_min_pixels,
                    "max_pixels": image_max_pixels,
                },
            ]
        )
    content.append({"type": "text", "text": prompt})
    return content


def _reset_peak_cuda_memory() -> None:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def _peak_cuda_memory() -> dict[str, int | None]:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
    except Exception:  # noqa: BLE001
        pass
    return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}


__all__ = [
    "QwenRunnerError",
    "QwenVlmBatchSession",
    "run_qwen_vlm",
    "run_qwen_vlm_batches",
]
