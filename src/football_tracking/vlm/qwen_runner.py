"""Optional Qwen-VL execution for prepared tracking context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.vlm.config import VlmTrackingConfig
from football_tracking.vlm.model_loader import (
    VlmModelLoadError,
    first_model_device,
    load_qwen_model,
)
from football_tracking.vlm.quantization import normalize_quantization


class QwenRunnerError(RuntimeError):
    """Raised when local Qwen inference cannot run."""


def run_qwen_vlm(
    config: VlmTrackingConfig,
    prompt: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
    except ImportError as exc:
        raise QwenRunnerError(
            "Missing Qwen VLM dependencies. Install them with: "
            "pip install -r requirements/vlm.txt"
        ) from exc

    try:
        model, processor = load_qwen_model(config)
    except VlmModelLoadError as exc:
        raise QwenRunnerError(
            str(exc)
        ) from exc

    messages = [
        {
            "role": "user",
            "content": [
                *[
                    {
                        "type": "image",
                        "image": str(path.resolve()),
                    }
                    for path in image_paths
                ],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
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

    return {
        "status": "ok",
        "model_id": config.model_id,
        "quantization": normalize_quantization(config.quantization),
        "torch_dtype": config.torch_dtype,
        "image_count": len(image_paths),
        "answer": answer,
    }


__all__ = ["QwenRunnerError", "run_qwen_vlm"]
