"""Optional Qwen-VL execution for prepared tracking context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.vlm.config import VlmTrackingConfig
from football_tracking.vlm.quantization import (
    QuantizationConfigError,
    build_bitsandbytes_config,
    normalize_quantization,
    quantized_device_map,
)


class QwenRunnerError(RuntimeError):
    """Raised when local Qwen inference cannot run."""


def run_qwen_vlm(
    config: VlmTrackingConfig,
    prompt: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
        from transformers import AutoProcessor  # type: ignore[import-not-found]
    except ImportError as exc:
        raise QwenRunnerError(
            "Missing Qwen VLM dependencies. Install them with: "
            "pip install -r requirements/vlm.txt"
        ) from exc

    model_class = _load_transformers_model_class()
    try:
        processor = _from_pretrained_with_local_cache(
            AutoProcessor,
            config.model_id,
            trust_remote_code=True,
        )
        model_kwargs = _model_load_kwargs(torch, config)
        model = _from_pretrained_with_local_cache(model_class, config.model_id, **model_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise QwenRunnerError(
            f"Could not load Qwen model '{config.model_id}'. "
            "Check that the model exists locally or can be downloaded. "
            f"Root error: {type(exc).__name__}: {exc}"
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
        device = _first_model_device(model)
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


def _load_transformers_model_class() -> Any:
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore[import-not-found]

        return AutoModelForImageTextToText
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore[import-not-found]

            return AutoModelForVision2Seq
        except ImportError as exc:
            raise QwenRunnerError(
                "Your transformers version does not provide a VLM auto model class. "
                "Upgrade transformers via requirements/vlm.txt."
            ) from exc


def _from_pretrained_with_local_cache(loader: Any, model_id: str, **kwargs: Any) -> Any:
    try:
        return loader.from_pretrained(
            model_id,
            local_files_only=True,
            **kwargs,
        )
    except Exception as local_exc:  # noqa: BLE001
        try:
            return loader.from_pretrained(model_id, **kwargs)
        except Exception as remote_exc:  # noqa: BLE001
            raise QwenRunnerError(
                "Could not load from local Hugging Face cache or remote Hub. "
                f"Local error: {type(local_exc).__name__}: {local_exc}; "
                f"remote error: {type(remote_exc).__name__}: {remote_exc}"
            ) from remote_exc


def _model_load_kwargs(torch: Any, config: VlmTrackingConfig) -> dict[str, Any]:
    quantization = normalize_quantization(config.quantization)
    try:
        quantization_config = build_bitsandbytes_config(
            torch_module=torch,
            quantization=quantization,
            torch_dtype=config.torch_dtype,
        )
    except QuantizationConfigError as exc:
        raise QwenRunnerError(str(exc)) from exc
    kwargs: dict[str, Any] = {
        "dtype": _torch_dtype(torch, config.torch_dtype),
        "device_map": _device_map(config.device),
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = quantized_device_map(config.device)
    return kwargs


def _torch_dtype(torch: Any, value: str) -> Any:
    normalized = value.lower()
    if normalized == "auto":
        return "auto"
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in aliases:
        raise QwenRunnerError(f"Unsupported torch dtype: {value}")
    return aliases[normalized]


def _device_map(device: str) -> Any:
    normalized = device.lower()
    if normalized == "auto":
        return "auto"
    return {"": device}


def _first_model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise QwenRunnerError("Loaded Qwen model has no parameters.") from exc


__all__ = ["QwenRunnerError", "run_qwen_vlm"]
