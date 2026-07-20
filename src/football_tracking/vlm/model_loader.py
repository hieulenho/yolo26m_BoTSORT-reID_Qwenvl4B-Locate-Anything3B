"""Shared, quantization-aware Qwen vision-language model loading."""

from __future__ import annotations

import gc
from typing import Any

from football_tracking.vlm.quantization import (
    QuantizationConfigError,
    build_bitsandbytes_config,
    normalize_quantization,
    quantized_device_map,
)


class VlmModelLoadError(RuntimeError):
    """Raised when a local or remote VLM checkpoint cannot be loaded."""


def load_qwen_model(config: Any) -> tuple[Any, Any]:
    """Load the configured Qwen VLM and processor, preferring the local HF cache."""
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoProcessor  # type: ignore[import-not-found]
    except ImportError as exc:
        raise VlmModelLoadError(
            "Missing Qwen dependencies. Install requirements/vlm.txt."
        ) from exc
    model_class = _load_model_class()
    model_id = str(config.model_id)
    try:
        processor = _from_pretrained_with_local_cache(
            AutoProcessor,
            model_id,
            trust_remote_code=True,
        )
        kwargs = model_load_kwargs(
            torch,
            device=str(config.device),
            torch_dtype=str(config.torch_dtype),
            quantization=str(config.quantization),
        )
        model = _from_pretrained_with_local_cache(model_class, model_id, **kwargs)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        raise VlmModelLoadError(
            f"Could not load Qwen model '{model_id}'. Root error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return model, processor


def model_load_kwargs(
    torch_module: Any,
    *,
    device: str,
    torch_dtype: str,
    quantization: str,
) -> dict[str, Any]:
    mode = normalize_quantization(quantization)
    try:
        quantization_config = build_bitsandbytes_config(
            torch_module=torch_module,
            quantization=mode,
            torch_dtype=torch_dtype,
        )
    except QuantizationConfigError as exc:
        raise VlmModelLoadError(str(exc)) from exc
    kwargs: dict[str, Any] = {
        "dtype": resolve_torch_dtype(torch_module, torch_dtype),
        "device_map": resolve_device_map(device),
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = quantized_device_map(device)
    return kwargs


def resolve_torch_dtype(torch_module: Any, value: str) -> Any:
    normalized = str(value).lower()
    if normalized == "auto":
        return "auto"
    aliases = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if normalized not in aliases:
        raise VlmModelLoadError(f"Unsupported torch dtype: {value}")
    return aliases[normalized]


def resolve_device_map(device: str) -> Any:
    normalized = str(device).lower()
    if normalized == "auto":
        return "auto"
    return {"": device}


def first_model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    try:
        for parameter in model.parameters():
            if str(parameter.device) != "meta":
                return parameter.device
    except StopIteration as exc:
        raise VlmModelLoadError("Loaded Qwen model has no parameters.") from exc
    raise VlmModelLoadError("Loaded Qwen model exposes only meta-device parameters.")


def release_model_memory(*objects: Any) -> None:
    """Release model references owned by a short-lived stage and clear CUDA cache."""
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


def _load_model_class() -> Any:
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore[import-not-found]

        return AutoModelForImageTextToText
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore[import-not-found]

            return AutoModelForVision2Seq
        except ImportError as exc:
            raise VlmModelLoadError(
                "Transformers does not provide a VLM auto model class."
            ) from exc


def _from_pretrained_with_local_cache(loader: Any, model_id: str, **kwargs: Any) -> Any:
    try:
        return loader.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception as local_exc:  # noqa: BLE001
        try:
            return loader.from_pretrained(model_id, **kwargs)
        except Exception as remote_exc:  # noqa: BLE001
            raise VlmModelLoadError(
                "Could not load from the local Hugging Face cache or Hub. "
                f"Local error: {type(local_exc).__name__}: {local_exc}; "
                f"remote error: {type(remote_exc).__name__}: {remote_exc}"
            ) from remote_exc


__all__ = [
    "VlmModelLoadError",
    "first_model_device",
    "load_qwen_model",
    "model_load_kwargs",
    "release_model_memory",
    "resolve_device_map",
    "resolve_torch_dtype",
]
