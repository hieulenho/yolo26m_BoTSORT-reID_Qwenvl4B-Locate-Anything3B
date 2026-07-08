"""Shared model quantization helpers for VLM backends."""

from __future__ import annotations

from typing import Any


SUPPORTED_QUANTIZATION = ("none", "8bit", "4bit")


class QuantizationConfigError(RuntimeError):
    """Raised when a requested model quantization mode is invalid."""


def normalize_quantization(value: str | None) -> str:
    """Return the canonical quantization mode name."""

    normalized = str(value or "none").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "": "none",
        "none": "none",
        "false": "none",
        "off": "none",
        "no": "none",
        "0": "none",
        "int8": "8bit",
        "8": "8bit",
        "8bit": "8bit",
        "bnb8": "8bit",
        "bnb8bit": "8bit",
        "int4": "4bit",
        "4": "4bit",
        "4bit": "4bit",
        "nf4": "4bit",
        "bnb4": "4bit",
        "bnb4bit": "4bit",
    }
    if normalized not in aliases:
        supported = ", ".join(SUPPORTED_QUANTIZATION)
        raise QuantizationConfigError(
            f"Unsupported quantization mode: {value}. Supported modes: {supported}."
        )
    return aliases[normalized]


def build_bitsandbytes_config(
    *,
    torch_module: Any,
    quantization: str | None,
    torch_dtype: str | None,
) -> Any | None:
    """Build a Transformers BitsAndBytesConfig, or None for full precision loading."""

    mode = normalize_quantization(quantization)
    if mode == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise QuantizationConfigError(
            "Transformers BitsAndBytesConfig is required for 4-bit/8-bit "
            "quantization. Install requirements/vlm.txt."
        ) from exc
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    compute_dtype = _bnb_compute_dtype(torch_module, torch_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def quantized_device_map(device: str) -> Any:
    """Use Accelerate device maps for quantized models."""

    normalized = str(device or "auto").strip().lower()
    if normalized in {"auto", "cuda", "gpu"}:
        return "auto"
    return {"": device}


def _bnb_compute_dtype(torch_module: Any, torch_dtype: str | None) -> Any:
    normalized = str(torch_dtype or "auto").strip().lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch_module.float32
    return torch_module.float16


__all__ = [
    "QuantizationConfigError",
    "SUPPORTED_QUANTIZATION",
    "build_bitsandbytes_config",
    "normalize_quantization",
    "quantized_device_map",
]
