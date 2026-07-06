"""LocateAnything backend for standalone image phrase grounding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.grounding.backend import BackendGroundingResponse

DEFAULT_LOCATEANYTHING_MODEL_ID = "nvidia/LocateAnything-3B"


class LocateAnythingBackendError(RuntimeError):
    """Raised when LocateAnything cannot be loaded or executed."""


def _extract_generated_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("generated_text", "answer", "text", "output", "content"):
            if key in output:
                return _extract_generated_text(output[key])
        return ""
    if isinstance(output, list | tuple):
        pieces = [_extract_generated_text(item) for item in output]
        return "\n".join(piece for piece in pieces if piece)
    return str(output)


def _resolve_dtype(dtype: str) -> Any:
    normalized = str(dtype or "auto").lower()
    if normalized == "auto":
        return "auto"
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise LocateAnythingBackendError(
            "PyTorch is required when LocateAnything torch_dtype is not auto."
        ) from exc
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in aliases:
        raise LocateAnythingBackendError(f"Unsupported LocateAnything dtype: {dtype}")
    return aliases[normalized]


class LocateAnythingBackend:
    """Lazy Hugging Face LocateAnything wrapper.

    The model is not loaded until ``ground`` is called.  Importing this module
    does not require transformers, torch, PIL, CUDA, or model weights.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_LOCATEANYTHING_MODEL_ID,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 4096,
        prompt_template: str = "Locate the object described by this phrase: {query}",
        trust_remote_code: bool = True,
    ) -> None:
        self._model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_template = prompt_template
        self.trust_remote_code = bool(trust_remote_code)
        self._pipeline: Any | None = None

    @property
    def name(self) -> str:
        return "locate_anything"

    @property
    def model_id(self) -> str:
        return self._model_id

    def inference_config(self) -> dict[str, object]:
        return {
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "max_new_tokens": self.max_new_tokens,
            "prompt_template": self.prompt_template,
            "trust_remote_code": self.trust_remote_code,
        }

    def load_model(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise LocateAnythingBackendError(
                "LocateAnything dependencies are not installed. Install the optional "
                "LocateAnything dependencies before using the real grounding backend."
            ) from exc
        kwargs: dict[str, object] = {
            "model": self.model_id,
            "trust_remote_code": self.trust_remote_code,
            "dtype": _resolve_dtype(self.torch_dtype),
        }
        if self.device and self.device.lower() != "cpu":
            kwargs["device_map"] = "auto" if self.device.lower() == "cuda" else self.device
        try:
            self._pipeline = pipeline("image-text-to-text", **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LocateAnythingBackendError(
                f"Failed to load LocateAnything model {self.model_id}: {exc}"
            ) from exc
        return self._pipeline

    def ground(self, image_path: Path, query: str) -> BackendGroundingResponse:
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise LocateAnythingBackendError(
                "Pillow is required to load images for LocateAnything grounding."
            ) from exc
        model = self.load_model()
        prompt = self.prompt_template.format(query=query)
        with Image.open(image_path) as image:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.convert("RGB")},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                output = model(
                    text=messages,
                    max_new_tokens=self.max_new_tokens,
                    return_full_text=False,
                )
            except TypeError:
                output = model(messages, max_new_tokens=self.max_new_tokens)
            except Exception as exc:  # noqa: BLE001
                raise LocateAnythingBackendError(
                    f"LocateAnything grounding failed: {exc}"
                ) from exc
        return BackendGroundingResponse(
            raw_response=_extract_generated_text(output),
            metadata={
                "prompt": prompt,
            },
        )
