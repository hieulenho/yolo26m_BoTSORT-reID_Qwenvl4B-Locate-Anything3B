"""LocateAnything backend for standalone image phrase grounding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.grounding.backend import BackendGroundingResponse

DEFAULT_LOCATEANYTHING_MODEL_ID = "nvidia/LocateAnything-3B"
DEFAULT_LOCATEANYTHING_PROMPT_TEMPLATE = (
    "Locate a single instance that matches the following description: {query}."
)


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
        prompt_template: str = DEFAULT_LOCATEANYTHING_PROMPT_TEMPLATE,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_template = prompt_template
        self.trust_remote_code = bool(trust_remote_code)
        self._worker: _LocateAnythingWorker | None = None

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
        if self._worker is not None:
            return self._worker
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModel,
                AutoProcessor,
                AutoTokenizer,
            )
        except Exception as exc:  # noqa: BLE001
            raise LocateAnythingBackendError(
                "LocateAnything dependencies are not installed. Install the optional "
                "LocateAnything dependencies before using the real grounding backend."
            ) from exc
        dtype = _resolve_dtype(self.torch_dtype)
        model_kwargs: dict[str, object] = {
            "trust_remote_code": self.trust_remote_code,
        }
        if dtype != "auto":
            model_kwargs["dtype"] = dtype
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            try:
                model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            except TypeError:
                if "dtype" in model_kwargs:
                    model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
                model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            model = _move_model(model, self.device)
            self._worker = _LocateAnythingWorker(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                device=self.device,
                dtype=dtype,
                max_new_tokens=self.max_new_tokens,
                prompt_template=self.prompt_template,
            )
        except Exception as exc:  # noqa: BLE001
            raise LocateAnythingBackendError(
                f"Failed to load LocateAnything model {self.model_id}: {exc}"
            ) from exc
        return self._worker

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
            try:
                output = model.ground_single(image.convert("RGB"), query)
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


def _move_model(model: Any, device: str) -> Any:
    if not device or device.lower() == "auto":
        return model.eval()
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


class _LocateAnythingWorker:
    """Direct LocateAnything worker using the model's custom AutoModel mapping."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        processor: Any,
        device: str,
        dtype: Any,
        max_new_tokens: int,
        prompt_template: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_template = prompt_template

    def ground_single(self, image: Any, query: str) -> str:
        prompt = self.prompt_template.format(query=query)
        result = self.predict(image, prompt)
        return _extract_generated_text(result.get("answer", result))

    def predict(self, image: Any, question: str) -> dict[str, Any]:
        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise LocateAnythingBackendError("PyTorch is required for LocateAnything.") from exc
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(_input_device(self.device))
        pixel_values = inputs["pixel_values"]
        if self.dtype != "auto":
            pixel_values = pixel_values.to(self.dtype)
        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                generation_mode="hybrid",
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )
        result: dict[str, Any] = {
            "answer": response[0] if isinstance(response, tuple) else response,
        }
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result


def _input_device(device: str) -> str:
    if not device or device.lower() == "auto":
        return "cuda"
    return device
