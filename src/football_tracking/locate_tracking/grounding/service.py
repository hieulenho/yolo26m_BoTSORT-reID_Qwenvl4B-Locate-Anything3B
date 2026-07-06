"""Service orchestration for standalone image grounding."""

from __future__ import annotations

import json
import time
from pathlib import Path

from football_tracking.locate_tracking.grounding.backend import GroundingBackend
from football_tracking.locate_tracking.grounding.cache import GroundingCache
from football_tracking.locate_tracking.grounding.parser import parse_locate_anything_response
from football_tracking.locate_tracking.grounding.schemas import (
    GroundingRequest,
    GroundingResult,
    GroundingRuntimeInfo,
)


class GroundingServiceError(RuntimeError):
    """Raised when standalone grounding cannot be completed."""


def _read_image_size(image_path: Path) -> tuple[int, int]:
    if not image_path.is_file():
        raise GroundingServiceError(f"Image does not exist: {image_path}")
    try:
        import cv2  # type: ignore[import-not-found]

        image = cv2.imread(str(image_path))
        if image is not None:
            height, width = image.shape[:2]
            return int(width), int(height)
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(image_path) as image:
            width, height = image.size
            return int(width), int(height)
    except Exception as exc:  # noqa: BLE001
        raise GroundingServiceError(f"Could not read image dimensions: {image_path}") from exc


def save_grounding_result(
    result: GroundingResult,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise GroundingServiceError(f"Grounding output exists and overwrite=false: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


class GroundingService:
    def __init__(
        self,
        *,
        backend: GroundingBackend,
        cache: GroundingCache | None = None,
        overwrite: bool = False,
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.overwrite = bool(overwrite)

    def _request(self, image_path: Path, query: str) -> GroundingRequest:
        return GroundingRequest(
            image_path=image_path,
            query=query,
            backend=self.backend.name,
            model_id=self.backend.model_id,
            inference_config=self.backend.inference_config(),
        )

    def ground_image(
        self,
        *,
        image_path: str | Path,
        query: str,
        output_path: str | Path | None = None,
        overwrite: bool | None = None,
    ) -> GroundingResult:
        path = Path(image_path).resolve()
        image_width, image_height = _read_image_size(path)
        request = self._request(path, query)
        cache_warning: str | None = None
        if self.cache is not None:
            lookup = self.cache.get(request)
            if lookup.cache_hit and lookup.result is not None:
                result = lookup.result
                if output_path is not None:
                    save_grounding_result(
                        result,
                        output_path,
                        overwrite=self.overwrite if overwrite is None else overwrite,
                    )
                return result
            cache_warning = lookup.error

        started = time.perf_counter()
        backend_response = self.backend.ground(path, query)
        latency = time.perf_counter() - started
        parsed = parse_locate_anything_response(
            backend_response.raw_response,
            query=query,
            image_width=image_width,
            image_height=image_height,
        )
        cache_key = (
            self.cache.cache_key(request)
            if self.cache is not None and self.cache.enabled
            else None
        )
        warnings = (cache_warning,) if cache_warning else ()
        runtime = GroundingRuntimeInfo(
            backend=self.backend.name,
            model_id=self.backend.model_id,
            latency_seconds=latency,
            cache_key=cache_key,
            cache_status="miss" if self.cache is not None and self.cache.enabled else "disabled",
            warnings=warnings,
            errors=parsed.errors,
            metadata=backend_response.metadata,
        )
        result = GroundingResult(
            request=request,
            image_width=image_width,
            image_height=image_height,
            boxes=parsed.boxes,
            raw_response=backend_response.raw_response,
            runtime_info=runtime,
            cache_hit=False,
        )
        if self.cache is not None and not result.has_errors:
            self.cache.set(result)
        if output_path is not None:
            save_grounding_result(
                result,
                output_path,
                overwrite=self.overwrite if overwrite is None else overwrite,
            )
        return result
