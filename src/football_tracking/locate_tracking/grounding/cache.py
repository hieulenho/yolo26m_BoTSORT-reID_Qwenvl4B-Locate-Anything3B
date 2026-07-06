"""Deterministic JSON cache for standalone grounding results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.grounding.schemas import (
    GroundingRequest,
    GroundingResult,
)


class GroundingCacheError(RuntimeError):
    """Raised when grounding cache data cannot be written safely."""


@dataclass(frozen=True)
class CacheLookup:
    result: GroundingResult | None
    cache_hit: bool
    cache_key: str | None
    error: str | None = None


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GroundingCache:
    def __init__(
        self,
        directory: str | Path,
        *,
        enabled: bool = True,
        overwrite: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.enabled = bool(enabled)
        self.overwrite = bool(overwrite)

    def cache_key(self, request: GroundingRequest) -> str:
        payload = {
            "image_sha256": _image_sha256(request.image_path),
            "query": request.query,
            "backend": request.backend,
            "model_id": request.model_id,
            "inference_config": _jsonable(request.inference_config),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def cache_path(self, request: GroundingRequest) -> Path:
        return self.directory / f"{self.cache_key(request)}.json"

    def get(self, request: GroundingRequest) -> CacheLookup:
        if not self.enabled:
            return CacheLookup(result=None, cache_hit=False, cache_key=None)
        key = self.cache_key(request)
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return CacheLookup(result=None, cache_hit=False, cache_key=key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result = GroundingResult.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            return CacheLookup(
                result=None,
                cache_hit=False,
                cache_key=key,
                error=f"Invalid grounding cache entry {path}: {exc}",
            )
        return CacheLookup(result=result.with_cache_hit(key), cache_hit=True, cache_key=key)

    def set(self, result: GroundingResult) -> Path | None:
        if not self.enabled:
            return None
        if result.has_errors:
            raise GroundingCacheError("Refusing to cache grounding result with errors.")
        key = self.cache_key(result.request)
        path = self.directory / f"{key}.json"
        if path.exists() and not self.overwrite:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return path

