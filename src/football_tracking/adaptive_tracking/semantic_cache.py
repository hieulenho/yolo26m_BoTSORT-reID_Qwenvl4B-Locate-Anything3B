"""Deterministic semantic discovery cache keyed by video content and model settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from football_tracking.adaptive_tracking.schemas import SceneDiscovery


def video_fingerprint(path: str | Path, *, sample_bytes: int = 1024 * 1024) -> str:
    video = Path(path)
    stat = video.stat()
    digest = hashlib.sha256()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    with video.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes * 2:
            handle.seek(max(stat.st_size // 2 - sample_bytes // 2, 0))
            digest.update(handle.read(sample_bytes))
            handle.seek(max(stat.st_size - sample_bytes, 0))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def discovery_cache_key(
    video_path: str | Path,
    *,
    model_id: str,
    prompt_version: str,
    sampling: dict[str, Any],
) -> str:
    payload = {
        "video": video_fingerprint(video_path),
        "model_id": model_id,
        "prompt_version": prompt_version,
        "sampling": sampling,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class SemanticCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, cache_key: str) -> Path:
        return self.root / cache_key[:2] / f"{cache_key}.json"

    def load(self, cache_key: str) -> SceneDiscovery | None:
        path = self.path_for(cache_key)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return SceneDiscovery.from_dict(data["discovery"])

    def save(self, cache_key: str, discovery: SceneDiscovery) -> Path:
        path = self.path_for(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cache_key": cache_key, "discovery": discovery.to_dict()}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
