"""Stable color helpers."""

from __future__ import annotations

from hashlib import blake2b


def stable_color(value: int | str) -> tuple[int, int, int]:
    digest = blake2b(str(value).encode("utf-8"), digest_size=3).digest()
    return int(digest[0]), int(digest[1]), int(digest[2])
