"""Small tqdm wrapper for long-running CLI loops."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator


def progress_iter[T](
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = False,
) -> Iterator[T]:
    """Yield an iterable with a terminal progress bar when appropriate."""
    mode = os.environ.get("FOOTBALL_TRACKING_PROGRESS", "auto").strip().lower()
    if mode in {"0", "false", "no", "off", "none"}:
        yield from iterable
        return
    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        yield from iterable
        return
    disable = mode == "auto" and not sys.stderr.isatty()
    yield from tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        disable=disable,
    )
