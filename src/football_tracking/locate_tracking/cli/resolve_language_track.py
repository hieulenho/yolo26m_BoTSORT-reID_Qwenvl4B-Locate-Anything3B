"""Convenience entrypoint for the multi-frame language-track CLI."""

from __future__ import annotations

from collections.abc import Sequence

from football_tracking.locate_tracking.cli.__main__ import main as namespace_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or ())
    return namespace_main(["resolve-language-track", *args])


if __name__ == "__main__":
    raise SystemExit(main())
