"""JSONL persistence for uncertainty events."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.events.schemas import UncertaintyEvent, event_json_line


class EventStoreError(RuntimeError):
    """Raised when uncertainty events cannot be persisted."""


def write_events_jsonl(
    events: tuple[UncertaintyEvent, ...],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise EventStoreError(f"Event output exists and overwrite=false: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(event_json_line(event) for event in events)
    output.write_text(text + ("\n" if text else ""), encoding="utf-8")
    return output


def read_events_jsonl(path: str | Path) -> tuple[UncertaintyEvent, ...]:
    resolved = Path(path)
    if not resolved.is_file():
        raise EventStoreError(f"Event JSONL does not exist: {resolved}")
    events: list[UncertaintyEvent] = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EventStoreError(f"Invalid JSONL at {resolved}:{line_number}: {exc}") from exc
        events.append(UncertaintyEvent.from_dict(data))
    return tuple(events)
