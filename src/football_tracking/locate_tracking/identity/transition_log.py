"""Append-only transition log for semantic target identities."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.identity.schemas import IdentityStateTransition


class TransitionLogError(RuntimeError):
    """Raised when identity transition logs cannot be used."""


def read_transition_log(path: str | Path) -> tuple[IdentityStateTransition, ...]:
    resolved = Path(path)
    if not resolved.is_file():
        return ()
    transitions: list[IdentityStateTransition] = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TransitionLogError(
                f"Invalid transition JSONL at {resolved}:{line_number}"
            ) from exc
        transitions.append(IdentityStateTransition.from_dict(data))
    return tuple(transitions)


def append_transition(
    transition: IdentityStateTransition,
    path: str | Path,
) -> Path:
    output = Path(path)
    existing = read_transition_log(output)
    if any(item.transition_id == transition.transition_id for item in existing):
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(transition.to_dict(), sort_keys=True, default=str))
        handle.write("\n")
    return output
