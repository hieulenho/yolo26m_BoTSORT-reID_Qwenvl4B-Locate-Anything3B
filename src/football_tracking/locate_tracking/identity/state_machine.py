"""Explicit state machine for semantic target identity."""

from __future__ import annotations

from football_tracking.locate_tracking.identity.schemas import IdentityState


class IdentityStateMachineError(ValueError):
    """Raised when an identity state transition is invalid."""


ALLOWED_TRANSITIONS: dict[IdentityState, set[IdentityState]] = {
    "UNRESOLVED": {"ACTIVE", "REJECTED"},
    "ACTIVE": {"UNCERTAIN", "TERMINATED"},
    "UNCERTAIN": {"ACTIVE", "LOST"},
    "LOST": {"ACTIVE", "REACQUIRING", "TERMINATED"},
    "REACQUIRING": {"ACTIVE", "REACQUIRING", "PROBATION", "LOST"},
    "PROBATION": {"ACTIVE", "LOST", "REJECTED"},
    "REJECTED": {"TERMINATED"},
    "TERMINATED": set(),
}


def validate_transition(from_state: IdentityState, to_state: IdentityState) -> None:
    if to_state not in ALLOWED_TRANSITIONS[from_state]:
        raise IdentityStateMachineError(
            f"Invalid identity state transition: {from_state}->{to_state}"
        )
