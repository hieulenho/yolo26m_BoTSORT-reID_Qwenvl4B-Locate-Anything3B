"""Backend abstractions for standalone grounding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BackendGroundingResponse:
    raw_response: str
    metadata: dict[str, object] = field(default_factory=dict)


class GroundingBackend(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def model_id(self) -> str:
        ...

    def inference_config(self) -> dict[str, object]:
        ...

    def ground(self, image_path: Path, query: str) -> BackendGroundingResponse:
        ...


class MockGroundingBackend:
    """Deterministic backend for tests and local plumbing checks."""

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        *,
        default_response: str = "<box>none</box>",
        model_id: str = "mock-grounding",
    ) -> None:
        self._responses = dict(responses or {})
        self._default_response = default_response
        self._model_id = model_id
        self.call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return self._model_id

    def inference_config(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "response_count": len(self._responses),
            "default_response": self._default_response,
        }

    def ground(self, image_path: Path, query: str) -> BackendGroundingResponse:
        self.call_count += 1
        return BackendGroundingResponse(
            raw_response=self._responses.get(query, self._default_response),
            metadata={
                "image_path": str(image_path),
                "mock": True,
            },
        )

