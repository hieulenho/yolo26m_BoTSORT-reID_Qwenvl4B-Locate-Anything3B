"""Incremental builder for candidate semantic memories."""

from __future__ import annotations

from collections.abc import Iterable

from football_tracking.locate_tracking.association.schemas import FrameQueryResolution
from football_tracking.locate_tracking.semantic_memory.aggregator import build_semantic_memory
from football_tracking.locate_tracking.semantic_memory.schemas import (
    SemanticMemory,
    SemanticMemoryConfig,
)


class SemanticMemoryBuilderError(RuntimeError):
    """Raised when a semantic memory builder receives invalid updates."""


class SemanticMemoryBuilder:
    """Small incremental wrapper that rejects duplicate frame updates."""

    def __init__(self, *, query: str, config: SemanticMemoryConfig | None = None) -> None:
        self.query = query
        self.config = config or SemanticMemoryConfig()
        self._frame_resolutions: list[FrameQueryResolution | dict[str, object]] = []
        self._seen_frames: set[int] = set()

    def add_frame_resolution(
        self,
        resolution: FrameQueryResolution | dict[str, object],
    ) -> None:
        frame_index = int(
            resolution.frame_index
            if hasattr(resolution, "frame_index")
            else resolution["frame_index"]
        )
        if frame_index in self._seen_frames:
            raise SemanticMemoryBuilderError(
                f"Duplicate frame update rejected for frame {frame_index}."
            )
        self._seen_frames.add(frame_index)
        self._frame_resolutions.append(resolution)

    def extend(
        self,
        resolutions: Iterable[FrameQueryResolution | dict[str, object]],
    ) -> None:
        for resolution in resolutions:
            self.add_frame_resolution(resolution)

    def build(self, *, sampled_frames: tuple[int, ...] | None = None) -> SemanticMemory:
        return build_semantic_memory(
            query=self.query,
            frame_resolutions=tuple(self._frame_resolutions),
            config=self.config,
            sampled_frames=sampled_frames,
        )
