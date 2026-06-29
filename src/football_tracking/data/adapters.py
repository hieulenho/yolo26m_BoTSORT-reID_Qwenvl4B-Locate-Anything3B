"""Dataset adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from football_tracking.data.schemas import FrameAnnotation, SequenceCandidate, SequenceInfo


class DatasetAdapterError(RuntimeError):
    """Raised when an adapter cannot load a dataset layout."""


class DatasetAdapter(ABC):
    """Interface for raw tracking dataset adapters."""

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Return whether this adapter recognizes the dataset layout."""

    @abstractmethod
    def discover_sequences(self, path: Path) -> list[SequenceCandidate]:
        """Discover sequence candidates below a raw dataset root."""

    @abstractmethod
    def load_sequence(self, path: Path) -> SequenceInfo:
        """Load a single sequence directory into the internal data model."""

    @abstractmethod
    def load_annotations(self, path: Path) -> list[FrameAnnotation]:
        """Load frame annotations for a single sequence directory."""
