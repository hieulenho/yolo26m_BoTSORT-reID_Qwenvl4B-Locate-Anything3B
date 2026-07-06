"""Schemas for deterministic frame sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SamplingMode = Literal["uniform", "explicit"]
SelectionReason = Literal["uniform_temporal_sample", "explicit_user_frame"]


class FrameSamplingError(ValueError):
    """Raised when a frame sampling request is invalid."""


@dataclass(frozen=True)
class SelectedFrame:
    frame_index: int
    selection_order: int
    selection_reason: SelectionReason

    def __post_init__(self) -> None:
        if int(self.frame_index) < 1:
            raise FrameSamplingError("selected frame_index must be >= 1.")
        if int(self.selection_order) < 1:
            raise FrameSamplingError("selection_order must be >= 1.")
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "selection_order", int(self.selection_order))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "selection_order": self.selection_order,
            "selection_reason": self.selection_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelectedFrame:
        return cls(
            frame_index=int(data["frame_index"]),
            selection_order=int(data["selection_order"]),
            selection_reason=str(data["selection_reason"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class FrameSamplingRequest:
    total_frames: int
    max_frames: int = 5
    start_frame: int = 1
    end_frame: int | None = None
    explicit_frames: tuple[int, ...] = ()
    mode: SamplingMode = "uniform"

    def __post_init__(self) -> None:
        total_frames = int(self.total_frames)
        max_frames = int(self.max_frames)
        start_frame = int(self.start_frame)
        end_frame = int(self.end_frame) if self.end_frame is not None else total_frames
        explicit_frames = tuple(int(item) for item in self.explicit_frames)
        if total_frames < 1:
            raise FrameSamplingError("total_frames must be >= 1.")
        if max_frames < 1:
            raise FrameSamplingError("max_frames must be >= 1.")
        if start_frame < 1:
            raise FrameSamplingError("start_frame must be >= 1.")
        if end_frame < 1:
            raise FrameSamplingError("end_frame must be >= 1.")
        if start_frame > end_frame:
            raise FrameSamplingError("start_frame must be <= end_frame.")
        if end_frame > total_frames:
            raise FrameSamplingError("end_frame must be <= total_frames.")
        if self.mode not in {"uniform", "explicit"}:
            raise FrameSamplingError("mode must be 'uniform' or 'explicit'.")
        object.__setattr__(self, "total_frames", total_frames)
        object.__setattr__(self, "max_frames", max_frames)
        object.__setattr__(self, "start_frame", start_frame)
        object.__setattr__(self, "end_frame", end_frame)
        object.__setattr__(self, "explicit_frames", explicit_frames)

    @property
    def effective_mode(self) -> SamplingMode:
        return "explicit" if self.explicit_frames else self.mode

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "effective_mode": self.effective_mode,
            "total_frames": self.total_frames,
            "max_frames": self.max_frames,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "explicit_frames": list(self.explicit_frames),
        }


@dataclass(frozen=True)
class FrameSamplingPlan:
    request: FrameSamplingRequest
    selected_frames: tuple[SelectedFrame, ...]

    def __post_init__(self) -> None:
        selected = tuple(self.selected_frames)
        if not selected:
            raise FrameSamplingError("sampling plan must contain at least one frame.")
        frame_indices = [item.frame_index for item in selected]
        if frame_indices != sorted(set(frame_indices)):
            raise FrameSamplingError("sampling plan frames must be sorted and unique.")
        for expected_order, frame in enumerate(selected, 1):
            if frame.selection_order != expected_order:
                raise FrameSamplingError("selection_order must be contiguous from 1.")
        object.__setattr__(self, "selected_frames", selected)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(item.frame_index for item in self.selected_frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "selected_frame_count": len(self.selected_frames),
            "selected_frames": [item.to_dict() for item in self.selected_frames],
            "frame_indices": list(self.frame_indices),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameSamplingPlan:
        request_data = data["request"]
        return cls(
            request=FrameSamplingRequest(
                total_frames=int(request_data["total_frames"]),
                max_frames=int(request_data.get("max_frames", data.get("selected_frame_count", 1))),
                start_frame=int(request_data.get("start_frame", 1)),
                end_frame=int(request_data.get("end_frame", request_data["total_frames"])),
                explicit_frames=tuple(request_data.get("explicit_frames", ())),
                mode=str(request_data.get("mode", "uniform")),  # type: ignore[arg-type]
            ),
            selected_frames=tuple(
                SelectedFrame.from_dict(item) for item in data["selected_frames"]
            ),
        )
