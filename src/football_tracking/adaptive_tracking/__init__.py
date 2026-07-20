"""Adaptive multi-domain detection, tracking, and semantic orchestration."""

from football_tracking.adaptive_tracking.router import (
    DetectorRoute,
    build_detector_route,
)
from football_tracking.adaptive_tracking.schemas import (
    DiscoveredObject,
    SceneDiscovery,
)

__all__ = [
    "DetectorRoute",
    "DiscoveredObject",
    "SceneDiscovery",
    "build_detector_route",
]
