"""Route a normalized scene vocabulary to the most suitable detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from football_tracking.adaptive_tracking.schemas import SceneDiscovery

FOOTBALL_DOMAINS = {"football", "soccer", "football_match", "sports_football"}
FOOTBALL_PERSON_CLASSES = {
    "person",
    "player",
    "football player",
    "goalkeeper",
    "referee",
    "coach",
}


@dataclass(frozen=True)
class DetectorRoute:
    route_name: str
    backend: str
    checkpoint: str
    class_names: tuple[str, ...]
    class_ids: tuple[int, ...]
    reason: str
    deferred_classes: tuple[str, ...] = ()
    profile: str = "realtime"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_detector_route(
    discovery: SceneDiscovery,
    *,
    profile: str = "realtime",
    football_checkpoint: str = "models/detector/football/yolo26m_best.pt",
    coco_checkpoint: str | None = None,
    open_checkpoint: str | None = None,
) -> DetectorRoute:
    """Choose a detector without dropping unknown classes from the discovery record."""
    normalized_profile = str(profile).strip().lower()
    if normalized_profile not in {"realtime", "balanced", "accuracy"}:
        raise ValueError(f"Unsupported adaptive profile: {profile}")
    coco_checkpoint = coco_checkpoint or (
        "yolo26n.pt" if normalized_profile == "realtime" else "yolo26s.pt"
    )
    open_checkpoint = open_checkpoint or (
        "yoloe-26n-seg.pt"
        if normalized_profile == "realtime"
        else "yoloe-26s-seg.pt"
    )
    detector_objects = discovery.detector_objects
    if not detector_objects:
        return DetectorRoute(
            route_name="coco_fallback",
            backend="ultralytics",
            checkpoint=coco_checkpoint,
            class_names=("person",),
            class_ids=(0,),
            reason="No detector class survived normalization; use a conservative person fallback.",
            profile=normalized_profile,
        )

    names = tuple(item.canonical_name for item in detector_objects)
    domain = discovery.domain
    if domain in FOOTBALL_DOMAINS or "football" in domain or "soccer" in domain:
        person_names = tuple(name for name in names if name in FOOTBALL_PERSON_CLASSES)
        if not person_names:
            person_names = ("player",)
        deferred = tuple(name for name in names if name not in FOOTBALL_PERSON_CLASSES)
        return DetectorRoute(
            route_name="football_finetuned",
            backend="ultralytics",
            checkpoint=football_checkpoint,
            class_names=("player",),
            class_ids=(0,),
            reason=(
                "Football domain uses the fine-tuned person detector; semantic roles and "
                "teams are assigned after tracking."
            ),
            deferred_classes=deferred,
            profile=normalized_profile,
        )

    if all(item.coco_id is not None for item in detector_objects):
        ids_and_names = sorted(
            {(int(item.coco_id), item.canonical_name) for item in detector_objects}
        )
        return DetectorRoute(
            route_name="coco_pretrained",
            backend="ultralytics",
            checkpoint=coco_checkpoint,
            class_names=tuple(name for _class_id, name in ids_and_names),
            class_ids=tuple(class_id for class_id, _name in ids_and_names),
            reason="All requested classes map to COCO, so the pretrained YOLO route is fastest.",
            profile=normalized_profile,
        )

    return DetectorRoute(
        route_name="open_vocabulary",
        backend="ultralytics_yoloe",
        checkpoint=open_checkpoint,
        class_names=names,
        class_ids=tuple(range(len(names))),
        reason="At least one class is outside COCO; use YOLOE with the discovered vocabulary.",
        profile=normalized_profile,
    )


def checkpoint_exists_or_downloadable(route: DetectorRoute, project_root: Path) -> bool:
    path = Path(route.checkpoint)
    if path.is_absolute():
        return path.is_file()
    if (project_root / path).is_file():
        return True
    return path.name.startswith(("yolo26", "yoloe-26"))
