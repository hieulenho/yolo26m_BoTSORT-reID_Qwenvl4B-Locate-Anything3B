"""Route a normalized scene vocabulary to the most suitable detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from football_tracking.adaptive_tracking.ontology import COCO80_CLASSES
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
    tracker_class_ids: tuple[int, ...]
    reason: str
    deferred_classes: tuple[str, ...] = ()
    supplemental_detectors: tuple[dict[str, Any], ...] = ()
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
            tracker_class_ids=(0,),
            reason="No detector class survived normalization; use a conservative person fallback.",
            profile=normalized_profile,
        )

    if _is_football_scene(discovery):
        non_person_tracks = tuple(
            item
            for item in discovery.tracking_objects
            if item.canonical_name not in FOOTBALL_PERSON_CLASSES
        )
        if non_person_tracks:
            return _general_route(
                detector_objects,
                profile=normalized_profile,
                coco_checkpoint=coco_checkpoint,
                open_checkpoint=open_checkpoint,
                reason_prefix=(
                    "Football contains non-person classes that require persistent IDs; "
                ),
            )
        supplemental, supplemental_names, supplemental_ids = _football_supplemental_detectors(
            detector_objects,
            profile=normalized_profile,
            coco_checkpoint=coco_checkpoint,
            open_checkpoint=open_checkpoint,
        )
        return DetectorRoute(
            route_name="football_finetuned",
            backend="ultralytics",
            checkpoint=football_checkpoint,
            class_names=("player", *supplemental_names),
            class_ids=(0, *supplemental_ids),
            tracker_class_ids=(0,),
            reason=(
                "Football uses the fine-tuned person detector for tracking. Detect-only "
                "COCO or open classes use supplemental routed detectors."
            ),
            supplemental_detectors=supplemental,
            profile=normalized_profile,
        )

    return _general_route(
        detector_objects,
        profile=normalized_profile,
        coco_checkpoint=coco_checkpoint,
        open_checkpoint=open_checkpoint,
    )


def _general_route(
    detector_objects: tuple[Any, ...],
    *,
    profile: str,
    coco_checkpoint: str,
    open_checkpoint: str,
    reason_prefix: str = "",
) -> DetectorRoute:
    names = tuple(item.canonical_name for item in detector_objects)
    if all(item.coco_id is not None for item in detector_objects):
        class_ids = tuple(sorted({int(item.coco_id) for item in detector_objects}))
        tracking_ids = tuple(
            sorted(
                {
                    int(item.coco_id)
                    for item in detector_objects
                    if item.action == "track" and item.coco_id is not None
                }
            )
        )
        return DetectorRoute(
            route_name="coco_pretrained",
            backend="ultralytics",
            checkpoint=coco_checkpoint,
            class_names=tuple(COCO80_CLASSES[class_id] for class_id in class_ids),
            class_ids=class_ids,
            tracker_class_ids=tracking_ids,
            reason=(
                reason_prefix
                + "All requested classes map to COCO, so the pretrained YOLO route is fastest."
            ),
            profile=profile,
        )

    tracking_ids = tuple(
        index for index, item in enumerate(detector_objects) if item.action == "track"
    )
    return DetectorRoute(
        route_name="open_vocabulary",
        backend="ultralytics_yoloe",
        checkpoint=open_checkpoint,
        class_names=names,
        class_ids=tuple(range(len(names))),
        tracker_class_ids=tracking_ids,
        reason=(
            reason_prefix
            + "At least one class is outside COCO; use YOLOE with the discovered vocabulary."
        ),
        profile=profile,
    )


def _is_football_scene(discovery: SceneDiscovery) -> bool:
    if discovery.domain in FOOTBALL_DOMAINS:
        return True
    semantic_text = " ".join(
        (
            discovery.domain,
            discovery.description,
            *(
                value
                for item in discovery.objects
                for value in (
                    item.canonical_name,
                    item.display_name,
                    *item.aliases,
                    *item.source_names,
                )
            ),
        )
    ).lower()
    return "football" in semantic_text or "soccer" in semantic_text


def _football_supplemental_detectors(
    detector_objects: tuple[Any, ...],
    *,
    profile: str,
    coco_checkpoint: str,
    open_checkpoint: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], tuple[int, ...]]:
    detect_only = [
        item
        for item in detector_objects
        if item.action != "track" and item.canonical_name not in FOOTBALL_PERSON_CLASSES
    ]
    coco_items = [item for item in detect_only if item.coco_id is not None]
    open_items = [item for item in detect_only if item.coco_id is None]
    detectors: list[dict[str, Any]] = []
    names: list[str] = []
    output_ids: list[int] = []
    coco_interval = {"realtime": 6, "balanced": 3, "accuracy": 1}[profile]
    open_interval = {"realtime": 12, "balanced": 6, "accuracy": 1}[profile]
    if coco_items:
        input_ids = [int(item.coco_id) for item in coco_items]
        detectors.append(
            {
                "name": "football_coco_supplemental",
                "backend": "ultralytics",
                "checkpoint": coco_checkpoint,
                "input_class_ids": input_ids,
                "output_class_ids": input_ids,
                "class_names": [item.canonical_name for item in coco_items],
                "every_n_frames": coco_interval,
            }
        )
        names.extend(item.canonical_name for item in coco_items)
        output_ids.extend(input_ids)
    if open_items:
        first_open_id = max([999, *output_ids]) + 1
        open_output_ids = [first_open_id + index for index in range(len(open_items))]
        detectors.append(
            {
                "name": "football_open_supplemental",
                "backend": "ultralytics_yoloe",
                "checkpoint": open_checkpoint,
                "text_classes": [item.canonical_name for item in open_items],
                "input_class_ids": list(range(len(open_items))),
                "output_class_ids": open_output_ids,
                "class_names": [item.canonical_name for item in open_items],
                "every_n_frames": open_interval,
            }
        )
        names.extend(item.canonical_name for item in open_items)
        output_ids.extend(open_output_ids)
    return tuple(detectors), tuple(names), tuple(output_ids)


def checkpoint_exists_or_downloadable(route: DetectorRoute, project_root: Path) -> bool:
    path = Path(route.checkpoint)
    if path.is_absolute():
        return path.is_file()
    if (project_root / path).is_file():
        return True
    return path.name.startswith(("yolo26", "yoloe-26"))
