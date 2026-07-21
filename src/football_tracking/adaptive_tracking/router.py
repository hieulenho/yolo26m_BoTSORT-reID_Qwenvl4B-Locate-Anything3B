"""Route a normalized scene vocabulary to the most suitable detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from football_tracking.adaptive_tracking.ontology import COCO80_CLASSES
from football_tracking.adaptive_tracking.schemas import SceneDiscovery

FOOTBALL_DOMAINS = {"football", "soccer", "football_match", "sports_football"}
REALTIME_PROFILES = {"realtime", "realtime_stable"}
FOOTBALL_PERSON_CLASSES = {
    "person",
    "player",
    "football player",
    "goalkeeper",
    "referee",
    "coach",
}
TRACKABLE_COCO_IDS = {
    0,  # person and semantic person roles
    1, 2, 3, 4, 5, 6, 7, 8,  # vehicles
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # animals
    32,  # sports ball
}
PERSISTENT_NAME_TOKENS = {
    "animal",
    "athlete",
    "bird",
    "car",
    "cyclist",
    "doctor",
    "drone",
    "fish",
    "nurse",
    "patient",
    "person",
    "player",
    "robot",
    "student",
    "teacher",
    "vehicle",
}
PERSISTENT_TAXONOMY_HINTS = {
    "animal",
    "human",
    "person",
    "role",
    "species",
    "vehicle",
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
    primary_class_ids: tuple[int, ...] = ()
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
    if normalized_profile not in {*REALTIME_PROFILES, "balanced", "accuracy"}:
        raise ValueError(f"Unsupported adaptive profile: {profile}")
    coco_checkpoint = coco_checkpoint or (
        "yolo26n.pt" if normalized_profile in REALTIME_PROFILES else "yolo26s.pt"
    )
    open_checkpoint = open_checkpoint or "yoloe-26s-seg.pt"
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
        (
            supplemental,
            supplemental_names,
            supplemental_ids,
            supplemental_tracker_ids,
        ) = _football_supplemental_detectors(
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
            tracker_class_ids=(0, *supplemental_tracker_ids),
            reason=(
                "Football keeps the fine-tuned person detector. Non-person COCO/open "
                "classes use routed supplemental detectors and enter tracking only when "
                "their discovered action is track."
            ),
            primary_class_ids=(0,),
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
    tracking_names, tracking_fallback = _tracking_names(detector_objects)
    promoted_names = ", ".join(sorted(tracking_names))
    fallback_reason = (
        " No class was explicitly marked for tracking, so persistent entities "
        f"[{promoted_names}] were promoted."
        if tracking_fallback
        else ""
    )
    if all(item.coco_id is not None for item in detector_objects):
        class_ids = tuple(sorted({int(item.coco_id) for item in detector_objects}))
        tracking_ids = tuple(
            sorted(
                {
                    int(item.coco_id)
                    for item in detector_objects
                    if item.canonical_name in tracking_names and item.coco_id is not None
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
                + fallback_reason
            ),
            profile=profile,
        )

    coco_items = [item for item in detector_objects if item.coco_id is not None]
    open_items = [item for item in detector_objects if item.coco_id is None]
    if coco_items and open_items:
        coco_ids = tuple(sorted({int(item.coco_id) for item in coco_items}))
        open_output_ids = tuple(1000 + index for index in range(len(open_items)))
        coco_tracking_ids = {
            int(item.coco_id)
            for item in coco_items
            if item.canonical_name in tracking_names
        }
        open_tracking_ids = {
            output_id
            for item, output_id in zip(open_items, open_output_ids, strict=True)
            if item.canonical_name in tracking_names
        }
        open_classes_need_identity = any(item.action == "track" for item in open_items)
        interval = (
            {
                "realtime": 6,
                "realtime_stable": 6,
                "balanced": 3,
                "accuracy": 1,
            }[profile]
            if open_classes_need_identity
            else {
                "realtime": 30,
                "realtime_stable": 30,
                "balanced": 10,
                "accuracy": 1,
            }[profile]
        )
        return DetectorRoute(
            route_name="coco_open_composite",
            backend="ultralytics",
            checkpoint=coco_checkpoint,
            class_names=(
                *(COCO80_CLASSES[class_id] for class_id in coco_ids),
                *(item.canonical_name for item in open_items),
            ),
            class_ids=(*coco_ids, *open_output_ids),
            tracker_class_ids=tuple(sorted(coco_tracking_ids | open_tracking_ids)),
            reason=(
                reason_prefix
                + "COCO classes use pretrained YOLO while unseen classes use a YOLOE "
                "supplemental detector."
                + fallback_reason
            ),
            primary_class_ids=coco_ids,
            supplemental_detectors=(
                {
                    "name": "open_vocabulary_supplemental",
                    "backend": "ultralytics_yoloe",
                    "checkpoint": open_checkpoint,
                    "text_classes": [item.canonical_name for item in open_items],
                    "input_class_ids": list(range(len(open_items))),
                    "output_class_ids": list(open_output_ids),
                    "class_names": [item.canonical_name for item in open_items],
                    "every_n_frames": interval,
                    "half": False,
                },
            ),
            profile=profile,
        )

    tracking_ids = tuple(
        index
        for index, item in enumerate(detector_objects)
        if item.canonical_name in tracking_names
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
            + fallback_reason
        ),
        profile=profile,
    )


def _tracking_names(detector_objects: tuple[Any, ...]) -> tuple[set[str], bool]:
    explicit = {
        item.canonical_name for item in detector_objects if item.action == "track"
    }
    if explicit:
        return explicit, False
    persistent = {
        item.canonical_name
        for item in detector_objects
        if _is_persistent_entity(item)
    }
    if persistent:
        return persistent, True
    promoted = max(
        detector_objects,
        key=lambda item: (float(item.confidence), -len(item.canonical_name)),
    )
    return {promoted.canonical_name}, True


def _is_persistent_entity(item: Any) -> bool:
    if item.coco_id is not None and int(item.coco_id) in TRACKABLE_COCO_IDS:
        return True
    name_tokens = set(str(item.canonical_name).lower().replace("-", " ").split())
    if name_tokens & PERSISTENT_NAME_TOKENS:
        return True
    taxonomy = str(getattr(item, "taxonomy_hint", "")).strip().lower()
    return taxonomy in PERSISTENT_TAXONOMY_HINTS


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
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    supplemental_items = [
        item
        for item in detector_objects
        if item.canonical_name not in FOOTBALL_PERSON_CLASSES
    ]
    coco_items_by_id: dict[int, Any] = {}
    for item in supplemental_items:
        if item.coco_id is None:
            continue
        class_id = int(item.coco_id)
        existing = coco_items_by_id.get(class_id)
        if existing is None or (
            existing.action != "track" and item.action == "track"
        ):
            coco_items_by_id[class_id] = item
    coco_items = list(coco_items_by_id.values())
    open_items = [item for item in supplemental_items if item.coco_id is None]
    detectors: list[dict[str, Any]] = []
    names: list[str] = []
    output_ids: list[int] = []
    tracker_ids: list[int] = []
    coco_interval = (
        {"realtime": 3, "realtime_stable": 3, "balanced": 2, "accuracy": 1}[
            profile
        ]
        if any(item.action == "track" for item in coco_items)
        else {"realtime": 6, "realtime_stable": 6, "balanced": 3, "accuracy": 1}[
            profile
        ]
    )
    open_interval = (
        {"realtime": 6, "realtime_stable": 6, "balanced": 3, "accuracy": 1}[
            profile
        ]
        if any(item.action == "track" for item in open_items)
        else {"realtime": 12, "realtime_stable": 12, "balanced": 6, "accuracy": 1}[
            profile
        ]
    )
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
                "half": False,
            }
        )
        names.extend(item.canonical_name for item in coco_items)
        output_ids.extend(input_ids)
        tracker_ids.extend(
            int(item.coco_id) for item in coco_items if item.action == "track"
        )
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
                "half": False,
            }
        )
        names.extend(item.canonical_name for item in open_items)
        output_ids.extend(open_output_ids)
        tracker_ids.extend(
            output_id
            for item, output_id in zip(open_items, open_output_ids, strict=True)
            if item.action == "track"
        )
    return tuple(detectors), tuple(names), tuple(output_ids), tuple(tracker_ids)


def checkpoint_exists_or_downloadable(route: DetectorRoute, project_root: Path) -> bool:
    path = Path(route.checkpoint)
    if path.is_absolute():
        return path.is_file()
    if (project_root / path).is_file():
        return True
    return path.name.startswith(("yolo26", "yoloe-26"))
