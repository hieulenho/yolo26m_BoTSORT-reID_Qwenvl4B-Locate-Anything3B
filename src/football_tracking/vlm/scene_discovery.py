"""Dynamic scene, domain, and object discovery with Qwen3-VL."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from football_tracking.adaptive_tracking.ontology import (
    COCO80_CLASSES,
    COCO_ID_BY_NAME,
    VocabularyRegistry,
    normalize_objects,
)
from football_tracking.adaptive_tracking.schemas import SceneDiscovery
from football_tracking.adaptive_tracking.shot_sampling import sample_shot_keyframes
from football_tracking.detection.serialization import file_sha256
from football_tracking.paths import get_project_root, resolve_project_path
from football_tracking.vlm.model_loader import first_model_device

LOGGER = logging.getLogger(__name__)
PROMPT_VERSION = "dynamic-v2"
DEFAULT_REGISTRY = Path("configs/ontology/vocabulary_registry.yaml")

DISCOVERY_PROMPT = """You analyze a small set of representative shots from one video.

Infer the domain from visual evidence. Do not assume football, traffic, medicine, education,
or any fixed dataset. Build a compact object vocabulary that a detector and tracker should use.

Rules:
1. Separate a base object class from attributes. Example: class="car", attributes=["red"],
   not class="red car".
2. Use action="track" for persistent moving entities, "detect" for useful objects that do not
   need identity over time, and "context" for background regions.
3. Include classes outside COCO when they are visually supported. Do not force an unknown class
   into a COCO label.
4. Merge synonyms and use a singular, short English canonical_name.
5. Do not include speculative or invisible objects. Return at most {max_classes} object entries.
6. Confidence is a number from 0 to 1.

Return one valid JSON object only:
{{
  "domain": {{
    "name": "short_domain_name",
    "confidence": 0.0,
    "description": "one factual sentence"
  }},
  "objects": [
    {{
      "canonical_name": "object class",
      "display_name": "readable label",
      "action": "track|detect|context",
      "attributes": ["visible attribute"],
      "confidence": 0.0
    }}
  ],
  "background_regions": ["region name"]
}}
"""


SceneDiscoveryResult = SceneDiscovery


def extract_sample_frames(
    video_path: str | Path,
    n: int = 3,
    start_fraction: float = 0.05,
    end_fraction: float = 0.85,
) -> list[Any]:
    """Compatibility helper that returns uniformly sampled BGR frames."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        start = int(total * start_fraction)
        end = max(int(total * end_fraction), start)
        positions = [
            start + int((end - start) * index / max(n - 1, 1))
            for index in range(max(n, 1))
        ]
        frames: list[Any] = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")
    return frames


def resolve_coco_names(raw_names: list[str]) -> tuple[list[str], list[int]]:
    """Compatibility mapping used by older callers; unmatched names are not returned."""
    registry = VocabularyRegistry.load(resolve_project_path(DEFAULT_REGISTRY))
    normalized = normalize_objects(
        [{"name": name, "action": "track", "confidence": 1.0} for name in raw_names],
        registry=registry,
        max_classes=max(len(raw_names), 1),
    )
    pairs = sorted(
        {
            (int(item.coco_id), COCO80_CLASSES[int(item.coco_id)])
            for item in normalized
            if item.coco_id is not None
        }
    )
    return [name for _class_id, name in pairs], [class_id for class_id, _name in pairs]


def discover_scene(
    video_path: str | Path,
    *,
    model: Any,
    processor: Any,
    n_frames: int = 6,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    registry_path: str | Path = DEFAULT_REGISTRY,
    max_classes: int = 24,
    sample_fps: float = 2.0,
    transition_threshold: float = 0.45,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    max_new_tokens: int = 768,
) -> SceneDiscovery:
    """Discover a dynamic vocabulary from shot-aware keyframes."""
    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    destination = Path(output_path).resolve() if output_path is not None else None
    if destination is not None and destination.exists() and not overwrite:
        return SceneDiscovery.from_dict(json.loads(destination.read_text(encoding="utf-8")))

    root = (
        destination.parent
        if destination is not None
        else get_project_root() / "outputs/discovery"
    )
    keyframes_dir = root / "keyframes"
    probe, sampled = sample_shot_keyframes(
        source,
        keyframes_dir,
        max_keyframes=max(n_frames, 1),
        sample_fps=sample_fps,
        transition_threshold=transition_threshold,
    )
    prompt = DISCOVERY_PROMPT.format(max_classes=max_classes)
    started = time.perf_counter()
    raw_response = _call_qwen(
        model,
        processor,
        [Path(item.path) for item in sampled],
        prompt,
        max_new_tokens=max_new_tokens,
    )
    latency = time.perf_counter() - started
    parsed, parse_warning = _parse_or_fallback(raw_response)
    domain_data = parsed.get("domain", {})
    if not isinstance(domain_data, dict):
        domain_data = {"name": domain_data}
    raw_objects = parsed.get("objects", [])
    if not isinstance(raw_objects, list):
        raw_objects = []
    legacy_objects = parsed.get("objects_to_track", [])
    if not raw_objects and isinstance(legacy_objects, list):
        raw_objects = [
            {
                "name": item.get("coco_class", "") if isinstance(item, dict) else item,
                "action": "track",
                "confidence": 0.5,
            }
            for item in legacy_objects
        ]
    registry = VocabularyRegistry.load(resolve_project_path(registry_path))
    objects = normalize_objects(
        [item for item in raw_objects if isinstance(item, dict)],
        registry=registry,
        max_classes=max_classes,
    )
    warnings = [parse_warning] if parse_warning else []
    if not any(item.action in {"track", "detect"} for item in objects):
        warnings.append("No trackable class returned; added conservative person fallback.")
        objects = normalize_objects(
            [{"name": "person", "action": "track", "confidence": 0.25}],
            registry=registry,
            max_classes=max_classes,
        )
    background = parsed.get("background_regions", [])
    if not isinstance(background, list):
        background = []
    result = SceneDiscovery(
        source_video=str(source),
        domain=str(domain_data.get("name", "unknown")),
        domain_confidence=domain_data.get("confidence", 0.5),
        description=str(domain_data.get("description", parsed.get("context", ""))),
        objects=objects,
        background_regions=tuple(str(item) for item in background),
        keyframes=tuple(item.to_dict() for item in sampled),
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        raw_response=raw_response,
        created_at=datetime.now(UTC).isoformat(),
        warnings=tuple(warnings),
        metadata={
            "video": {
                "width": probe.width,
                "height": probe.height,
                "fps": probe.fps,
                "frame_count": probe.frame_count,
                "duration_seconds": probe.duration_seconds,
            },
            "inference_seconds": latency,
            "max_classes": max_classes,
            "sample_fps": sample_fps,
            "transition_threshold": transition_threshold,
            "registry_sha256": file_sha256(resolve_project_path(registry_path)),
        },
    )
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(destination)
    return result


def load_scene_discovery(path: str | Path) -> SceneDiscovery:
    return SceneDiscovery.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _call_qwen(
    model: Any,
    processor: Any,
    image_paths: list[Path],
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    try:
        import torch  # type: ignore[import-not-found]
        from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install requirements/vlm.txt before scene discovery.") from exc
    messages = [
        {
            "role": "user",
            "content": [
                *[
                    {"type": "image", "image": str(path.resolve())}
                    for path in image_paths
                ],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(first_model_device(model))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _parse_or_fallback(response: str) -> tuple[dict[str, Any], str | None]:
    try:
        return _parse_vlm_response(response), None
    except ValueError as exc:
        LOGGER.warning("Could not parse discovery response: %s", exc)
        return (
            {
                "domain": {
                    "name": "unknown",
                    "confidence": 0.1,
                    "description": "Scene discovery response could not be parsed.",
                },
                "objects": [],
                "background_regions": [],
            },
            str(exc),
        )


def _parse_vlm_response(response: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", response.strip(), flags=re.I)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Qwen response.")
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Qwen JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Qwen response root must be an object.")
    return parsed


__all__ = [
    "COCO80_CLASSES",
    "COCO_ID_BY_NAME",
    "DISCOVERY_PROMPT",
    "PROMPT_VERSION",
    "SceneDiscoveryResult",
    "discover_scene",
    "extract_sample_frames",
    "load_scene_discovery",
    "resolve_coco_names",
]
