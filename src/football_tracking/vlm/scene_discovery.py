"""
scene_discovery.py
------------------
Bước 1 của VLM-Guided Tracking Pipeline:
  - Trích xuất frames đại diện từ video gốc
  - Gửi ảnh cho Qwen3-VL-4B để phân tích ngữ cảnh
  - Sinh ra danh sách COCO class IDs cần track, lưu vào scene_discovery.json

Đầu ra (scene_discovery.json):
{
  "context":         "Mô tả bối cảnh video",
  "context_short":   "Từ khóa ngắn (e.g. football, traffic, film)",
  "objects_found":   ["car", "motorcycle", "person", ...],
  "coco_class_ids":  [0, 2, 3],      // ID trong COCO80
  "coco_class_names": {"0": "person", "2": "car", "3": "motorcycle"},
  "confidence":      "high" | "medium" | "low",
  "raw_response":    "<toàn bộ câu trả lời của VLM>"
}
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COCO80 class list (index = class_id)
# ---------------------------------------------------------------------------
COCO80_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

COCO80_BY_NAME: dict[str, int] = {name: idx for idx, name in enumerate(COCO80_CLASSES)}

# Aliases để VLM có thể trả về tên gần đúng
COCO80_ALIASES: dict[str, str] = {
    "motorbike":       "motorcycle",
    "bike":            "bicycle",
    "truck":           "truck",
    "lorry":           "truck",
    "van":             "car",
    "automobile":      "car",
    "pedestrian":      "person",
    "footballer":      "person",
    "player":          "person",
    "referee":         "person",
    "athlete":         "person",
    "traffic light":   "traffic light",
    "traffic_light":   "traffic light",
    "stoplight":       "traffic light",
    "airplane":        "airplane",
    "aeroplane":       "airplane",
    "plane":           "airplane",
}

# Prompt gốc cho VLM
_DISCOVERY_PROMPT = """You are a video scene analyzer.
You will receive 3 sampled video frames.

Your task has TWO steps:

STEP 1 — Describe the scene context:
- What is the setting? (e.g., football match, city traffic, film scene, wildlife)
- Write a short 1-sentence summary in English.

STEP 2 — Identify trackable moving objects:
- List all distinct moving objects visible in the frames that should be tracked.
- IMPORTANT: You MUST map EVERY object to the exact COCO dataset class name from this list:
  {coco_classes}
- Only list COCO classes. Do NOT invent new class names.
- If multiple different class types need tracking, list them all.

Respond ONLY with valid JSON in this EXACT format (no extra text outside the JSON):
{{
  "context": "<one sentence scene description>",
  "context_short": "<one or two keywords, e.g.: football, city_traffic, wildlife>",
  "objects_to_track": [
    {{"coco_class": "<exact COCO class name>", "reason": "<why it should be tracked>"}}
  ],
  "confidence": "<high|medium|low>"
}}"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SceneDiscoveryResult:
    context: str
    context_short: str
    objects_found: list[str]
    coco_class_ids: list[int]
    coco_class_names: dict[str, str]  # str(id) -> name
    confidence: str
    raw_response: str
    frames_sampled: int
    source_video: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_sample_frames(
    video_path: str | Path,
    n: int = 3,
    start_fraction: float = 0.05,
    end_fraction: float = 0.85,
) -> list[Any]:
    """Trích xuất n frames phân bổ đều từ video (tránh fade-in/out)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    start = int(total * start_fraction)
    end = int(total * end_fraction)
    positions = [start + int((end - start) * i / max(n - 1, 1)) for i in range(n)]

    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            log.warning("Could not read frame %d from %s", pos, video_path)
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not extract any frames from {video_path}")
    log.info("Extracted %d/%d requested frames from %s", len(frames), n, Path(video_path).name)
    return frames


# ---------------------------------------------------------------------------
# COCO class resolution
# ---------------------------------------------------------------------------

def resolve_coco_names(raw_names: list[str]) -> tuple[list[str], list[int]]:
    """Ánh xạ tên VLM trả về sang COCO class name chuẩn và ID."""
    resolved_names: list[str] = []
    resolved_ids: list[int] = []
    for raw in raw_names:
        normalized = raw.strip().lower()
        # Thử alias trước
        if normalized in COCO80_ALIASES:
            normalized = COCO80_ALIASES[normalized]
        # Tìm exact match
        if normalized in COCO80_BY_NAME:
            class_id = COCO80_BY_NAME[normalized]
            if class_id not in resolved_ids:
                resolved_ids.append(class_id)
                resolved_names.append(normalized)
            continue
        # Thử partial match
        matches = [name for name in COCO80_BY_NAME if normalized in name or name in normalized]
        if matches:
            best = matches[0]
            class_id = COCO80_BY_NAME[best]
            if class_id not in resolved_ids:
                resolved_ids.append(class_id)
                resolved_names.append(best)
            log.debug("Resolved '%s' -> '%s' (id=%d) via partial match", raw, best, class_id)
        else:
            log.warning("Could not map '%s' to any COCO80 class. Skipping.", raw)

    return sorted(resolved_names), sorted(resolved_ids)


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------

def _call_qwen(model: Any, processor: Any, frames: list[Any], prompt: str) -> str:
    """Gọi Qwen3-VL với danh sách frames và prompt, trả về text output."""
    import torch

    # Encode ảnh sang PIL
    from PIL import Image as PILImage
    pil_frames = [
        PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        for f in frames
    ]

    # Xây dựng conversation multi-image
    image_entries = [{"type": "image", "image": img} for img in pil_frames]
    messages = [
        {
            "role": "user",
            "content": image_entries + [{"type": "text", "text": prompt}],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = processor.process_vision_info(messages)  # type: ignore[attr-defined]
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    output_ids = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _parse_vlm_response(response: str) -> dict[str, Any]:
    """Trích xuất JSON từ response của VLM (có thể có text rác xung quanh)."""
    # Tìm JSON block đầu tiên
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in VLM response:\n{response[:500]}")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse VLM JSON: {e}\nRaw: {response[:500]}") from e


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def discover_scene(
    video_path: str | Path,
    *,
    model: Any,
    processor: Any,
    n_frames: int = 3,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> SceneDiscoveryResult:
    """
    Phân tích bối cảnh video bằng VLM, trả về SceneDiscoveryResult.

    Args:
        video_path:   Đường dẫn video gốc (chưa có bounding box).
        model:        Qwen model đã load (transformers AutoModelForVision2Seq).
        processor:    Qwen processor đã load.
        n_frames:     Số frames lấy mẫu để phân tích.
        output_path:  Lưu kết quả ra file JSON (tùy chọn).
        overwrite:    Ghi đè nếu file output đã tồn tại.

    Returns:
        SceneDiscoveryResult
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            log.info("Scene discovery output exists, loading: %s", output_path)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            return SceneDiscoveryResult(**data)

    prompt = _DISCOVERY_PROMPT.format(coco_classes=", ".join(COCO80_CLASSES))

    log.info("Extracting %d sample frames from %s ...", n_frames, video_path.name)
    frames = extract_sample_frames(video_path, n=n_frames)

    log.info("Calling Qwen3-VL for scene discovery (%d frames) ...", len(frames))
    t0 = time.time()
    raw_response = _call_qwen(model, processor, frames, prompt)
    elapsed = time.time() - t0
    log.info("VLM responded in %.1fs: %s", elapsed, raw_response[:200].replace("\n", " "))

    try:
        parsed = _parse_vlm_response(raw_response)
    except ValueError as exc:
        log.error("Failed to parse VLM response: %s", exc)
        # Fallback: track persons only
        parsed = {
            "context": "Unknown (VLM parse failed)",
            "context_short": "unknown",
            "objects_to_track": [{"coco_class": "person", "reason": "fallback"}],
            "confidence": "low",
        }

    raw_names = [item.get("coco_class", "") for item in parsed.get("objects_to_track", [])]
    resolved_names, resolved_ids = resolve_coco_names(raw_names)

    # Luôn đảm bảo có ít nhất 1 class
    if not resolved_ids:
        log.warning("No COCO classes resolved from VLM output. Defaulting to person (0).")
        resolved_ids = [0]
        resolved_names = ["person"]

    result = SceneDiscoveryResult(
        context=str(parsed.get("context", "")),
        context_short=str(parsed.get("context_short", "unknown")),
        objects_found=resolved_names,
        coco_class_ids=resolved_ids,
        coco_class_names={str(cid): COCO80_CLASSES[cid] for cid in resolved_ids},
        confidence=str(parsed.get("confidence", "medium")),
        raw_response=raw_response,
        frames_sampled=len(frames),
        source_video=str(video_path),
        created_at=datetime.now(UTC).isoformat(),
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Scene discovery saved -> %s", output_path)

    return result


def load_scene_discovery(path: str | Path) -> SceneDiscoveryResult:
    """Load SceneDiscoveryResult từ file JSON đã lưu."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SceneDiscoveryResult(**data)
