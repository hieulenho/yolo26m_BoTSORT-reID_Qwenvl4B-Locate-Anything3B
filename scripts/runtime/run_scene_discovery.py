"""Run shot-aware, dynamic scene and object discovery with Qwen3-VL-4B."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

from football_tracking.vlm.model_loader import VlmModelLoadError, load_qwen_model
from football_tracking.vlm.scene_discovery import discover_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover a video's domain and open object vocabulary with Qwen3-VL."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    parser.add_argument(
        "--quantization",
        default="8bit",
        choices=("none", "8bit", "4bit"),
    )
    parser.add_argument("--n-frames", "--max-keyframes", dest="max_keyframes", type=int, default=6)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--transition-threshold", type=float, default=0.45)
    parser.add_argument("--max-classes", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/ontology/vocabulary_registry.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if not args.video.is_file():
        sys.stderr.write(f"Video not found: {args.video}\n")
        return 2
    if args.output.exists() and not args.overwrite:
        _print_summary(json.loads(args.output.read_text(encoding="utf-8")))
        return 0
    model_config = SimpleNamespace(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        quantization=args.quantization,
    )
    try:
        model, processor = load_qwen_model(model_config)
        result = discover_scene(
            args.video,
            model=model,
            processor=processor,
            n_frames=args.max_keyframes,
            output_path=args.output,
            overwrite=args.overwrite,
            registry_path=args.registry,
            max_classes=args.max_classes,
            sample_fps=args.sample_fps,
            transition_threshold=args.transition_threshold,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
        )
    except (VlmModelLoadError, RuntimeError, ValueError, OSError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    _print_summary(result.to_dict())
    return 0


def _print_summary(data: dict) -> None:
    domain = data.get("domain", {})
    objects = data.get("objects", [])
    print(json.dumps(
        {
            "status": "ok",
            "domain": domain,
            "track_classes": [
                item.get("canonical_name")
                for item in objects
                if item.get("action") == "track"
            ],
            "detect_classes": [
                item.get("canonical_name")
                for item in objects
                if item.get("action") == "detect"
            ],
            "open_vocabulary_classes": [
                item.get("canonical_name")
                for item in objects
                if item.get("open_vocabulary")
            ],
            "keyframe_count": len(data.get("keyframes", [])),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
