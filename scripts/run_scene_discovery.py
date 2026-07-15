"""
run_scene_discovery.py
----------------------
Script CLI Bước 1 — Scene Discovery:
  python scripts/run_scene_discovery.py --video F:/videos/1.mp4 --output outputs/discovery/1/scene_discovery.json

Không cần tracking trước, chỉ cần video gốc và Qwen model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bước 1 VLM-Guided Tracking: phân tích bối cảnh video bằng Qwen3-VL-4B,"
            " sinh ra danh sách COCO class cần track."
        )
    )
    parser.add_argument("--video", type=Path, required=True, help="Video gốc đầu vào")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="File JSON đầu ra (scene_discovery.json)",
    )
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3-VL-4B-Instruct",
        help="Hugging Face model ID cho Qwen",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype", default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--quantization", default="none",
        choices=["none", "8bit", "4bit"],
    )
    parser.add_argument("--n-frames", type=int, default=3, help="Số frames lấy mẫu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    video = args.video
    if not video.is_file():
        log.error("Video not found: %s", video)
        sys.exit(1)

    output_path = args.output
    if output_path.exists() and not args.overwrite:
        log.info("Output already exists (use --overwrite to regenerate): %s", output_path)
        data = json.loads(output_path.read_text(encoding="utf-8"))
        log.info("Loaded existing discovery: context_short=%s, classes=%s",
                 data.get("context_short"), data.get("coco_class_names"))
        _print_summary(data)
        sys.exit(0)

    # ---- Load Qwen model ----
    log.info("Loading Qwen model: %s (quant=%s, dtype=%s)", args.model_id, args.quantization, args.torch_dtype)
    model, processor = _load_qwen(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        quantization=args.quantization,
    )

    # ---- Run discovery ----
    from football_tracking.vlm.scene_discovery import discover_scene
    result = discover_scene(
        video_path=video,
        model=model,
        processor=processor,
        n_frames=args.n_frames,
        output_path=output_path,
        overwrite=args.overwrite,
    )

    _print_summary(result.to_dict())
    log.info("Done. Output saved: %s", output_path)


def _print_summary(data: dict) -> None:
    print()
    print("==> Scene Discovery Result")
    print(f"    Context     : {data.get('context', '')}")
    print(f"    Short       : {data.get('context_short', '')}")
    print(f"    Confidence  : {data.get('confidence', '')}")
    print(f"    COCO IDs    : {data.get('coco_class_ids', [])}")
    print(f"    COCO Names  : {data.get('coco_class_names', {})}")
    print()


def _load_qwen(*, model_id: str, device: str, torch_dtype: str, quantization: str):
    """Load Qwen3-VL model và processor."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # Resolve dtype
    if torch_dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Quantization config
    bnb_config = None
    if quantization == "8bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs: dict = {"torch_dtype": dtype}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    elif device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = device

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, **model_kwargs
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


if __name__ == "__main__":
    main()
