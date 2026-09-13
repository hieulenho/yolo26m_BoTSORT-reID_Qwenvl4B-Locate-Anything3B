"""Prefetch public weights. Quantization is applied when Qwen is loaded, not downloaded."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", action="store_true", help="Also download the Qwen 4B snapshot.")
    args = parser.parse_args()
    from ultralytics import YOLO

    YOLO("yolo26n.pt")
    if args.qwen:
        from huggingface_hub import snapshot_download

        path = snapshot_download("Qwen/Qwen3-VL-4B-Instruct")
        print(f"Qwen snapshot cached at {path}; the semantic worker loads it in 8-bit.")
    print("Model prefetch complete.")


if __name__ == "__main__":
    main()
