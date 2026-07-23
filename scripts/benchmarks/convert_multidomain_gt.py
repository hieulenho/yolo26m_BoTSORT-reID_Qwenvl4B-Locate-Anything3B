"""Convert external tracking annotations into a class-aware MOT layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from football_tracking.data.multidomain_gt import (
    MultiDomainGtError,
    convert_multidomain_gt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        choices=(
            "bdd100k_scalabel",
            "tao_coco_video",
            "animaltrack_mot",
            "ctc_masks_lineage",
            "ua_detrac_xml",
        ),
        required=True,
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--media-root",
        type=Path,
        default=None,
        help="Optional video/image root used to write exact MOT seqinfo.ini metadata.",
    )
    parser.add_argument(
        "--category-map",
        type=Path,
        default=None,
        help="Optional JSON/YAML ID-to-name map or one-name-per-line text file.",
    )
    parser.add_argument(
        "--media-fps",
        type=float,
        default=None,
        help="FPS for image-sequence media when files do not carry timing metadata.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optionally normalize only the first N annotated frames.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = convert_multidomain_gt(
            source_format=args.format,
            annotation_path=args.annotations,
            output_dir=args.output_dir,
            category_map_path=args.category_map,
            media_root=args.media_root,
            media_fps=args.media_fps,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
    except (MultiDomainGtError, OSError, ValueError, KeyError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
