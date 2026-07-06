"""Namespace CLI for LocateAnything tracking milestones."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from football_tracking.locate_tracking.association.config import (
    AssociationConfigError,
    load_frame_association_config,
)
from football_tracking.locate_tracking.association.matcher import GroundingTrackMatcher
from football_tracking.locate_tracking.association.service import (
    FrameTrackQueryService,
    FrameTrackQueryServiceError,
)
from football_tracking.locate_tracking.cli.locate_image import (
    LocateImageError,
    _build_backend,
    load_locate_image_config,
    run_locate_image,
)
from football_tracking.locate_tracking.grounding.cache import GroundingCache
from football_tracking.locate_tracking.grounding.service import GroundingService
from football_tracking.locate_tracking.sampling.explicit_selector import (
    parse_explicit_frames,
)
from football_tracking.locate_tracking.semantic_memory.config import (
    SemanticMemoryConfigError,
    load_semantic_memory_config,
)
from football_tracking.locate_tracking.semantic_memory.service import (
    SemanticMemoryService,
    SemanticMemoryServiceError,
    build_sampling_request_for_video,
)
from football_tracking.locate_tracking.visualization.semantic_summary import (
    write_semantic_summary,
)


def _add_locate_image(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/locate_tracking/locateanything_grounding.yaml"),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_match_grounding_frame(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--association-config",
        type=Path,
        default=Path("configs/locate_tracking/frame_association.yaml"),
    )
    parser.add_argument("--grounding-result", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--frame-width", type=int, required=True)
    parser.add_argument("--frame-height", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_query_track_frame(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--grounding-config",
        type=Path,
        default=Path("configs/locate_tracking/locateanything_grounding.yaml"),
    )
    parser.add_argument(
        "--association-config",
        type=Path,
        default=Path("configs/locate_tracking/frame_association.yaml"),
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_aggregate_language_track(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=Path("configs/locate_tracking/semantic_memory.yaml"),
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--frame-resolution",
        type=Path,
        action="append",
        dest="frame_resolutions",
        required=True,
        help="Path to a single-frame association JSON. Repeat for multiple frames.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, default=None)
    parser.add_argument("--tracks", type=Path, default=None)
    parser.add_argument("--sampled-frames", default=None)
    parser.add_argument("--query-mode", choices=("single_target", "multi_target"), default=None)
    parser.add_argument(
        "--aggregation-strategy", choices=("weighted", "majority_support"), default=None
    )
    parser.add_argument("--min-usable-frames", type=int, default=None)
    parser.add_argument("--min-support-frames", type=int, default=None)
    parser.add_argument("--min-support-ratio", type=float, default=None)
    parser.add_argument("--min-aggregate-score", type=float, default=None)
    parser.add_argument("--winner-margin", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_resolve_language_track(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--grounding-config",
        type=Path,
        default=Path("configs/locate_tracking/locateanything_grounding.yaml"),
    )
    parser.add_argument(
        "--association-config",
        type=Path,
        default=Path("configs/locate_tracking/frame_association.yaml"),
    )
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=Path("configs/locate_tracking/semantic_memory.yaml"),
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default=None, help="Comma-separated explicit frame list.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--query-mode", choices=("single_target", "multi_target"), default=None)
    parser.add_argument(
        "--aggregation-strategy", choices=("weighted", "majority_support"), default=None
    )
    parser.add_argument("--min-usable-frames", type=int, default=None)
    parser.add_argument("--min-support-frames", type=int, default=None)
    parser.add_argument("--min-support-ratio", type=float, default=None)
    parser.add_argument("--min-aggregate-score", type=float, default=None)
    parser.add_argument("--winner-margin", type=float, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m football_tracking.locate_tracking.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    locate = subparsers.add_parser("locate-image", help="Ground one image with a query.")
    _add_locate_image(locate)
    match = subparsers.add_parser(
        "match-grounding-frame",
        help="Associate an existing grounding JSON with MOT tracks at one frame.",
    )
    _add_match_grounding_frame(match)
    query = subparsers.add_parser(
        "query-track-frame",
        help="Extract one video frame, ground a query, and match active tracks.",
    )
    _add_query_track_frame(query)
    aggregate = subparsers.add_parser(
        "aggregate-language-track",
        help="Aggregate existing single-frame association JSON files into semantic memory.",
    )
    _add_aggregate_language_track(aggregate)
    resolve = subparsers.add_parser(
        "resolve-language-track",
        help="Sample video frames, ground a language query, match tracks, and aggregate memory.",
    )
    _add_resolve_language_track(resolve)
    return parser


def _grounding_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "backend_name": getattr(args, "backend", None),
        "model_id": getattr(args, "model_id", None),
        "device": getattr(args, "device", None),
        "torch_dtype": getattr(args, "torch_dtype", None),
        "max_new_tokens": getattr(args, "max_new_tokens", None),
        "overwrite": True if getattr(args, "overwrite", False) else None,
    }


def _association_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "output_dir": getattr(args, "output_dir", None),
        "overwrite": True if getattr(args, "overwrite", False) else None,
    }


def _semantic_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "output_dir": getattr(args, "output_dir", None),
        "overwrite": True if getattr(args, "overwrite", False) else None,
        "explicit_frames": getattr(args, "frames", None)
        if getattr(args, "frames", None) is not None
        else getattr(args, "sampled_frames", None),
        "max_frames": getattr(args, "max_frames", None),
        "start_frame": getattr(args, "start_frame", None),
        "end_frame": getattr(args, "end_frame", None),
        "query_mode": getattr(args, "query_mode", None),
        "aggregation_strategy": getattr(args, "aggregation_strategy", None),
        "min_usable_frames": getattr(args, "min_usable_frames", None),
        "min_support_frames": getattr(args, "min_support_frames", None),
        "min_support_ratio": getattr(args, "min_support_ratio", None),
        "min_aggregate_score": getattr(args, "min_aggregate_score", None),
        "winner_margin": getattr(args, "winner_margin", None),
    }


def _build_query_service(args: argparse.Namespace) -> FrameTrackQueryService:
    grounding_config = load_locate_image_config(
        args.grounding_config,
        overrides=_grounding_overrides(args),
    )
    association_config = load_frame_association_config(
        args.association_config,
        overrides=_association_overrides(args),
    )
    grounding_service = GroundingService(
        backend=_build_backend(grounding_config),
        cache=GroundingCache(
            grounding_config.cache_directory,
            enabled=grounding_config.cache_enabled,
            overwrite=grounding_config.overwrite,
        ),
        overwrite=grounding_config.overwrite,
    )
    return FrameTrackQueryService(
        matcher=GroundingTrackMatcher(association_config.association),
        grounding_service=grounding_service,
        overwrite=association_config.overwrite,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "locate-image":
            result = run_locate_image(
                args.config,
                image=args.image,
                query=args.query,
                output=args.output,
                overrides=_grounding_overrides(args),
                dry_run=args.dry_run,
            )
            sys.stdout.write(json.dumps(result, indent=2, default=str))
            sys.stdout.write("\n")
            return 0
        if args.command == "match-grounding-frame":
            association_config = load_frame_association_config(
                args.association_config,
                overrides={"overwrite": True if args.overwrite else None},
            )
            service = FrameTrackQueryService(
                matcher=GroundingTrackMatcher(association_config.association),
                overwrite=association_config.overwrite,
            )
            result = service.match_existing_grounding(
                grounding_result_path=args.grounding_result,
                tracks_path=args.tracks,
                frame_index=args.frame_index,
                frame_width=args.frame_width,
                frame_height=args.frame_height,
                output_path=args.output,
            )
            sys.stdout.write(json.dumps(result.to_dict(), indent=2, default=str))
            sys.stdout.write("\n")
            return 0
        if args.command == "aggregate-language-track":
            semantic_config = load_semantic_memory_config(
                args.semantic_config,
                overrides=_semantic_overrides(args),
            )
            service = SemanticMemoryService(
                config=semantic_config.semantic_memory,
                overwrite=semantic_config.overwrite,
            )
            sampled_frames = (
                parse_explicit_frames(args.sampled_frames)
                if args.sampled_frames is not None
                else None
            )
            session = service.aggregate_frame_resolutions(
                query=args.query,
                frame_resolution_paths=tuple(args.frame_resolutions),
                output_dir=args.output_dir,
                sampled_frames=sampled_frames,
                source_video=str(args.source_video) if args.source_video else None,
                tracks_path=str(args.tracks) if args.tracks else None,
            )
            write_semantic_summary(
                session.semantic_memory,
                session.final_resolution,
                args.output_dir / "semantic_summary.md",
            )
            sys.stdout.write(json.dumps(session.final_resolution.to_dict(), indent=2, default=str))
            sys.stdout.write("\n")
            return 0
        if args.command == "resolve-language-track":
            semantic_config = load_semantic_memory_config(
                args.semantic_config,
                overrides=_semantic_overrides(args),
            )
            query_service = _build_query_service(args)
            sampling_request = build_sampling_request_for_video(
                source_video=args.source_video,
                sampling_config=semantic_config.sampling,
            )
            service = SemanticMemoryService(
                config=semantic_config.semantic_memory,
                frame_query_service=query_service,
                overwrite=semantic_config.overwrite,
            )
            session = service.resolve_language_track(
                source_video=args.source_video,
                tracks_path=args.tracks,
                query=args.query,
                sampling_request=sampling_request,
                output_dir=args.output_dir,
                save_overlay=args.save_overlay,
            )
            write_semantic_summary(
                session.semantic_memory,
                session.final_resolution,
                args.output_dir / "semantic_summary.md",
            )
            sys.stdout.write(json.dumps(session.final_resolution.to_dict(), indent=2, default=str))
            sys.stdout.write("\n")
            return 0
        service = _build_query_service(args)
        result = service.query_track_frame(
            source_video=args.source_video,
            tracks_path=args.tracks,
            frame_index=args.frame_index,
            query=args.query,
            output_dir=args.output_dir,
            save_overlay=args.save_overlay,
        )
        sys.stdout.write(json.dumps(result.to_dict(), indent=2, default=str))
        sys.stdout.write("\n")
        return 0
    except (
        AssociationConfigError,
        FrameTrackQueryServiceError,
        LocateImageError,
        SemanticMemoryConfigError,
        SemanticMemoryServiceError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
