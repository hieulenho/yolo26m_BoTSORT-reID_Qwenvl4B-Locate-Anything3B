"""CLI for cached scene discovery and adaptive detector planning."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from football_tracking.adaptive_tracking.config_builder import (
    build_tracking_payload,
    write_adaptive_plan,
)
from football_tracking.adaptive_tracking.grounding_verification import (
    build_grounding_plan,
    execute_grounding_plan,
)
from football_tracking.adaptive_tracking.router import build_detector_route
from football_tracking.adaptive_tracking.run_report import build_adaptive_run_report
from football_tracking.adaptive_tracking.schemas import SceneDiscovery
from football_tracking.adaptive_tracking.semantic_cache import (
    SemanticCache,
    discovery_cache_key,
)
from football_tracking.adaptive_tracking.semantic_fusion import fuse_semantic_files
from football_tracking.adaptive_tracking.semantic_render import render_semantic_video
from football_tracking.detection.serialization import file_sha256
from football_tracking.paths import resolve_project_path
from football_tracking.vlm.model_loader import VlmModelLoadError, load_qwen_model
from football_tracking.vlm.scene_discovery import PROMPT_VERSION, discover_scene


def _positive_track_ids(value: str) -> tuple[int, ...]:
    try:
        track_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Track IDs must be comma-separated integers."
        ) from exc
    if not track_ids or any(track_id <= 0 for track_id in track_ids):
        raise argparse.ArgumentTypeError("Track IDs must be positive integers.")
    if len(set(track_ids)) != len(track_ids):
        raise argparse.ArgumentTypeError("Track IDs must be unique.")
    return track_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover",
        help="Sample shots and discover a dynamic domain/object vocabulary.",
    )
    discover.add_argument("--source", type=Path, required=True)
    discover.add_argument("--output", type=Path, required=True)
    discover.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/semantic_discovery"),
    )
    discover.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    discover.add_argument("--device", default="auto")
    discover.add_argument("--torch-dtype", default="auto")
    discover.add_argument(
        "--quantization",
        choices=("none", "8bit", "4bit"),
        default="4bit",
    )
    discover.add_argument("--max-keyframes", type=int, default=4)
    discover.add_argument("--sample-fps", type=float, default=2.0)
    discover.add_argument("--transition-threshold", type=float, default=0.45)
    discover.add_argument("--max-classes", type=int, default=24)
    discover.add_argument("--max-new-tokens", type=int, default=768)
    discover.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/ontology/vocabulary_registry.yaml"),
    )
    discover.add_argument("--overwrite", action="store_true")
    discover.add_argument("--refresh-cache", action="store_true")

    plan = subparsers.add_parser(
        "build-plan",
        help="Route a discovery result and write a runnable tracking config.",
    )
    plan.add_argument("--source", type=Path, required=True)
    plan.add_argument("--discovery", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--output-video", type=Path, required=True)
    plan.add_argument(
        "--profile",
        choices=("realtime", "realtime_stable", "balanced", "accuracy"),
        default="realtime",
    )
    plan.add_argument(
        "--tracker",
        choices=(
            "fasttrack",
            "tracktrack",
            "ocsort",
            "deepocsort_reid",
            "botsort_reid",
        ),
        default=None,
        help="Override the profile-specific tracker selection.",
    )
    plan.add_argument(
        "--tracker-config",
        default=None,
        help="Override the profile-specific tracker config.",
    )
    plan.add_argument("--device", default="auto")
    plan.add_argument("--max-frames", type=int, default=None)
    plan.add_argument("--overwrite", action="store_true")

    grounding_plan = subparsers.add_parser(
        "build-grounding-plan",
        help="Schedule LocateAnything only for uncertain or open classes.",
    )
    grounding_plan.add_argument("--discovery", type=Path, required=True)
    grounding_plan.add_argument("--output", type=Path, required=True)
    grounding_plan.add_argument("--confidence-trigger", type=float, default=0.65)
    grounding_plan.add_argument("--max-classes", type=int, default=4)
    grounding_plan.add_argument("--max-keyframes-per-class", type=int, default=2)
    grounding_plan.add_argument("--max-expected-tracks-per-class", type=int, default=3)
    grounding_plan.add_argument("--reacquisition-min-gap-frames", type=int, default=15)
    grounding_plan.add_argument("--max-reacquisition-tracks", type=int, default=3)
    grounding_plan.add_argument("--qwen-answer", type=Path, default=None)
    grounding_plan.add_argument(
        "--semantic-context",
        type=Path,
        default=None,
        help="Optional VLM context providing representative frames/crops per track.",
    )
    grounding_plan.add_argument(
        "--verify-track-ids",
        type=_positive_track_ids,
        default=None,
        help="Comma-separated track IDs for a controlled Locate-only benchmark.",
    )
    grounding_plan.add_argument("--overwrite", action="store_true")

    grounding = subparsers.add_parser(
        "execute-grounding-plan",
        help="Run LocateAnything and associate grounded boxes with track IDs.",
    )
    grounding.add_argument("--plan", type=Path, required=True)
    grounding.add_argument("--tracks", type=Path, required=True)
    grounding.add_argument("--output", type=Path, required=True)
    grounding.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/cache/locateanything"),
    )
    grounding.add_argument("--model-id", default="nvidia/LocateAnything-3B")
    grounding.add_argument("--device", default="cuda")
    grounding.add_argument("--torch-dtype", default="auto")
    grounding.add_argument("--quantization", choices=("none", "8bit", "4bit"), default="8bit")
    grounding.add_argument("--max-new-tokens", type=int, default=512)
    grounding.add_argument("--minimum-iou", type=float, default=0.10)
    grounding.add_argument("--target-crop-padding", type=float, default=1.0)
    grounding.add_argument("--target-crop-size", type=int, default=384)
    grounding.add_argument("--overwrite", action="store_true")

    fusion = subparsers.add_parser(
        "fuse-semantics",
        help="Fuse Qwen/LocateAnything evidence with unknown rejection.",
    )
    fusion.add_argument("--qwen-answer", type=Path, default=None)
    fusion.add_argument("--locate-result", type=Path, default=None)
    fusion.add_argument("--output", type=Path, required=True)
    fusion.add_argument("--unknown-threshold", type=float, default=0.45)
    fusion.add_argument("--minimum-margin", type=float, default=0.10)
    fusion.add_argument("--fine-unknown-threshold", type=float, default=0.85)
    fusion.add_argument("--fine-minimum-margin", type=float, default=0.15)
    fusion.add_argument("--semantic-memory", type=Path, default=None)
    fusion.add_argument("--memory-context-id", default=None)
    fusion.add_argument("--max-memory-observations", type=int, default=32)
    fusion.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/ontology/vocabulary_registry.yaml"),
    )
    fusion.add_argument("--overwrite", action="store_true")

    render = subparsers.add_parser(
        "render-semantics",
        help="Render fused labels while keeping rejected tracks as unknown.",
    )
    render.add_argument("--source", type=Path, required=True)
    render.add_argument("--tracks", type=Path, required=True)
    render.add_argument("--semantics", type=Path, required=True)
    render.add_argument("--output-video", type=Path, required=True)
    render.add_argument("--max-frames", type=int, default=None)
    render.add_argument("--hide-confidence", action="store_true")
    render.add_argument("--overwrite", action="store_true")

    report = subparsers.add_parser(
        "build-run-report",
        help="Consolidate timing, VRAM, coverage, and artifact provenance.",
    )
    report.add_argument("--run-root", type=Path, required=True)
    report.add_argument("--tracking-metadata", type=Path, required=True)
    report.add_argument("--semantic-metadata", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--overwrite", action="store_true")
    return parser


def _discover(args: argparse.Namespace) -> dict:
    if not args.source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {args.source}")
    registry_path = resolve_project_path(args.registry)
    if not registry_path.is_file():
        raise FileNotFoundError(f"Vocabulary registry does not exist: {registry_path}")
    sampling = {
        "max_keyframes": args.max_keyframes,
        "sample_fps": args.sample_fps,
        "transition_threshold": args.transition_threshold,
        "max_classes": args.max_classes,
        "max_new_tokens": args.max_new_tokens,
        "quantization": args.quantization,
        "torch_dtype": args.torch_dtype,
        "registry_sha256": file_sha256(registry_path),
    }
    cache_key = discovery_cache_key(
        args.source,
        model_id=args.model_id,
        prompt_version=PROMPT_VERSION,
        sampling=sampling,
    )
    cache = SemanticCache(args.cache_root)
    cached = None if args.refresh_cache else cache.load(cache_key)
    if cached is not None and not all(
        Path(row.get("path", "")).is_file() for row in cached.keyframes
    ):
        cached = None
    if cached is not None:
        cached = _materialize_cached_keyframes(cached, args.output.parent / "keyframes")
        _write_json_atomic(args.output, cached.to_dict())
        return _discovery_summary(cached, "hit", cache_key, args.output)
    if args.output.exists() and not args.overwrite:
        existing = SceneDiscovery.from_dict(
            json.loads(args.output.read_text(encoding="utf-8"))
        )
        if _discovery_matches_request(existing, args.source, args, registry_path):
            return _discovery_summary(
                existing,
                "output_reused",
                cache_key,
                args.output,
            )
        raise FileExistsError(
            "Discovery output exists but does not match the requested source/model/"
            f"sampling settings: {args.output}. Pass --overwrite."
        )
    model_config = SimpleNamespace(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        quantization=args.quantization,
    )
    _reset_cuda_peak_memory()
    model_load_started = time.perf_counter()
    model, processor = load_qwen_model(model_config)
    model_load_seconds = time.perf_counter() - model_load_started
    discovery = discover_scene(
        args.source,
        model=model,
        processor=processor,
        n_frames=args.max_keyframes,
        output_path=args.output,
        overwrite=args.overwrite,
        max_classes=args.max_classes,
        sample_fps=args.sample_fps,
        transition_threshold=args.transition_threshold,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        registry_path=registry_path,
    )
    metadata = dict(discovery.metadata)
    metadata.update(
        {
            "model_load_seconds": model_load_seconds,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "quantization": args.quantization,
            "cuda_memory": _cuda_memory_metrics(),
        }
    )
    discovery = replace(discovery, metadata=metadata)
    _write_json_atomic(args.output, discovery.to_dict())
    cache_path = cache.save(cache_key, discovery)
    summary = _discovery_summary(discovery, "miss", cache_key, args.output)
    summary["cache_path"] = str(cache_path)
    return summary


def _discovery_matches_request(
    discovery: SceneDiscovery,
    source: Path,
    args: argparse.Namespace,
    registry_path: Path,
) -> bool:
    metadata = discovery.metadata
    return (
        Path(discovery.source_video).resolve() == source.resolve()
        and discovery.model_id == args.model_id
        and discovery.prompt_version == PROMPT_VERSION
        and int(metadata.get("max_classes", -1)) == args.max_classes
        and int(metadata.get("max_new_tokens", -1)) == args.max_new_tokens
        and metadata.get("quantization") == args.quantization
        and metadata.get("torch_dtype") == args.torch_dtype
        and float(metadata.get("sample_fps", -1.0)) == args.sample_fps
        and float(metadata.get("transition_threshold", -1.0))
        == args.transition_threshold
        and metadata.get("registry_sha256") == file_sha256(registry_path)
    )


def _materialize_cached_keyframes(
    discovery: SceneDiscovery,
    destination: Path,
) -> SceneDiscovery:
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for row in discovery.keyframes:
        source = Path(row["path"])
        target = destination / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        updated = dict(row)
        updated["path"] = str(target.resolve())
        records.append(updated)
    return replace(discovery, keyframes=tuple(records))


def _discovery_summary(
    discovery: SceneDiscovery,
    cache_status: str,
    cache_key: str,
    output: Path,
) -> dict:
    return {
        "status": "ok",
        "cache_status": cache_status,
        "cache_key": cache_key,
        "output": str(output),
        "domain": discovery.domain,
        "classes": [item.canonical_name for item in discovery.detector_objects],
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _reset_cuda_peak_memory() -> None:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        return


def _cuda_memory_metrics() -> dict[str, int | None]:
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return {
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "total_bytes": None,
            }
        properties = torch.cuda.get_device_properties(0)
        return {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "total_bytes": int(properties.total_memory),
        }
    except (ImportError, RuntimeError):
        return {
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "total_bytes": None,
        }


def _build_plan(args: argparse.Namespace) -> dict:
    if not args.source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {args.source}")
    if not args.discovery.is_file():
        raise FileNotFoundError(f"Discovery result does not exist: {args.discovery}")
    discovery = SceneDiscovery.from_dict(
        json.loads(args.discovery.read_text(encoding="utf-8"))
    )
    route = build_detector_route(discovery, profile=args.profile)
    payload = build_tracking_payload(
        source_video=args.source,
        output_video=args.output_video,
        route=route,
        tracker_name=args.tracker,
        tracker_config=args.tracker_config,
        device=args.device,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
    )
    paths = write_adaptive_plan(
        output_dir=args.output_dir,
        discovery=discovery,
        route=route,
        tracking_payload=payload,
        overwrite=args.overwrite,
    )
    return {
        "status": "ok",
        "domain": discovery.domain,
        "route": route.to_dict(),
        "paths": {key: str(path) for key, path in paths.items()},
    }


def _build_grounding_plan(args: argparse.Namespace) -> dict:
    discovery = SceneDiscovery.from_dict(
        json.loads(args.discovery.read_text(encoding="utf-8"))
    )
    return build_grounding_plan(
        discovery,
        output_path=args.output,
        confidence_trigger=args.confidence_trigger,
        max_classes=args.max_classes,
        max_keyframes_per_class=args.max_keyframes_per_class,
        max_expected_tracks_per_class=args.max_expected_tracks_per_class,
        qwen_answer=args.qwen_answer,
        semantic_context=args.semantic_context,
        verify_track_ids=args.verify_track_ids,
        reacquisition_min_gap_frames=args.reacquisition_min_gap_frames,
        max_reacquisition_tracks=args.max_reacquisition_tracks,
        overwrite=args.overwrite,
    )


def _execute_grounding_plan(args: argparse.Namespace) -> dict:
    return execute_grounding_plan(
        plan_path=args.plan,
        tracks_path=args.tracks,
        output_path=args.output,
        cache_dir=args.cache_dir,
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
        minimum_iou=args.minimum_iou,
        target_crop_padding=args.target_crop_padding,
        target_crop_size=args.target_crop_size,
        overwrite=args.overwrite,
    )


def _fuse_semantics(args: argparse.Namespace) -> dict:
    return fuse_semantic_files(
        qwen_answer=args.qwen_answer,
        locate_result=args.locate_result,
        output_path=args.output,
        unknown_threshold=args.unknown_threshold,
        minimum_margin=args.minimum_margin,
        fine_unknown_threshold=args.fine_unknown_threshold,
        fine_minimum_margin=args.fine_minimum_margin,
        registry_path=args.registry,
        memory_path=args.semantic_memory,
        memory_context_id=args.memory_context_id,
        max_memory_observations_per_track=args.max_memory_observations,
        overwrite=args.overwrite,
    )


def _render_semantics(args: argparse.Namespace) -> dict:
    return render_semantic_video(
        source_video=args.source,
        tracks_path=args.tracks,
        semantics_path=args.semantics,
        output_video=args.output_video,
        overwrite=args.overwrite,
        show_confidence=not args.hide_confidence,
        max_frames=args.max_frames,
    )


def _build_run_report(args: argparse.Namespace) -> dict:
    return build_adaptive_run_report(
        run_root=args.run_root,
        tracking_metadata=args.tracking_metadata,
        semantic_metadata=args.semantic_metadata,
        output_path=args.output,
        overwrite=args.overwrite,
    )


def _run_command(args: argparse.Namespace) -> dict:
    commands = {
        "discover": _discover,
        "build-plan": _build_plan,
        "build-grounding-plan": _build_grounding_plan,
        "execute-grounding-plan": _execute_grounding_plan,
        "fuse-semantics": _fuse_semantics,
        "render-semantics": _render_semantics,
        "build-run-report": _build_run_report,
    }
    return commands[args.command](args)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run_command(args)
    except (
        FileNotFoundError,
        FileExistsError,
        VlmModelLoadError,
        RuntimeError,
        ValueError,
    ) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
