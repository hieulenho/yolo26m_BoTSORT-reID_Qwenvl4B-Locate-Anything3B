"""Process bounded realtime semantic batches with one persistent Qwen session."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from football_tracking.adaptive_tracking.semantic_queue import (
    SemanticQueueError,
    prepare_pending_events_with_locate,
    process_semantic_queue,
)
from football_tracking.vlm.config import load_vlm_tracking_config
from football_tracking.vlm.qwen_runner import QwenVlmBatchSession


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _pending_count(queue_dir: Path) -> int:
    return len(list((queue_dir / "pending").glob("*.json")))


def _process_is_alive(process_id: int) -> bool:
    if process_id <= 0:
        return True
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            process_id,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    return True


def _shutdown_reason(args: argparse.Namespace) -> str | None:
    if args.stop_file is not None and args.stop_file.exists():
        return "stop_requested"
    if args.parent_pid > 0 and not _process_is_alive(args.parent_pid):
        return "parent_process_exited"
    return None


def _write_worker_status(
    args: argparse.Namespace,
    state: str,
    *,
    detail: str | None = None,
) -> None:
    if args.status_file is None:
        return
    _write_json(
        args.status_file,
        {
            "status": state,
            "detail": detail,
            "pid": os.getpid(),
            "parent_pid": args.parent_pid or None,
            "pending_event_count": _pending_count(args.queue_dir),
            "updated_unix_seconds": time.time(),
        },
    )


def _recover_processing_events(queue_dir: Path) -> int:
    pending_dir = queue_dir / "pending"
    processing_dir = queue_dir / "processing"
    pending_dir.mkdir(parents=True, exist_ok=True)
    recovered = 0
    for path in sorted(processing_dir.glob("*.json")):
        target = pending_dir / path.name
        if target.exists():
            path.unlink()
        else:
            path.replace(target)
        recovered += 1
    return recovered


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    totals = {
        "processed": 0,
        "failed": 0,
        "batches": 0,
        "qwen_calls": 0,
        "grouped_events": 0,
    }
    results: list[dict[str, Any]] = []
    recovered = _recover_processing_events(args.queue_dir)
    exit_reason = "completed"

    if args.watch:
        _write_worker_status(args, "waiting_for_events")
        while _pending_count(args.queue_dir) == 0:
            shutdown_reason = _shutdown_reason(args)
            if shutdown_reason is not None:
                summary = _worker_summary(args, totals, results, started)
                summary["exit_reason"] = shutdown_reason
                summary["recovered_processing_events"] = recovered
                return summary
            time.sleep(args.poll_interval)

    if _pending_count(args.queue_dir) == 0:
        result = {"status": "idle", "processed_event_count": 0}
        results.append(result)
        summary = _worker_summary(args, totals, results, started)
        summary["exit_reason"] = "queue_empty"
        summary["recovered_processing_events"] = recovered
        return summary

    locate_result: dict[str, Any] | None = None
    if args.locate_first:
        locate_limit = (
            args.max_total_events
            if args.max_total_events > 0
            else _pending_count(args.queue_dir)
        )
        locate_result = prepare_pending_events_with_locate(
            queue_dir=args.queue_dir,
            max_events=locate_limit,
            model_id=args.locate_model_id,
            device=args.locate_device,
            quantization=args.locate_quantization,
            max_new_tokens=args.locate_max_new_tokens,
            image_max_pixels=args.locate_image_max_pixels,
            minimum_association_score=args.locate_minimum_association_score,
        )
        results.append({"locateanything": locate_result})

    shutdown_reason = _shutdown_reason(args)
    if shutdown_reason is not None:
        summary = _worker_summary(args, totals, results, started)
        summary["exit_reason"] = shutdown_reason
        summary["recovered_processing_events"] = recovered
        return summary

    _write_worker_status(args, "loading_qwen")
    config = load_vlm_tracking_config(args.vlm_config, overrides={"run_model": True})
    with QwenVlmBatchSession(config) as session:
        _write_worker_status(args, "ready")
        while True:
            shutdown_reason = _shutdown_reason(args)
            if shutdown_reason is not None:
                exit_reason = shutdown_reason
                break
            _write_worker_status(args, "processing")
            result = process_semantic_queue(
                queue_dir=args.queue_dir,
                vlm_config_path=args.vlm_config,
                semantic_output=args.semantic_output,
                memory_path=args.memory,
                registry_path=args.registry,
                max_events=args.max_events,
                group_events=args.group_events,
                max_group_images=args.max_group_images,
                runner=session.run,
            )
            results.append(result)
            totals["processed"] += int(result.get("processed_event_count", 0))
            totals["failed"] += int(result.get("failed_event_count", 0))
            totals["qwen_calls"] += int(result.get("qwen_call_count", 0))
            if bool(result.get("grouped_events")):
                totals["grouped_events"] += int(
                    result.get("processed_event_count", 0)
                )
            if int(result.get("qwen_call_count", 0)) > 0:
                totals["batches"] += 1

            pending = _pending_count(args.queue_dir)
            shutdown_reason = _shutdown_reason(args)
            reached_limit = (
                args.max_total_events > 0
                and totals["processed"] + totals["failed"] >= args.max_total_events
            )
            if reached_limit:
                exit_reason = "event_limit_reached"
                break
            if shutdown_reason is not None:
                exit_reason = shutdown_reason
                break
            if not args.watch and not args.drain:
                exit_reason = "single_batch_complete"
                break
            if not args.watch and pending == 0:
                exit_reason = "queue_drained"
                break
            if pending == 0:
                _write_worker_status(args, "waiting_for_events")
                time.sleep(args.poll_interval)
            elif result.get("status") in {"idle", "waiting_for_evidence"}:
                _write_worker_status(args, "collecting_evidence")
                time.sleep(
                    min(
                        max(
                            float(result.get("next_ready_seconds", args.poll_interval)),
                            0.05,
                        ),
                        args.poll_interval,
                    )
                )

    summary = _worker_summary(args, totals, results, started)
    summary["locateanything"] = locate_result
    summary["exit_reason"] = exit_reason
    summary["recovered_processing_events"] = recovered
    return summary


def _worker_summary(
    args: argparse.Namespace,
    totals: dict[str, int],
    results: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    return {
        "status": "ok" if totals["failed"] == 0 else "completed_with_failures",
        "mode": "watch" if args.watch else "drain" if args.drain else "single_batch",
        "processed_event_count": totals["processed"],
        "failed_event_count": totals["failed"],
        "batch_count": totals["batches"],
        "qwen_call_count": totals["qwen_calls"],
        "grouped_event_count": totals["grouped_events"],
        "remaining_event_count": _pending_count(args.queue_dir),
        "elapsed_seconds": time.perf_counter() - started,
        "last_result": results[-1] if results else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--vlm-config", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--group-events", action="store_true")
    parser.add_argument("--max-group-images", type=int, default=2)
    parser.add_argument("--max-total-events", type=int, default=0)
    parser.add_argument("--locate-first", action="store_true")
    parser.add_argument(
        "--locate-model-id",
        default="nvidia/LocateAnything-3B",
    )
    parser.add_argument("--locate-device", default="cuda")
    parser.add_argument(
        "--locate-quantization",
        choices=("none", "8bit", "4bit"),
        default="8bit",
    )
    parser.add_argument("--locate-max-new-tokens", type=int, default=256)
    parser.add_argument("--locate-image-max-pixels", type=int, default=256 * 256)
    parser.add_argument(
        "--locate-minimum-association-score",
        type=float,
        default=0.10,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--drain", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--pid-file", type=Path, default=None)
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/ontology/vocabulary_registry.yaml"),
    )
    args = parser.parse_args()
    if args.max_events < 1:
        parser.error("--max-events must be positive.")
    if args.max_total_events < 0:
        parser.error("--max-total-events must be non-negative.")
    if args.max_group_images < 1:
        parser.error("--max-group-images must be positive.")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive.")
    if args.parent_pid < 0:
        parser.error("--parent-pid must be non-negative.")
    if args.locate_first and args.watch:
        parser.error(
            "--locate-first is a two-phase deferred mode and cannot be combined "
            "with --watch. Use a separate Locate service for live semantics."
        )
    if args.locate_max_new_tokens < 1:
        parser.error("--locate-max-new-tokens must be positive.")
    if args.locate_image_max_pixels < 4096:
        parser.error("--locate-image-max-pixels must be at least 4096.")
    if not 0.0 <= args.locate_minimum_association_score <= 1.0:
        parser.error("--locate-minimum-association-score must be in [0, 1].")
    if args.pid_file is not None:
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(str(os.getpid()), encoding="utf-8")
    _write_worker_status(args, "starting")
    try:
        result = run_worker(args)
    except (SemanticQueueError, RuntimeError, ValueError, OSError) as exc:
        result = {
            "status": "failed",
            "error": str(exc),
            "remaining_event_count": _pending_count(args.queue_dir),
        }
        if args.report is not None:
            _write_json(args.report, result)
        _write_worker_status(args, "failed", detail=str(exc))
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    finally:
        if args.pid_file is not None and args.pid_file.is_file():
            try:
                owner = int(args.pid_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                owner = 0
            if owner == os.getpid():
                args.pid_file.unlink(missing_ok=True)
    if args.report is not None:
        _write_json(args.report, result)
    _write_worker_status(args, "stopped", detail=str(result.get("exit_reason", "")))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
