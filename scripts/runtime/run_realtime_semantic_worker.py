"""Process bounded realtime semantic batches with one persistent Qwen session."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from football_tracking.adaptive_tracking.semantic_queue import (
    SemanticQueueError,
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


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    totals = {"processed": 0, "failed": 0, "batches": 0}
    results: list[dict[str, Any]] = []

    if args.watch:
        while _pending_count(args.queue_dir) == 0:
            if args.stop_file is not None and args.stop_file.exists():
                return _worker_summary(args, totals, results, started)
            time.sleep(args.poll_interval)

    if _pending_count(args.queue_dir) == 0:
        result = {"status": "idle", "processed_event_count": 0}
        results.append(result)
        return _worker_summary(args, totals, results, started)

    config = load_vlm_tracking_config(args.vlm_config, overrides={"run_model": True})
    with QwenVlmBatchSession(config) as session:
        while True:
            result = process_semantic_queue(
                queue_dir=args.queue_dir,
                vlm_config_path=args.vlm_config,
                semantic_output=args.semantic_output,
                memory_path=args.memory,
                registry_path=args.registry,
                max_events=args.max_events,
                runner=session.run,
            )
            results.append(result)
            totals["processed"] += int(result.get("processed_event_count", 0))
            totals["failed"] += int(result.get("failed_event_count", 0))
            if result.get("status") != "idle":
                totals["batches"] += 1

            pending = _pending_count(args.queue_dir)
            stop_requested = args.stop_file is not None and args.stop_file.exists()
            reached_limit = (
                args.max_total_events > 0
                and totals["processed"] + totals["failed"] >= args.max_total_events
            )
            if reached_limit or (stop_requested and pending == 0):
                break
            if not args.watch and not args.drain:
                break
            if not args.watch and pending == 0:
                break
            if pending == 0:
                time.sleep(args.poll_interval)

    return _worker_summary(args, totals, results, started)


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
    parser.add_argument("--max-total-events", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--drain", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--stop-file", type=Path, default=None)
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
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive.")
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
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    if args.report is not None:
        _write_json(args.report, result)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
