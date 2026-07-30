from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.runtime.run_realtime_semantic_worker import (
    _recover_processing_events,
    _shutdown_reason,
)


def test_worker_recovers_an_event_claimed_before_an_interrupted_run(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue"
    processing = queue / "processing"
    processing.mkdir(parents=True)
    claimed = processing / "event.json"
    claimed.write_text("{}", encoding="utf-8")

    assert _recover_processing_events(queue) == 1
    assert not claimed.exists()
    assert (queue / "pending" / "event.json").is_file()


def test_worker_honors_stop_file_without_draining_the_queue(tmp_path: Path) -> None:
    stop_file = tmp_path / "worker.stop"
    stop_file.touch()
    args = Namespace(stop_file=stop_file, parent_pid=0)

    assert _shutdown_reason(args) == "stop_requested"
