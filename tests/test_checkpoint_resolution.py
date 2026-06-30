from pathlib import Path

import pytest

from football_tracking.detection.checkpoint import (
    CheckpointError,
    compute_file_hash,
    copy_final_checkpoints,
    resolve_best_checkpoint,
    resolve_last_checkpoint,
    validate_checkpoint,
)


def test_checkpoint_resolution_hash_and_metadata(tmp_path: Path) -> None:
    weights = tmp_path / "run" / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    last = weights / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")

    copied = copy_final_checkpoints(tmp_path / "run", tmp_path / "models", "demo")

    assert resolve_best_checkpoint(tmp_path / "run") == best
    assert resolve_last_checkpoint(tmp_path / "run") == last
    assert compute_file_hash(best) == compute_file_hash(copied["best"])
    assert copied["metadata"].is_file()


def test_validate_checkpoint_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        validate_checkpoint(tmp_path / "missing.pt")
