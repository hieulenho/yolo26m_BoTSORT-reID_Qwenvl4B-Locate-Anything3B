"""Train/validation/test splitting by sequence."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

from football_tracking.data.schemas import SequenceInfo, SplitManifest


class SplitError(RuntimeError):
    """Raised when sequence split configuration is invalid."""


LOGGER = logging.getLogger(__name__)
VALID_SPLITS = {"train", "val", "test"}


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"Split ratios must sum to 1.0, got {total:.6f}.")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise SplitError("Split ratios must be non-negative.")


def _read_predefined_split(path: Path, seed: int, strategy: str) -> SplitManifest:
    if not path.is_file():
        raise SplitError(f"Predefined split file does not exist: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SplitError(f"Predefined split must be a mapping: {path}")
    return SplitManifest(
        seed=int(raw.get("seed", seed)),
        strategy=str(raw.get("strategy", strategy)),
        train=list(raw.get("train", [])),
        val=list(raw.get("val", [])),
        test=list(raw.get("test", [])),
    )


def _counts_for_small_dataset(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total >= 3 and train_ratio > 0 and val_ratio > 0 and test_ratio > 0:
        train = max(1, round(total * train_ratio))
        val = max(1, round(total * val_ratio))
        test = max(1, total - train - val)
        while train + val + test > total:
            if train >= val and train >= test and train > 1:
                train -= 1
            elif val >= test and val > 1:
                val -= 1
            else:
                test -= 1
        return train, val, test

    train = round(total * train_ratio)
    val = round(total * val_ratio)
    test = total - train - val
    return max(0, train), max(0, val), max(0, test)


def split_sequences(
    sequences: list[SequenceInfo],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    strategy: str = "sequence",
    predefined_split_file: Path | None = None,
) -> SplitManifest:
    if strategy != "sequence":
        raise SplitError(f"Unsupported split strategy: {strategy}")
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    sequence_names = [sequence.name for sequence in sequences]
    if len(sequence_names) != len(set(sequence_names)):
        raise SplitError("Sequence names must be unique before splitting.")

    if predefined_split_file is not None:
        manifest = _read_predefined_split(predefined_split_file, seed, strategy)
        validate_split_manifest(manifest, sequence_names)
        return manifest

    metadata_split = _split_from_sequence_metadata(sequences, seed, strategy)
    if metadata_split is not None:
        validate_split_manifest(metadata_split, sequence_names)
        return metadata_split

    shuffled = list(sequence_names)
    random.Random(seed).shuffle(shuffled)
    train_count, val_count, _test_count = _counts_for_small_dataset(
        len(shuffled),
        train_ratio,
        val_ratio,
        test_ratio,
    )
    train = sorted(shuffled[:train_count])
    val = sorted(shuffled[train_count : train_count + val_count])
    test = sorted(shuffled[train_count + val_count :])
    if len(shuffled) < 3:
        LOGGER.warning("Dataset has fewer than 3 sequences; some splits may be empty.")
    manifest = SplitManifest(seed=seed, strategy=strategy, train=train, val=val, test=test)
    validate_split_manifest(manifest, sequence_names)
    return manifest


def _split_from_sequence_metadata(
    sequences: list[SequenceInfo],
    seed: int,
    strategy: str,
) -> SplitManifest | None:
    split_values: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sequence in sequences:
        split_name = sequence.metadata.get("split")
        if split_name is None:
            return None
        normalized = str(split_name).lower()
        if normalized not in VALID_SPLITS:
            return None
        split_values[normalized].append(sequence.name)
    return SplitManifest(
        seed=seed,
        strategy=f"{strategy}_metadata",
        train=sorted(split_values["train"]),
        val=sorted(split_values["val"]),
        test=sorted(split_values["test"]),
    )


def validate_split_manifest(manifest: SplitManifest, expected_sequence_names: list[str]) -> None:
    split_sets = {name: set(values) for name, values in manifest.as_mapping().items()}
    overlaps: list[tuple[str, str, set[str]]] = []
    for left_name, left_values in split_sets.items():
        for right_name, right_values in split_sets.items():
            if left_name >= right_name:
                continue
            overlap = left_values & right_values
            if overlap:
                overlaps.append((left_name, right_name, overlap))
    if overlaps:
        raise SplitError(f"Sequence split leakage detected: {overlaps}")

    expected = set(expected_sequence_names)
    actual = set().union(*split_sets.values()) if split_sets else set()
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SplitError(f"Split mismatch. Missing={missing}; extra={extra}")


def write_split_manifest(manifest: SplitManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "seed": manifest.seed,
        "strategy": manifest.strategy,
        "train": manifest.train,
        "val": manifest.val,
        "test": manifest.test,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
