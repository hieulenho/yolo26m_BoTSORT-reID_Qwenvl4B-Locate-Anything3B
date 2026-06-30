from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from football_tracking.data.sportsmot_adapter import (
    create_local_split,
    find_sportsmot_root,
    football_records,
    prepare_sportsmot,
    read_football_sequences,
    validate_records,
)


def _write_sequence(
    root: Path,
    split: str,
    name: str,
    rows: list[str] | None = None,
    frames: int = 4,
) -> Path:
    sequence_dir = root / split / name
    image_dir = sequence_dir / "img1"
    gt_dir = sequence_dir / "gt"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(1, frames + 1):
        (image_dir / f"{frame:06d}.jpg").write_bytes(b"not-a-real-jpeg")
    (sequence_dir / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={name}",
                "imDir=img1",
                "frameRate=30",
                f"seqLength={frames}",
                "imWidth=100",
                "imHeight=50",
                "imExt=.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )
    gt_rows = rows or [
        "1,7,10,5,20,10,1,1,1",
        "3,7,12,6,20,10,1,1,0.9",
        "3,8,40,10,15,18,1,1,0.8",
    ]
    (gt_dir / "gt.txt").write_text("\n".join(gt_rows) + "\n", encoding="utf-8")
    return sequence_dir


def _sportsmot_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "raw" / "sportsmot" / "SportsMOT-v1"
    (root / "splits_txt").mkdir(parents=True, exist_ok=True)
    football_names = [
        "train/football_alpha_c001",
        "football_beta_c001",
        "football_gamma_c001",
        "val/football_eval_c001",
    ]
    (root / "splits_txt" / "football.txt").write_text(
        "\n".join(football_names) + "\n",
        encoding="utf-8",
    )
    for name in ("football_alpha_c001", "football_beta_c001", "football_gamma_c001"):
        _write_sequence(root, "train", name)
    _write_sequence(root, "train", "basketball_alpha_c001")
    _write_sequence(root, "val", "football_eval_c001")
    return root


def _sportsmot_config(tmp_path: Path, raw_dir: Path) -> Path:
    config = tmp_path / "sportsmot_data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "sportsmot_football",
                    "adapter": "sportsmot",
                    "raw_dir": str(raw_dir),
                    "interim_dir": str(tmp_path / "interim"),
                },
                "split": {"seed": 42, "local_val_ratio": 0.20},
                "yolo": {
                    "output_dir": str(tmp_path / "yolo"),
                    "smoke_output_dir": str(tmp_path / "yolo_smoke"),
                    "decimal_places": 6,
                    "prefer_symlink": False,
                },
                "mot": {"output_dir": str(tmp_path / "mot")},
                "smoke": {
                    "max_train_sequences": 1,
                    "max_val_sequences": 1,
                    "max_train_frames": 1,
                    "max_val_frames": 1,
                },
                "runtime": {"overwrite": True, "dry_run": False},
            }
        ),
        encoding="utf-8",
    )
    return config


def test_sportsmot_discovers_football_only_and_grouped_splits(tmp_path: Path) -> None:
    root = _sportsmot_fixture(tmp_path)

    discovered = find_sportsmot_root(root.parent)
    names = read_football_sequences(discovered)
    records, summary = football_records(discovered)
    split_manifest, groups, warnings = create_local_split(records, seed=42, local_val_ratio=0.20)

    assert discovered == root
    assert "football_alpha_c001" in names
    assert {record.name for record in records} == {
        "football_alpha_c001",
        "football_beta_c001",
        "football_gamma_c001",
        "football_eval_c001",
    }
    assert "basketball_alpha_c001" in summary["non_football_train"]
    assert set(split_manifest.train).isdisjoint(split_manifest.val)
    assert set(split_manifest.train).isdisjoint(split_manifest.test)
    assert set(split_manifest.val).isdisjoint(split_manifest.test)
    assert all(sequence_names for sequence_names in groups.values())
    assert warnings == []


def test_sportsmot_prepare_writes_yolo_mot_and_smoke_outputs(tmp_path: Path) -> None:
    root = _sportsmot_fixture(tmp_path)
    config = _sportsmot_config(tmp_path, root.parent)

    result = prepare_sportsmot(config, overwrite=True)

    yolo_root = tmp_path / "yolo"
    smoke_root = tmp_path / "yolo_smoke"
    mot_root = tmp_path / "mot"
    assert (yolo_root / "dataset.yaml").is_file()
    assert (smoke_root / "dataset.yaml").is_file()
    assert (mot_root / "manifest.json").is_file()
    assert result["football_sequence_count"] == 4

    label_files = sorted((yolo_root / "labels").rglob("*.txt"))
    assert label_files
    assert any(path.read_text(encoding="utf-8") == "" for path in label_files)
    non_empty_label = next(path for path in label_files if path.read_text(encoding="utf-8").strip())
    label_fields = non_empty_label.read_text(encoding="utf-8").splitlines()[0].split()
    assert len(label_fields) == 5
    assert label_fields[0] == "0"

    mot_gt_text = "\n".join(path.read_text(encoding="utf-8") for path in mot_root.rglob("gt.txt"))
    assert "1,7,10,5,20,10,1,1,1" in mot_gt_text

    splits = result["splits"]
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["test"]) == {"football_eval_c001"}


def test_sportsmot_validation_reports_duplicate_frame_track(tmp_path: Path) -> None:
    root = _sportsmot_fixture(tmp_path)
    _write_sequence(
        root,
        "train",
        "football_alpha_c001",
        rows=[
            "1,7,10,5,20,10,1,1,1",
            "1,7,12,6,20,10,1,1,1",
        ],
    )
    records, _summary = football_records(root)

    report = validate_records(records)

    assert report.has_errors
    assert any("duplicate frame-track pair" in issue.message for issue in report.issues)


def test_download_sportsmot_dry_run_does_not_create_downloader_venv(tmp_path: Path) -> None:
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is required for the downloader script test.")
    script = Path("scripts/download_sportsmot.ps1").resolve()
    downloader_venv = Path("tools") / ".venv-download"
    existed_before = downloader_venv.exists()

    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Split",
            "train,val",
            "-Output",
            str(tmp_path / "raw" / "sportsmot"),
            "-CacheDir",
            str(tmp_path / "cache"),
            "-DryRun",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Would run: trackers download sportsmot" in result.stdout
    assert downloader_venv.exists() == existed_before
