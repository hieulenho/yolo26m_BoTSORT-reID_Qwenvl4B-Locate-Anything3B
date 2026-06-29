# Football Player Detection and Multi-Object Tracking

This project will use YOLOv8m to detect football players and DeepSORT to keep stable track IDs across video frames.

Current scope: Milestone 2 data preparation. The repository contains project structure, Python package configuration, base YAML config, logging, CLI health checks, PowerShell environment setup, foundational tests, and a data pipeline that prepares detection and tracking ground truth.

Planned architecture:

```text
video -> YOLOv8m -> DeepSORT -> MOT evaluation
```

Not implemented in this milestone: dataset download, YOLOv8m training, YOLO inference, DeepSORT tracking integration, TrackEval, SORT baseline, dashboards, mAP, HOTA, MOTA, or IDF1 metrics.

## Requirements

- Windows 10 or Windows 11
- Python 3.12.x
- Python Launcher for Windows
- Git
- NVIDIA GPU is optional

PyTorch is intentionally not listed in `requirements/base.txt`. Install PyTorch separately for your CPU/CUDA environment from the official PyTorch instructions so an existing CUDA-compatible install is not overwritten by accident.

## Check Python

From PowerShell:

```powershell
py -0p
py -3.12 --version
```

Expected Python version:

```text
Python 3.12.x
```

## Setup

Recommended setup:

```powershell
cd F:\Tracking
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_env.ps1
```

Manual environment creation:

```powershell
cd F:\Tracking
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Check the active interpreter:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

Expected result:

```text
Python 3.12.x
F:\Tracking\.venv\Scripts\python.exe
```

If an older Python 3.11 virtual environment exists, recreate it. Python cannot be upgraded inside an existing virtual environment.

To keep the old environment as a backup:

```powershell
deactivate
Rename-Item .venv .venv-py311-backup
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

After the Python 3.12 environment is ready, install project dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/dev.txt
python -m pip install --editable .
```

Install PyTorch separately for the CUDA version on this machine before treating the environment as fully complete.

## Doctor

```powershell
python -m football_tracking.cli doctor
```

The doctor command checks the project root, Python 3.12 runtime, Python executable location, operating system, config loading, output writability, PyTorch/CUDA, Ultralytics, OpenCV, NumPy, and DeepSORT availability.

## Milestone 2 Data Pipeline

The data pipeline converts football tracking annotations into two ground-truth formats:

YOLO detection labels:

```text
class_id x_center y_center width height
```

MOTChallenge tracking labels:

```text
frame,track_id,left,top,width,height,confidence,class,visibility
```

YOLO ground truth is needed for detector training/evaluation such as mAP. MOT ground truth keeps `track_id`, so later milestones can evaluate tracking metrics such as HOTA, MOTA, and IDF1.

The MVP class set is intentionally single-class:

```text
0: player
```

Player-like source labels such as `player`, `player team left`, `player home`, and `goalkeeper` are mapped to `player` through `configs/class_mapping.yaml`. Referees, staff, ball, audience, and unknown classes are skipped unless the mapping config is changed.

Place real raw datasets under:

```text
data/raw/
```

The SoccerNet adapter currently discovers sequence directories that contain a frame directory plus an explicit annotation JSON file. Real SoccerNet layouts can vary; the adapter validates the discovered layout and raises a descriptive error when it cannot recognize the annotation schema. It does not download datasets and does not invent missing annotations.

Run a dry-run on the fixture config:

```powershell
python -m football_tracking.cli prepare-data --config configs/data_test.yaml --dry-run
```

Run the fixture pipeline:

```powershell
python -m football_tracking.cli prepare-data --config configs/data_test.yaml
```

Run against the default raw dataset location:

```powershell
python -m football_tracking.cli prepare-data --config configs/data.yaml --dry-run
python -m football_tracking.cli prepare-data --config configs/data.yaml
python -m football_tracking.cli validate-data --config configs/data.yaml
python -m football_tracking.cli audit-data --config configs/data.yaml
python -m football_tracking.cli visualize-annotations --config configs/data.yaml --num-samples 10
```

Generated outputs:

```text
data/interim/      split files and dataset manifest
data/yolo/         YOLO images, labels, dataset.yaml, manifest
data/mot/          MOTChallenge train/val/test sequences and seqmaps
outputs/metrics/  validation and audit reports
outputs/figures/  annotation visualization samples
```

Splits are made by sequence, not by frame, so one video/sequence cannot leak across train, validation, and test.

## Tests

```powershell
python -m pytest -q
```

Optional lint check:

```powershell
python -m ruff check src tests
```

## Next Milestones

Future milestones may add dataset preparation, detector training, tracking, evaluation, and reporting. Those workflows are not implemented here yet.
