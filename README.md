# Football Player Detection and Multi-Object Tracking

This project will use YOLOv8m to detect football players and DeepSORT to keep stable track IDs across video frames.

Current scope: Milestone 1 environment migration only. The repository contains project structure, Python package configuration, base YAML config, logging, CLI health checks, PowerShell environment setup, and foundational tests.

Planned architecture:

```text
video -> YOLOv8m -> DeepSORT -> MOT evaluation
```

Not implemented in this milestone: dataset download, annotation conversion, YOLOv8m training, DeepSORT tracking integration, TrackEval, SORT baseline, dashboards, mAP, HOTA, or IDF1 metrics.

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
