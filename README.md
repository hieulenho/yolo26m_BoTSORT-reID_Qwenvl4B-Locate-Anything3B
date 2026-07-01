# Football Player Detection and Multi-Object Tracking

YOLOv8m detector training, SORT/DeepSORT tracking, TrackEval evaluation, benchmark tables, figures, and demo video rendering for football player tracking.

## Overview

This repository is a reproducible benchmark pipeline for football player detection and multi-object tracking. It prepares SportsMOT football sequences, fine-tunes YOLOv8m, caches detections once, compares SORT and DeepSORT from the same detector outputs, evaluates with TrackEval, renders annotated videos, and generates final reports.

Core flow:

```text
SportsMOT frames + GT
-> YOLO dataset / MOT dataset
-> YOLOv8m fine-tuning and detector evaluation
-> shared detection cache
-> SORT and DeepSORT
-> TrackEval metrics, figures, videos, reports
```

## Features

- SportsMOT football adapter with YOLO and MOTChallenge exports.
- YOLOv8m training, resume, preflight validation, and evaluation commands.
- Shared detection cache so SORT and DeepSORT compare against identical boxes.
- SORT baseline and DeepSORT tracker integration without Ultralytics built-in tracking.
- Official TrackEval integration for HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, and FN.
- OpenCV video rendering with bbox, track id, confidence, FPS, frame, tracker, and sequence overlays.
- Matplotlib figures for detector, tracker, speed-vs-quality, and per-sequence metrics.
- CSV, JSON, Markdown benchmark outputs and final Markdown report generation.
- Demo scripts, GitHub Actions, Dockerfile, docker-compose, tests, ruff, and MIT license.

Main CLI commands:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli doctor
.\.venv\Scripts\python.exe -m football_tracking.cli prepare-dataset --help
.\.venv\Scripts\python.exe -m football_tracking.cli train-detector --help
.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-detector --help
.\.venv\Scripts\python.exe -m football_tracking.cli cache-detections --help
.\.venv\Scripts\python.exe -m football_tracking.cli track --help
.\.venv\Scripts\python.exe -m football_tracking.cli compare-trackers --help
.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-tracking --help
.\.venv\Scripts\python.exe -m football_tracking.cli render-video --help
.\.venv\Scripts\python.exe -m football_tracking.cli benchmark --help
.\.venv\Scripts\python.exe -m football_tracking.cli generate-report --help
.\.venv\Scripts\python.exe -m football_tracking.cli summarize-experiments --help
```

## Project Structure

```text
configs/                 YAML configs for data, training, cache, tracking, reports
data/raw/sportsmot/      downloaded SportsMOT source data
data/yolo/               prepared YOLO datasets
data/mot/                prepared MOTChallenge datasets
models/detector/         exported detector checkpoints, ignored by Git
src/football_tracking/   package source
outputs/detections/      predictions and shared detection cache
outputs/tracks/          MOT tracker outputs
outputs/videos/          rendered MP4 demos
outputs/metrics/         JSON/CSV/Markdown metrics and reports
outputs/figures/         matplotlib figures
outputs/reports/         final project reports
demo/                    one-command demo scripts
tests/                   pytest suite
```

## Installation

Windows / PowerShell:

```powershell
cd F:\Tracking
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pip install --editable .
.\.venv\Scripts\python.exe -m football_tracking.cli doctor
```

Install PyTorch for your CUDA/CPU environment from the official PyTorch instructions before GPU training. The project does not pin PyTorch in `requirements/base.txt` so it does not overwrite a working CUDA install.

## Dataset

SportsMOT is the recommended real dataset because it provides football videos, MOT-style ground truth, track IDs, and official football sequence lists.

Download train and val:

```powershell
.\scripts\download_sportsmot.ps1 -Split "train,val"
```

Prepare football-only YOLO and MOT data:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli prepare-dataset `
  --config configs/sportsmot_data.yaml `
  --overwrite
```

Validate and audit:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli validate-data --config configs/sportsmot_data.yaml
.\.venv\Scripts\python.exe -m football_tracking.cli audit-data --config configs/sportsmot_data.yaml
```

## Training

Run preflight first:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli preflight-training `
  --config configs/yolov8m_sportsmot_train.yaml
```

Smoke training:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli train-detector `
  --config configs/yolov8m_sportsmot_smoke.yaml `
  --device 0 `
  --overwrite
```

Full training:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli train-detector `
  --config configs/yolov8m_sportsmot_train.yaml `
  --device 0 `
  --overwrite
```

Resume:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli resume-detector `
  --checkpoint outputs/training/yolov8m_players/weights/last.pt
```

## Evaluation

Detector validation:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-detector `
  --config configs/yolov8m_sportsmot_eval.yaml
```

Detector test evaluation, after validation decisions are frozen:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-detector `
  --config configs/yolov8m_sportsmot_test.yaml
```

Evaluate existing tracker outputs with TrackEval:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-tracking `
  --config configs/compare_trackers.yaml
```

## Tracking

Create the shared detector cache. Use `--overwrite` when rerunning after an interrupted or older cache run:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli cache-detections `
  --config configs/detection_cache.yaml `
  --overwrite
```

Compare SORT and DeepSORT from the same cache:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli compare-trackers `
  --config configs/compare_trackers.yaml `
  --overwrite
```

Run the DeepSORT tracking pipeline directly:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli track `
  --config configs/track_sportsmot.yaml `
  --overwrite
```

## Benchmark

Generate benchmark artifacts:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli benchmark `
  --config configs/benchmark.yaml
```

Outputs:

```text
outputs/metrics/benchmark/benchmark.csv
outputs/metrics/benchmark/benchmark.json
outputs/metrics/benchmark/benchmark.md
outputs/figures/benchmark/
```

The benchmark table includes detector, tracker, mAP50, mAP50-95, precision, recall, HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN, and FPS.

## Visualization

Render annotated videos:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli render-video `
  --config configs/render_video.yaml `
  --overwrite
```

Generated videos are written under `outputs/videos/rendered/<tracker>/<split>/`. The renderer keeps the source FPS and resolution and overlays bbox, track id, confidence, FPS, frame number, tracker name, and sequence name.

Figures are written with matplotlib only, under `outputs/figures/`.

## Demo

Quick smoke pipeline:

```powershell
.\demo\demo.ps1 -Mode smoke -Device 0
```

Full pipeline:

```powershell
.\demo\demo.ps1 -Mode full -Device 0
```

Bash:

```bash
./demo/demo.sh smoke
./demo/demo.sh full
```

## Results

Current full validation comparison on the prepared SportsMOT football validation split:

| Tracker | Frames | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Tracker FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SORT | 2900 | 43.060 | 60.344 | 30.860 | 60.960 | 40.727 | 788 | 10246 | 1554 | 190.171 |
| DeepSORT | 2900 | 50.570 | 62.218 | 41.331 | 60.886 | 49.022 | 539 | 10572 | 1501 | 15.001 |

DeepSORT improves HOTA, AssA, and IDF1 on this validation run, while SORT is much faster. TrackEval is the source of truth for official tracking metrics.

Generate the final report:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli generate-report `
  --config configs/report.yaml
```

## Citation

If you use this project, cite the upstream datasets and libraries used in your experiment, including SportsMOT, YOLO/Ultralytics, SORT, DeepSORT, and TrackEval.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
