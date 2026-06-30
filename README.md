# Football Player Detection and Multi-Object Tracking

This project will use YOLOv8m to detect football players and DeepSORT to keep stable track IDs across video frames.

Current scope: Milestone 3 dataset QA and YOLOv8m pretrained detection baseline. The repository contains the Milestone 1/2 project setup and data conversion pipeline, plus audit reports, dataset plots, annotation samples, pretrained detector inference, prediction serialization, timing, and baseline reporting.

Planned architecture:

```text
video -> YOLOv8m -> DeepSORT -> MOT evaluation
```

Not implemented in this milestone: dataset download, YOLOv8m fine-tuning/training, DeepSORT tracking integration, TrackEval, SORT baseline, dashboards, HOTA, MOTA, or IDF1 metrics.

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
.\.venv\Scripts\python.exe -m football_tracking.cli doctor
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
.\.venv\Scripts\python.exe -m football_tracking.cli prepare-data --config configs/data_test.yaml --dry-run
```

Run the fixture pipeline:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli prepare-data --config configs/data_test.yaml
```

Run against the default raw dataset location:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli prepare-data --config configs/data.yaml --dry-run
.\.venv\Scripts\python.exe -m football_tracking.cli prepare-data --config configs/data.yaml
.\.venv\Scripts\python.exe -m football_tracking.cli validate-data --config configs/data.yaml
.\.venv\Scripts\python.exe -m football_tracking.cli audit-data --config configs/data.yaml
.\.venv\Scripts\python.exe -m football_tracking.cli visualize-annotations --config configs/data.yaml --num-samples 10
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

## Milestone 3 Dataset Audit

The dataset audit checks the prepared YOLO/MOT data and the mapped annotations:

- sequence, frame, object, and track counts;
- bounding-box width, height, area ratio, aspect ratio, clipping, invalid boxes, and boundary-touch boxes;
- track length, single-frame tracks, frame gaps, continuous tracks, and duplicate track IDs within one frame;
- split counts and split leakage;
- source classes, target classes, ignored classes, unknown classes, goalkeeper-to-player mapping, and referee ignore counts;
- ground-truth annotation samples under split and sequence folders.

The default audit config uses the mini fixture when no real dataset exists:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli audit-data `
  --config configs/audit.yaml
```

Main audit outputs:

```text
outputs/metrics/dataset_audit_summary.json
outputs/metrics/dataset_audit_per_sequence.csv
outputs/metrics/dataset_audit_per_split.csv
outputs/metrics/dataset_audit_tracks.csv
outputs/metrics/dataset_audit_errors.json
outputs/figures/dataset_audit/
outputs/figures/annotation_samples/<split>/<sequence>/
```

JSON audit reports use `null` plus a reason when a statistic cannot be computed; they do not write NaN.

## Milestone 3 YOLOv8m Pretrained Baseline

The baseline uses `yolov8m.pt` pretrained on COCO. It keeps only COCO class `person` and maps it to the project target class:

```text
0: player
```

This is a pretrained baseline, not the final football detector. It is useful because it gives a reproducible reference before fine-tuning in Milestone 4. A fine-tuned model learns the football-specific annotation policy; the pretrained COCO model only knows the broad `person` category.

COCO `person` can include players, goalkeepers, referees, staff, coaches, and people outside the field. The MVP ground truth maps player-like labels and goalkeepers to `player`, while referees and staff are ignored. Because predictions must not be silently corrected with ground truth, this baseline can produce false positives on referees or staff.

Run a safe dry-run with no inference and no checkpoint download:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli run-baseline `
  --config configs/yolov8m_baseline.yaml `
  --dry-run
```

Run a small CPU smoke test after `yolov8m.pt` is available or Ultralytics is allowed to download it:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli run-baseline `
  --config configs/yolov8m_baseline.yaml `
  --device cpu `
  --max-images 20 `
  --overwrite
```

Run on CUDA device 0:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli run-baseline `
  --config configs/yolov8m_baseline.yaml `
  --device 0 `
  --max-images 100 `
  --overwrite
```

Inference-only and evaluation-only commands are also available:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli baseline-detect `
  --config configs/yolov8m_baseline.yaml `
  --max-images 20 `
  --overwrite

.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-baseline `
  --config configs/yolov8m_baseline.yaml
```

The baseline report distinguishes:

- `mAP@50`: AP at IoU threshold 0.50.
- `mAP@50:95`: mean AP over IoU thresholds 0.50 through 0.95 in steps of 0.05.

Metrics come from the Ultralytics validator when it can run. If it cannot run, the report writes `not available` / `null` with the reason rather than filling fake zeros.

Baseline outputs:

```text
outputs/detections/yolov8m_pretrained/predictions.jsonl
outputs/detections/yolov8m_pretrained/yolo_labels/
outputs/detections/yolov8m_pretrained/predictions_summary.csv
outputs/detections/yolov8m_pretrained/run_metadata.json
outputs/metrics/yolov8m_pretrained_baseline.json
outputs/metrics/yolov8m_pretrained_baseline.csv
outputs/metrics/yolov8m_pretrained_report.md
outputs/figures/yolov8m_pretrained/
```

Fine-tuning is planned for Milestone 4. DeepSORT tracking and tracking metrics are planned for Milestone 5.

## Milestone 4 YOLOv8m Fine-Tuning

Milestone 4 adds the detector fine-tuning and evaluation pipeline. It still handles object detection only; DeepSORT, SORT, TrackEval, HOTA, MOTA, and IDF1 are not implemented here.

The intended protocol is:

```text
train split -> training
val split   -> monitoring and best.pt selection
test split  -> one final evaluation after configuration is fixed
```

Do not tune thresholds or hyperparameters on the test split.

## SportsMOT Football Dataset

SportsMOT is the recommended real dataset path for this milestone because it
contains MOT-style annotations, track IDs, frames, and an official football
sequence list. The project prepares only football sequences:

```text
official train football -> local train/val by sequence group
official val football   -> local test
```

First check the downloader command without downloading anything:

```powershell
.\scripts\download_sportsmot.ps1 -Split "train,val" -DryRun
```

Download train and val when you are ready for a long network/disk operation:

```powershell
.\scripts\download_sportsmot.ps1 -Split "train,val"
```

The downloader uses a separate environment at `tools/.venv-download`, writes
raw data under `data/raw/sportsmot`, and caches ZIPs under `.cache/sportsmot`.
If the official downloader asks for terms or authentication, complete that step
through the official source and rerun the script; do not bypass access controls.

Prepare SportsMOT football YOLO and MOT outputs:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli prepare-sportsmot `
  --config configs/sportsmot_data.yaml `
  --overwrite
```

Validation and audit:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli validate-data `
  --config configs/sportsmot_data.yaml

.\.venv\Scripts\python.exe -m football_tracking.cli audit-data `
  --config configs/sportsmot_data.yaml
```

Main SportsMOT outputs:

```text
data/yolo/sportsmot_football/dataset.yaml
data/yolo/sportsmot_football_smoke/dataset.yaml
data/mot/sportsmot_football/
outputs/metrics/sportsmot_download_validation.json
outputs/metrics/sportsmot_football_audit.json
outputs/metrics/sportsmot_football_per_sequence.csv
```

Run training preflight:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli preflight-training `
  --config configs/yolov8m_train.yaml
```

Dry-run training without loading the full model:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli train-detector `
  --config configs/yolov8m_train.yaml `
  --dry-run
```

Smoke training:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli train-detector `
  --config configs/yolov8m_smoke.yaml
```

The default smoke config uses the checked-in mini fixture at
`data/yolo/mini_tracking_fixture/dataset.yaml`. After SportsMOT is prepared, use
the SportsMOT smoke config:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli train-detector `
  --config configs/yolov8m_sportsmot_smoke.yaml `
  --device 0
```

Validate the SportsMOT smoke checkpoint:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-detector `
  --config configs/yolov8m_sportsmot_smoke_eval.yaml
```

Full training:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli train-detector `
  --config configs/yolov8m_sportsmot_train.yaml `
  --device 0
```

Full training can take hours. Run the SportsMOT smoke command first and keep the
test split untouched until final reporting.

Resume from `last.pt`:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli resume-detector `
  --checkpoint outputs/training/yolov8m_players/weights/last.pt
```

Validation and test evaluation:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-detector `
  --config configs/yolov8m_sportsmot_eval.yaml

.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-detector `
  --config configs/yolov8m_sportsmot_test.yaml
```

You can also evaluate an explicit checkpoint without editing the YAML:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-detector `
  --config configs/yolov8m_eval.yaml `
  --checkpoint models/detector/yolov8m_players_best.pt
```

Compare pretrained and fine-tuned reports:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli compare-detectors
```

PowerShell wrappers are available and do not require activating the virtual environment:

```powershell
.\scripts\train_detector.ps1
.\scripts\resume_detector.ps1 -Checkpoint outputs/training/yolov8m_players/weights/last.pt
.\scripts\evaluate_detector.ps1 -Config configs/yolov8m_eval.yaml
```

For an RTX 4060 Laptop, start with `imgsz=960` and `batch=-1`. If CUDA runs out of memory, try `batch=4`, then `batch=2`, then `imgsz=640`, keep `cache=false`, and close other GPU applications. The repository does not assume one fixed batch size will always fit.

Metric names:

- `mAP@50`: AP at IoU 0.50.
- `mAP@50:95`: AP averaged from IoU 0.50 to 0.95 in steps of 0.05.

Do not use names like `mAP@95:50` or `mAP@9550`.

Fine-tuned detector outputs:

```text
outputs/training/<run_name>/experiment_manifest.json
outputs/training/<run_name>/results.csv
outputs/training/<run_name>/weights/best.pt
outputs/training/<run_name>/weights/last.pt
models/detector/*_best.pt
models/detector/*_last.pt
outputs/metrics/yolov8m_finetuned_val.json
outputs/metrics/yolov8m_finetuned_test.json
outputs/metrics/yolov8m_finetuned_report.md
outputs/figures/yolov8m_finetuned/
```

Model weights are ignored by Git.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Optional lint check:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
```

## Next Milestones

Milestone 5 may add DeepSORT tracking integration and MOT-style tracking evaluation. Those workflows are intentionally not part of Milestone 4.
