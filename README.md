# Football Tracking

Reusable YOLO detection, multi-object tracking, TrackEval evaluation, and Qwen VLM analysis.

The project started as a football-player tracking benchmark, but the current structure is designed
to be reused across domains: football, generic people, vehicles, or any workflow where you want:

```text
video or image sequence
  -> detector
  -> tracker
  -> metrics / rendered video / MOT files
  -> VLM reasoning layer
```

The recommended current football stack is:

```text
YOLO26m fine-tuned on SportsMOT football
  -> BoT-SORT ReID
  -> TrackEval metrics
  -> optional Qwen3-VL-4B-Instruct analysis
```

## Current Status

Environment health is currently clean on the local Windows setup:

```text
Python 3.12.10
PyTorch 2.11.0+cu128
CUDA available
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Ultralytics 8.4.82
OpenCV 4.13.0
DeepSORT realtime 1.3.2
```

Latest full checks run successfully:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Result:

```text
ruff: all checks passed
pytest: 338 passed
```

## Latest Reference Metrics

Fine-tuned detector on `data/yolo/sportsmot_football/dataset.yaml`, validation split:

| Model | Checkpoint | Precision | Recall | mAP50 | mAP50-95 | mAP75 |
|---|---|---:|---:|---:|---:|---:|
| YOLO26m fine-tuned | `models/detector/football/yolo26m_best.pt` | 0.9595 | 0.9601 | 0.9793 | 0.8306 | 0.9536 |

Stable BoT-SORT ReID on all 30 SportsMOT football sequences:

| Tracker | Sequences | Frames | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BoT-SORT ReID identity-stable | 30 | 20171 | 68.503 | 80.080 | 58.643 | 88.451 | 71.352 | 895 | 8638 | 20764 | 13.672 |

Compared with the previous BoT-SORT ReID preset, the identity-stable preset reduced ID switches
from `1064` to `895` while keeping HOTA and IDF1 essentially unchanged. The trade-off is more FN.

## Project Layout

```text
configs/                  YAML configs and tracker presets
docs/                     deeper design notes and runbooks
requirements/             dependency groups
scripts/                  PowerShell wrappers for common workflows
src/football_tracking/    Python package
tests/                    pytest suite
```

Generated or large artifacts are ignored by Git:

```text
data/                     downloaded and converted datasets
models/                   promoted detector checkpoints
outputs/                  metrics, videos, tracks, caches, VLM artifacts
runs/                     Ultralytics run outputs
```

More detail:

- [Project structure](docs/PROJECT_STRUCTURE.md)
- [Config guide](configs/README.md)
- [Script guide](scripts/README.md)
- [Detector fine-tuning](docs/detector_finetuning.md)
- [Domain optimization](docs/domain_optimization.md)
- [Qwen VLM pipeline](docs/vlm_qwen4b_pipeline.md)

## Installation

Windows / PowerShell:

```powershell
cd F:\Tracking
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements\dev.txt
.\.venv\Scripts\python.exe -m pip install --editable .
.\.venv\Scripts\python.exe -m football_tracking.cli doctor
```

Install PyTorch for your CUDA or CPU environment before heavy training. The base requirements do
not pin PyTorch so a working CUDA install is not accidentally replaced.

Optional Qwen VLM dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\vlm.txt
```

## Most Common Commands

### Track One Video

This is the simplest command for a normal video:

```powershell
.\scripts\track_video.ps1 `
  -Source F:\videos\1.mp4 `
  -OutputVideo F:\videos\1_Tracking.mp4 `
  -Overwrite
```

Outputs are written beside the output video:

```text
F:\videos\1_Tracking.mp4
F:\videos\1_Tracking.txt
F:\videos\1_Tracking.metadata.json
```

### Track One Video And Prepare VLM Artifacts

This runs tracking, then creates keyframes, crops, `vlm_context.json`, and `prompt.md`.
It does not run Qwen unless `-RunModel` is included.

```powershell
.\scripts\track_video_qwen_vlm.ps1 `
  -Source F:\videos\1.mp4 `
  -OutputVideo F:\videos\1_Tracking.mp4 `
  -Overwrite
```

### Track One Video And Run Qwen

```powershell
.\scripts\track_video_qwen_vlm.ps1 `
  -Source F:\videos\1.mp4 `
  -OutputVideo F:\videos\1_Tracking.mp4 `
  -RunModel `
  -MaxKeyframes 4 `
  -MaxTracks 20 `
  -Overwrite
```

For an 8GB laptop GPU, keep `MaxKeyframes` and `MaxTracks` modest.

### Analyze Existing Tracking Output With Qwen

Use this when the video was already tracked:

```powershell
.\scripts\analyze_tracking_vlm.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -TrackedVideo F:\videos\1_Tracking.mp4 `
  -Tracks F:\videos\1_Tracking.txt `
  -Metadata F:\videos\1_Tracking.metadata.json `
  -OutputDir F:\videos\1_vlm `
  -RunModel `
  -MaxKeyframes 4 `
  -MaxTracks 20 `
  -Overwrite
```

## VLM Artifacts

The VLM layer is downstream of tracking. It does not replace YOLO or the tracker.

```text
video
  -> YOLO detector
  -> BoT-SORT ReID tracker
  -> MOT tracks + annotated video
  -> keyframes/crops/context JSON
  -> Qwen report
```

Typical output folder:

```text
F:\videos\1_vlm\
  vlm_context.json
  prompt.md
  keyframes/
  crops/
  vlm_answer.md
  vlm_answer.json
```

- `keyframes/`: full-frame images with tracking IDs drawn on top.
- `crops/`: cropped object images grouped by `track_id`.
- `vlm_context.json`: structured tracking metadata and `tracking_diagnostics` for downstream reasoning.
- `prompt.md`: the prompt sent to Qwen.
- `vlm_answer.md/json`: Qwen output, only present after `-RunModel`.

The Qwen runner prefers the local Hugging Face cache first. If the cache is incomplete, it may need
network access to download missing files.

On the local RTX 4060 Laptop 8GB setup, use 2 keyframes before trying anything larger:

```powershell
.\scripts\analyze_tracking_vlm.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -TrackedVideo F:\videos\1_Tracking_qwen.mp4 `
  -Tracks F:\videos\1_Tracking_qwen.txt `
  -Metadata F:\videos\1_Tracking_qwen.metadata.json `
  -OutputDir F:\videos\1_vlm_tracking_report `
  -RunModel `
  -TorchDtype float16 `
  -MaxKeyframes 2 `
  -MaxTracks 10 `
  -MaxCropsPerTrack 1 `
  -MaxNewTokens 768 `
  -Overwrite
```

Smoke check Qwen without touching your video-side VLM folder:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli analyze-tracking-vlm `
  --config configs\vlm_qwen4b_tracking.yaml `
  --source-video F:\videos\1.mp4 `
  --tracked-video F:\videos\1_Tracking_vlm.mp4 `
  --tracks F:\videos\1_Tracking_vlm.txt `
  --metadata F:\videos\1_Tracking_vlm.metadata.json `
  --output-dir outputs\vlm\qwen4b\smoke_check `
  --max-keyframes 1 `
  --max-tracks 5 `
  --max-crops-per-track 1 `
  --max-new-tokens 64 `
  --run-model `
  --overwrite
```

Expected result:

```text
model_result.status = ok
outputs/vlm/qwen4b/smoke_check/vlm_answer.md exists
```

## Dataset Workflow

SportsMOT is the recommended football dataset because it provides MOT-style ground truth and track
IDs.

Download train and validation splits:

```powershell
.\scripts\download_sportsmot.ps1 -Split "train,val"
```

Prepare YOLO and MOTChallenge formats:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli prepare-dataset `
  --config configs\sportsmot_data.yaml `
  --overwrite
```

Validate and audit:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli validate-data `
  --config configs\sportsmot_data.yaml

.\.venv\Scripts\python.exe -m football_tracking.cli audit-data `
  --config configs\sportsmot_data.yaml
```

## Detector Training

Preflight:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli preflight-training `
  --config configs\yolo26m_sportsmot_football_train.yaml
```

Full training:

```powershell
.\scripts\train_football_detector.ps1
```

Equivalent direct command:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli train-detector `
  --config configs\yolo26m_sportsmot_football_train.yaml `
  --device 0 `
  --overwrite
```

Fast local experiments:

```powershell
.\scripts\train_football_detector.ps1 `
  -Epochs 1 `
  -Fraction 0.1 `
  -Workers 0 `
  -NoVal `
  -Overwrite
```

Evaluate the active checkpoint:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli evaluate-detector `
  --config configs\yolo26m_sportsmot_football_eval.yaml
```

Important output paths:

```text
models/detector/football/yolo26m_best.pt
outputs/metrics/football/yolo26m/yolo26m_val.json
outputs/metrics/football/yolo26m/yolo26m_val.csv
```

## Tracking Evaluation

Create or refresh the shared detector cache:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli cache-detections `
  --config configs\detection_cache_yolo26m_all.yaml `
  --overwrite
```

Run the current stable BoT-SORT ReID evaluation across all 30 SportsMOT football sequences:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli compare-trackers `
  --config configs\compare_trackers_yolo26m_botsort_identity_stable_all.yaml `
  --overwrite
```

Dry-run the same experiment without running inference:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli compare-trackers `
  --config configs\compare_trackers_yolo26m_botsort_identity_stable_all.yaml `
  --dry-run
```

Current metric output:

```text
outputs/metrics/experiments/yolo26m_botsort_identity_stable_all/
  best_tracker_result.json
  tracker_summary.json
  tracker_sequence_metrics.csv
  tracker_comparison_report.md
  botsort_reid_mot_validation.json
```

Open `best_tracker_result.json` first when you only need the selected best tracker.
Use `tracker_summary.json` for the full configured tracker table and
`tracker_sequence_metrics.csv` when you need per-sequence diagnostics.

## Tracker Presets

Recommended presets live in `configs/trackers/`.

| Preset | Use |
|---|---|
| `botsort_reid_identity_stable.yaml` | Preferred when fewer ID switches matter most. |
| `botsort_balanced.yaml` | Balanced baseline. |
| `botsort_high_recall.yaml` | More permissive, useful when missed objects hurt more. |
| `botsort_high_identity.yaml` | Stricter association for crowded identity cases. |
| `bytetrack_fast.yaml` | Fast non-ReID baseline. |

The default video tracker config is:

```text
configs/botsort_reid.yaml
```

## Reusable Domain Configs

Domain profiles live in:

```text
configs/domains/
  football.yaml
  generic_person.yaml
  vehicle.yaml
```

Generate reusable configs for a domain:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli build-domain-configs `
  --domain configs\domains\football.yaml `
  --output-dir configs\generated\football `
  --overwrite
```

Generated configs are ignored by Git and can be regenerated at any time.

## Reports And Benchmarks

Generate a benchmark:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli benchmark `
  --config configs\benchmark.yaml
```

Generate the final Markdown report:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli generate-report `
  --config configs\report.yaml
```

## CLI Reference

All commands are available through:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli --help
```

Important commands:

```text
doctor
prepare-dataset
preflight-training
train-detector
resume-detector
evaluate-detector
cache-detections
track-video
track
track-from-cache
compare-trackers
evaluate-tracking
render-video
plan-tracker-grid
build-domain-configs
analyze-tracking-vlm
benchmark
generate-report
summarize-experiments
```

## Testing

Run everything:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Run focused VLM tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_vlm_tracking_context.py
```

Run environment health check:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli doctor
```

## Troubleshooting

### `Output run directory exists and overwrite=false`

Pass `--overwrite` or use the PowerShell script's `-Overwrite` flag.

### Qwen says `status: failed`

Check `vlm_answer.json`. Common causes:

- `requirements/vlm.txt` was not installed.
- The Qwen model cache is incomplete.
- The model needs more VRAM/RAM than available for the selected keyframes.

Start with a small smoke run:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli analyze-tracking-vlm `
  --config configs\vlm_qwen4b_tracking.yaml `
  --source-video F:\videos\1.mp4 `
  --tracked-video F:\videos\1_Tracking_vlm.mp4 `
  --tracks F:\videos\1_Tracking_vlm.txt `
  --metadata F:\videos\1_Tracking_vlm.metadata.json `
  --output-dir outputs\vlm\qwen4b\smoke_check `
  --max-keyframes 1 `
  --max-tracks 5 `
  --max-crops-per-track 1 `
  --max-new-tokens 64 `
  --run-model `
  --overwrite
```

### Too many ID switches in rendered video

Use the identity-stable BoT-SORT preset:

```text
configs/trackers/botsort_reid_identity_stable.yaml
```

For video tracking, `configs/botsort_reid.yaml` is already tuned toward fewer ID switches.

### Tracking is slow

BoT-SORT ReID is slower than SORT because it uses appearance features. Reduce video resolution,
lower `detector.imgsz`, or switch to a faster tracker preset when identity stability is less
important.

### Terminal cannot show Vietnamese text

The CLI writes JSON with escaped Unicode, but ad-hoc `print()` calls can fail on some Windows
codepages. Prefer reading `vlm_answer.md` or use:

```powershell
chcp 65001
```

## Language-Guided Semantic Tracking

`locate_tracking` is an optional parallel subsystem. Normal YOLO + BoT-SORT tracking works
without LocateAnything, Qwen, or the language benchmark.

The language pipeline is artifact-based:

```text
saved video frames + MOT TXT + language query
  -> LocateAnything grounding
  -> frame-to-track association
  -> multi-frame semantic memory
  -> appearance verification
  -> uncertainty monitoring
  -> event-triggered grounding
  -> semantic reacquisition
  -> stable semantic target identity
  -> language benchmark/report
```

Smoke benchmark:

```powershell
.\scripts\run_language_benchmark_smoke.ps1 -Overwrite
```

Install optional LocateAnything dependencies before running the real NVIDIA backend:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\locate_tracking.txt
```

Create a real-video subset template:

```powershell
.\scripts\create_language_subset_template.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -Tracks F:\videos\1_Tracking.txt `
  -GroundTruth data\mot\sportsmot_football\val\YOUR_SEQUENCE\gt\gt.txt `
  -FrameCount 1200 `
  -Query "the goalkeeper wearing green" `
  -TargetGtTrackId 3 `
  -RawTrackId 7 `
  -OutputDir data\language_tracking\subset\video_1 `
  -Overwrite
```

Ablation smoke:

```powershell
.\scripts\run_language_ablation.ps1 -Overwrite
```

Generate the language report:

```powershell
.\scripts\generate_language_report.ps1 -Overwrite
```

Important artifacts:

```text
data/language_tracking/benchmark_manifest.json
configs/locate_tracking/experiments/ablation_manifest.yaml
outputs/locate_tracking/benchmark/
outputs/locate_tracking/reports/
```

Documentation:

- [Locate tracking system overview](docs/locate_tracking/system_overview.md)
- [Language benchmark annotation guide](docs/locate_tracking/language_benchmark_annotation_guide.md)
- [Windows benchmark runbook](docs/locate_tracking/benchmark_runbook_windows.md)
- [Technical report template](docs/locate_tracking/technical_report.md)

Current limitation: the checked-in benchmark is a tiny synthetic smoke fixture. Real
research claims require manually annotated real sequences, frozen thresholds, and a
separate final evaluation split.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
