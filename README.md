# Adaptive Multi-Domain Visual Tracking

This repository turns a video or live stream into class-aware tracks and deeper semantic
labels without fixing the vocabulary to football.

The production path is:

```text
video / webcam
  -> shot-aware keyframes
  -> Qwen3-VL-4B scene and class discovery
  -> normalized vocabulary and detector routing
  -> YOLO26 / YOLOE detection
  -> profile-selected multi-object tracker
  -> track crops and event-triggered semantic analysis
  -> unknown rejection, MP4, MOT TXT, JSON, and metrics
```

![Adaptive architecture](docs/assets/benchmarks/adaptive_architecture.png)

## Why Tracking Comes Before Deep Semantics

The detector and tracker process every frame. Qwen and LocateAnything process a compact set of
keyframes, uncertain tracks, and multi-time crops. A semantic result is then attached to the
stable `track_id` instead of running a 3B/4B model on every frame. This keeps the live path
responsive and makes every semantic claim auditable.

## Dynamic Vocabulary

1. Shot boundaries are estimated from sampled frame differences.
2. Representative full-frame keyframes are sent to `Qwen/Qwen3-VL-4B-Instruct`.
3. Qwen returns the domain, visible objects, and an action for each object: `track`, `detect`,
   or `context`.
4. The ontology registry merges synonyms, removes attribute-only class names, limits class
   count, and maps known classes to COCO IDs.
5. The detector router selects a checkpoint per class group.

| Route | Detector | Use |
|---|---|---|
| Football | fine-tuned YOLO26m | people on football footage |
| COCO | YOLO26n/s pretrained | known general objects |
| Open vocabulary | YOLOE-26n/s | classes outside the COCO vocabulary |

Football uses a hybrid route: the fine-tuned model tracks people, while a small COCO detector
samples detect-only classes such as the ball. The realtime profile runs this supplemental
detector every six frames and never reuses stale box coordinates.

## Tracker Profiles

| Profile | Tracker | Intended use |
|---|---|---|
| `realtime` | OC-SORT | live streams and lowest end-to-end latency |
| `balanced` | TrackTrack | strongest measured HOTA with moderate speed |
| `accuracy` | BoT-SORT ReID stable | lowest official IDSW among the evaluated identity trackers |

OC-SORT is the realtime default because the local 30-sequence benchmark places it ahead of
the other low-latency candidates on the measured quality-speed trade-off. Appearance-CNN
trackers remain available, but Deep OC-SORT ReID and BoT-SORT ReID were slower on this 8 GB
laptop GPU.

## Semantic Roles

- **Qwen** discovers classes, reads global context, and assigns open semantic labels from
  full-frame keyframes plus multi-time track crops.
- **LocateAnything** is called on uncertain or query-relevant cases to ground a description
  spatially; it is not run continuously on every frame.
- **Fusion** combines accepted evidence and emits `unknown` when confidence or score margin is
  insufficient.

Models run sequentially and are quantized by default (`Qwen` 4-bit, `LocateAnything` 8-bit),
so their VRAM footprints do not add together.

## Installation

Windows PowerShell, Python 3.12:

```powershell
cd F:\Tracking
.\scripts\setup_env.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements\vlm.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\open_vocab.txt
```

Install a CUDA-enabled PyTorch build appropriate for the machine before running the large
models. Check the environment with:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli doctor
nvidia-smi
```

## Run One Video

Full adaptive path on `F:\videos\1.mp4`:

```powershell
.\scripts\run_adaptive_tracking.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -OutputVideo F:\videos\1_adaptive_tracking.mp4 `
  -SemanticOutputVideo F:\videos\1_adaptive_semantic.mp4 `
  -Profile realtime `
  -QwenQuantization 4bit `
  -Device cuda `
  -Overwrite
```

Change only `-SourceVideo` and output names for `2.mp4`, `3.webm`, traffic footage, classroom
footage, or another domain. Do not reuse a discovery cache across different source videos.

For a short plumbing check:

```powershell
.\scripts\run_adaptive_tracking.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -OutputVideo F:\videos\1_adaptive_smoke.mp4 `
  -SemanticOutputVideo F:\videos\1_adaptive_smoke_semantic.mp4 `
  -Profile realtime `
  -MaxFrames 120 `
  -SemanticMaxTracks 4 `
  -SemanticMaxImages 4 `
  -Overwrite
```

## Run A Webcam

The camera is calibrated for a few seconds, Qwen creates one vocabulary cache, and the stream
then runs with detector plus tracker only. Deep semantics can be triggered asynchronously in a
future deployment.

```powershell
.\scripts\run_realtime_adaptive.ps1 `
  -Source 0 `
  -RunName webcam_01 `
  -CalibrationSeconds 8 `
  -QwenQuantization 4bit `
  -Device cuda `
  -Overwrite
```

Use an RTSP URL instead of `0` for a network camera.

## Outputs

Video products are written to the paths supplied on the command line. Reproducibility artifacts
are stored under `outputs/adaptive_runs/<video_stem>/`:

```text
discovery/scene_discovery.json       domain, objects, actions, evidence
plan/adaptive_plan.json              detector route and tracker profile
plan/tracking.generated.yaml         exact generated runtime config
qwen_track_semantics/                keyframes, crops, prompt, model answer
locate_verification/                 event plan and grounding result
fused_track_semantics.json           accepted labels and unknown decisions
adaptive_run_report.json             timings, VRAM, coverage, and provenance
```

The MOT text rows use:

```text
frame, track_id, x, y, width, height, confidence, class_id, visibility, reserved
```

Detection-only classes are rendered as `DET | class` and never receive a fake track ID.

## Verified Results

All detector and tracker quality scores below use SportsMOT ground truth. Runtime measurements
use the same RTX 4060 Laptop GPU (8 GB), rendering enabled, and a 120-frame file source.

### Detector

| Detector | Precision | Recall | mAP50 | mAP50-95 | Detector FPS |
|---|---:|---:|---:|---:|---:|
| YOLO26m fine-tuned | 0.9595 | 0.9601 | 0.9793 | 0.8306 | 55.89 |
| YOLO26m pretrained | 0.8662 | 0.9026 | 0.8935 | 0.7361 | 53.20 |
| YOLOv8m pretrained | 0.8555 | 0.9139 | 0.8932 | 0.7229 | 6.65 |
| YOLO26n pretrained | 0.7865 | 0.8377 | 0.8401 | 0.5894 | 58.65 |

![Detector benchmark](docs/assets/benchmarks/detector_quality_speed.png)

### Tracking

| Tracker | HOTA | IDF1 | Official IDSW | Cached pipeline FPS |
|---|---:|---:|---:|---:|
| TrackTrack | 71.058 | 71.341 | 1042 | 21.66 |
| BoT-SORT ReID stable | 68.503 | 71.352 | 895 | 11.66 |
| OC-SORT | 59.379 | 66.108 | 2186 | 79.40 |
| FastTracker | 58.702 | 64.325 | 2220 | 51.03 |
| ByteTrack | 58.032 | 64.106 | 1828 | 83.33 |

![Tracker quality-speed trade-off](docs/assets/benchmarks/tracker_quality_speed.png)

The complete eight-tracker table and the diagnostic five-class IDSW decomposition are in the
[final experiment report](docs/benchmarks/final_experiment_report.md).

![IDSW taxonomy](docs/assets/benchmarks/idsw_taxonomy.png)

### Semantic A/B/C Ablation

The current semantic GT contains 31 manually reviewed tracks from one football video.

| Pipeline | End-to-end accuracy | Macro F1 | Coverage | Cold time | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| A: Qwen | 51.61% | 77.12% | 51.61% | 197.74 s | 4.01 GiB |
| B: LocateAnything | 16.13% | 11.90% | 16.13% | 108.10 s | 4.46 GiB |
| C: Qwen + event Locate | 64.52% | 81.87% | 64.52% | 267.30 s | 4.46 GiB |

![Semantic benchmark](docs/assets/benchmarks/semantic_quality_cost.png)

### Realtime Routes

| Route | E2E FPS | Steady processing FPS | Startup |
|---|---:|---:|---:|
| Football hybrid | 28.57 | 58.29 | 0.54 s |
| COCO pretrained | 26.97 | 62.21 | 0.58 s |
| Open vocabulary | 21.25 | 42.49 | 2.62 s |

![Realtime route benchmark](docs/assets/benchmarks/realtime_route_fps.png)

`E2E FPS` includes video open/write overhead. `Steady processing FPS` excludes startup and the
first five warm-up frames. These are short file-source measurements, not long-duration webcam
claims.

## Reproduce And Verify

```powershell
.\scripts\run_tracking_benchmark.ps1 -Smoke -Overwrite
.\.venv\Scripts\python.exe scripts\build_final_benchmark_report.py `
  --config configs\benchmarks\final_report.yaml `
  --overwrite
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

The verified local state is `379 passed`. Canonical reports:

- [Final report](docs/benchmarks/final_experiment_report.md)
- [Artifact audit](docs/benchmarks/artifact_audit.json)
- [Runtime CSV](docs/benchmarks/realtime_route_summary.csv)
- [Five-pass engineering audit](docs/benchmarks/five_pass_audit.md)
- [All terminal commands](commands.txt)

## Measurement Limits

- Cross-domain routing is implemented and runtime-tested, but traffic, medical, and education
  accuracy require new human ground truth before a valid accuracy claim can be made.
- Semantic accuracy currently covers 31 tracks from one football video.
- The five IDSW categories are deterministic diagnostic heuristics. Official tracker ranking
  uses TrackEval IDSW, HOTA, AssA, and IDF1.
- Functional position labels such as striker or midfielder generally require temporal and field
  context; a visible jersey crop alone is not sufficient ground truth.

## Repository Layout

```text
configs/                    source configs, profiles, ontology, benchmarks
data/                       local datasets and reviewed manifests
docs/                       design notes and publishable figures
models/                     local promoted checkpoints
outputs/                    generated runs, caches, metrics, and reports
requirements/               base, development, VLM, and open-vocabulary dependencies
scripts/                    reproducible PowerShell and Python entry points
src/football_tracking/      package implementation
tests/                      regression and benchmark-contract tests
```

See [project structure](docs/PROJECT_STRUCTURE.md), [config guide](configs/README.md), and
[script guide](scripts/README.md).

## License

See [LICENSE](LICENSE). Model checkpoints and datasets retain their original licenses.
