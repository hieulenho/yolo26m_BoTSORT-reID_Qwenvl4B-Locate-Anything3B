# Script Guide

Supported entry points are grouped by purpose. Run them from `F:\Tracking` in PowerShell.

## Adaptive Video Pipeline

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

This is the primary offline entry point. It performs scene discovery, class normalization,
detector routing, tracking, Qwen track semantics, event-triggered LocateAnything, fusion,
unknown rejection, rendering, and run-report consolidation.

Useful controls:

| Parameter | Meaning |
|---|---|
| `-Profile realtime` | class-routed OC-SORT; special motion config for small fast objects |
| `-Profile realtime_stable` | YOLO26n/640 plus TrackTrack; fewer fragmented IDs at lower FPS |
| `-Profile balanced` | class-routed TrackTrack; OC-SORT for small fast objects |
| `-Profile accuracy` | class-routed identity-stable BoT-SORT ReID |
| `-MaxFrames 120` | bounded smoke run |
| `-SemanticMaxTracks 0` | analyze all tracks; use a positive value only for a bounded smoke run |
| `-RunTrackSemantics $false` | skip downstream Qwen track labeling |
| `-RunLocateVerification $false` | skip LocateAnything verification |
| `-RefreshSemanticCache` | rerun discovery for the same source |

## Realtime Stream

```powershell
.\scripts\run_realtime_adaptive.ps1 `
  -Source 0 `
  -RunName webcam_01 `
  -CalibrationSeconds 8 `
  -DiscoveryKeyframes 2 `
  -QwenQuantization 4bit `
  -Device cuda `
  -Overwrite
```

The script captures a short calibration clip, discovers the vocabulary once, builds a realtime
plan, and starts the camera/RTSP/file stream. Track crops are written to a non-blocking semantic
queue. Use `-NoWindow` for headless measurement.

`-DiscoveryKeyframes 2` is the 8 GB GPU default. Increase it only for videos with several
visually different shots. `-DiscoveryMaxNewTokens 768` protects complete structured JSON; the
semantic cache makes this cold cost one-time for a matching source and configuration.

On an 8 GB GPU, process the queue after the realtime session so Qwen does not compete with the
detector for VRAM:

```powershell
.\.venv\Scripts\python.exe scripts\run_realtime_semantic_worker.py `
  --queue-dir outputs\adaptive_realtime\webcam_01\semantic_queue `
  --vlm-config configs\vlm_dynamic_track_semantics.yaml `
  --semantic-output outputs\adaptive_realtime\webcam_01\semantic_cache.json `
  --memory outputs\adaptive_realtime\webcam_01\semantic_memory.json `
  --max-events 8
```

The worker atomically claims each event. A model/runtime exception returns the event to
`pending/`; an invalid answer is moved to `failed/` with its failure reason instead of being
retried forever. Run only one worker per queue on a single-GPU machine.

## Benchmarking

Fast tracker plumbing check:

```powershell
.\scripts\run_tracking_benchmark.ps1 -Smoke -SmokeFrames 300 -Overwrite
```

Consolidate existing full benchmark sources:

```powershell
.\scripts\build_tracking_benchmark_report.ps1 -Overwrite
.\.venv\Scripts\python.exe scripts\consolidate_detector_benchmark.py `
  --config configs\benchmarks\detector_sportsmot.yaml `
  --overwrite
.\.venv\Scripts\python.exe scripts\build_final_benchmark_report.py `
  --config configs\benchmarks\final_report.yaml `
  --overwrite
```

The final command validates source hashes, counts, ranges, hardware compatibility, semantic GT
scope, and writes report-ready figures.

Licensed public multi-domain trial:

```powershell
.\.venv\Scripts\python.exe scripts\download_multidomain_samples.py
.\.venv\Scripts\python.exe scripts\build_multidomain_trial_report.py `
  --manifest data\samples\multidomain\samples_manifest.json `
  --run-root outputs\adaptive_runs\multidomain_long `
  --output-dir outputs\adaptive_runs\multidomain_long\summary `
  --overwrite
```

The canonical wildlife, traffic, and classroom clips are 37.9, 35.0, and 84.3 seconds. Run
`run_adaptive_tracking.ps1` on each clip before building the report. Video-level domain/class
discovery is kept separate from per-track semantic accuracy, which needs human annotation.

Prepare the human-review package, then finalize and merge the reviewed manifests:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_semantic_gt.py finalize `
  --package-dir data\semantic_benchmark\review\traffic_street

.\.venv\Scripts\python.exe scripts\prepare_semantic_gt.py merge `
  --manifest data\semantic_benchmark\review\wildlife_black_noddies\manifest.reviewed.yaml `
  --manifest data\semantic_benchmark\review\traffic_street\manifest.reviewed.yaml `
  --manifest data\semantic_benchmark\review\education_classroom_long\manifest.reviewed.yaml `
  --output-manifest data\semantic_benchmark\multidomain.reviewed.yaml `
  --overwrite
```

`finalize` intentionally fails until every track row and the video-level review block have been
marked as reviewed by a named annotator.

Build the measured long-stream realtime comparison:

```powershell
.\.venv\Scripts\python.exe scripts\build_realtime_benchmark.py `
  --run baseline_fp32=outputs\adaptive_realtime\traffic_long\realtime_metrics.json `
  --run optimized_no_drop=outputs\adaptive_realtime\traffic_final\realtime_metrics.json `
  --run bounded_live=outputs\adaptive_realtime\traffic_bounded\realtime_metrics.json `
  --output-dir outputs\benchmarks\realtime\traffic_35s `
  --overwrite
```

## Validation

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli doctor
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

## Compatibility Scripts

The following scripts support earlier football-only experiments and are not the primary adaptive
entry point:

- `run_raw_video_semantic_experiments.ps1`
- `run_vlm_guided_pipeline.ps1`
- `run_team_position_benchmark.ps1`
- `render_team_position_video.py`

Use them only when reproducing an older report whose manifest explicitly references those paths.
