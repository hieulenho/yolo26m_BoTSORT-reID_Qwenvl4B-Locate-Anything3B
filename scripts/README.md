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
| `-Profile realtime` | OC-SORT and small detector routes |
| `-Profile balanced` | TrackTrack and medium routes |
| `-Profile accuracy` | identity-stable BoT-SORT ReID |
| `-MaxFrames 120` | bounded smoke run |
| `-SemanticMaxTracks 4` | limit expensive Qwen track analysis |
| `-RunTrackSemantics $false` | skip downstream Qwen track labeling |
| `-RunLocateVerification $false` | skip LocateAnything verification |
| `-RefreshSemanticCache` | rerun discovery for the same source |

## Realtime Stream

```powershell
.\scripts\run_realtime_adaptive.ps1 `
  -Source 0 `
  -RunName webcam_01 `
  -CalibrationSeconds 8 `
  -QwenQuantization 4bit `
  -Device cuda `
  -Overwrite
```

The script captures a short calibration clip, discovers the vocabulary once, builds a realtime
plan, and starts the camera/RTSP/file stream. Use `-NoWindow` for headless measurement.

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
