# Script Guide

The cleaned raw-video experiment flow is:

```text
raw video -> YOLO26m -> BoT-SORT ReID -> A/B/C semantic pipelines
```

The full terminal runbook is:

```text
outputs\reports\focused_pipeline\run_all_team_position_commands.txt
```

## Main Raw-Video Command

Change `-SourceVideo` to switch between `1.mp4`, `2.mp4`, `3.mp4`, etc.

```powershell
.\scripts\run_raw_video_semantic_experiments.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -Query "the goalkeeper wearing green" `
  -Pipelines A,B,C `
  -LocateBackend locate_anything `
  -RunQwenModel `
  -Device cuda `
  -TorchDtype auto `
  -MaxKeyframes 2 `
  -MaxTracks 20 `
  -MaxCropsPerTrack 1 `
  -MaxNewTokens 512 `
  -LocateMaxFrames 6 `
  -OutputRoot outputs\semantic_video_experiments `
  -Overwrite
```

Pipelines:

- `A`: YOLO26m + BoT-SORT ReID + Qwen3-VL 4B.
- `B`: YOLO26m + BoT-SORT ReID + LocateAnything 3B.
- `C`: YOLO26m + BoT-SORT ReID + LocateAnything 3B + Qwen3-VL 4B.

Use a fast plumbing check before loading large models:

```powershell
.\scripts\run_raw_video_semantic_experiments.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -Query "the goalkeeper wearing green" `
  -Pipelines A,B,C `
  -LocateBackend mock `
  -OutputRoot outputs\semantic_video_experiments `
  -Overwrite
```

## Benchmark Commands

Use these only when you already have a benchmark manifest and saved prediction
manifests.

```powershell
.\scripts\run_team_position_benchmark.ps1 `
  -Manifest data\team_benchmark\video_1\benchmark_manifest_expanded.json `
  -PipelineA data\team_benchmark\video_1\pipeline_a_yolo26m_botsort_reid_qwen4b_expanded_bootstrap.json `
  -PipelineC data\team_benchmark\video_1\pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b_expanded_bootstrap.json `
  -OutputDir outputs\team_benchmark\focused\video_1_available `
  -Overwrite
```

## Supporting Commands

Analyze ID switch failure types across trackers:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_idsw_taxonomy.py `
  --mot-root data\mot\sportsmot_football `
  --seqmap data\mot\sportsmot_football\seqmaps\all.txt `
  --output-dir outputs\reports\focused_pipeline\idsw_taxonomy `
  --overwrite
```

Train detector:

```powershell
.\scripts\train_football_detector.ps1
```

Evaluate detector:

```powershell
.\scripts\evaluate_detector.ps1 -Config configs\yolo26m_sportsmot_football_eval.yaml
```

Compare MOT trackers:

```powershell
.\scripts\compare_trackers.ps1 -Config configs\compare_trackers_yolo26m_botsort_identity_stable_all.yaml -Overwrite
```

Run tests:

```powershell
.\scripts\run_tests.ps1
```
