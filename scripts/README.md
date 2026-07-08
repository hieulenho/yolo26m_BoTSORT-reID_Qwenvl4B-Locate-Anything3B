# Script Guide

The project is focused on:

```text
YOLO26m -> BoT-SORT ReID -> team/position semantics -> quantitative benchmark
```

The full terminal runbook is:

```text
outputs\reports\focused_pipeline\run_all_team_position_commands.txt
```

## Main Commands

Track one raw video and run Qwen VLM:

```powershell
.\scripts\track_video_qwen_vlm.ps1 `
  -Source F:\videos\1.mp4 `
  -OutputVideo F:\videos\1_A_qwen.mp4 `
  -RunModel `
  -MaxKeyframes 2 `
  -MaxTracks 20 `
  -MaxCropsPerTrack 1 `
  -MaxNewTokens 512 `
  -Overwrite
```

Analyze ID switch failure types across trackers:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_idsw_taxonomy.py `
  --mot-root data\mot\sportsmot_football `
  --seqmap data\mot\sportsmot_football\seqmaps\all.txt `
  --output-dir outputs\reports\focused_pipeline\idsw_taxonomy `
  --overwrite
```

Run team/position benchmark:

```powershell
.\scripts\run_team_position_benchmark.ps1 `
  -Manifest data\team_benchmark\video_1\benchmark_manifest_expanded.json `
  -PipelineA data\team_benchmark\video_1\pipeline_a_yolo26m_botsort_reid_qwen4b_expanded_bootstrap.json `
  -PipelineC data\team_benchmark\video_1\pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b_expanded_bootstrap.json `
  -OutputDir outputs\team_benchmark\focused\video_1_available `
  -Overwrite
```

Render `bbox + ID + team + role` to video:

```powershell
.\.venv\Scripts\python.exe scripts\render_team_position_video.py `
  --source-video F:\videos\1.mp4 `
  --tracks F:\videos\1_Tracking_qwen.txt `
  --predictions data\team_benchmark\video_1\pipeline_a_yolo26m_botsort_reid_qwen4b_expanded_bootstrap.json `
  --sequence-name video_1 `
  --output-video F:\videos\1_A_team_position.mp4 `
  --overwrite
```

## Supporting Commands

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
