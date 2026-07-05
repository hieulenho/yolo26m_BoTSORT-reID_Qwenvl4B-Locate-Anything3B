# Script Guide

PowerShell scripts are thin wrappers around:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli
```

Use scripts for common Windows workflows and use the CLI directly for precise automation.

## Daily Video Tracking

Track one video:

```powershell
.\scripts\track_video.ps1 -Source F:\videos\1.mp4 -OutputVideo F:\videos\1_Tracking.mp4 -Overwrite
```

Track one video and prepare Qwen VLM artifacts:

```powershell
.\scripts\track_video_qwen_vlm.ps1 -Source F:\videos\1.mp4 -OutputVideo F:\videos\1_Tracking.mp4 -Overwrite
```

Run Qwen after tracking:

```powershell
.\scripts\track_video_qwen_vlm.ps1 -Source F:\videos\1.mp4 -OutputVideo F:\videos\1_Tracking.mp4 -RunModel -Overwrite
```

Analyze existing tracking outputs without rerunning tracking:

```powershell
.\scripts\analyze_tracking_vlm.ps1 `
  -SourceVideo F:\videos\1.mp4 `
  -TrackedVideo F:\videos\1_Tracking.mp4 `
  -Tracks F:\videos\1_Tracking.txt `
  -Metadata F:\videos\1_Tracking.metadata.json `
  -OutputDir F:\videos\1_vlm `
  -RunModel `
  -Overwrite
```

## Training And Evaluation

Train the current football detector:

```powershell
.\scripts\train_football_detector.ps1
```

Evaluate detector only:

```powershell
.\scripts\evaluate_detector.ps1 -Config configs\yolo26m_sportsmot_football_eval.yaml
```

## Dataset And Experiments

Download SportsMOT:

```powershell
.\scripts\download_sportsmot.ps1 -Split "train,val"
```

Compare trackers:

```powershell
.\scripts\compare_trackers.ps1 -Config configs\compare_trackers_yolo26m_botsort_identity_stable_all.yaml -Overwrite
```

## Maintenance

Run tests:

```powershell
.\scripts\run_tests.ps1
```

For uncommon commands, prefer CLI help:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.cli --help
```
