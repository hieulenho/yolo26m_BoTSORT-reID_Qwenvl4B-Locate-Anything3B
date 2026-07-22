# Windows Runbook - Language Tracking Benchmark

Run from the project root:

```powershell
cd F:\Tracking
```

Validate annotations:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli validate-language-benchmark `
  --manifest data\language_tracking\benchmark_manifest.json `
  --output outputs\locate_tracking\benchmark\smoke\validation.json
```

Smoke benchmark:

```powershell
.\scripts\run_language_benchmark_smoke.ps1 -Overwrite
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
  -EvaluationStartFrame 1 `
  -EvaluationEndFrame 1200 `
  -SequenceName video_1 `
  -QueryId q_goalkeeper_green `
  -OutputDir data\language_tracking\subset\video_1 `
  -Overwrite
```

This writes:

```text
data\language_tracking\subset\video_1\benchmark_manifest.json
data\language_tracking\subset\video_1\predictions_a5_full_system.json
data\language_tracking\subset\video_1\artifacts\semantic_target.json
```

Review the generated benchmark manifest before using the numbers. In particular,
`target_gt_track_ids` must be GT IDs from the annotation file, while `RawTrackId`
is the predicted tracker ID used by the semantic target artifact.

Subset/full benchmark:

```powershell
.\scripts\run_language_benchmark.ps1 `
  -Mode subset `
  -Manifest data\language_tracking\subset\video_1\benchmark_manifest.json `
  -Predictions data\language_tracking\subset\video_1\predictions_a5_full_system.json `
  -OutputDir outputs\locate_tracking\benchmark\subset\a5_full_system `
  -Overwrite

.\scripts\run_language_benchmark.ps1 `
  -Mode full `
  -Predictions data\language_tracking\predictions_full.json `
  -OutputDir outputs\locate_tracking\benchmark\full\a5_full_system `
  -Overwrite
```

The subset/full commands need saved prediction manifests produced by the M1-M6
language-guided pipeline. The wrapper fails early if these files do not exist:

```text
data\language_tracking\predictions_subset.json
data\language_tracking\predictions_full.json
```

Ablation:

```powershell
.\scripts\run_language_ablation.ps1 -Overwrite
```

Failure analysis:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli analyze-language-failures `
  --evaluation outputs\locate_tracking\benchmark\smoke\a5_full_system `
  --output-dir outputs\locate_tracking\benchmark\smoke\failures `
  --overwrite
```

Report:

```powershell
.\scripts\generate_language_report.ps1 -Overwrite
```

Demo manifest:

```powershell
.\.venv\Scripts\python.exe -m football_tracking.locate_tracking.cli build-language-demo `
  --evaluation outputs\locate_tracking\benchmark\smoke\a5_full_system `
  --output-dir outputs\locate_tracking\demo\smoke `
  --overwrite
```

The benchmark reads saved prediction artifacts. It does not rerun YOLO, BoT-SORT,
LocateAnything, or Qwen.
