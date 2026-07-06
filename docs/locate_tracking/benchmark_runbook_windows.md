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

Subset/full benchmark:

```powershell
.\scripts\run_language_benchmark.ps1 `
  -Mode subset `
  -Predictions data\language_tracking\predictions_subset.json `
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
