param(
    [ValidateSet("smoke", "full")]
    [string] $Mode = "smoke",
    [string] $Device = "0"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

function Invoke-PipelineStep {
    param(
        [string] $Name,
        [string[]] $Arguments
    )
    Write-Host ""
    Write-Host "==> $Name"
    & $Python -m football_tracking.cli @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

if ($Mode -eq "full") {
    $TrainConfig = "configs/legacy/football/yolov8m_sportsmot_train.yaml"
    $EvalConfig = "configs/legacy/football/yolov8m_sportsmot_eval.yaml"
    $CacheConfig = "configs/legacy/football/detection_cache.yaml"
    $CompareConfig = "configs/legacy/football/compare_trackers.yaml"
    $RenderConfig = "configs/legacy/football/render_video.yaml"
    $BenchmarkConfig = "configs/legacy/football/benchmark.yaml"
    $ReportConfig = "configs/legacy/football/report.yaml"
} else {
    $TrainConfig = "configs/legacy/football/yolov8m_sportsmot_smoke.yaml"
    $EvalConfig = "configs/legacy/football/yolov8m_sportsmot_smoke_eval.yaml"
    $CacheConfig = "configs/legacy/football/detection_cache_smoke.yaml"
    $CompareConfig = "configs/legacy/football/compare_trackers_smoke.yaml"
    $RenderConfig = "configs/legacy/football/render_video_smoke.yaml"
    $BenchmarkConfig = "configs/legacy/football/benchmark_smoke.yaml"
    $ReportConfig = "configs/legacy/football/report_smoke.yaml"
}

Set-Location $Root
Invoke-PipelineStep "doctor" @("doctor")
Invoke-PipelineStep "prepare dataset" @(
    "prepare-dataset", "--config", "configs/legacy/football/sportsmot_data.yaml", "--overwrite"
)
Invoke-PipelineStep "train detector" @(
    "train-detector", "--config", $TrainConfig, "--device", $Device, "--overwrite"
)
Invoke-PipelineStep "evaluate detector" @("evaluate-detector", "--config", $EvalConfig)
Invoke-PipelineStep "cache detections" @(
    "cache-detections", "--config", $CacheConfig, "--device", $Device, "--overwrite"
)
Invoke-PipelineStep "compare trackers" @(
    "compare-trackers", "--config", $CompareConfig, "--overwrite"
)
Invoke-PipelineStep "evaluate tracking" @("evaluate-tracking", "--config", $CompareConfig)
Invoke-PipelineStep "render video" @("render-video", "--config", $RenderConfig, "--overwrite")
Invoke-PipelineStep "benchmark" @("benchmark", "--config", $BenchmarkConfig)
Invoke-PipelineStep "generate report" @("generate-report", "--config", $ReportConfig)

Write-Host ""
Write-Host "Demo pipeline completed in $Mode mode."
