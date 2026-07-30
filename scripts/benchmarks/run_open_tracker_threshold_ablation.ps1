<# Compare TrackTrack initialization thresholds for low-score open-vocabulary detections. #>

param(
    [string]$Source = "F:\videos\multidomain_wildlife_black_noddies_38s.mp4",
    [string]$TaskConfig = "configs\tasks\wildlife_birds.yaml",
    [string]$OutputRoot = "outputs\benchmarks\tracking\open_threshold_ablation",
    [ValidateRange(1, 1000000)]
    [int]$MaxFrames = 300,
    [string]$Device = "cuda",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment Python does not exist: $Python"
}
if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    throw "Source video does not exist: $Source"
}

$Profiles = @(
    @{
        Id = "open_t010"
        Config = "configs\benchmarks\tracker_profiles\tracktrack_open_t010.yaml"
    },
    @{
        Id = "open_t015"
        Config = "configs\benchmarks\tracker_profiles\tracktrack_open_t015.yaml"
    },
    @{
        Id = "open_t020"
        Config = "configs\benchmarks\tracker_profiles\tracktrack_open_t020.yaml"
    }
)
$ReportRuns = @()
for ($Index = 0; $Index -lt $Profiles.Count; $Index++) {
    $Profile = $Profiles[$Index]
    $RunId = [string]$Profile.Id
    Write-Progress `
        -Activity "Open-vocabulary tracker threshold ablation" `
        -Status "[$($Index + 1)/$($Profiles.Count)] $RunId" `
        -PercentComplete ([int](100 * $Index / $Profiles.Count))
    Write-Host "==> [$($Index + 1)/$($Profiles.Count)] $RunId"
    $Arguments = @{
        TaskConfig = $TaskConfig
        TrackerConfig = [string]$Profile.Config
        Source = $Source
        RunName = $RunId
        OutputRoot = $OutputRoot
        SemanticWorkerMode = "disabled"
        Device = $Device
        MaxFrames = $MaxFrames
        NoWindow = $true
        Quiet = $true
    }
    if ($Overwrite) { $Arguments.Overwrite = $true }
    & (Join-Path $ProjectRoot "scripts\run_task_realtime.ps1") @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Threshold ablation failed for $RunId with exit code $LASTEXITCODE."
    }
    $Metrics = Join-Path $ProjectRoot (
        Join-Path $OutputRoot (Join-Path $RunId "realtime_metrics.json")
    )
    if (-not (Test-Path -LiteralPath $Metrics -PathType Leaf)) {
        throw "Threshold ablation did not produce metrics: $Metrics"
    }
    $ReportRuns += "${RunId}=${Metrics}"
}
Write-Progress -Activity "Open-vocabulary tracker threshold ablation" -Completed

$ReportDir = Join-Path $ProjectRoot (Join-Path $OutputRoot "comparison")
$ReportArgs = @("scripts\benchmarks\build_realtime_benchmark.py")
foreach ($RunValue in $ReportRuns) {
    $ReportArgs += @("--run", $RunValue)
}
$ReportArgs += @("--output-dir", $ReportDir)
if ($Overwrite) { $ReportArgs += "--overwrite" }
& $Python @ReportArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Threshold report: $(Join-Path $ReportDir 'realtime_benchmark.md')"
