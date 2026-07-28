<# Rebuild the three 120-frame detector-router runtime benchmarks. #>

param(
    [string]$FootballSource = "F:\videos\1.mp4",
    [int]$MaxFrames = 120,
    [string]$OutputRoot = "outputs\benchmarks\runtime\adaptive_routes",
    [string]$Device = "cuda",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

$Routes = @(
    @{
        Id = "football_120"
        Name = "Football fine-tuned"
        Source = $FootballSource
        Config = ""
    },
    @{
        Id = "coco_120"
        Name = "COCO pretrained"
        Source = "F:\videos\multidomain_wildlife_black_noddies_38s.mp4"
        Config = "outputs\adaptive_runs\multidomain_suite_8bit\multidomain_wildlife_black_noddies_38s\plan\tracking.generated.yaml"
    },
    @{
        Id = "open_vocab_120"
        Name = "Open vocabulary"
        Source = "F:\videos\multidomain_wildlife_black_noddies_38s.mp4"
        Config = ""
    }
)

foreach ($Route in $Routes) {
    $RouteRoot = Join-Path $OutputRoot $Route.Id
    New-Item -ItemType Directory -Force -Path $RouteRoot | Out-Null
    $Config = $Route.Config

    if ($Route.Id -in @("football_120", "open_vocab_120")) {
        $PlanRoot = Join-Path $RouteRoot "plan"
        $Discovery = if ($Route.Id -eq "football_120") {
            "configs\benchmarks\discovery_football_runtime.json"
        }
        else {
            "configs\benchmarks\discovery_open_runtime.json"
        }
        $PlanArgs = @(
            "-m", "football_tracking.adaptive_tracking.cli", "build-plan",
            "--source", $Route.Source,
            "--discovery", $Discovery,
            "--output-dir", $PlanRoot,
            "--output-video", (Join-Path $RouteRoot "tracked.mp4"),
            "--profile", "realtime_stable",
            "--device", $Device,
            "--max-frames", "$MaxFrames"
        )
        if ($Overwrite) { $PlanArgs += "--overwrite" }
        & $Python @PlanArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $Config = Join-Path $PlanRoot "tracking.generated.yaml"
    }

    foreach ($Required in @($Route.Source, $Config)) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            throw "Required benchmark input does not exist: $Required"
        }
    }

    Write-Host "==> $($Route.Name): $MaxFrames frames"
    $Video = Join-Path $RouteRoot "tracked.mp4"
    $Mot = Join-Path $RouteRoot "tracks.txt"
    $Metrics = Join-Path $RouteRoot "metrics.json"
    $Validation = Join-Path $RouteRoot "validation.json"
    $ValidationMd = Join-Path $RouteRoot "validation.md"
    $RunArgs = @(
        "scripts\runtime\run_realtime_adaptive.py",
        "--config", $Config,
        "--source", $Route.Source,
        "--output-video", $Video,
        "--output-mot", $Mot,
        "--metadata", $Metrics,
        "--max-frames", "$MaxFrames",
        "--disable-frame-dropping",
        "--no-window"
    )
    if ($Overwrite) { $RunArgs += "--overwrite" }
    & $Python @RunArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $DiagnoseArgs = @(
        "scripts\data\diagnose_video_tracks.py",
        "--tracks", $Mot,
        "--metadata", $Metrics,
        "--source-video", $Route.Source,
        "--output-json", $Validation,
        "--output-md", $ValidationMd
    )
    if ($Overwrite) { $DiagnoseArgs += "--overwrite" }
    & $Python @DiagnoseArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "==> Runtime router benchmarks complete: $OutputRoot"
