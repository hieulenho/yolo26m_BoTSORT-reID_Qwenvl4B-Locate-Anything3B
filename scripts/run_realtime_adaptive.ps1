<# Calibrate dynamic vocabulary once, then run low-latency adaptive tracking. #>

param(
    [string]$Source = "0",
    [string]$RunName = "realtime_session",
    [string]$OutputRoot = "outputs\adaptive_realtime",
    [double]$CalibrationSeconds = 8.0,
    [int]$MaxFrames = 0,
    [ValidateSet("none", "8bit", "4bit")]
    [string]$QwenQuantization = "4bit",
    [string]$Device = "cuda",
    [switch]$NoWindow,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RunRoot = Join-Path $ProjectRoot (Join-Path $OutputRoot $RunName)
$CalibrationVideo = Join-Path $RunRoot "calibration.mp4"
$Discovery = Join-Path $RunRoot "discovery\scene_discovery.json"
$PlanRoot = Join-Path $RunRoot "plan"
$GeneratedConfig = Join-Path $PlanRoot "tracking.generated.yaml"
$OutputVideo = Join-Path $RunRoot "realtime_tracked.mp4"
$OutputMot = Join-Path $RunRoot "realtime_tracks.txt"
$Metadata = Join-Path $RunRoot "realtime_metrics.json"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

Write-Host "[1/4] Capture $CalibrationSeconds-second calibration clip"
$CaptureArgs = @(
    "scripts\capture_calibration_clip.py",
    "--source", $Source,
    "--output", $CalibrationVideo,
    "--seconds", "$CalibrationSeconds"
)
if ($Overwrite) { $CaptureArgs += "--overwrite" }
& $Python @CaptureArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/4] Discover domain and dynamic vocabulary with Qwen"
$DiscoveryArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "discover",
    "--source", $CalibrationVideo,
    "--output", $Discovery,
    "--quantization", $QwenQuantization,
    "--device", $Device,
    "--max-keyframes", "4"
)
if ($Overwrite) { $DiscoveryArgs += "--overwrite" }
& $Python @DiscoveryArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/4] Build realtime YOLO/YOLOE + OC-SORT plan"
$PlanArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "build-plan",
    "--source", $CalibrationVideo,
    "--discovery", $Discovery,
    "--output-dir", $PlanRoot,
    "--output-video", $OutputVideo,
    "--profile", "realtime",
    "--device", $Device
)
if ($Overwrite) { $PlanArgs += "--overwrite" }
& $Python @PlanArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/4] Start realtime stream"
$RealtimeArgs = @(
    "scripts\run_realtime_adaptive.py",
    "--config", $GeneratedConfig,
    "--source", $Source,
    "--output-video", $OutputVideo,
    "--output-mot", $OutputMot,
    "--metadata", $Metadata
)
if ($MaxFrames -gt 0) { $RealtimeArgs += @("--max-frames", "$MaxFrames") }
if ($NoWindow) { $RealtimeArgs += "--no-window" }
if ($Overwrite) { $RealtimeArgs += "--overwrite" }
& $Python @RealtimeArgs
exit $LASTEXITCODE
