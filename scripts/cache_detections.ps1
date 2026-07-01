param(
    [string]$Config = "configs/detection_cache.yaml",
    [string]$Device = "auto",
    [int]$MaxSequences = 0,
    [int]$MaxFrames = 0,
    [switch]$Overwrite,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ArgsList = @("-m", "football_tracking.cli", "cache-detections", "--config", $Config, "--device", $Device)
if ($MaxSequences -gt 0) { $ArgsList += @("--max-sequences", $MaxSequences) }
if ($MaxFrames -gt 0) { $ArgsList += @("--max-frames", $MaxFrames) }
if ($Overwrite) { $ArgsList += "--overwrite" }
if ($DryRun) { $ArgsList += "--dry-run" }
& $Python @ArgsList
