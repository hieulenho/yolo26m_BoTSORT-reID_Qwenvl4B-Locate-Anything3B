param(
    [string]$Source,
    [string]$Config = "configs/track_video.yaml",
    [string]$Checkpoint = "",
    [string]$Device = "auto",
    [int]$MaxFrames = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Args = @("-m", "football_tracking.cli", "track-video", "--config", $Config, "--device", $Device)
if ($Source) { $Args += @("--source", $Source) }
if ($Checkpoint) { $Args += @("--checkpoint", $Checkpoint) }
if ($MaxFrames -gt 0) { $Args += @("--max-frames", "$MaxFrames") }
if ($Overwrite) { $Args += "--overwrite" }

& $Python @Args
exit $LASTEXITCODE
