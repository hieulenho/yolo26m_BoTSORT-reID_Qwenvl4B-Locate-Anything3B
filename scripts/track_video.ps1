param(
    [string]$Source,
    [string]$OutputVideo = "",
    [string]$Config = "configs/track_video_yolo26m_botsort.yaml",
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
if ($OutputVideo) { $Args += @("--output-video", $OutputVideo) }
if ($Checkpoint) { $Args += @("--checkpoint", $Checkpoint) }
if ($MaxFrames -gt 0) { $Args += @("--max-frames", "$MaxFrames") }
if ($Overwrite) { $Args += "--overwrite" }

& $Python @Args
exit $LASTEXITCODE
