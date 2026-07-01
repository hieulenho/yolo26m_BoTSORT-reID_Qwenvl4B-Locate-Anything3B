param(
    [string]$Config = "configs/track_sportsmot.yaml",
    [string]$Device = "auto",
    [int]$MaxSequences = 0,
    [int]$MaxFrames = 0,
    [switch]$Render,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Args = @("-m", "football_tracking.cli", "track-sportsmot", "--config", $Config, "--device", $Device)
if ($MaxSequences -gt 0) { $Args += @("--max-sequences", "$MaxSequences") }
if ($MaxFrames -gt 0) { $Args += @("--max-frames", "$MaxFrames") }
if ($Render) { $Args += "--render" }
if ($Overwrite) { $Args += "--overwrite" }

& $Python @Args
exit $LASTEXITCODE
