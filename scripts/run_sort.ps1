param(
    [string]$Config = "configs/compare_trackers.yaml",
    [int]$MaxSequences = 0,
    [int]$MaxFrames = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ArgsList = @(
    "-m", "football_tracking.cli",
    "track-from-cache",
    "--tracker", "sort",
    "--experiment-config", $Config
)
if ($MaxSequences -gt 0) { $ArgsList += @("--max-sequences", $MaxSequences) }
if ($MaxFrames -gt 0) { $ArgsList += @("--max-frames", $MaxFrames) }
if ($Overwrite) { $ArgsList += "--overwrite" }
& $Python @ArgsList
