param(
    [string]$Config = "configs/compare_trackers.yaml",
    [int]$MaxSequences = 0,
    [int]$MaxFrames = 0,
    [switch]$Overwrite,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

& $Python -m football_tracking.cli doctor

$ArgsList = @("-m", "football_tracking.cli", "compare-trackers", "--config", $Config)
if ($MaxSequences -gt 0) { $ArgsList += @("--max-sequences", $MaxSequences) }
if ($MaxFrames -gt 0) { $ArgsList += @("--max-frames", $MaxFrames) }
if ($Overwrite) { $ArgsList += "--overwrite" }
if ($DryRun) { $ArgsList += "--dry-run" }
& $Python @ArgsList
