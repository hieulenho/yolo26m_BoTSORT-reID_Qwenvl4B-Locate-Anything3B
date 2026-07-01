param(
    [string]$Config = "configs/tracker_ablation.yaml",
    [int]$MaxExperiments = 0,
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ArgsList = @("-m", "football_tracking.cli", "run-tracker-ablation", "--config", $Config)
if ($MaxExperiments -gt 0) { $ArgsList += @("--max-experiments", $MaxExperiments) }
if ($Resume) { $ArgsList += "--resume" }
if ($DryRun) { $ArgsList += "--dry-run" }
& $Python @ArgsList
