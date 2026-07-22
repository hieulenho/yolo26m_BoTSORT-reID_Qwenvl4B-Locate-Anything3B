<# Build the validated six-tracker SportsMOT report and figures. #>

param(
    [string]$Config = "configs\benchmarks\tracking_full_report.yaml",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

$Arguments = @("scripts\benchmarks\consolidate_tracking_benchmark.py", "--config", $Config)
if ($Overwrite) {
    $Arguments += "--overwrite"
}

Write-Host "==> Validate and consolidate full tracking benchmark"
& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "==> Report: outputs\benchmarks\tracking\sportsmot_yolo26m\final\tracker_benchmark_report.md"
