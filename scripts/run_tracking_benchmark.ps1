<#
.SYNOPSIS
    Compare realtime MOT trackers fairly on a shared detector cache.
#>

param(
    [string]$Config = "configs\benchmarks\tracking_sportsmot_yolo26m.yaml",
    [string]$SmokeConfig = "configs\benchmarks\tracking_sportsmot_yolo26m_smoke.yaml",
    [switch]$Smoke,
    [int]$SmokeFrames = 300,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python does not exist: $Python"
}
$ActiveConfig = $Config
if ($Smoke -and -not $PSBoundParameters.ContainsKey("Config")) {
    $ActiveConfig = $SmokeConfig
}
if (-not (Test-Path -LiteralPath $ActiveConfig)) {
    throw "Benchmark config does not exist: $ActiveConfig"
}

$Arguments = @(
    "-m", "football_tracking.cli",
    "compare-trackers",
    "--config", $ActiveConfig
)
if ($Smoke) {
    $Arguments += @("--max-sequences", "1", "--max-frames", "$SmokeFrames")
}
if ($Overwrite) {
    $Arguments += "--overwrite"
}

Write-Host "==> Shared-cache tracker benchmark"
Write-Host "==> Config: $ActiveConfig"
if ($Smoke) {
    Write-Host "==> Mode: smoke (1 sequence, up to $SmokeFrames frames)"
} else {
    Write-Host "==> Mode: full SportsMOT split"
}

& $Python @Arguments
exit $LASTEXITCODE
