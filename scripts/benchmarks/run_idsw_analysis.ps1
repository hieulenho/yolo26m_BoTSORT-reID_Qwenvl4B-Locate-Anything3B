<#
.SYNOPSIS
    Run ID switch taxonomy analysis for one or more tracker outputs.

.DESCRIPTION
    Produces JSON, CSV, a full Markdown report, and a standalone Markdown table
    with total IDSW and the percentage of each diagnostic failure type.
#>

param(
    [string]$MotRoot = "data/mot/sportsmot_football",
    [string]$Seqmap = "data/mot/sportsmot_football/seqmaps/all.txt",
    [string[]]$Trackers = @(),
    [string]$OutputDir = "outputs/reports/idsw_taxonomy",
    [float]$IouThreshold = 0.5,
    [int]$ReidGap = 10,
    [int]$SwapWindow = 5,
    [float]$CrowdScale = 1.5,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $MotRoot)) {
    throw "MOT root does not exist: $MotRoot"
}
if (-not (Test-Path -LiteralPath $Seqmap)) {
    throw "Seqmap does not exist: $Seqmap"
}

$Args = @(
    "scripts\benchmarks\analyze_idsw_taxonomy.py",
    "--mot-root", $MotRoot,
    "--seqmap", $Seqmap,
    "--output-dir", $OutputDir,
    "--iou-threshold", "$IouThreshold",
    "--reid-gap", "$ReidGap",
    "--swap-window", "$SwapWindow",
    "--crowd-scale", "$CrowdScale"
)

foreach ($tracker in $Trackers) {
    $Args += @("--tracker", $tracker)
}
if ($Overwrite) {
    $Args += "--overwrite"
}

$TrackerLabel = "(default project trackers)"
if ($Trackers.Count -gt 0) {
    $TrackerLabel = $Trackers -join ", "
}

Write-Host ""
Write-Host "==> Run IDSW taxonomy analysis"
Write-Host "==> MOT root : $MotRoot"
Write-Host "==> Seqmap   : $Seqmap"
Write-Host "==> Output   : $OutputDir"
Write-Host "==> Trackers : $TrackerLabel"
Write-Host ""

& $Python @Args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$ReportPath = Join-Path $OutputDir "idsw_taxonomy_report.md"
$TablePath = Join-Path $OutputDir "idsw_taxonomy_table.md"
$SummaryPath = Join-Path $OutputDir "idsw_taxonomy_summary.csv"

Write-Host ""
Write-Host "==> Done."
Write-Host "==> Markdown report : $ReportPath"
Write-Host "==> Markdown table  : $TablePath"
Write-Host "==> CSV summary     : $SummaryPath"
Write-Host ""

if (Test-Path -LiteralPath $TablePath) {
    Write-Host "--- IDSW Taxonomy Table ---"
    Get-Content -LiteralPath $TablePath |
        Where-Object { $_ -match "^\|" -or $_ -match "^#" } |
        ForEach-Object { Write-Host $_ }
}
