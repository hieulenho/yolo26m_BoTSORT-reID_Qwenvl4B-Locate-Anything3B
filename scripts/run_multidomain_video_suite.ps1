<#
.SYNOPSIS
    Run the adaptive 8-bit pipeline over the normalized multi-domain suite.
#>

param(
    [string]$Manifest = "F:\videos\multidomain_video_suite_manifest.json",
    [string]$OutputDir = "F:\videos",
    [string]$OutputRoot = "outputs\adaptive_runs\multidomain_suite_8bit",
    [ValidateSet("realtime", "realtime_stable", "balanced", "accuracy")]
    [string]$Profile = "realtime_stable",
    [string[]]$VideoIds = @(),
    [ValidateRange(0, 100000)]
    [int]$SemanticMaxTracks = 0,
    [ValidateRange(2, 64)]
    [int]$SemanticMaxImages = 2,
    [ValidateRange(1, 32)]
    [int]$SemanticMaxTracksPerBatch = 1,
    [ValidateRange(64, 4096)]
    [int]$SemanticMaxNewTokens = 192,
    [ValidateRange(65536, 1048576)]
    [int]$SemanticImageMaxPixels = 196608,
    [ValidateRange(65536, 1048576)]
    [int]$QwenImageMaxPixels = 262144,
    [ValidateRange(65536, 1048576)]
    [int]$LocateImageMaxPixels = 262144,
    [ValidateRange(0, 100000)]
    [int]$LocateMaxTracks = 0,
    [switch]$SkipSemantics,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment Python does not exist: $Python"
}
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Video suite manifest does not exist: $Manifest"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Payload = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$Videos = if ($Payload.samples) { @($Payload.samples) } else { @($Payload.videos) }
if ($VideoIds.Count -gt 0) {
    $Videos = @($Videos | Where-Object { $VideoIds -contains ([string]$_.video_id) })
}
if ($Videos.Count -eq 0) {
    throw "No videos matched the requested suite selection."
}

for ($Index = 0; $Index -lt $Videos.Count; $Index++) {
    $Video = $Videos[$Index]
    $Source = [string]$Video.path
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Suite source video does not exist: $Source"
    }
    $Stem = [System.IO.Path]::GetFileNameWithoutExtension($Source)
    $TrackingVideo = Join-Path $OutputDir ($Stem + "_8bit_" + $Profile + "_tracking.mp4")
    $SemanticVideo = Join-Path $OutputDir ($Stem + "_8bit_" + $Profile + "_semantic.mp4")
    $Percent = [int](100 * $Index / $Videos.Count)
    Write-Progress -Activity "Adaptive multi-domain suite" `
        -Status "[$($Index + 1)/$($Videos.Count)] $($Video.video_id)" `
        -PercentComplete $Percent
    Write-Host ""
    Write-Host "==> [$($Index + 1)/$($Videos.Count)] $($Video.video_id)"

    $Arguments = @{
        SourceVideo = $Source
        OutputVideo = $TrackingVideo
        SemanticOutputVideo = $SemanticVideo
        OutputRoot = $OutputRoot
        Profile = $Profile
        QwenQuantization = "8bit"
        Device = "cuda"
        MaxKeyframes = 4
        DiscoveryMaxNewTokens = 768
        QwenImageMaxPixels = $QwenImageMaxPixels
        SemanticMaxTracks = $SemanticMaxTracks
        SemanticMaxImages = $SemanticMaxImages
        SemanticMaxTracksPerBatch = $SemanticMaxTracksPerBatch
        SemanticImageMaxPixels = $SemanticImageMaxPixels
        SemanticMaxNewTokens = $SemanticMaxNewTokens
        LocateMaxTracks = $LocateMaxTracks
        LocateImageMaxPixels = $LocateImageMaxPixels
        RunTrackSemantics = (-not $SkipSemantics)
        RunLocateVerification = (-not $SkipSemantics)
    }
    if ($Overwrite) { $Arguments.Overwrite = $true }
    & (Join-Path $PSScriptRoot "run_adaptive_tracking.ps1") @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Adaptive pipeline failed for $($Video.video_id) with exit code $LASTEXITCODE."
    }
}
Write-Progress -Activity "Adaptive multi-domain suite" -Completed

if (-not $SkipSemantics) {
    $SummaryDir = Join-Path $OutputRoot "summary"
    $ReportArgs = @(
        "scripts\benchmarks\build_multidomain_trial_report.py",
        "--manifest", $Manifest,
        "--run-root", $OutputRoot,
        "--output-dir", $SummaryDir
    )
    if ($Overwrite) { $ReportArgs += "--overwrite" }
    & $Python @ReportArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "Multi-domain suite completed."
Write-Host "  Videos : $OutputDir"
Write-Host "  Runs   : $(Join-Path $ProjectRoot $OutputRoot)"
Write-Host "  Report : $(Join-Path $ProjectRoot (Join-Path $OutputRoot 'summary\multidomain_trial_report.md'))"
