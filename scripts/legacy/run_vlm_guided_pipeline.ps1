<#
.SYNOPSIS
    Compatibility wrapper for the former VLM-guided Pipeline D command.

.DESCRIPTION
    The old implementation duplicated scene discovery and fixed detection to COCO classes.
    It now forwards to the adaptive router, which supports football, COCO, and YOLOE classes.
    New commands should call run_adaptive_tracking.ps1 directly.
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SourceVideo,

    [string]$OutputRoot = "outputs\adaptive_runs",

    [string]$ModelId = "Qwen/Qwen3-VL-4B-Instruct",

    [ValidateSet("none", "8bit", "4bit")]
    [string]$Quantization = "4bit",

    [string]$Device = "cuda",
    [string]$TorchDtype = "auto",
    [int]$NFrames = 4,
    [string]$TrackingConfig = "",
    [int]$MaxFrames = 0,
    [switch]$SkipDiscovery,
    [switch]$SkipTracking,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $SourceVideo -PathType Leaf)) {
    throw "Source video does not exist: $SourceVideo"
}

$SourcePath = (Resolve-Path -LiteralPath $SourceVideo).Path
$Stem = [System.IO.Path]::GetFileNameWithoutExtension($SourcePath)
$SourceDirectory = Split-Path -Parent $SourcePath
$TrackingVideo = Join-Path $SourceDirectory ($Stem + "_adaptive_tracking.mp4")
$SemanticVideo = Join-Path $SourceDirectory ($Stem + "_pipeline_D_vlm_guided.mp4")
$AdaptiveScript = Join-Path $ProjectRoot "scripts\run_adaptive_tracking.ps1"

if ($ModelId -ne "Qwen/Qwen3-VL-4B-Instruct") {
    Write-Warning "-ModelId is retained for compatibility; adaptive config selects the model."
}
if ($TorchDtype -ne "auto") {
    Write-Warning "-TorchDtype is retained for compatibility and is not forwarded."
}
if ($TrackingConfig) {
    Write-Warning "-TrackingConfig is superseded by the selected adaptive profile."
}

Write-Warning (
    "run_vlm_guided_pipeline.ps1 is deprecated; forwarding to " +
    "run_adaptive_tracking.ps1."
)

$ForwardArgs = @(
    "-SourceVideo", $SourcePath,
    "-OutputVideo", $TrackingVideo,
    "-SemanticOutputVideo", $SemanticVideo,
    "-OutputRoot", $OutputRoot,
    "-Profile", "realtime",
    "-QwenQuantization", $Quantization,
    "-Device", $Device,
    "-MaxKeyframes", "$NFrames"
)
if ($MaxFrames -gt 0) {
    $ForwardArgs += @("-MaxFrames", "$MaxFrames")
}
if ($SkipDiscovery) {
    $ForwardArgs += "-SkipDiscovery"
}
if ($SkipTracking) {
    $ForwardArgs += "-SkipTracking"
}
if ($Overwrite) {
    $ForwardArgs += "-Overwrite"
}

& $AdaptiveScript @ForwardArgs
exit $LASTEXITCODE
