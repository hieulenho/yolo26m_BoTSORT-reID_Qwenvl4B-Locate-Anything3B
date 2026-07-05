param(
    [Parameter(Mandatory=$true)]
    [string]$Source,
    [string]$OutputVideo = "",
    [string]$TrackingConfig = "configs/track_video_yolo26m_botsort.yaml",
    [string]$VlmConfig = "configs/vlm_qwen4b_tracking.yaml",
    [string]$Checkpoint = "",
    [string]$ModelId = "Qwen/Qwen3-VL-4B-Instruct",
    [string]$Device = "auto",
    [int]$MaxFrames = 0,
    [double]$KeyframeInterval = 1.0,
    [int]$MaxKeyframes = 12,
    [int]$MaxTracks = 40,
    [int]$MaxCropsPerTrack = 3,
    [switch]$RunModel,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not $OutputVideo) {
    $sourceDir = Split-Path $Source -Parent
    $sourceStem = [System.IO.Path]::GetFileNameWithoutExtension($Source)
    $OutputVideo = Join-Path $sourceDir "${sourceStem}_tracked.mp4"
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrackArgs = @(
    "-m", "football_tracking.cli",
    "track-video",
    "--config", $TrackingConfig,
    "--source", $Source,
    "--output-video", $OutputVideo,
    "--device", $Device
)
if ($Checkpoint) { $TrackArgs += @("--checkpoint", $Checkpoint) }
if ($MaxFrames -gt 0) { $TrackArgs += @("--max-frames", "$MaxFrames") }
if ($Overwrite) { $TrackArgs += "--overwrite" }

& $Python @TrackArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$Tracks = [System.IO.Path]::ChangeExtension($OutputVideo, ".txt")
$trackedStem = [System.IO.Path]::GetFileNameWithoutExtension($OutputVideo)
$Metadata = Join-Path (Split-Path $OutputVideo -Parent) "${trackedStem}.metadata.json"
$sourceStem = [System.IO.Path]::GetFileNameWithoutExtension($Source)
$OutputDir = Join-Path (Split-Path $Source -Parent) "${sourceStem}_vlm"

$VlmArgs = @(
    "-m", "football_tracking.cli",
    "analyze-tracking-vlm",
    "--config", $VlmConfig,
    "--source-video", $Source,
    "--tracked-video", $OutputVideo,
    "--tracks", $Tracks,
    "--metadata", $Metadata,
    "--output-dir", $OutputDir,
    "--model-id", $ModelId,
    "--device", $Device,
    "--keyframe-interval", "$KeyframeInterval",
    "--max-keyframes", "$MaxKeyframes",
    "--max-tracks", "$MaxTracks",
    "--max-crops-per-track", "$MaxCropsPerTrack"
)
if ($RunModel) { $VlmArgs += "--run-model" }
if ($Overwrite) { $VlmArgs += "--overwrite" }

& $Python @VlmArgs
exit $LASTEXITCODE
