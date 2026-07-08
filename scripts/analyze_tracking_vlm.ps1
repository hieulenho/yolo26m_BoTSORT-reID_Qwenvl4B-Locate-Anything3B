param(
    [Parameter(Mandatory=$true)]
    [string]$SourceVideo,
    [string]$TrackedVideo = "",
    [string]$Tracks = "",
    [string]$Metadata = "",
    [string]$OutputDir = "",
    [string]$Config = "configs/vlm_qwen4b_tracking.yaml",
    [string]$ModelId = "Qwen/Qwen3-VL-4B-Instruct",
    [string]$Device = "auto",
    [string]$TorchDtype = "auto",
    [ValidateSet("none", "8bit", "4bit")]
    [string]$Quantization = "8bit",
    [int]$MaxNewTokens = 512,
    [double]$KeyframeInterval = 1.0,
    [int]$MaxKeyframes = 12,
    [int]$MaxTracks = 40,
    [int]$MaxCropsPerTrack = 3,
    [switch]$RunModel,
    [switch]$Overwrite,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not $OutputDir) {
    $sourceName = [System.IO.Path]::GetFileNameWithoutExtension($SourceVideo)
    $OutputDir = Join-Path (Split-Path $SourceVideo -Parent) "${sourceName}_vlm"
}

if (-not $Tracks -and $TrackedVideo) {
    $Tracks = [System.IO.Path]::ChangeExtension($TrackedVideo, ".txt")
}

if (-not $Metadata -and $TrackedVideo) {
    $trackedStem = [System.IO.Path]::GetFileNameWithoutExtension($TrackedVideo)
    $Metadata = Join-Path (Split-Path $TrackedVideo -Parent) "${trackedStem}.metadata.json"
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Args = @(
    "-m", "football_tracking.cli",
    "analyze-tracking-vlm",
    "--config", $Config,
    "--source-video", $SourceVideo,
    "--output-dir", $OutputDir,
    "--model-id", $ModelId,
    "--device", $Device,
    "--torch-dtype", $TorchDtype,
    "--quantization", $Quantization,
    "--max-new-tokens", "$MaxNewTokens",
    "--keyframe-interval", "$KeyframeInterval",
    "--max-keyframes", "$MaxKeyframes",
    "--max-tracks", "$MaxTracks",
    "--max-crops-per-track", "$MaxCropsPerTrack"
)
if ($TrackedVideo) { $Args += @("--tracked-video", $TrackedVideo) }
if ($Tracks) { $Args += @("--tracks", $Tracks) }
if ($Metadata) { $Args += @("--metadata", $Metadata) }
if ($RunModel) { $Args += "--run-model" }
if ($Overwrite) { $Args += "--overwrite" }
if ($DryRun) { $Args += "--dry-run" }

& $Python @Args
exit $LASTEXITCODE
