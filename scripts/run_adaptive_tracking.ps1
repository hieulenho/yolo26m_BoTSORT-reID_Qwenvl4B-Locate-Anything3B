<#
.SYNOPSIS
    Dynamic multi-domain tracking from one video path.

.DESCRIPTION
    Stage 1 runs Qwen scene discovery in a short-lived Python process.
    Stage 2 routes to football YOLO26m, COCO YOLO26, or open-vocabulary YOLOE-26.
    Stage 3 uses OC-SORT for realtime, TrackTrack for balanced, or
    BoT-SORT ReID for identity-focused accuracy.
    Qwen exits before detector loading, so both models do not occupy VRAM together.
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SourceVideo,

    [string]$OutputVideo = "",
    [string]$OutputRoot = "outputs\adaptive_runs",

    [ValidateSet("realtime", "balanced", "accuracy")]
    [string]$Profile = "realtime",

    [ValidateSet("auto", "none", "8bit", "4bit")]
    [string]$QwenQuantization = "auto",

    [string]$Device = "cuda",
    [int]$MaxKeyframes = 4,
    [int]$MaxClasses = 24,
    [int]$MaxFrames = 0,
    [int]$SemanticMaxTracks = 20,
    [int]$SemanticMaxImages = 8,

    [bool]$RunLocateVerification = $true,
    [bool]$RunTrackSemantics = $true,
    [string]$SemanticOutputVideo = "",

    [switch]$SkipDiscovery,
    [switch]$SkipTracking,
    [switch]$RefreshSemanticCache,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment Python does not exist: $Python"
}
if (-not (Test-Path -LiteralPath $SourceVideo -PathType Leaf)) {
    throw "Source video does not exist: $SourceVideo"
}

$SourcePath = (Resolve-Path -LiteralPath $SourceVideo).Path
$Stem = [System.IO.Path]::GetFileNameWithoutExtension($SourcePath)
$SourceDirectory = Split-Path -Parent $SourcePath
$RunRoot = Join-Path $ProjectRoot (Join-Path $OutputRoot $Stem)
$DiscoveryWork = Join-Path $RunRoot "discovery"
$DiscoveryPath = Join-Path $DiscoveryWork "scene_discovery.json"
$PlanRoot = Join-Path $RunRoot "plan"
$GeneratedConfig = Join-Path $PlanRoot "tracking.generated.yaml"

if (-not $OutputVideo) {
    $OutputVideo = Join-Path $SourceDirectory ($Stem + "_adaptive_tracking.mp4")
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputVideo)
$MotPath = [System.IO.Path]::ChangeExtension($OutputPath, ".txt")
$TrackingMetadataName = [System.IO.Path]::GetFileNameWithoutExtension($OutputPath) + ".metadata.json"
$TrackingMetadataPath = [System.IO.Path]::Combine((Split-Path $OutputPath), $TrackingMetadataName)
$GroundingRoot = Join-Path $RunRoot "locate_verification"
$GroundingPlan = Join-Path $GroundingRoot "grounding_plan.json"
$LocateResult = Join-Path $GroundingRoot "grounding_verification.json"
$QwenSemanticRoot = Join-Path $RunRoot "qwen_track_semantics"
$QwenAnswer = Join-Path $QwenSemanticRoot "vlm_answer.json"
$FusedSemantics = Join-Path $RunRoot "fused_track_semantics.json"
if (-not $SemanticOutputVideo) {
    $SemanticOutputVideo = Join-Path $SourceDirectory ($Stem + "_adaptive_semantic.mp4")
}
$SemanticOutputPath = [System.IO.Path]::GetFullPath($SemanticOutputVideo)
$SemanticMetadataPath = [System.IO.Path]::Combine(
    (Split-Path $SemanticOutputPath),
    ([System.IO.Path]::GetFileNameWithoutExtension($SemanticOutputPath) + ".semantic.metadata.json")
)
$RunReport = Join-Path $RunRoot "adaptive_run_report.json"
if ($OutputPath -eq $SourcePath -or $SemanticOutputPath -eq $SourcePath) {
    throw "Output video must not overwrite the source video: $SourcePath"
}
New-Item -ItemType Directory -Force -Path $DiscoveryWork, $PlanRoot | Out-Null

if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}
$EffectiveQwenQuantization = $QwenQuantization
if ($EffectiveQwenQuantization -eq "auto") {
    $EffectiveQwenQuantization = "4bit"
}

Write-Host ""
Write-Host "==> Adaptive tracking: $SourcePath"
Write-Host "    profile : $Profile"
Write-Host "    Qwen    : $EffectiveQwenQuantization"
Write-Host "    output  : $OutputPath"

if (-not $SkipDiscovery) {
    Write-Host ""
    Write-Host "[1/8] Qwen scene discovery and semantic cache"
    $DiscoveryArgs = @(
        "-m", "football_tracking.adaptive_tracking.cli", "discover",
        "--source", $SourcePath,
        "--output", $DiscoveryPath,
        "--quantization", $EffectiveQwenQuantization,
        "--device", $Device,
        "--max-keyframes", "$MaxKeyframes",
        "--max-classes", "$MaxClasses"
    )
    if ($RefreshSemanticCache) { $DiscoveryArgs += "--refresh-cache" }
    if ($Overwrite) { $DiscoveryArgs += "--overwrite" }
    & $Python @DiscoveryArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} elseif (-not (Test-Path -LiteralPath $DiscoveryPath -PathType Leaf)) {
    throw "SkipDiscovery was requested but discovery is missing: $DiscoveryPath"
}

Write-Host ""
Write-Host "[2/8] Vocabulary normalization and detector routing"
$PlanArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "build-plan",
    "--source", $SourcePath,
    "--discovery", $DiscoveryPath,
    "--output-dir", $PlanRoot,
    "--output-video", $OutputPath,
    "--profile", $Profile,
    "--device", $Device
)
if ($MaxFrames -gt 0) { $PlanArgs += @("--max-frames", "$MaxFrames") }
if ($Overwrite) { $PlanArgs += "--overwrite" }
& $Python @PlanArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipTracking) {
    Write-Host ""
    Write-Host "[3/8] Routed detector and profile-selected tracker"
    $TrackArgs = @(
        "-m", "football_tracking.cli", "track-video",
        "--config", $GeneratedConfig,
        "--source", $SourcePath,
        "--output-video", $OutputPath,
        "--device", $Device,
        "--save-mot"
    )
    if ($MaxFrames -gt 0) { $TrackArgs += @("--max-frames", "$MaxFrames") }
    if ($Overwrite) { $TrackArgs += "--overwrite" }
    & $Python @TrackArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path -LiteralPath $MotPath -PathType Leaf)) {
    throw "Tracking MOT output does not exist: $MotPath"
}

if ($RunTrackSemantics) {
    Write-Host ""
    Write-Host "[4/8] Qwen batched multi-time track semantics"
    $DiscoveryData = Get-Content -LiteralPath $DiscoveryPath -Raw | ConvertFrom-Json
    $DiscoveredClasses = @($DiscoveryData.objects | ForEach-Object { $_.canonical_name }) -join ", "
    $TaskPrompt = "Domain discovered as '$($DiscoveryData.domain.name)'. Candidate object vocabulary: $DiscoveredClasses. Independently label every visible track using visual evidence; preserve unseen classes and reject uncertainty."
    $QwenArgs = @(
        "-m", "football_tracking.cli", "analyze-tracking-vlm",
        "--config", "configs\vlm_dynamic_track_semantics.yaml",
        "--source-video", $SourcePath,
        "--tracked-video", $OutputPath,
        "--tracks", $MotPath,
        "--output-dir", $QwenSemanticRoot,
        "--task-prompt", $TaskPrompt,
        "--device", $Device,
        "--quantization", $EffectiveQwenQuantization,
        "--max-keyframes", "2",
        "--max-tracks", "$SemanticMaxTracks",
        "--max-crops-per-track", "2",
        "--max-model-images", "$SemanticMaxImages",
        "--max-new-tokens", "768",
        "--output-schema", "dynamic",
        "--run-model"
    )
    if (Test-Path -LiteralPath $TrackingMetadataPath -PathType Leaf) {
        $QwenArgs += @("--metadata", $TrackingMetadataPath)
    }
    if ($Overwrite) { $QwenArgs += "--overwrite" }
    & $Python @QwenArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "[5/8] Build event-triggered LocateAnything plan"
$GroundingPlanArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "build-grounding-plan",
    "--discovery", $DiscoveryPath,
    "--output", $GroundingPlan,
    "--max-keyframes-per-class", "1",
    "--max-expected-tracks-per-class", "$SemanticMaxTracks"
)
if ($RunTrackSemantics -and (Test-Path -LiteralPath $QwenAnswer -PathType Leaf)) {
    $GroundingPlanArgs += @("--qwen-answer", $QwenAnswer)
}
if ($Overwrite) { $GroundingPlanArgs += "--overwrite" }
& $Python @GroundingPlanArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($RunLocateVerification) {
    Write-Host ""
    Write-Host "[6/8] Event-triggered LocateAnything grounding"
    $LocateArgs = @(
        "-m", "football_tracking.adaptive_tracking.cli", "execute-grounding-plan",
        "--plan", $GroundingPlan,
        "--tracks", $MotPath,
        "--output", $LocateResult,
        "--device", $Device,
        "--quantization", "8bit",
        "--max-new-tokens", "256"
    )
    if ($Overwrite) { $LocateArgs += "--overwrite" }
    & $Python @LocateArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "[7/8] Semantic fusion, unknown rejection, and final render"
$FusionArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "fuse-semantics",
    "--output", $FusedSemantics,
    "--unknown-threshold", "0.45",
    "--minimum-margin", "0.10"
)
if ($RunTrackSemantics -and (Test-Path -LiteralPath $QwenAnswer -PathType Leaf)) {
    $FusionArgs += @("--qwen-answer", $QwenAnswer)
}
if ($RunLocateVerification -and (Test-Path -LiteralPath $LocateResult -PathType Leaf)) {
    $FusionArgs += @("--locate-result", $LocateResult)
}
if ($Overwrite) { $FusionArgs += "--overwrite" }
& $Python @FusionArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$RenderArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "render-semantics",
    "--source", $SourcePath,
    "--tracks", $MotPath,
    "--semantics", $FusedSemantics,
    "--output-video", $SemanticOutputPath
)
if ($MaxFrames -gt 0) { $RenderArgs += @("--max-frames", "$MaxFrames") }
if ($Overwrite) { $RenderArgs += "--overwrite" }
& $Python @RenderArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "[8/8] Consolidate timing, VRAM, coverage, and provenance"
$ReportArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "build-run-report",
    "--run-root", $RunRoot,
    "--tracking-metadata", $TrackingMetadataPath,
    "--semantic-metadata", $SemanticMetadataPath,
    "--output", $RunReport
)
if ($Overwrite) { $ReportArgs += "--overwrite" }
& $Python @ReportArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Adaptive tracking completed."
Write-Host "  Tracking video  : $OutputPath"
Write-Host "  Semantic video  : $SemanticOutputPath"
Write-Host "  MOT             : $MotPath"
Write-Host "  Track metadata  : $TrackingMetadataPath"
Write-Host "  Fused semantics : $FusedSemantics"
Write-Host "  Plan            : $(Join-Path $PlanRoot 'adaptive_plan.json')"
Write-Host "  Run report      : $RunReport"
