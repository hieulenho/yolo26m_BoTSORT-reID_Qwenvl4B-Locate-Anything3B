<#
.SYNOPSIS
    Dynamic multi-domain tracking from one video path.

.DESCRIPTION
    Stage 1 runs Qwen scene discovery in a short-lived Python process.
    Stage 2 routes to football YOLO26m, COCO YOLO26, or open-vocabulary YOLOE-26.
    Stage 3 uses OC-SORT for realtime, TrackTrack for balanced, or
    BoT-SORT ReID for identity-focused accuracy.
    After tracking, LocateAnything verifies each track region before Qwen assigns open,
    fine-grained semantics. Large models run sequentially to stay within an 8 GB GPU.
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SourceVideo,

    [string]$OutputVideo = "",
    [string]$OutputRoot = "outputs\adaptive_runs",

    [ValidateSet("realtime", "realtime_stable", "balanced", "accuracy")]
    [string]$Profile = "realtime_stable",

    [ValidateSet("auto", "none", "8bit", "4bit")]
    [string]$QwenQuantization = "auto",

    [string]$Device = "cuda",
    [int]$MaxKeyframes = 4,
    [int]$MaxClasses = 24,
    [ValidateRange(128, 1024)]
    [int]$DiscoveryMaxNewTokens = 768,
    [ValidateRange(65536, 16777216)]
    [int]$QwenImageMaxPixels = 524288,
    [int]$MaxFrames = 0,
    [ValidateRange(0, 100000)]
    [int]$SemanticMaxTracks = 0,
    [ValidateRange(2, 64)]
    [int]$SemanticMaxImages = 2,
    [ValidateRange(1, 16)]
    [int]$SemanticMaxTracksPerBatch = 1,
    [ValidateRange(65536, 16777216)]
    [int]$SemanticImageMaxPixels = 196608,
    [ValidateRange(128, 1024)]
    [int]$SemanticMaxNewTokens = 192,
    [ValidateRange(0, 100000)]
    [int]$LocateMaxTracks = 0,
    [ValidateRange(4096, 4194304)]
    [int]$LocateImageMaxPixels = 262144,
    [ValidateRange(0.0, 2.0)]
    [double]$LocateCropPadding = 0.35,

    [ValidateRange(0.0, 1.0)]
    [double]$FineUnknownThreshold = 0.82,

    [bool]$RunLocateVerification = $true,
    [bool]$RunTrackSemantics = $true,
    [bool]$LocateFirstAllTracks = $true,
    [bool]$ReuseLocateVerification = $false,
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
$LocateContextRoot = Join-Path $RunRoot "locate_context"
$LocateContext = Join-Path $LocateContextRoot "vlm_context.json"
$QwenSemanticRoot = Join-Path $RunRoot "qwen_track_semantics"
$QwenAnswer = Join-Path $QwenSemanticRoot "vlm_answer.json"
$FusedSemantics = Join-Path $RunRoot "fused_track_semantics.json"
$SemanticMemory = Join-Path $RunRoot "semantic_memory.json"
$TrackDiagnosticsJson = Join-Path $RunRoot "track_diagnostics.json"
$TrackDiagnosticsMd = Join-Path $RunRoot "track_diagnostics.md"
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
    $EffectiveQwenQuantization = "8bit"
}

Write-Host ""
Write-Host "==> Adaptive tracking: $SourcePath"
Write-Host "    profile : $Profile"
Write-Host "    Qwen    : $EffectiveQwenQuantization"
Write-Host "    output  : $OutputPath"

if (-not $SkipDiscovery) {
    Write-Host ""
    Write-Host "[1/9] Qwen scene discovery and semantic cache"
    $DiscoveryArgs = @(
        "-m", "football_tracking.adaptive_tracking.cli", "discover",
        "--source", $SourcePath,
        "--output", $DiscoveryPath,
        "--quantization", $EffectiveQwenQuantization,
        "--device", $Device,
        "--max-keyframes", "$MaxKeyframes",
        "--max-classes", "$MaxClasses",
        "--max-new-tokens", "$DiscoveryMaxNewTokens",
        "--image-max-pixels", "$QwenImageMaxPixels"
    )
    if ($RefreshSemanticCache) { $DiscoveryArgs += "--refresh-cache" }
    if ($Overwrite) { $DiscoveryArgs += "--overwrite" }
    & $Python @DiscoveryArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} elseif (-not (Test-Path -LiteralPath $DiscoveryPath -PathType Leaf)) {
    throw "SkipDiscovery was requested but discovery is missing: $DiscoveryPath"
}

Write-Host ""
Write-Host "[2/9] Vocabulary normalization and detector routing"
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
    Write-Host "[3/9] Routed detector and profile-selected tracker"
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

if ($Overwrite -or -not (Test-Path -LiteralPath $TrackDiagnosticsJson -PathType Leaf)) {
    Write-Host "    refreshing track diagnostics"
    $DiagnosticsArgs = @(
        "scripts\data\diagnose_video_tracks.py",
        "--tracks", $MotPath,
        "--source-video", $SourcePath,
        "--output-json", $TrackDiagnosticsJson,
        "--output-md", $TrackDiagnosticsMd
    )
    if (Test-Path -LiteralPath $TrackingMetadataPath -PathType Leaf) {
        $DiagnosticsArgs += @("--metadata", $TrackingMetadataPath)
    }
    if ($Overwrite) { $DiagnosticsArgs += "--overwrite" }
    & $Python @DiagnosticsArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$TrackDiagnostics = Get-Content -LiteralPath $TrackDiagnosticsJson -Raw | ConvertFrom-Json
$TrackingTrackCount = [int]$TrackDiagnostics.unique_track_count

if ($RunLocateVerification -and -not $ReuseLocateVerification) {
    Write-Host ""
    Write-Host "[4/9] Prepare all-track crops for LocateAnything"
    $PrepareArgs = @(
        "-m", "football_tracking.cli", "analyze-tracking-vlm",
        "--config", "configs\semantics\dynamic_track.yaml",
        "--source-video", $SourcePath,
        "--tracked-video", $OutputPath,
        "--tracks", $MotPath,
        "--output-dir", $LocateContextRoot,
        "--task-prompt", "Prepare track crops for LocateAnything spatial verification.",
        "--max-keyframes", "2",
        "--max-tracks", "$SemanticMaxTracks",
        "--max-crops-per-track", "2",
        "--max-model-images", "$SemanticMaxImages",
        "--max-tracks-per-batch", "$SemanticMaxTracksPerBatch",
        "--no-run-model"
    )
    if (Test-Path -LiteralPath $TrackingMetadataPath -PathType Leaf) {
        $PrepareArgs += @("--metadata", $TrackingMetadataPath)
    }
    if ($Overwrite) { $PrepareArgs += "--overwrite" }
    & $Python @PrepareArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $PreparedContext = Get-Content -LiteralPath $LocateContext -Raw | ConvertFrom-Json
    $PreparedTrackCount = @($PreparedContext.tracks).Count
    Write-Host (
        "    prepared tracks={0}, crops={1}, batches={2}" -f
        $PreparedTrackCount,
        @($PreparedContext.crops).Count,
        @($PreparedContext.model_batches).Count
    )
    if ($SemanticMaxTracks -eq 0 -and $PreparedTrackCount -ne $TrackingTrackCount) {
        throw (
            "All-track Locate preparation is incomplete: prepared {0}/{1} track IDs." -f
            $PreparedTrackCount,
            $TrackingTrackCount
        )
    }
}

if ($RunLocateVerification -and -not $ReuseLocateVerification) {
    Write-Host ""
    Write-Host "[5/9] Build post-tracking LocateAnything plan"
    $GroundingPlanArgs = @(
        "-m", "football_tracking.adaptive_tracking.cli", "build-grounding-plan",
        "--discovery", $DiscoveryPath,
        "--output", $GroundingPlan,
        "--semantic-context", $LocateContext,
        "--tracking-metadata", $TrackingMetadataPath,
        "--max-classes", "$MaxClasses",
        "--max-keyframes-per-class", "1",
        "--max-expected-tracks-per-class", "$LocateMaxTracks"
    )
    if ($LocateFirstAllTracks) { $GroundingPlanArgs += "--verify-all-tracks" }
    if ($Overwrite) { $GroundingPlanArgs += "--overwrite" }
    & $Python @GroundingPlanArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $GroundingPlanData = Get-Content -LiteralPath $GroundingPlan -Raw | ConvertFrom-Json
    Write-Host (
        "    Locate requests={0}, all_tracks={1}" -f
        $GroundingPlanData.summary.request_count,
        $GroundingPlanData.summary.all_track_verification
    )

    Write-Host ""
    Write-Host "[6/9] LocateAnything 3B 8-bit spatial verification"
    $LocateArgs = @(
        "-m", "football_tracking.adaptive_tracking.cli", "execute-grounding-plan",
        "--plan", $GroundingPlan,
        "--tracks", $MotPath,
        "--output", $LocateResult,
        "--device", $Device,
        "--quantization", "8bit",
        "--max-new-tokens", "256",
        "--image-max-pixels", "$LocateImageMaxPixels",
        "--target-crop-padding", "$LocateCropPadding",
        "--minimum-association-score", "0.10",
        "--minimum-identity-margin", "0.05"
    )
    if ($Overwrite) { $LocateArgs += "--overwrite" }
    & $Python @LocateArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $LocateData = Get-Content -LiteralPath $LocateResult -Raw | ConvertFrom-Json
    Write-Host (
        "    Locate accepted={0}/{1}, time={2:N1}s, peak_vram={3:N2}GB" -f
        $LocateData.summary.accepted_association_count,
        $LocateData.summary.request_count,
        $LocateData.timing.total_seconds,
        ($LocateData.cuda_memory.peak_allocated_bytes / 1GB)
    )
}
elseif ($ReuseLocateVerification) {
    foreach ($Required in @($LocateContext, $LocateResult)) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            throw "ReuseLocateVerification requires an existing artifact: $Required"
        }
    }
    Write-Host ""
    Write-Host "[4-6/9] Reusing all-track LocateAnything artifacts"
    Write-Host "    context : $LocateContext"
    Write-Host "    result  : $LocateResult"
}

if ($RunTrackSemantics) {
    Write-Host ""
    $LocateEvidenceAvailable = (
        ($RunLocateVerification -or $ReuseLocateVerification) -and
        (Test-Path -LiteralPath $LocateResult -PathType Leaf)
    )
    if ($LocateEvidenceAvailable) {
        Write-Host "[7/9] Qwen semantics using LocateAnything-verified regions"
        $LocateInstruction = (
            "LocateAnything identity-scoped spatial evidence is provided. " +
            "Use it as geometric support only; independently verify semantic labels from pixels."
        )
    }
    else {
        Write-Host "[7/9] Qwen semantics without LocateAnything evidence"
        $LocateInstruction = (
            "No LocateAnything evidence is provided for this run. " +
            "Use only the supplied target-local frame and multi-time crop panel, " +
            "and reject any semantic label not supported by those pixels."
        )
    }
    $DiscoveryData = Get-Content -LiteralPath $DiscoveryPath -Raw | ConvertFrom-Json
    $SemanticProfiles = @($DiscoveryData.objects | ForEach-Object {
        $BaseClass = ([string]$_.canonical_name).Replace(" ", "_")
        $Taxonomy = ([string]$_.taxonomy_hint).Replace(" ", "_")
        $Facets = (@($_.semantic_facets) | ForEach-Object { ([string]$_).Replace(" ", "_") }) -join ","
        "base=$BaseClass;taxonomy=$Taxonomy;facets=$Facets"
    }) -join " | "
    $TaskPrompt = (
        "Domain discovered as '$($DiscoveryData.domain.name)'. " +
        "Structural semantic profiles: $SemanticProfiles. $LocateInstruction " +
        "Assign every requested track a stable detector-compatible base class and, " +
        "only when visual pixels support it, a high-level role, subtype, species, " +
        "breed, make, or model. Do not inherit speculative fine-class names from " +
        "scene discovery. For a classroom use person as the base class and " +
        "student/teacher as the fine label. Preserve unseen classes and reject " +
        "uncertain fine labels."
    )
    $QwenArgs = @(
        "-m", "football_tracking.cli", "analyze-tracking-vlm",
        "--config", "configs\semantics\dynamic_track.yaml",
        "--source-video", $SourcePath,
        "--tracked-video", $OutputPath,
        "--tracks", $MotPath,
        "--output-dir", $QwenSemanticRoot,
        "--task-prompt", $TaskPrompt,
        "--device", $Device,
        "--quantization", $EffectiveQwenQuantization,
        "--max-keyframes", "1",
        "--max-tracks", "$SemanticMaxTracks",
        "--max-crops-per-track", "2",
        "--max-model-images", "$SemanticMaxImages",
        "--max-tracks-per-batch", "$SemanticMaxTracksPerBatch",
        "--max-new-tokens", "$SemanticMaxNewTokens",
        "--image-max-pixels", "$SemanticImageMaxPixels",
        "--output-schema", "dynamic",
        "--run-model"
    )
    if (Test-Path -LiteralPath $TrackingMetadataPath -PathType Leaf) {
        $QwenArgs += @("--metadata", $TrackingMetadataPath)
    }
    if ($LocateEvidenceAvailable) {
        $QwenArgs += @("--grounding", $LocateResult)
    }
    if ($Overwrite) { $QwenArgs += "--overwrite" }
    & $Python @QwenArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $QwenData = Get-Content -LiteralPath $QwenAnswer -Raw | ConvertFrom-Json
    Write-Host (
        "    Qwen predicted={0}/{1}, batches={2}, images={3}, time={4:N1}s, peak_vram={5:N2}GB" -f
        $QwenData.coverage.predicted_track_count,
        $QwenData.coverage.expected_track_count,
        $QwenData.batch_count,
        $QwenData.image_count,
        $QwenData.timing.inference_seconds,
        ($QwenData.cuda_memory.peak_allocated_bytes / 1GB)
    )
    if ($SemanticMaxTracks -eq 0) {
        $QwenExpected = [int]$QwenData.coverage.expected_track_count
        $QwenPredicted = [int]$QwenData.coverage.predicted_track_count
        if (
            $QwenExpected -ne $TrackingTrackCount -or
            $QwenPredicted -ne $TrackingTrackCount
        ) {
            throw (
                (
                    "All-track semantic coverage is incomplete: tracking={0}, " +
                    "Qwen expected={1}, predicted={2}. No semantic video was rendered."
                ) -f $TrackingTrackCount, $QwenExpected, $QwenPredicted
            )
        }
        Write-Host "    all-track semantic inference verified: $QwenPredicted/$TrackingTrackCount"
    }
}

Write-Host ""
Write-Host "[8/9] Semantic fusion, unknown rejection, and final render"
$FusionDiscovery = Get-Content -LiteralPath $DiscoveryPath -Raw | ConvertFrom-Json
if ($Overwrite -and (Test-Path -LiteralPath $SemanticMemory -PathType Leaf)) {
    # An offline overwrite is a fresh experiment. Reusing memory here would
    # count evidence from previous runs more than once.
    Remove-Item -LiteralPath $SemanticMemory -Force
}
$FusionArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "fuse-semantics",
    "--output", $FusedSemantics,
    "--semantic-memory", $SemanticMemory,
    "--memory-context-id", $SourcePath,
    "--unknown-threshold", "0.45",
    "--minimum-margin", "0.10",
    "--fine-unknown-threshold", "$FineUnknownThreshold",
    "--fine-minimum-margin", "0.15",
    "--domain", ([string]$FusionDiscovery.domain.name)
)
if ($RunTrackSemantics -and (Test-Path -LiteralPath $QwenAnswer -PathType Leaf)) {
    $FusionArgs += @("--qwen-answer", $QwenAnswer)
}
if (
    ($RunLocateVerification -or $ReuseLocateVerification) -and
    (Test-Path -LiteralPath $LocateResult -PathType Leaf)
) {
    $FusionArgs += @("--locate-result", $LocateResult)
}
if ($Overwrite) { $FusionArgs += "--overwrite" }
& $Python @FusionArgs | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$FusedData = Get-Content -LiteralPath $FusedSemantics -Raw | ConvertFrom-Json
if (
    $RunTrackSemantics -and
    $SemanticMaxTracks -eq 0 -and
    [int]$FusedData.summary.track_count -ne $TrackingTrackCount
) {
    throw (
        (
            "Semantic fusion is incomplete: fused {0}/{1} track IDs. " +
            "No semantic video was rendered."
        ) -f $FusedData.summary.track_count, $TrackingTrackCount
    )
}
Write-Host (
    "    semantic accepted={0}/{1}, fine={2}/{1}, unknown={3}" -f
    $FusedData.summary.accepted_count,
    $FusedData.summary.track_count,
    $FusedData.summary.fine_accepted_count,
    $FusedData.summary.unknown_count
)

$RenderArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "render-semantics",
    "--source", $SourcePath,
    "--tracks", $MotPath,
    "--semantics", $FusedSemantics,
    "--tracking-metadata", $TrackingMetadataPath,
    "--output-video", $SemanticOutputPath
)
if ($MaxFrames -gt 0) { $RenderArgs += @("--max-frames", "$MaxFrames") }
if ($Overwrite) { $RenderArgs += "--overwrite" }
& $Python @RenderArgs | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $RenderData = Get-Content -LiteralPath $SemanticMetadataPath -Raw | ConvertFrom-Json
Write-Host (
    "    rendered frames={0}, label_coverage={1:P1}, render_fps={2:N1}" -f
    $RenderData.video.rendered_frame_count,
    $RenderData.semantics_summary.label_box_coverage,
    $RenderData.timing.render_fps
)

Write-Host ""
Write-Host "[9/9] Consolidate timing, VRAM, coverage, and provenance"
$ReportArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "build-run-report",
    "--run-root", $RunRoot,
    "--tracking-metadata", $TrackingMetadataPath,
    "--semantic-metadata", $SemanticMetadataPath,
    "--output", $RunReport
)
if ($Overwrite) { $ReportArgs += "--overwrite" }
& $Python @ReportArgs | Out-Null
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
