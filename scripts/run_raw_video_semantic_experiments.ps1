param(
    [Parameter(Mandatory = $true)]
    [string]$SourceVideo,

    [string[]]$Pipelines = @("A", "B", "C"),

    [string]$Query = "the goalkeeper wearing green",

    [string]$OutputRoot = "",

    [string]$TrackingConfig = "configs/track_video_yolo26m_botsort.yaml",

    # Dùng file tracking sẵn có, bỏ qua bước track lại
    [string]$Tracks = "",

    [string]$TrackedVideo = "",

    [string]$TrackingMetadata = "",

    [switch]$SkipTracking,

    [string]$QwenModelId = "Qwen/Qwen3-VL-4B-Instruct",

    [string]$LocateModelId = "nvidia/LocateAnything-3B",

    [ValidateSet("locate_anything", "mock")]
    [string]$LocateBackend = "locate_anything",

    [string]$Device = "auto",

    [string]$TorchDtype = "auto",

    [int]$MaxNewTokens = 512,

    [int]$MaxKeyframes = 2,

    [int]$MaxTracks = 20,

    [int]$MaxCropsPerTrack = 1,

    [int]$LocateMaxFrames = 6,

    [int]$MaxFrames = 0,

    [bool]$RenderPipelineVideos = $true,

    [bool]$UseBenchmarkLabelsForRender = $true,

    [switch]$HideUnlabeledRenderTracks,

    [switch]$RunQwenModel,

    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $SourceVideo -PathType Leaf)) {
    throw "Source video does not exist: $SourceVideo"
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SourcePath = Resolve-Path -LiteralPath $SourceVideo
$SourceDir = Split-Path $SourcePath -Parent
$Stem = [System.IO.Path]::GetFileNameWithoutExtension($SourcePath)
$SequenceName = if ($Stem -match '^\d+$') { "video_$Stem" } else { $Stem }

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $ProjectRoot "outputs\semantic_video_experiments"
}

$RunRoot = Join-Path $OutputRoot $Stem
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$BenchmarkDir = Join-Path $ProjectRoot "data\team_benchmark\$SequenceName"
$AnnotationCsv = Join-Path $BenchmarkDir "track_annotation_expanded.csv"
$PreferredManifestA = Join-Path $BenchmarkDir "pipeline_a_yolo26m_botsort_reid_qwen4b_expanded_bootstrap.json"
$PreferredManifestB = Join-Path $BenchmarkDir "pipeline_b_yolo26m_botsort_reid_locateanything3b_expanded_bootstrap.json"
$PreferredManifestC = Join-Path $BenchmarkDir "pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b_expanded_bootstrap.json"

$TrackedVideo = Join-Path $SourceDir "${Stem}_yolo26m_botsort_reid.mp4"
$DefaultTracks = [System.IO.Path]::ChangeExtension($TrackedVideo, ".txt")
# Dùng -Tracks override nếu có (ví dụ: F:\videos\1_Tracking_qwen.txt)
if ($Tracks) {
    $TracksResolved = if (Test-Path -LiteralPath $Tracks) {
        (Resolve-Path -LiteralPath $Tracks).Path
    } else {
        $Tracks
    }
} else {
    $TracksResolved = $DefaultTracks
}
$TrackingMetadata = Join-Path $SourceDir "${Stem}_yolo26m_botsort_reid.metadata.json"

Write-Host "==> Source video: $SourcePath"
Write-Host "==> Run root    : $RunRoot"
Write-Host "==> Shared tracks: $TracksResolved"
Write-Host "==> Sequence name: $SequenceName"
if ($SkipTracking) { Write-Host "==> [SkipTracking] Bỏ qua bước track, dùng file có sẵn." }

function Invoke-PipelineVideoRender {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Pipeline,
        [Parameter(Mandatory = $true)]
        [string]$PipelineOutputDir,
        [string]$PreferredManifest = "",
        [string]$LocateFinalResolution = "",
        [string]$QwenAnswer = ""
    )

    if (-not $RenderPipelineVideos) {
        return $null
    }

    $SuffixByPipeline = @{
        A = "qwen4b"
        B = "locateanything3b"
        C = "locateanything3b_qwen4b"
    }
    $TitleByPipeline = @{
        A = "Pipeline A | YOLO26m + BoT-SORT ReID + Qwen3-VL 4B"
        B = "Pipeline B | YOLO26m + BoT-SORT ReID + LocateAnything 3B"
        C = "Pipeline C | YOLO26m + BoT-SORT ReID + LocateAnything 3B + Qwen3-VL 4B"
    }
    $Predictions = Join-Path $PipelineOutputDir "render_predictions.json"
    $OutputVideo = Join-Path $SourceDir ("{0}_pipeline_{1}_{2}.mp4" -f $Stem, $Pipeline, $SuffixByPipeline[$Pipeline])

    Write-Host ""
    Write-Host "==> Render Pipeline $Pipeline video: $OutputVideo"

    $PredictionArgs = @(
        "scripts\build_pipeline_render_predictions.py",
        "--pipeline", $Pipeline,
        "--sequence-name", $SequenceName,
        "--query", $Query,
        "--tracks", "$TracksResolved",
        "--output", "$Predictions"
    )
    if ($PreferredManifest -and (Test-Path -LiteralPath $PreferredManifest -PathType Leaf)) {
        $PredictionArgs += @("--preferred-manifest", "$PreferredManifest")
    }
    if ($UseBenchmarkLabelsForRender -and (Test-Path -LiteralPath $AnnotationCsv -PathType Leaf)) {
        $PredictionArgs += @("--annotation-csv", "$AnnotationCsv", "--use-annotation-labels")
    }
    if ($LocateFinalResolution -and (Test-Path -LiteralPath $LocateFinalResolution -PathType Leaf)) {
        $PredictionArgs += @("--locate-final-resolution", "$LocateFinalResolution")
    }
    if ($QwenAnswer -and (Test-Path -LiteralPath $QwenAnswer -PathType Leaf)) {
        $PredictionArgs += @("--qwen-answer", "$QwenAnswer")
    }
    if ($Overwrite) { $PredictionArgs += "--overwrite" }
    & $Python @PredictionArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $RenderArgs = @(
        "scripts\render_team_position_video.py",
        "--source-video", "$SourcePath",
        "--tracks", "$TracksResolved",
        "--predictions", "$Predictions",
        "--sequence-name", $SequenceName,
        "--output-video", "$OutputVideo",
        "--title", $TitleByPipeline[$Pipeline]
    )
    if ($HideUnlabeledRenderTracks) { $RenderArgs += "--hide-unlabeled" }
    if ($Overwrite) { $RenderArgs += "--overwrite" }
    & $Python @RenderArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    return [ordered]@{
        video = "$OutputVideo"
        metadata = "$([System.IO.Path]::ChangeExtension($OutputVideo, ".metadata.json"))"
        predictions = "$Predictions"
    }
}

$TrackArgs = @(
    "-m", "football_tracking.cli",
    "track-video",
    "--config", $TrackingConfig,
    "--source", "$SourcePath",
    "--output-video", "$TrackedVideo",
    "--device", $Device
)
if ($Overwrite) { $TrackArgs += "--overwrite" }
if ($MaxFrames -gt 0) { $TrackArgs += @("--max-frames", "$MaxFrames") }

Write-Host ""
if ($SkipTracking) {
    Write-Host "==> Step 1/4: SKIP (dùng tracks có sẵn: $TracksResolved)"
    if (-not (Test-Path -LiteralPath $TracksResolved -PathType Leaf)) {
        throw "Tracks file khong ton tai: $TracksResolved"
    }
} else {
    Write-Host "==> Step 1/4: YOLO26m + BoT-SORT ReID tracking"
    & $Python @TrackArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$NormalizedPipelines = @($Pipelines | ForEach-Object { $_.ToUpperInvariant() })
$RenderedVideos = [ordered]@{}

if ($NormalizedPipelines -contains "A") {
    $AOut = Join-Path $RunRoot "pipeline_a_yolo26m_botsort_reid_qwen4b"
    Write-Host ""
    Write-Host "==> Pipeline A: YOLO26m + BoT-SORT ReID + Qwen3-VL 4B"
    .\scripts\analyze_tracking_vlm.ps1 `
        -SourceVideo "$SourcePath" `
        -TrackedVideo "$TrackedVideo" `
        -Tracks "$TracksResolved" `
        -Metadata "$TrackingMetadata" `
        -OutputDir "$AOut" `
        -ModelId $QwenModelId `
        -Device $Device `
        -TorchDtype $TorchDtype `
        -MaxNewTokens $MaxNewTokens `
        -MaxKeyframes $MaxKeyframes `
        -MaxTracks $MaxTracks `
        -MaxCropsPerTrack $MaxCropsPerTrack `
        -RunModel:$RunQwenModel `
        -Overwrite:$Overwrite
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $RenderedVideos["A"] = Invoke-PipelineVideoRender `
        -Pipeline "A" `
        -PipelineOutputDir "$AOut" `
        -PreferredManifest "$PreferredManifestA" `
        -QwenAnswer (Join-Path $AOut "vlm_answer.json")
}

if ($NormalizedPipelines -contains "B") {
    $BOut = Join-Path $RunRoot "pipeline_b_yolo26m_botsort_reid_locateanything3b"
    Write-Host ""
    Write-Host "==> Pipeline B: YOLO26m + BoT-SORT ReID + LocateAnything 3B"
    $LocateArgs = @(
        "-m", "football_tracking.locate_tracking.cli",
        "resolve-language-track",
        "--source-video", "$SourcePath",
        "--tracks", "$TracksResolved",
        "--query", $Query,
        "--output-dir", "$BOut",
        "--backend", $LocateBackend,
        "--model-id", $LocateModelId,
        "--device", $Device,
        "--torch-dtype", $TorchDtype,
        "--max-new-tokens", "$MaxNewTokens",
        "--max-frames", "$LocateMaxFrames",
        "--query-mode", "single_target"
    )
    if ($Overwrite) { $LocateArgs += "--overwrite" }
    & $Python @LocateArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $RenderedVideos["B"] = Invoke-PipelineVideoRender `
        -Pipeline "B" `
        -PipelineOutputDir "$BOut" `
        -PreferredManifest "$PreferredManifestB" `
        -LocateFinalResolution (Join-Path $BOut "final_resolution.json")
}

if ($NormalizedPipelines -contains "C") {
    $COut = Join-Path $RunRoot "pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b"
    $CLocateOut = Join-Path $COut "locateanything"
    $CQwenOut = Join-Path $COut "qwen4b"
    Write-Host ""
    Write-Host "==> Pipeline C: YOLO26m + BoT-SORT ReID + LocateAnything 3B + Qwen3-VL 4B"
    $LocateArgs = @(
        "-m", "football_tracking.locate_tracking.cli",
        "resolve-language-track",
        "--source-video", "$SourcePath",
        "--tracks", "$TracksResolved",
        "--query", $Query,
        "--output-dir", "$CLocateOut",
        "--backend", $LocateBackend,
        "--model-id", $LocateModelId,
        "--device", $Device,
        "--torch-dtype", $TorchDtype,
        "--max-new-tokens", "$MaxNewTokens",
        "--max-frames", "$LocateMaxFrames",
        "--query-mode", "single_target"
    )
    if ($Overwrite) { $LocateArgs += "--overwrite" }
    & $Python @LocateArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    .\scripts\analyze_tracking_vlm.ps1 `
        -SourceVideo "$SourcePath" `
        -TrackedVideo "$TrackedVideo" `
        -Tracks "$TracksResolved" `
        -Metadata "$TrackingMetadata" `
        -OutputDir "$CQwenOut" `
        -ModelId $QwenModelId `
        -Device $Device `
        -TorchDtype $TorchDtype `
        -MaxNewTokens $MaxNewTokens `
        -MaxKeyframes $MaxKeyframes `
        -MaxTracks $MaxTracks `
        -MaxCropsPerTrack $MaxCropsPerTrack `
        -RunModel:$RunQwenModel `
        -Overwrite:$Overwrite
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $RenderedVideos["C"] = Invoke-PipelineVideoRender `
        -Pipeline "C" `
        -PipelineOutputDir "$COut" `
        -PreferredManifest "$PreferredManifestC" `
        -LocateFinalResolution (Join-Path $CLocateOut "final_resolution.json") `
        -QwenAnswer (Join-Path $CQwenOut "vlm_answer.json")
}

$Summary = [ordered]@{
    source_video = "$SourcePath"
    sequence_name = $SequenceName
    query = $Query
    run_root = "$RunRoot"
    tracked_video = "$TrackedVideo"
    tracks = "$TracksResolved"
    tracking_metadata = "$TrackingMetadata"
    pipelines = $NormalizedPipelines
    qwen_model_loaded = [bool]$RunQwenModel
    locate_backend = $LocateBackend
    render_pipeline_videos = [bool]$RenderPipelineVideos
    use_benchmark_labels_for_render = [bool]$UseBenchmarkLabelsForRender
    pipeline_videos = $RenderedVideos
}
$SummaryPath = Join-Path $RunRoot "run_summary.json"
$Summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8

Write-Host ""
Write-Host "==> Done."
Write-Host "Summary: $SummaryPath"
