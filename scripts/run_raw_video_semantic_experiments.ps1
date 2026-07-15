<#
.SYNOPSIS
    Chạy thực nghiệm semantic video cho 3 pipeline phân loại cầu thủ:
      A : YOLO26m + BoT-SORT ReID + Qwen3-VL 4B
      B : YOLO26m + BoT-SORT ReID + LocateAnything 3B
      C : YOLO26m + BoT-SORT ReID + LocateAnything 3B + Qwen3-VL 4B

.PARAMETER SourceVideo
    Đường dẫn đến file video gốc (bắt buộc). Ví dụ: F:\videos\1.mp4

.PARAMETER Pipelines
    Danh sách pipeline cần chạy. Mặc định: A,B,C

.PARAMETER Query
    Câu query ngôn ngữ tự nhiên cho Pipeline B/C (LocateAnything).
    Ví dụ: "the goalkeeper wearing green"

.PARAMETER SkipTracking
    Bỏ qua bước tracking, dùng tracks file có sẵn (cần -Tracks).

.PARAMETER Tracks
    Đường dẫn đến file MOT .txt có sẵn khi SkipTracking=true.

.PARAMETER RunQwenModel
    Bật inference Qwen VLM thực sự. Nếu bỏ qua, chỉ chuẩn bị prompt/context.

.PARAMETER AnnotationCsv
    Ground truth CSV để đánh giá định lượng (tùy chọn).
    Nếu cung cấp, sẽ chạy evaluate_pipeline_results.py sau render.

.EXAMPLE
    # Chạy đầy đủ từ video gốc:
    .\scripts\run_raw_video_semantic_experiments.ps1 `
        -SourceVideo "F:\videos\1.mp4" `
        -Query "the goalkeeper wearing green" `
        -RunQwenModel `
        -Pipelines A,B,C `
        -Overwrite

.EXAMPLE
    # Skip tracking, dùng tracks có sẵn:
    .\scripts\run_raw_video_semantic_experiments.ps1 `
        -SourceVideo "F:\videos\1.mp4" `
        -Tracks "F:\videos\1_yolo26m_botsort_reid.txt" `
        -SkipTracking `
        -Query "the goalkeeper wearing green" `
        -RunQwenModel `
        -Overwrite
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SourceVideo,

    [string[]]$Pipelines = @("A", "B", "C"),

    [string]$Query = "the goalkeeper wearing green",

    [string]$OutputRoot = "",

    [string]$TrackingConfig = "configs/track_video_yolo26m_botsort.yaml",

    [ValidateSet("football", "football_high_recall", "general_person")]
    [string]$TrackingProfile = "football",

    [string]$TrackCheckpoint = "",

    [double]$TrackConf = -1.0,

    [int]$TrackImgsz = 0,

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

    [ValidateSet("none", "8bit", "4bit")]
    [string]$Quantization = "8bit",

    [int]$MaxNewTokens = 512,

    [int]$MaxKeyframes = 6,

    [int]$MaxTracks = 30,

    [int]$MaxCropsPerTrack = 2,

    [int]$LocateMaxFrames = 6,

    [int]$MaxFrames = 0,

    [bool]$RunTrackDiagnostics = $true,

    [bool]$RenderPipelineVideos = $true,

    [bool]$UseBenchmarkLabelsForRender = $true,

    [bool]$CompleteRenderLabels = $true,

    [int]$RenderLabelSamplesPerTrack = 7,

    # Ground truth annotation CSV để đánh giá định lượng (tùy chọn)
    [string]$AnnotationCsv = "",

    [switch]$HideUnlabeledRenderTracks,

    [switch]$RunQwenModel,

    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$TrackingConfigWasProvided = $PSBoundParameters.ContainsKey("TrackingConfig")
if (-not $TrackingConfigWasProvided) {
    switch ($TrackingProfile) {
        "football_high_recall" {
            $TrackingConfig = "configs/track_video_yolo26m_botsort_high_recall.yaml"
        }
        "general_person" {
            $TrackingConfig = "configs/track_video_yolo26m_pretrained_person_botsort_high_recall.yaml"
        }
        default {
            $TrackingConfig = "configs/track_video_yolo26m_botsort.yaml"
        }
    }
}

if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}

if (-not (Test-Path -LiteralPath $SourceVideo -PathType Leaf)) {
    throw "Source video does not exist: $SourceVideo"
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SourcePath = (Resolve-Path -LiteralPath $SourceVideo).Path
$SourceDir = Split-Path $SourcePath -Parent
$Stem = [System.IO.Path]::GetFileNameWithoutExtension($SourcePath)
$SequenceName = if ($Stem -match '^\d+$') { "video_$Stem" } else { $Stem }

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $ProjectRoot "outputs\semantic_video_experiments"
}

$RunRoot = Join-Path $OutputRoot $Stem
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

# Ground truth annotation: ưu tiên tham số, sau đó dùng default benchmark dir
$BenchmarkDir = Join-Path $ProjectRoot "data\team_benchmark\$SequenceName"
if (-not $AnnotationCsv) {
    $DefaultAnnotationCsv = Join-Path $BenchmarkDir "track_annotation_expanded.csv"
    if (Test-Path -LiteralPath $DefaultAnnotationCsv -PathType Leaf) {
        $AnnotationCsv = $DefaultAnnotationCsv
    }
}

# Preferred manifests (bootstrap predictions có sẵn để dùng làm fallback render labels)
$PreferredManifestA = Join-Path $BenchmarkDir "pipeline_a_yolo26m_botsort_reid_qwen4b_expanded_bootstrap.json"
$PreferredManifestB = Join-Path $BenchmarkDir "pipeline_b_yolo26m_botsort_reid_locateanything3b_expanded_bootstrap.json"
$PreferredManifestC = Join-Path $BenchmarkDir "pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b_expanded_bootstrap.json"
$CompletionManifest = Join-Path $RunRoot "shared\visual_label_completion.json"

# Tracking output paths
$DefaultTrackedStem = if ($TrackingProfile -eq "general_person") {
    "${Stem}_yolo26m_pretrained_person_botsort_high_recall"
} elseif ($TrackingProfile -eq "football_high_recall") {
    "${Stem}_yolo26m_football_botsort_high_recall"
} else {
    "${Stem}_yolo26m_botsort_reid"
}
$DefaultTrackedVideo = Join-Path $SourceDir "${DefaultTrackedStem}.mp4"
$DefaultTracks = Join-Path $SourceDir "${DefaultTrackedStem}.txt"

if ($SkipTracking) {
    if (-not $Tracks) {
        throw "SkipTracking=true requires -Tracks path."
    }
    if (-not (Test-Path -LiteralPath $Tracks -PathType Leaf)) {
        throw "Tracks file does not exist: $Tracks"
    }
    $TracksResolved = (Resolve-Path -LiteralPath $Tracks).Path

    if ($TrackedVideo) {
        if (-not (Test-Path -LiteralPath $TrackedVideo -PathType Leaf)) {
            throw "Tracked video does not exist: $TrackedVideo"
        }
        $TrackedVideo = (Resolve-Path -LiteralPath $TrackedVideo).Path
    } else {
        $CandidateTrackedVideo = [System.IO.Path]::ChangeExtension($TracksResolved, ".mp4")
        $TrackedVideo = if (Test-Path -LiteralPath $CandidateTrackedVideo -PathType Leaf) {
            (Resolve-Path -LiteralPath $CandidateTrackedVideo).Path
        } else { "" }
    }

    if ($TrackingMetadata) {
        if (-not (Test-Path -LiteralPath $TrackingMetadata -PathType Leaf)) {
            throw "Tracking metadata does not exist: $TrackingMetadata"
        }
        $TrackingMetadata = (Resolve-Path -LiteralPath $TrackingMetadata).Path
    } else {
        $CandidateMetadata = [System.IO.Path]::ChangeExtension($TracksResolved, ".metadata.json")
        $TrackingMetadata = if (Test-Path -LiteralPath $CandidateMetadata -PathType Leaf) {
            (Resolve-Path -LiteralPath $CandidateMetadata).Path
        } else { "" }
    }
} else {
    $TrackedVideo = if ($TrackedVideo) { $TrackedVideo } else { $DefaultTrackedVideo }
    $TracksResolved = if ($Tracks) {
        $Tracks
    } else {
        [System.IO.Path]::ChangeExtension($TrackedVideo, ".txt")
    }
    $TrackingMetadata = if ($TrackingMetadata) {
        $TrackingMetadata
    } else {
        Join-Path (Split-Path $TrackedVideo -Parent) "$([System.IO.Path]::GetFileNameWithoutExtension($TrackedVideo)).metadata.json"
    }
}

Write-Host ""
Write-Host "==> Source video : $SourcePath"
Write-Host "==> Run root     : $RunRoot"
Write-Host "==> Sequence name: $SequenceName"
Write-Host "==> Track profile: $TrackingProfile"
Write-Host "==> Track config : $TrackingConfig"
Write-Host "==> Tracks       : $TracksResolved"
if ($TrackedVideo) { Write-Host "==> Tracked video: $TrackedVideo" }
if ($TrackingMetadata) { Write-Host "==> Metadata     : $TrackingMetadata" }
if ($SkipTracking) { Write-Host "==> SkipTracking : true" }
if ($TrackCheckpoint) { Write-Host "==> Track ckpt   : $TrackCheckpoint" }
if ($TrackConf -ge 0.0) { Write-Host "==> Track conf   : $TrackConf" }
if ($TrackImgsz -gt 0) { Write-Host "==> Track imgsz  : $TrackImgsz" }
Write-Host "==> Quantization : $Quantization"
if ($AnnotationCsv) { Write-Host "==> Annotation   : $AnnotationCsv" }

# ---------------------------------------------------------------------------
# Hàm render video cho một pipeline
# ---------------------------------------------------------------------------
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
        A = "pipeline_A_qwen4b"
        B = "pipeline_B_locateanything3b"
        C = "pipeline_C_locateanything3b_qwen4b"
    }
    $TitleByPipeline = @{
        A = "Pipeline A | YOLO26m + BoT-SORT ReID + Qwen3-VL 4B"
        B = "Pipeline B | YOLO26m + BoT-SORT ReID + LocateAnything 3B"
        C = "Pipeline C | YOLO26m + BoT-SORT ReID + LocateAnything 3B + Qwen3-VL 4B"
    }

    $Predictions = Join-Path $PipelineOutputDir "render_predictions.json"
    # Output thẳng vào thư mục video gốc F:/videos/
    $OutputVideo = Join-Path $SourceDir ("{0}_{1}.mp4" -f $Stem, $SuffixByPipeline[$Pipeline])

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
    if ($CompleteRenderLabels -and (Test-Path -LiteralPath $CompletionManifest -PathType Leaf)) {
        $PredictionArgs += @("--completion-manifest", "$CompletionManifest")
    }
    if (
        $UseBenchmarkLabelsForRender -and
        $PreferredManifest -and
        (Test-Path -LiteralPath $PreferredManifest -PathType Leaf)
    ) {
        $PredictionArgs += @("--preferred-manifest", "$PreferredManifest")
    }
    if (
        $UseBenchmarkLabelsForRender -and
        $AnnotationCsv -and
        (Test-Path -LiteralPath $AnnotationCsv -PathType Leaf)
    ) {
        $PredictionArgs += @("--annotation-csv", "$AnnotationCsv", "--use-annotation-labels")
    }
    if ($LocateFinalResolution -and (Test-Path -LiteralPath $LocateFinalResolution -PathType Leaf)) {
        $PredictionArgs += @("--locate-final-resolution", "$LocateFinalResolution")
    }
    if ($QwenAnswer -and (Test-Path -LiteralPath $QwenAnswer -PathType Leaf)) {
        $PredictionArgs += @("--qwen-answer", "$QwenAnswer")
    }
    if ($Overwrite) { $PredictionArgs += "--overwrite" }
    & $Python @PredictionArgs | Out-Host
    $PredictionExitCode = $LASTEXITCODE
    if ($PredictionExitCode -ne 0) { exit $PredictionExitCode }

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
    & $Python @RenderArgs | Out-Host
    $RenderExitCode = $LASTEXITCODE
    if ($RenderExitCode -ne 0) { exit $RenderExitCode }

    return [ordered]@{
        video       = "$OutputVideo"
        predictions = "$Predictions"
    }
}

# ---------------------------------------------------------------------------
# Step 1: Tracking
# ---------------------------------------------------------------------------
if ($SkipTracking) {
    Write-Host ""
    Write-Host "==> Step 1/4: skipped tracking; using existing tracks"
} else {
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
    if ($TrackCheckpoint) { $TrackArgs += @("--checkpoint", "$TrackCheckpoint") }
    if ($TrackConf -ge 0.0) { $TrackArgs += @("--conf", "$TrackConf") }
    if ($TrackImgsz -gt 0) { $TrackArgs += @("--imgsz", "$TrackImgsz") }

    Write-Host ""
    Write-Host "==> Step 1/4: YOLO26m + BoT-SORT ReID tracking"
    & $Python @TrackArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (Test-Path -LiteralPath $TracksResolved -PathType Leaf) {
    $TracksResolved = (Resolve-Path -LiteralPath $TracksResolved).Path
}
if ($TrackingMetadata -and (Test-Path -LiteralPath $TrackingMetadata -PathType Leaf)) {
    $TrackingMetadata = (Resolve-Path -LiteralPath $TrackingMetadata).Path
}

$DiagnosticsJson = Join-Path $RunRoot "tracking_diagnostics.json"
$DiagnosticsMd = Join-Path $RunRoot "tracking_diagnostics.md"
if ($RunTrackDiagnostics -and (Test-Path -LiteralPath $TracksResolved -PathType Leaf)) {
    Write-Host ""
    Write-Host "==> Running tracking diagnostics"
    $DiagnosticsArgs = @(
        "scripts\diagnose_video_tracks.py",
        "--tracks", "$TracksResolved",
        "--source-video", "$SourcePath",
        "--output-json", "$DiagnosticsJson",
        "--output-md", "$DiagnosticsMd"
    )
    if ($TrackingMetadata -and (Test-Path -LiteralPath $TrackingMetadata -PathType Leaf)) {
        $DiagnosticsArgs += @("--metadata", "$TrackingMetadata")
    }
    if ($Overwrite) { $DiagnosticsArgs += "--overwrite" }
    & $Python @DiagnosticsArgs | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Tracking diagnostics failed (exit $LASTEXITCODE). Continuing..."
    } else {
        Write-Host "==> Tracking diagnostics: $DiagnosticsMd"
    }
}

if ($CompleteRenderLabels) {
    Write-Host ""
    Write-Host "==> Preparing render label completion manifest"
    $CompletionArgs = @(
        "scripts\build_track_label_completion.py",
        "--sequence-name", $SequenceName,
        "--source-video", "$SourcePath",
        "--tracks", "$TracksResolved",
        "--output", "$CompletionManifest",
        "--samples-per-track", "$RenderLabelSamplesPerTrack"
    )
    if ($AnnotationCsv -and (Test-Path -LiteralPath $AnnotationCsv -PathType Leaf)) {
        $CompletionArgs += @("--annotation-csv", "$AnnotationCsv")
    }
    if ($Overwrite) { $CompletionArgs += "--overwrite" }
    & $Python @CompletionArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$NormalizedPipelines = @($Pipelines | ForEach-Object { $_.ToUpperInvariant() })
$RenderedVideos = [ordered]@{}

# ---------------------------------------------------------------------------
# Step 2: Pipeline A — Qwen3-VL 4B
# ---------------------------------------------------------------------------
if ($NormalizedPipelines -contains "A") {
    $AOut = Join-Path $RunRoot "pipeline_a_yolo26m_botsort_reid_qwen4b"
    Write-Host ""
    Write-Host "==> Step 2/4 Pipeline A: YOLO26m + BoT-SORT ReID + Qwen3-VL 4B"
    .\scripts\analyze_tracking_vlm.ps1 `
        -SourceVideo "$SourcePath" `
        -TrackedVideo "$TrackedVideo" `
        -Tracks "$TracksResolved" `
        -Metadata "$TrackingMetadata" `
        -OutputDir "$AOut" `
        -ModelId $QwenModelId `
        -Device $Device `
        -TorchDtype $TorchDtype `
        -Quantization $Quantization `
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

# ---------------------------------------------------------------------------
# Step 2: Pipeline B — LocateAnything 3B
# ---------------------------------------------------------------------------
if ($NormalizedPipelines -contains "B") {
    $BOut = Join-Path $RunRoot "pipeline_b_yolo26m_botsort_reid_locateanything3b"
    Write-Host ""
    Write-Host "==> Step 2/4 Pipeline B: YOLO26m + BoT-SORT ReID + LocateAnything 3B"
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
        "--quantization", $Quantization,
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

# ---------------------------------------------------------------------------
# Step 2: Pipeline C — LocateAnything 3B + Qwen3-VL 4B
# ---------------------------------------------------------------------------
if ($NormalizedPipelines -contains "C") {
    $COut = Join-Path $RunRoot "pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b"
    $CLocateOut = Join-Path $COut "locateanything"
    $CQwenOut = Join-Path $COut "qwen4b"
    $CLocateFinalResolution = ""
    Write-Host ""
    Write-Host "==> Step 2/4 Pipeline C: YOLO26m + BoT-SORT ReID + LocateAnything 3B + Qwen3-VL 4B"

    # Tái sử dụng kết quả LocateAnything từ Pipeline B nếu có
    $BFinalResolution = Join-Path (Join-Path $RunRoot "pipeline_b_yolo26m_botsort_reid_locateanything3b") "final_resolution.json"
    if (($NormalizedPipelines -contains "B") -and (Test-Path -LiteralPath $BFinalResolution -PathType Leaf)) {
        Write-Host "==> Pipeline C: reusing LocateAnything result from Pipeline B"
        New-Item -ItemType Directory -Force -Path $CLocateOut | Out-Null
        $ReuseNote = [ordered]@{
            status                   = "reused"
            source_pipeline          = "B"
            source_final_resolution  = "$BFinalResolution"
            reason                   = "Pipeline C uses the same language grounding target before Qwen reasoning."
        }
        $ReuseNote | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $CLocateOut "reuse_from_pipeline_b.json") -Encoding UTF8
        $CLocateFinalResolution = $BFinalResolution
    } else {
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
            "--quantization", $Quantization,
            "--max-new-tokens", "$MaxNewTokens",
            "--max-frames", "$LocateMaxFrames",
            "--query-mode", "single_target"
        )
        if ($Overwrite) { $LocateArgs += "--overwrite" }
        & $Python @LocateArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $CLocateFinalResolution = Join-Path $CLocateOut "final_resolution.json"
    }

    .\scripts\analyze_tracking_vlm.ps1 `
        -SourceVideo "$SourcePath" `
        -TrackedVideo "$TrackedVideo" `
        -Tracks "$TracksResolved" `
        -Metadata "$TrackingMetadata" `
        -OutputDir "$CQwenOut" `
        -ModelId $QwenModelId `
        -Device $Device `
        -TorchDtype $TorchDtype `
        -Quantization $Quantization `
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
        -LocateFinalResolution "$CLocateFinalResolution" `
        -QwenAnswer (Join-Path $CQwenOut "vlm_answer.json")
}

# ---------------------------------------------------------------------------
# Step 3: Đánh giá định lượng (nếu có annotation CSV)
# ---------------------------------------------------------------------------
Write-Host ""
if ($AnnotationCsv -and (Test-Path -LiteralPath $AnnotationCsv -PathType Leaf)) {
    Write-Host "==> Step 3/4: Đánh giá định lượng so với ground truth"
    $EvalOutputDir = Join-Path $RunRoot "evaluation"
    New-Item -ItemType Directory -Force -Path $EvalOutputDir | Out-Null

    $EvalArgs = @(
        "scripts\evaluate_pipeline_results.py",
        "--sequence-name", $SequenceName,
        "--annotation-csv", "$AnnotationCsv",
        "--output-dir", "$EvalOutputDir"
    )

    $PredA = Join-Path (Join-Path $RunRoot "pipeline_a_yolo26m_botsort_reid_qwen4b") "render_predictions.json"
    $PredB = Join-Path (Join-Path $RunRoot "pipeline_b_yolo26m_botsort_reid_locateanything3b") "render_predictions.json"
    $PredC = Join-Path (Join-Path $RunRoot "pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b") "render_predictions.json"

    if ($NormalizedPipelines -contains "A" -and (Test-Path -LiteralPath $PredA)) {
        $EvalArgs += @("--pipeline-a", "$PredA")
    }
    if ($NormalizedPipelines -contains "B" -and (Test-Path -LiteralPath $PredB)) {
        $EvalArgs += @("--pipeline-b", "$PredB")
    }
    if ($NormalizedPipelines -contains "C" -and (Test-Path -LiteralPath $PredC)) {
        $EvalArgs += @("--pipeline-c", "$PredC")
    }
    if ($Overwrite) { $EvalArgs += "--overwrite" }

    & $Python @EvalArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Evaluation step failed (exit $LASTEXITCODE). Continuing..."
    } else {
        Write-Host "==> Evaluation report: $EvalOutputDir\evaluation_report.md"
    }
} else {
    Write-Host "==> Step 3/4: Skipped evaluation (no annotation CSV found)"
    if (-not $AnnotationCsv) {
        Write-Host "    Tip: Cung cấp -AnnotationCsv để bật đánh giá định lượng."
    }
}

# ---------------------------------------------------------------------------
# Step 4: Tổng kết
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "==> Step 4/4: Lưu run summary"

$Summary = [ordered]@{
    source_video      = "$SourcePath"
    sequence_name     = $SequenceName
    query             = $Query
    run_root          = "$RunRoot"
    skip_tracking     = [bool]$SkipTracking
    tracking_profile  = $TrackingProfile
    tracking_config   = $TrackingConfig
    tracking_overrides = [ordered]@{
        checkpoint = if ($TrackCheckpoint) { $TrackCheckpoint } else { $null }
        conf       = if ($TrackConf -ge 0.0) { $TrackConf } else { $null }
        imgsz      = if ($TrackImgsz -gt 0) { $TrackImgsz } else { $null }
    }
    tracks            = "$TracksResolved"
    tracking_diagnostics = if ($RunTrackDiagnostics) { "$DiagnosticsJson" } else { $null }
    pipelines         = $NormalizedPipelines
    qwen_model_loaded = [bool]$RunQwenModel
    locate_backend    = $LocateBackend
    quantization      = $Quantization
    torch_dtype       = $TorchDtype
    annotation_csv    = if ($AnnotationCsv) { "$AnnotationCsv" } else { $null }
    render_pipeline_videos = [bool]$RenderPipelineVideos
    use_benchmark_labels_for_render = [bool]$UseBenchmarkLabelsForRender
    complete_render_labels = [bool]$CompleteRenderLabels
    render_label_samples_per_track = $RenderLabelSamplesPerTrack
    completion_manifest = if ($CompleteRenderLabels) { "$CompletionManifest" } else { $null }
    pipeline_videos   = $RenderedVideos
}
$SummaryPath = Join-Path $RunRoot "run_summary.json"
$Summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8

Write-Host ""
Write-Host "==> Done!"
Write-Host "==> Run summary : $SummaryPath"
Write-Host "==> Output dir  : $RunRoot"
Write-Host ""
Write-Host "==> Rendered videos:"
foreach ($key in $RenderedVideos.Keys) {
    if ($RenderedVideos[$key]) {
        Write-Host "    Pipeline $key : $($RenderedVideos[$key].video)"
    }
}
