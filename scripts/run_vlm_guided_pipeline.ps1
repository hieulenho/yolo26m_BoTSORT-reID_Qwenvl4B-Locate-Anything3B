<#
.SYNOPSIS
    Pipeline D: VLM-Guided Open-Vocabulary Tracking
    ------------------------------------------------
    Bước 1 (Scene Discovery): Qwen3-VL-4B phân tích video, tự sinh COCO class list
    Bước 2 (YOLO Detect + Track): YOLO gốc (yolov8m.pt) detect theo class VLM sinh ra, BoT-SORT track
    Bước 3 (Render Video): Vẽ bounding box + label class COCO lên video, xuất ra F:\videos\

.DESCRIPTION
    Đây là pipeline mới theo yêu cầu của Giảng viên:
      "Tự phát hiện bối cảnh trước (VLM), rồi detect + track bằng YOLO gốc (không bị khóa domain)"
    Không giới hạn cho bóng đá — hoạt động với giao thông, phim ảnh, thể thao bất kỳ.

.PARAMETER SourceVideo
    Đường dẫn file video gốc (bắt buộc). Ví dụ: F:\videos\traffic.mp4

.PARAMETER OutputRoot
    Thư mục gốc lưu kết quả. Mặc định: outputs\vlm_guided_experiments

.PARAMETER ModelId
    Hugging Face model ID của Qwen. Mặc định: Qwen/Qwen3-VL-4B-Instruct

.PARAMETER NFrames
    Số frames trích xuất để VLM phân tích. Mặc định: 3

.PARAMETER SkipDiscovery
    Bỏ qua Bước 1 (dùng scene_discovery.json có sẵn từ lần chạy trước)

.PARAMETER SkipTracking
    Bỏ qua Bước 2 (dùng file tracks .txt có sẵn)

.EXAMPLE
    # Chạy đầy đủ từ video gốc:
    .\scripts\run_vlm_guided_pipeline.ps1 -SourceVideo "F:\videos\traffic.mp4" -Overwrite

.EXAMPLE
    # Bỏ qua discovery, dùng kết quả cũ:
    .\scripts\run_vlm_guided_pipeline.ps1 `
        -SourceVideo "F:\videos\traffic.mp4" `
        -SkipDiscovery `
        -Overwrite
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SourceVideo,

    [string]$OutputRoot = "",

    # --- Qwen Model ---
    [string]$ModelId = "Qwen/Qwen3-VL-4B-Instruct",

    [ValidateSet("none", "8bit", "4bit")]
    [string]$Quantization = "8bit",

    [string]$Device = "auto",

    [string]$TorchDtype = "auto",

    [int]$NFrames = 3,

    # --- Tracking ---
    [string]$TrackingConfig = "configs/track_video_vlm_guided.yaml",

    [int]$MaxFrames = 0,

    # --- Skip flags ---
    [switch]$SkipDiscovery,

    [switch]$SkipTracking,

    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

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
    $OutputRoot = Join-Path $ProjectRoot "outputs\vlm_guided_experiments"
}

$RunRoot = Join-Path $OutputRoot $Stem
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

# Đường dẫn các file trung gian + output
$DiscoveryPath  = Join-Path $RunRoot "scene_discovery.json"
$TracksPath     = Join-Path $SourceDir "${Stem}_vlm_guided_botsort.txt"
$TrackedVideo   = Join-Path $SourceDir "${Stem}_vlm_guided_botsort.mp4"
$MetadataPath   = Join-Path $SourceDir "${Stem}_vlm_guided_botsort.metadata.json"
$OutputVideo    = Join-Path $SourceDir "${Stem}_pipeline_D_vlm_guided.mp4"   # xuất thẳng vào F:\videos\

Write-Host ""
Write-Host "===============================================" 
Write-Host "  Pipeline D: VLM-Guided Open-Vocabulary Tracking"
Write-Host "==============================================="
Write-Host "  Source video   : $SourcePath"
Write-Host "  Sequence name  : $SequenceName"
Write-Host "  Run root       : $RunRoot"
Write-Host "  Qwen model     : $ModelId ($Quantization)"
Write-Host "  Output video   : $OutputVideo"
Write-Host ""

# ---------------------------------------------------------------------------
# Bước 1: Scene Discovery
# ---------------------------------------------------------------------------
Write-Host ">>> Bước 1/3: Scene & Class Discovery (Qwen3-VL-4B)"
if ($SkipDiscovery) {
    if (-not (Test-Path -LiteralPath $DiscoveryPath -PathType Leaf)) {
        throw "SkipDiscovery=true nhưng không tìm thấy file: $DiscoveryPath"
    }
    Write-Host "    (Skipped — dùng scene_discovery.json có sẵn)"
} else {
    $DiscoveryArgs = @(
        "scripts\run_scene_discovery.py",
        "--video", "$SourcePath",
        "--output", "$DiscoveryPath",
        "--model-id", $ModelId,
        "--device", $Device,
        "--torch-dtype", $TorchDtype,
        "--quantization", $Quantization,
        "--n-frames", "$NFrames"
    )
    if ($Overwrite) { $DiscoveryArgs += "--overwrite" }
    & $Python @DiscoveryArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# Đọc class IDs từ discovery result
$DiscoveryData = Get-Content -LiteralPath $DiscoveryPath -Raw | ConvertFrom-Json
$CocoClassIds  = $DiscoveryData.coco_class_ids -join ","
$CocoNames     = ($DiscoveryData.coco_class_names.PSObject.Properties | ForEach-Object { $_.Value }) -join ", "

Write-Host ""
Write-Host "    Context : $($DiscoveryData.context)"
Write-Host "    Classes : [$CocoClassIds] -> $CocoNames"
Write-Host ""

# ---------------------------------------------------------------------------
# Bước 2: Tracking với YOLO gốc (yolov8m.pt)
# ---------------------------------------------------------------------------
Write-Host ">>> Bước 2/3: YOLO gốc (yolov8m.pt) + BoT-SORT Tracking"
if ($SkipTracking) {
    if (-not (Test-Path -LiteralPath $TracksPath -PathType Leaf)) {
        throw "SkipTracking=true nhưng không tìm thấy file: $TracksPath"
    }
    Write-Host "    (Skipped — dùng file tracks có sẵn: $TracksPath)"
} else {
    # Truyền class_ids từ VLM discovery vào tracker
    $TrackArgs = @(
        "-m", "football_tracking.cli",
        "track-video",
        "--config", $TrackingConfig,
        "--source", "$SourcePath",
        "--output-video", "$TrackedVideo",
        "--device", $Device,
        # Inject class IDs từ scene_discovery — YOLO chỉ detect các class này
        "--class-ids", $CocoClassIds
    )
    if ($Overwrite) { $TrackArgs += "--overwrite" }
    if ($MaxFrames -gt 0) { $TrackArgs += @("--max-frames", "$MaxFrames") }

    Write-Host "    Tracking class IDs: [$CocoClassIds] ($CocoNames)"
    & $Python @TrackArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ---------------------------------------------------------------------------
# Bước 3: Render Video đầu ra
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> Bước 3/3: Render Video với label COCO class"

$RenderArgs = @(
    "scripts\render_vlm_guided_video.py",
    "--source-video", "$SourcePath",
    "--tracks", "$TracksPath",
    "--discovery", "$DiscoveryPath",
    "--output-video", "$OutputVideo",
    "--title", "Pipeline D | VLM-Guided: $($DiscoveryData.context_short) | YOLO COCO + BoT-SORT"
)
if ($Overwrite) { $RenderArgs += "--overwrite" }
& $Python @RenderArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---------------------------------------------------------------------------
# Tổng kết
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "========================================"
Write-Host "  DONE! Pipeline D hoàn thành."
Write-Host "========================================"
Write-Host "  Context phát hiện : $($DiscoveryData.context)"
Write-Host "  Classes tracked   : [$CocoClassIds] -> $CocoNames"
Write-Host "  Tracks file       : $TracksPath"
Write-Host "  Output video      : $OutputVideo"
Write-Host ""

$Summary = [ordered]@{
    pipeline          = "D"
    description       = "VLM-Guided Open-Vocabulary Tracking"
    source_video      = "$SourcePath"
    sequence_name     = $SequenceName
    run_root          = "$RunRoot"
    scene_context     = $DiscoveryData.context
    context_short     = $DiscoveryData.context_short
    coco_class_ids    = $DiscoveryData.coco_class_ids
    coco_class_names  = $DiscoveryData.coco_class_names
    vlm_confidence    = $DiscoveryData.confidence
    tracks            = "$TracksPath"
    output_video      = "$OutputVideo"
    qwen_model        = $ModelId
    quantization      = $Quantization
    skip_discovery    = [bool]$SkipDiscovery
    skip_tracking     = [bool]$SkipTracking
}
$SummaryPath = Join-Path $RunRoot "run_summary.json"
$Summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
Write-Host "  Run summary       : $SummaryPath"
