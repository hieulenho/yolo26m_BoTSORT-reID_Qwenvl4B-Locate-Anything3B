<# Calibrate dynamic vocabulary once, then run low-latency adaptive tracking. #>

param(
    [string]$Source = "0",
    [string]$RunName = "realtime_session",
    [string]$OutputRoot = "outputs\adaptive_realtime",
    [double]$CalibrationSeconds = 8.0,
    [int]$MaxFrames = 0,
    [ValidateRange(1, 8)]
    [int]$DiscoveryKeyframes = 2,
    [ValidateRange(128, 1024)]
    [int]$DiscoveryMaxNewTokens = 768,
    [ValidateSet("none", "8bit", "4bit")]
    [string]$QwenQuantization = "4bit",
    [string]$Device = "cuda",
    [int]$SemanticEventIntervalFrames = 90,
    [int]$SemanticEventsPerFrame = 2,
    [int]$SemanticMaxPendingEvents = 256,
    [ValidateSet("deferred", "live", "disabled")]
    [string]$SemanticWorkerMode = "deferred",
    [ValidateRange(1, 64)]
    [int]$SemanticWorkerBatchSize = 8,
    [ValidateRange(0, 1000000)]
    [int]$SemanticWorkerMaxEvents = 64,
    [ValidateRange(1, 3600)]
    [int]$SemanticWorkerShutdownTimeoutSeconds = 600,
    [double]$SceneCutThreshold = 0.65,
    [int]$SceneCutMinGapFrames = 15,
    [int]$SceneCutCheckIntervalFrames = 5,
    [switch]$DisableDetectorPrewarm,
    [switch]$DisableFrameDropping,
    [int]$MaxCatchupFrames = 5,
    [int]$VideoWriteQueueSize = 128,
    [switch]$SynchronousVideoWrite,
    [switch]$DisableSceneCutReset,
    [switch]$DisableSemanticQueue,
    [switch]$NoWindow,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RunRoot = Join-Path $ProjectRoot (Join-Path $OutputRoot $RunName)
$CalibrationVideo = Join-Path $RunRoot "calibration.mp4"
$Discovery = Join-Path $RunRoot "discovery\scene_discovery.json"
$PlanRoot = Join-Path $RunRoot "plan"
$GeneratedConfig = Join-Path $PlanRoot "tracking.generated.yaml"
$OutputVideo = Join-Path $RunRoot "realtime_tracked.mp4"
$OutputMot = Join-Path $RunRoot "realtime_tracks.txt"
$Metadata = Join-Path $RunRoot "realtime_metrics.json"
$SemanticQueue = Join-Path $RunRoot "semantic_queue"
$SemanticCache = Join-Path $RunRoot "semantic_cache.json"
$SemanticMemory = Join-Path $RunRoot "semantic_memory.json"
$SemanticWorkerStop = Join-Path $RunRoot "semantic_worker.stop"
$SemanticWorkerReport = Join-Path $RunRoot "semantic_worker_report.json"
$SemanticWorkerStdout = Join-Path $RunRoot "semantic_worker.stdout.log"
$SemanticWorkerStderr = Join-Path $RunRoot "semantic_worker.stderr.log"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
if ($DisableSemanticQueue) { $SemanticWorkerMode = "disabled" }
if ($Overwrite) {
    foreach ($Path in @($SemanticWorkerStop, $SemanticWorkerReport, $SemanticWorkerStdout, $SemanticWorkerStderr)) {
        if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Force }
    }
}

Write-Host "[1/4] Capture $CalibrationSeconds-second calibration clip"
$CaptureArgs = @(
    "scripts\runtime\capture_calibration_clip.py",
    "--source", $Source,
    "--output", $CalibrationVideo,
    "--seconds", "$CalibrationSeconds"
)
if ($Overwrite) { $CaptureArgs += "--overwrite" }
& $Python @CaptureArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/4] Discover domain and dynamic vocabulary with Qwen"
$DiscoveryArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "discover",
    "--source", $CalibrationVideo,
    "--output", $Discovery,
    "--quantization", $QwenQuantization,
    "--device", $Device,
    "--max-keyframes", "$DiscoveryKeyframes",
    "--max-new-tokens", "$DiscoveryMaxNewTokens"
)
if ($Overwrite) { $DiscoveryArgs += "--overwrite" }
& $Python @DiscoveryArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/4] Build realtime YOLO/YOLOE + OC-SORT plan"
$PlanArgs = @(
    "-m", "football_tracking.adaptive_tracking.cli", "build-plan",
    "--source", $CalibrationVideo,
    "--discovery", $Discovery,
    "--output-dir", $PlanRoot,
    "--output-video", $OutputVideo,
    "--profile", "realtime",
    "--device", $Device
)
if ($Overwrite) { $PlanArgs += "--overwrite" }
& $Python @PlanArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/4] Start realtime stream"
$SemanticWorker = $null
$WorkerArgs = @(
    "scripts\runtime\run_realtime_semantic_worker.py",
    "--queue-dir", $SemanticQueue,
    "--vlm-config", "configs\semantics\dynamic_track.yaml",
    "--semantic-output", $SemanticCache,
    "--memory", $SemanticMemory,
    "--max-events", "$SemanticWorkerBatchSize",
    "--report", $SemanticWorkerReport
)
if ($SemanticWorkerMode -eq "live") {
    Write-Host "==> Starting non-blocking semantic worker"
    $LiveWorkerArgs = @($WorkerArgs) + @(
        "--watch",
        "--stop-file", $SemanticWorkerStop
    )
    $SemanticWorker = Start-Process `
        -FilePath $Python `
        -ArgumentList $LiveWorkerArgs `
        -RedirectStandardOutput $SemanticWorkerStdout `
        -RedirectStandardError $SemanticWorkerStderr `
        -WindowStyle Hidden `
        -PassThru
}
$RealtimeArgs = @(
    "scripts\runtime\run_realtime_adaptive.py",
    "--config", $GeneratedConfig,
    "--source", $Source,
    "--output-video", $OutputVideo,
    "--output-mot", $OutputMot,
    "--metadata", $Metadata
)
if ($SemanticWorkerMode -ne "disabled") {
    $RealtimeArgs += @(
        "--semantic-queue-dir", $SemanticQueue,
        "--semantic-cache", $SemanticCache,
        "--semantic-event-interval-frames", "$SemanticEventIntervalFrames",
        "--semantic-events-per-frame", "$SemanticEventsPerFrame",
        "--semantic-max-pending-events", "$SemanticMaxPendingEvents"
    )
}
if ($DisableSceneCutReset) { $RealtimeArgs += "--disable-scene-cut-reset" }
$RealtimeArgs += @(
    "--scene-cut-threshold", "$SceneCutThreshold",
    "--scene-cut-min-gap-frames", "$SceneCutMinGapFrames",
    "--scene-cut-check-interval-frames", "$SceneCutCheckIntervalFrames",
    "--video-write-queue-size", "$VideoWriteQueueSize",
    "--max-catchup-frames", "$MaxCatchupFrames"
)
if ($SynchronousVideoWrite) { $RealtimeArgs += "--synchronous-video-write" }
if ($DisableDetectorPrewarm) { $RealtimeArgs += "--disable-detector-prewarm" }
if ($DisableFrameDropping) { $RealtimeArgs += "--disable-frame-dropping" }
if ($MaxFrames -gt 0) { $RealtimeArgs += @("--max-frames", "$MaxFrames") }
if ($NoWindow) { $RealtimeArgs += "--no-window" }
if ($Overwrite) { $RealtimeArgs += "--overwrite" }
& $Python @RealtimeArgs
$RealtimeExitCode = $LASTEXITCODE
if ($SemanticWorkerMode -eq "live" -and $null -ne $SemanticWorker) {
    New-Item -ItemType File -Force -Path $SemanticWorkerStop | Out-Null
    try {
        Wait-Process -Id $SemanticWorker.Id -Timeout $SemanticWorkerShutdownTimeoutSeconds -ErrorAction Stop
    }
    catch {
        Write-Warning "Semantic worker did not stop within the timeout; stopping child process."
        Stop-Process -Id $SemanticWorker.Id -Force -ErrorAction SilentlyContinue
    }
    $SemanticWorker.Refresh()
    if ($SemanticWorker.HasExited -and $SemanticWorker.ExitCode -ne 0) {
        Write-Warning "Semantic worker exited with code $($SemanticWorker.ExitCode). See $SemanticWorkerStderr"
    }
}
elseif ($RealtimeExitCode -eq 0 -and $SemanticWorkerMode -eq "deferred") {
    Write-Host "==> Draining the semantic queue after capture (GPU-safe default)"
    $DrainWorkerArgs = @($WorkerArgs) + @(
        "--max-total-events", "$SemanticWorkerMaxEvents",
        "--drain"
    )
    & $Python @DrainWorkerArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Deferred semantic worker failed. Queued events were kept for retry."
    }
}
exit $RealtimeExitCode
