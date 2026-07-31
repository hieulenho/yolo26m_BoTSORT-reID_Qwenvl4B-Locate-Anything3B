<# Run one explicit detector/tracker/semantic task on a camera, stream, or video. #>

param(
    [Parameter(Mandatory = $true)]
    [string]$TaskConfig,
    [string]$Source = "0",
    [string]$RunName = "task_realtime",
    [string]$OutputRoot = "outputs\task_realtime",
    [string]$OutputVideo = "",
    [ValidateSet("live", "deferred", "disabled")]
    [string]$SemanticWorkerMode = "live",
    [string]$Device = "cuda",
    [string]$TrackerConfig = "",
    [ValidateSet("task_default", "none", "auto_low_light", "clahe")]
    [string]$PreprocessingMode = "task_default",
    [int]$MaxFrames = 0,
    [ValidateRange(1, 8)]
    [int]$SemanticWorkerBatchSize = 2,
    [ValidateRange(1, 2)]
    [int]$SemanticWorkerMaxGroupImages = 2,
    [ValidateRange(0, 1000000)]
    [int]$SemanticWorkerMaxEvents = 0,
    [ValidateRange(0, 1000000)]
    [int]$SemanticMaxPendingEvents = 0,
    [ValidateRange(1, 256)]
    [int]$LiveSemanticMaxPendingEvents = 4,
    [ValidateRange(1, 3600)]
    [int]$SemanticWorkerShutdownTimeoutSeconds = 600,
    [int]$SemanticCacheReloadFrames = 5,
    [switch]$DisableFrameDropping,
    [switch]$DisableSceneCutReset,
    [switch]$SynchronousVideoWrite,
    [switch]$SkipDeferredSemanticDrain,
    [switch]$NoWindow,
    [switch]$Quiet,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing. Run .\scripts\setup_webcam.ps1 first."
}
if (-not (Test-Path -LiteralPath $TaskConfig -PathType Leaf)) {
    throw "Task config does not exist: $TaskConfig"
}

$ResolvedOutputRoot = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    [System.IO.Path]::GetFullPath($OutputRoot)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputRoot))
}
$RunRoot = Join-Path $ResolvedOutputRoot $RunName
$RuntimeRoot = Join-Path $RunRoot "runtime"
if (-not $OutputVideo) {
    $OutputVideo = Join-Path $RunRoot "tracked_semantic.mp4"
}
$OutputMot = [System.IO.Path]::ChangeExtension($OutputVideo, ".txt")
$Metadata = Join-Path $RunRoot "realtime_metrics.json"
$GeneratedConfig = Join-Path $RuntimeRoot "tracking.generated.yaml"
$ResolvedTask = Join-Path $RuntimeRoot "task.resolved.json"
$SemanticQueue = Join-Path $RunRoot "semantic_queue"
$SemanticCache = Join-Path $RunRoot "semantic_cache.json"
$SemanticMemory = Join-Path $RunRoot "semantic_memory.json"
$WorkerStop = Join-Path $RunRoot "semantic_worker.stop"
$WorkerReport = Join-Path $RunRoot "semantic_worker_report.json"
$WorkerStdout = Join-Path $RunRoot "semantic_worker.stdout.log"
$WorkerStderr = Join-Path $RunRoot "semantic_worker.stderr.log"
$WorkerPidFile = Join-Path $RunRoot "semantic_worker.pid"
$WorkerStatus = Join-Path $RunRoot "semantic_worker.status.json"
$SemanticRenderMetadata = Join-Path $RunRoot "semantic_render.metadata.json"

function Remove-GeneratedRunArtifact {
    param([Parameter(Mandatory = $true)][string]$Path)

    $RunPrefix = [System.IO.Path]::GetFullPath($RunRoot).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $ResolvedArtifact = [System.IO.Path]::GetFullPath($Path)
    if (-not $ResolvedArtifact.StartsWith($RunPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove an artifact outside the run directory: $ResolvedArtifact"
    }
    if (Test-Path -LiteralPath $ResolvedArtifact) {
        Remove-Item -LiteralPath $ResolvedArtifact -Recurse -Force
    }
}

function Normalize-ProcessPathEnvironment {
    # Some IDE terminals inject both Path and PATH into the Windows environment
    # block. Start-Process rejects that duplicate before it can launch a worker.
    $PathValue = $env:Path
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        $PathValue = $env:PATH
    }
    [System.Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [System.Environment]::SetEnvironmentVariable("Path", $PathValue, "Process")
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
if ($Overwrite) {
    if (Test-Path -LiteralPath $WorkerPidFile -PathType Leaf) {
        $ExistingPidText = Get-Content -LiteralPath $WorkerPidFile -Raw
        $ExistingPid = 0
        if ([int]::TryParse($ExistingPidText.Trim(), [ref]$ExistingPid)) {
            $ExistingWorker = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue
            if ($null -ne $ExistingWorker -and $ExistingWorker.ProcessName -eq "python") {
                New-Item -ItemType File -Force -Path $WorkerStop | Out-Null
                try {
                    Wait-Process -Id $ExistingPid -Timeout 10 -ErrorAction Stop
                }
                catch {
                    Stop-Process -Id $ExistingPid -Force -ErrorAction SilentlyContinue
                }
            }
        }
    }
    foreach ($Path in @(
        $SemanticQueue,
        $SemanticCache,
        $SemanticMemory,
        $SemanticRenderMetadata,
        $WorkerStop,
        $WorkerReport,
        $WorkerStdout,
        $WorkerStderr,
        $WorkerPidFile,
        $WorkerStatus
    )) {
        Remove-GeneratedRunArtifact -Path $Path
    }
}

Write-Host "[1/3] Validate task and build detector/tracker runtime"
$BuildArgs = @(
    "-m", "football_tracking.task_pipeline.cli", "build",
    "--task", $TaskConfig,
    "--output-dir", $RuntimeRoot,
    "--output-video", $OutputVideo,
    "--device", $Device
)
if ($Overwrite) { $BuildArgs += "--overwrite" }
if ($PreprocessingMode -ne "task_default") {
    $BuildArgs += @("--preprocessing-mode", $PreprocessingMode)
}
if ($TrackerConfig) {
    if (-not (Test-Path -LiteralPath $TrackerConfig -PathType Leaf)) {
        throw "Tracker override config does not exist: $TrackerConfig"
    }
    $BuildArgs += @("--tracker-config", $TrackerConfig)
}
& $Python @BuildArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Task = Get-Content -LiteralPath $ResolvedTask -Raw | ConvertFrom-Json
if (-not [bool]$Task.semantic.enabled) { $SemanticWorkerMode = "disabled" }
$EventInterval = [int]$Task.semantic.event_interval_frames
$EventsPerFrame = [int]$Task.semantic.events_per_frame
$TaskMaxPending = [int]$Task.semantic.max_pending_events
$IsFileSource = Test-Path -LiteralPath $Source -PathType Leaf
if ($SemanticMaxPendingEvents -gt 0) {
    $MaxPending = $SemanticMaxPendingEvents
}
elseif ($SemanticWorkerMode -eq "deferred" -and $IsFileSource) {
    # Deferred file runs do not drain the queue until tracking ends. Keep one
    # bounded event per unique track without silently truncating ordinary videos.
    $MaxPending = [Math]::Max($TaskMaxPending, 4096)
}
elseif ($SemanticWorkerMode -eq "live") {
    $MaxPending = [Math]::Min($TaskMaxPending, $LiveSemanticMaxPendingEvents)
}
else {
    $MaxPending = $TaskMaxPending
}

$WorkerArgs = @(
    "scripts\runtime\run_realtime_semantic_worker.py",
    "--queue-dir", $SemanticQueue,
    "--vlm-config", "configs\semantics\dynamic_track.yaml",
    "--semantic-output", $SemanticCache,
    "--memory", $SemanticMemory,
    "--max-events", "$SemanticWorkerBatchSize",
    "--group-events",
    "--max-group-images", "$SemanticWorkerMaxGroupImages",
    "--report", $WorkerReport,
    "--pid-file", $WorkerPidFile,
    "--status-file", $WorkerStatus,
    "--parent-pid", "$PID"
)
$SemanticWorker = $null
if ($SemanticWorkerMode -eq "live") {
    Write-Host (
        "[2/3] Start persistent Qwen3-VL-4B 8-bit worker " +
        "(live queue capacity: $MaxPending)"
    )
    $LiveArgs = @($WorkerArgs) + @("--watch", "--stop-file", $WorkerStop)
    Normalize-ProcessPathEnvironment
    $SemanticWorker = Start-Process `
        -FilePath $Python `
        -ArgumentList $LiveArgs `
        -RedirectStandardOutput $WorkerStdout `
        -RedirectStandardError $WorkerStderr `
        -WindowStyle Hidden `
        -PassThru
}
else {
    Write-Host "[2/3] Semantic worker mode: $SemanticWorkerMode"
}

Write-Host "[3/3] Run realtime CV loop; press Q or Esc to stop"
$RealtimeArgs = @(
    "scripts\runtime\run_realtime_adaptive.py",
    "--config", $GeneratedConfig,
    "--task-config", $TaskConfig,
    "--source", $Source,
    "--output-video", $OutputVideo,
    "--output-mot", $OutputMot,
    "--metadata", $Metadata,
    "--semantic-cache-reload-frames", "$SemanticCacheReloadFrames"
)
if ($SemanticWorkerMode -ne "disabled") {
    $RealtimeArgs += @(
        "--semantic-queue-dir", $SemanticQueue,
        "--semantic-cache", $SemanticCache,
        "--semantic-event-interval-frames", "$EventInterval",
        "--semantic-events-per-frame", "$EventsPerFrame",
        "--semantic-max-pending-events", "$MaxPending"
    )
}
if ($MaxFrames -gt 0) { $RealtimeArgs += @("--max-frames", "$MaxFrames") }
if ($DisableFrameDropping) { $RealtimeArgs += "--disable-frame-dropping" }
if ($DisableSceneCutReset) { $RealtimeArgs += "--disable-scene-cut-reset" }
if ($SynchronousVideoWrite) { $RealtimeArgs += "--synchronous-video-write" }
if ($NoWindow) { $RealtimeArgs += "--no-window" }
if ($Quiet) { $RealtimeArgs += "--quiet" }
if ($Overwrite) { $RealtimeArgs += "--overwrite" }
$RealtimeExitCode = 0
try {
    & $Python @RealtimeArgs
    $RealtimeExitCode = if ($null -eq $LASTEXITCODE) { 1 } else { $LASTEXITCODE }
}
finally {
    if ($SemanticWorkerMode -eq "live" -and $null -ne $SemanticWorker) {
        New-Item -ItemType File -Force -Path $WorkerStop | Out-Null
        $WorkerTimedOut = $false
        try {
            Wait-Process `
                -Id $SemanticWorker.Id `
                -Timeout $SemanticWorkerShutdownTimeoutSeconds `
                -ErrorAction Stop
        }
        catch {
            Write-Warning "Qwen worker exceeded shutdown timeout; pending jobs remain on disk."
            Stop-Process -Id $SemanticWorker.Id -Force -ErrorAction SilentlyContinue
            $WorkerTimedOut = $true
            if ($RealtimeExitCode -eq 0) { $RealtimeExitCode = 2 }
        }
        if (-not $WorkerTimedOut) {
            $SemanticWorker.WaitForExit()
            $SemanticWorker.Refresh()
            $WorkerProcessExitCode = $SemanticWorker.ExitCode
            $WorkerFinalStatus = $null
            if (Test-Path -LiteralPath $WorkerStatus -PathType Leaf) {
                try {
                    $WorkerFinalStatus = (
                        Get-Content -LiteralPath $WorkerStatus -Raw |
                            ConvertFrom-Json
                    )
                }
                catch {
                    Write-Warning "Qwen worker status file could not be parsed: $WorkerStatus"
                }
            }
            if ($null -ne $WorkerFinalStatus -and $WorkerFinalStatus.status -eq "failed") {
                Write-Warning "Qwen worker failed: $($WorkerFinalStatus.detail)"
                if ($RealtimeExitCode -eq 0) {
                    $RealtimeExitCode = 2
                }
            }
            elseif ($null -ne $WorkerProcessExitCode -and $WorkerProcessExitCode -ne 0) {
                Write-Warning "Qwen worker exited with code $WorkerProcessExitCode."
                if ($RealtimeExitCode -eq 0) {
                    $RealtimeExitCode = $WorkerProcessExitCode
                }
            }
            elseif (
                $null -eq $WorkerFinalStatus -or
                $WorkerFinalStatus.status -ne "stopped"
            ) {
                Write-Warning "Qwen worker exited without a clean stopped status."
                if ($RealtimeExitCode -eq 0) {
                    $RealtimeExitCode = 2
                }
            }
        }
    }
}

if (
    $RealtimeExitCode -eq 0 -and
    $SemanticWorkerMode -eq "deferred" -and
    -not $SkipDeferredSemanticDrain
) {
    Write-Host "==> Process queued tracks with one Qwen 8-bit session"
    $DrainArgs = @($WorkerArgs) + @("--drain")
    if ($SemanticWorkerMaxEvents -gt 0) {
        $DrainArgs += @("--max-total-events", "$SemanticWorkerMaxEvents")
    }
    & $Python @DrainArgs
    $WorkerExitCode = $LASTEXITCODE
    if ($WorkerExitCode -ne 0) {
        Write-Warning "Qwen worker failed; queued jobs were kept for retry."
        if ($RealtimeExitCode -eq 0) { $RealtimeExitCode = $WorkerExitCode }
    }
    elseif ((Test-Path -LiteralPath $Source -PathType Leaf) -and (Test-Path -LiteralPath $SemanticCache -PathType Leaf)) {
        Write-Host "==> Render accepted Qwen labels on the completed file"
        $RenderArgs = @(
            "-m", "football_tracking.adaptive_tracking.cli", "render-semantics",
            "--source", $Source,
            "--tracks", $OutputMot,
            "--semantics", $SemanticCache,
            "--tracking-metadata", $Metadata,
            "--output-video", $OutputVideo,
            "--overwrite"
        )
        if ($MaxFrames -gt 0) { $RenderArgs += @("--max-frames", "$MaxFrames") }
        & $Python @RenderArgs
        if ($LASTEXITCODE -ne 0) {
            $RealtimeExitCode = $LASTEXITCODE
            Write-Warning "Semantic post-render failed. Tracking artifacts were preserved."
        }
        else {
            $GeneratedSemanticMetadata = [System.IO.Path]::Combine(
                [System.IO.Path]::GetDirectoryName($OutputVideo),
                ([System.IO.Path]::GetFileNameWithoutExtension($OutputVideo) + ".semantic.metadata.json")
            )
            if (Test-Path -LiteralPath $GeneratedSemanticMetadata -PathType Leaf) {
                Copy-Item -LiteralPath $GeneratedSemanticMetadata -Destination $SemanticRenderMetadata -Force
            }
        }
    }
}
elseif ($RealtimeExitCode -eq 0 -and $SemanticWorkerMode -eq "deferred") {
    Write-Host "==> Deferred semantic drain skipped; queued evidence was preserved."
}

Write-Host "Output video: $OutputVideo"
Write-Host "Runtime metrics: $Metadata"
Write-Host "Semantic cache: $SemanticCache"
if (Test-Path -LiteralPath $SemanticRenderMetadata -PathType Leaf) {
    Write-Host "Semantic render metrics: $SemanticRenderMetadata"
}
exit $RealtimeExitCode
