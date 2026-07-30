<# Run the measured foreground-path matrix for the explicit task pipeline. #>

param(
    [string]$Manifest = "configs\benchmarks\task_runtime_suite.json",
    [string]$OutputRoot = "outputs\benchmarks\task_runtime",
    [string[]]$RunIds = @(),
    [ValidateRange(0, 1000000)]
    [int]$MaxFrames = 0,
    [ValidateRange(1, 20)]
    [int]$Repeats = 1,
    [string]$Device = "cuda",
    [switch]$Resume,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment Python does not exist: $Python"
}
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Task runtime manifest does not exist: $Manifest"
}

$Payload = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$Runs = @($Payload.runs)
if ($RunIds.Count -gt 0) {
    $Runs = @($Runs | Where-Object { $RunIds -contains ([string]$_.id) })
}
if ($Runs.Count -eq 0) { throw "No task runtime runs were selected." }

$ReportRuns = @()
$TotalRuns = $Runs.Count * $Repeats
$CompletedRuns = 0
for ($Index = 0; $Index -lt $Runs.Count; $Index++) {
    $Run = $Runs[$Index]
    $RunId = [string]$Run.id
    $Source = [string]$Run.source
    $TaskConfig = [string]$Run.task_config
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Source video does not exist for ${RunId}: $Source"
    }
    if (-not (Test-Path -LiteralPath $TaskConfig -PathType Leaf)) {
        throw "Task config does not exist for ${RunId}: $TaskConfig"
    }
    $FrameLimit = if ($MaxFrames -gt 0) {
        [Math]::Min($MaxFrames, [int]$Run.max_frames)
    } else {
        [int]$Run.max_frames
    }
    for ($Repeat = 1; $Repeat -le $Repeats; $Repeat++) {
        $CompletedRuns += 1
        $MeasuredRunId = if ($Repeats -gt 1) {
            "{0}_r{1:d2}" -f $RunId, $Repeat
        } else {
            $RunId
        }
        $Percent = [int](100 * ($CompletedRuns - 1) / $TotalRuns)
        Write-Progress -Activity "Task runtime benchmark" `
            -Status "[$CompletedRuns/$TotalRuns] $MeasuredRunId" `
            -PercentComplete $Percent
        Write-Host "==> [$CompletedRuns/$TotalRuns] $MeasuredRunId"
        $Metrics = Join-Path $ProjectRoot (
            Join-Path $OutputRoot (Join-Path $MeasuredRunId "realtime_metrics.json")
        )
        if ($Resume -and (Test-Path -LiteralPath $Metrics -PathType Leaf)) {
            Write-Host "    reuse completed metrics: $Metrics"
            $ReportRuns += "${MeasuredRunId}=${Metrics}"
            continue
        }
        $Arguments = @{
            TaskConfig = $TaskConfig
            Source = $Source
            RunName = $MeasuredRunId
            OutputRoot = $OutputRoot
            SemanticWorkerMode = "disabled"
            Device = $Device
            MaxFrames = $FrameLimit
            PreprocessingMode = [string]$Run.preprocessing_mode
            NoWindow = $true
            Quiet = $true
        }
        if ($Overwrite) { $Arguments.Overwrite = $true }
        & (Join-Path $ProjectRoot "scripts\run_task_realtime.ps1") @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw (
                "Task runtime run failed for $MeasuredRunId with exit code " +
                "$LASTEXITCODE."
            )
        }
        if (-not (Test-Path -LiteralPath $Metrics -PathType Leaf)) {
            throw "Task runtime run did not produce metrics for ${MeasuredRunId}: $Metrics"
        }
        $ReportRuns += "${MeasuredRunId}=${Metrics}"
    }
}
Write-Progress -Activity "Task runtime benchmark" -Completed

$ReportDir = Join-Path $ProjectRoot (Join-Path $OutputRoot "comparison")
$ReportArgs = @("scripts\benchmarks\build_realtime_benchmark.py")
foreach ($RunValue in $ReportRuns) {
    $ReportArgs += @("--run", $RunValue)
}
$ReportArgs += @("--output-dir", $ReportDir)
if ($Overwrite) { $ReportArgs += "--overwrite" }
& $Python @ReportArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Task runtime report: $(Join-Path $ReportDir 'realtime_benchmark.md')"
