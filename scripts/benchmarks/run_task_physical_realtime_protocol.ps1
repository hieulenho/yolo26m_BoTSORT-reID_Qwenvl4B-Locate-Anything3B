<# Measure the current TaskConfig foreground path on a physical webcam or RTSP source. #>

param(
    [Parameter(Mandatory = $true)]
    [string]$TaskConfig,
    [string]$Source = "0",
    [string]$ProtocolName = "task_webcam_900_frames",
    [string]$OutputRoot = "outputs\benchmarks\realtime\task_physical",
    [ValidateRange(150, 1000000)]
    [int]$MaxFrames = 900,
    [ValidateRange(1, 10)]
    [int]$Repeats = 3,
    [string]$Device = "cuda",
    [ValidateSet("task_default", "none", "auto_low_light", "clahe")]
    [string]$PreprocessingMode = "task_default",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment Python does not exist: $Python"
}
if (-not (Test-Path -LiteralPath $TaskConfig -PathType Leaf)) {
    throw "Task config does not exist: $TaskConfig"
}

$ProtocolRoot = Join-Path $OutputRoot $ProtocolName
$ReportRuns = @()
$Profiles = @(
    @{
        Name = "bounded_tracking_only"
        SemanticMode = "disabled"
        DisableDropping = $false
    },
    @{
        Name = "bounded_semantic_queue"
        SemanticMode = "deferred"
        DisableDropping = $false
    },
    @{
        Name = "no_drop_semantic_queue"
        SemanticMode = "deferred"
        DisableDropping = $true
    }
)

$TotalRuns = $Repeats * $Profiles.Count
$CompletedRuns = 0
for ($Repeat = 1; $Repeat -le $Repeats; $Repeat++) {
    foreach ($Profile in $Profiles) {
        $CompletedRuns += 1
        $RepeatedName = "{0}_r{1:d2}" -f $Profile.Name, $Repeat
        Write-Progress -Activity "Physical TaskConfig realtime benchmark" `
            -Status "[$CompletedRuns/$TotalRuns] $RepeatedName" `
            -PercentComplete ([int](100 * ($CompletedRuns - 1) / $TotalRuns))
        Write-Host "==> [$CompletedRuns/$TotalRuns] $RepeatedName"
        $Arguments = @{
            TaskConfig = $TaskConfig
            Source = $Source
            RunName = (Join-Path $ProtocolName $RepeatedName)
            OutputRoot = $OutputRoot
            SemanticWorkerMode = [string]$Profile.SemanticMode
            Device = $Device
            PreprocessingMode = $PreprocessingMode
            MaxFrames = $MaxFrames
            SkipDeferredSemanticDrain = $true
            NoWindow = $true
            Quiet = $true
        }
        if ([bool]$Profile.DisableDropping) {
            $Arguments.DisableFrameDropping = $true
        }
        if ($Overwrite) { $Arguments.Overwrite = $true }
        & (Join-Path $ProjectRoot "scripts\run_task_realtime.ps1") @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Physical realtime run failed: $RepeatedName"
        }
        $Metrics = Join-Path $ProtocolRoot "$RepeatedName\realtime_metrics.json"
        if (-not (Test-Path -LiteralPath $Metrics -PathType Leaf)) {
            throw "Physical realtime metrics were not created: $Metrics"
        }
        $ReportRuns += "${RepeatedName}=${Metrics}"
    }
}
Write-Progress -Activity "Physical TaskConfig realtime benchmark" -Completed

$ReportDir = Join-Path $ProtocolRoot "comparison"
$ReportArgs = @("scripts\benchmarks\build_realtime_benchmark.py")
foreach ($RunValue in $ReportRuns) {
    $ReportArgs += @("--run", $RunValue)
}
$ReportArgs += @("--output-dir", $ReportDir)
if ($Overwrite) { $ReportArgs += "--overwrite" }
& $Python @ReportArgs
exit $LASTEXITCODE
