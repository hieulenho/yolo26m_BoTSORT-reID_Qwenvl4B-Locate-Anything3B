<# Run a reproducible webcam/RTSP realtime matrix and build the hardware-linked report. #>

param(
    [string]$Source = "0",
    [string]$ProtocolName = "camera_protocol",
    [string]$OutputRoot = "outputs\benchmarks\realtime\physical",
    [ValidateRange(150, 1000000)]
    [int]$MaxFrames = 900,
    [ValidateRange(1, 10)]
    [int]$Repeats = 3,
    [ValidateRange(2.0, 30.0)]
    [double]$CalibrationSeconds = 8.0,
    [ValidateSet("none", "8bit", "4bit")]
    [string]$QwenQuantization = "4bit",
    [string]$Device = "cuda",
    [string]$ReuseGeneratedConfig = "",
    [switch]$IncludeLiveSemantic,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ProtocolRoot = Join-Path $OutputRoot $ProtocolName
$SharedRunName = Join-Path $ProtocolName "_shared_setup"
$SharedConfig = Join-Path $ProtocolRoot "_shared_setup\plan\tracking.generated.yaml"

function Invoke-Profile {
    param(
        [string]$Name,
        [string]$SemanticMode,
        [bool]$DisableDropping,
        [int]$Repeat
    )
    $RepeatedName = "${Name}_r${Repeat}"
    Write-Host "==> Realtime profile: $RepeatedName"
    $Arguments = @{
        Source = $Source
        RunName = (Join-Path $ProtocolName $RepeatedName)
        OutputRoot = $OutputRoot
        MaxFrames = $MaxFrames
        Device = $Device
        SemanticWorkerMode = $SemanticMode
        ReuseGeneratedConfig = $SharedConfig
        SkipDeferredSemanticDrain = $true
        NoWindow = $true
    }
    if ($DisableDropping) { $Arguments["DisableFrameDropping"] = $true }
    if ($Overwrite) { $Arguments["Overwrite"] = $true }
    & (Join-Path $ProjectRoot "scripts\run_realtime_adaptive.ps1") @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($ReuseGeneratedConfig) {
    if (-not (Test-Path -LiteralPath $ReuseGeneratedConfig -PathType Leaf)) {
        throw "Reusable generated config does not exist: $ReuseGeneratedConfig"
    }
    $SharedConfig = (Resolve-Path -LiteralPath $ReuseGeneratedConfig).Path
    Write-Host "==> Reusing existing calibrated plan: $SharedConfig"
}
else {
    Write-Host "==> Calibrate and discover dynamic vocabulary once for all realtime runs"
    $SetupArguments = @{
        Source = $Source
        RunName = $SharedRunName
        OutputRoot = $OutputRoot
        CalibrationSeconds = $CalibrationSeconds
        QwenQuantization = $QwenQuantization
        Device = $Device
        SemanticWorkerMode = "disabled"
        SetupOnly = $true
        NoWindow = $true
    }
    if ($Overwrite) { $SetupArguments["Overwrite"] = $true }
    & (Join-Path $ProjectRoot "scripts\run_realtime_adaptive.ps1") @SetupArguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

for ($Repeat = 1; $Repeat -le $Repeats; $Repeat++) {
    Invoke-Profile -Name "bounded_tracking_only" -SemanticMode "disabled" -DisableDropping $false -Repeat $Repeat
    Invoke-Profile -Name "bounded_semantic_deferred" -SemanticMode "deferred" -DisableDropping $false -Repeat $Repeat
    Invoke-Profile -Name "no_drop_semantic_deferred" -SemanticMode "deferred" -DisableDropping $true -Repeat $Repeat
    if ($IncludeLiveSemantic) {
        Invoke-Profile -Name "bounded_semantic_live" -SemanticMode "live" -DisableDropping $false -Repeat $Repeat
    }
}

$ReportArgs = @(
    "scripts\benchmarks\build_realtime_benchmark.py"
)
for ($Repeat = 1; $Repeat -le $Repeats; $Repeat++) {
    foreach ($Name in @(
        "bounded_tracking_only",
        "bounded_semantic_deferred",
        "no_drop_semantic_deferred"
    )) {
        $RepeatedName = "${Name}_r${Repeat}"
        $ReportArgs += @(
            "--run",
            "$RepeatedName=$ProtocolRoot\$RepeatedName\realtime_metrics.json"
        )
    }
    if ($IncludeLiveSemantic) {
        $RepeatedName = "bounded_semantic_live_r${Repeat}"
        $ReportArgs += @(
            "--run",
            "$RepeatedName=$ProtocolRoot\$RepeatedName\realtime_metrics.json"
        )
    }
}
$ReportArgs += @("--output-dir", (Join-Path $ProtocolRoot "comparison"))
if ($Overwrite) { $ReportArgs += "--overwrite" }
& $Python @ReportArgs
exit $LASTEXITCODE
