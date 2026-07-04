param(
    [string]$Config = "configs\yolo26m_sportsmot_football_train.yaml",
    [string]$EvalConfig = "configs\yolo26m_sportsmot_football_eval.yaml",
    [string]$Device = "",
    [int]$Epochs = 0,
    [double]$Batch = 0,
    [int]$ImgSz = 0,
    [switch]$Overwrite,
    [switch]$SkipEval,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtualenv was not found: $python"
}

function Add-OptionalArgs {
    param(
        [string[]]$BaseArgs
    )
    $commandArgs = @($BaseArgs)
    if ($Device) {
        $commandArgs += @("--device", $Device)
    }
    if ($Epochs -gt 0) {
        $commandArgs += @("--epochs", "$Epochs")
    }
    if ($Batch -gt 0) {
        $commandArgs += @("--batch", "$Batch")
    }
    if ($ImgSz -gt 0) {
        $commandArgs += @("--imgsz", "$ImgSz")
    }
    if ($Overwrite) {
        $commandArgs += "--overwrite"
    }
    return $commandArgs
}

Write-Host "==> Preflight detector training"
$preflightArgs = Add-OptionalArgs -BaseArgs @(
    "-m", "football_tracking.cli",
    "preflight-training",
    "--config", $Config
)
& $python @preflightArgs

Write-Host "==> Training detector"
$trainArgs = Add-OptionalArgs -BaseArgs @(
    "-m", "football_tracking.cli",
    "train-detector",
    "--config", $Config
)
if ($DryRun) {
    $trainArgs += "--dry-run"
}
& $python @trainArgs

if (-not $SkipEval) {
    Write-Host "==> Evaluating detector"
    $evalArgs = @(
        "-m", "football_tracking.cli",
        "evaluate-detector",
        "--config", $EvalConfig,
        "--overwrite"
    )
    if ($Device) {
        $evalArgs += @("--device", $Device)
    }
    if ($Batch -gt 0) {
        $evalArgs += @("--batch", "$Batch")
    }
    if ($ImgSz -gt 0) {
        $evalArgs += @("--imgsz", "$ImgSz")
    }
    if ($DryRun) {
        $evalArgs += "--dry-run"
    }
    & $python @evalArgs
}
