param(
    [string]$Checkpoint,
    [string]$Config = "configs/yolov8m_train.yaml",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot

if (-not $Checkpoint) {
    Write-Error "Checkpoint path is required."
}

$argsList = @(
    "-m", "football_tracking.cli", "resume-detector",
    "--config", $Config,
    "--checkpoint", $Checkpoint
)
if ($Device -ne "auto") {
    $argsList += @("--device", $Device)
}

& $Python @argsList
exit $LASTEXITCODE
