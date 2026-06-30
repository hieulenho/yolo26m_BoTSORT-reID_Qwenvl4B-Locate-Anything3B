param(
    [string]$Config = "configs/yolov8m_eval.yaml",
    [string]$Split = "",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot

$argsList = @("-m", "football_tracking.cli", "evaluate-detector", "--config", $Config)
if ($Split) {
    $argsList += @("--split", $Split)
}
if ($Device -ne "auto") {
    $argsList += @("--device", $Device)
}

& $Python @argsList
exit $LASTEXITCODE
