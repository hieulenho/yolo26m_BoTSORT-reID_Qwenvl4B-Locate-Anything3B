param(
    [string]$Config = "configs/yolov8m_train.yaml",
    [string]$Device = "auto",
    [object]$Batch = 0,
    [int]$ImageSize = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot

$version = & $Python --version
if ($version -notmatch "Python 3\.12\.") {
    Write-Error "Python 3.12.x is required. Actual: $version"
}

& $Python -m football_tracking.cli preflight-training --config $Config
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$argsList = @("-m", "football_tracking.cli", "train-detector", "--config", $Config)
if ($Device -ne "auto") {
    $argsList += @("--device", $Device)
}
if ($Batch -ne 0) {
    $argsList += @("--batch", "$Batch")
}
if ($ImageSize -gt 0) {
    $argsList += @("--imgsz", "$ImageSize")
}

& $Python @argsList
exit $LASTEXITCODE
