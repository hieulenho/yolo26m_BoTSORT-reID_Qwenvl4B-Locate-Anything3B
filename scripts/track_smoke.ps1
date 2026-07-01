param(
    [string]$Config = "configs/track_sportsmot_smoke.yaml",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

& $Python -m football_tracking.cli doctor
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m football_tracking.cli track-sportsmot --config $Config --device $Device --dry-run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m football_tracking.cli track-sportsmot `
    --config $Config `
    --device $Device `
    --max-sequences 1 `
    --max-frames 100 `
    --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m football_tracking.cli validate-tracks --config $Config --max-sequences 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Tracks:"
Get-ChildItem outputs\tracks\deepsort -Recurse -Filter *.txt | Select-Object FullName
Write-Host "Videos:"
Get-ChildItem outputs\videos\deepsort -Recurse -Filter *.mp4 | Select-Object FullName
