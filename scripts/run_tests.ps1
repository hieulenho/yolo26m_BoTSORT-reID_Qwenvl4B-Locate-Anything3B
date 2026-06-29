$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TempRoot = Join-Path $ProjectRoot "outputs\pytest_tmp"
$RunId = "{0}-{1}" -f $PID, ([DateTimeOffset]::Now.ToUnixTimeMilliseconds())
$RunTemp = Join-Path $TempRoot ("env-" + $RunId)
$BaseTemp = Join-Path $TempRoot ("base-" + $RunId)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $RunTemp | Out-Null
New-Item -ItemType Directory -Force -Path $BaseTemp | Out-Null

$env:TEMP = $RunTemp
$env:TMP = $RunTemp
$env:TMPDIR = $RunTemp

Write-Host "[tests] TEMP=$env:TEMP"
Write-Host "[tests] BaseTemp=$BaseTemp"

& $Python -m pytest -q --basetemp="$BaseTemp" -p no:cacheprovider @args
exit $LASTEXITCODE
