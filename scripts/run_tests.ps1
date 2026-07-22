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

$ExitCode = 1
try {
    & $Python -m pytest -q --basetemp="$BaseTemp" -p no:cacheprovider @args
    $ExitCode = $LASTEXITCODE
}
finally {
    foreach ($Path in @($RunTemp, $BaseTemp)) {
        if (Test-Path -LiteralPath $Path) {
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    if ((Test-Path -LiteralPath $TempRoot) -and -not (Get-ChildItem -LiteralPath $TempRoot -Force)) {
        Remove-Item -LiteralPath $TempRoot -Force -ErrorAction SilentlyContinue
    }
}
exit $ExitCode
