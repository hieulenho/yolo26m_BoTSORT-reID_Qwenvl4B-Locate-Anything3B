param(
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
$overwriteFlag = @()
if ($Overwrite) {
    $overwriteFlag = @("--overwrite")
}

& $python -m football_tracking.locate_tracking.cli validate-language-benchmark `
    --manifest data\language_tracking\benchmark_manifest.json `
    --output outputs\locate_tracking\benchmark\smoke\validation.json

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m football_tracking.locate_tracking.cli run-language-benchmark `
    --manifest data\language_tracking\benchmark_manifest.json `
    --predictions data\language_tracking\smoke\predictions_full_system.json `
    --output-dir outputs\locate_tracking\benchmark\smoke\a5_full_system `
    @overwriteFlag

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
