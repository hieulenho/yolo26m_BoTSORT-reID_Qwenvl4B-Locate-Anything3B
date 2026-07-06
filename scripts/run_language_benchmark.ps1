param(
    [ValidateSet("smoke", "subset", "full")]
    [string]$Mode = "smoke",
    [string]$Manifest = "data\language_tracking\benchmark_manifest.json",
    [string]$Predictions = "",
    [string]$OutputDir = "",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
$overwriteFlag = @()
if ($Overwrite) {
    $overwriteFlag = @("--overwrite")
}

if (-not $Predictions) {
    if ($Mode -eq "smoke") {
        $Predictions = "data\language_tracking\smoke\predictions_full_system.json"
    } else {
        $Predictions = "data\language_tracking\predictions_$Mode.json"
    }
}

if (-not $OutputDir) {
    $OutputDir = "outputs\locate_tracking\benchmark\$Mode\a5_full_system"
}

if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Benchmark manifest does not exist: $Manifest"
}

if (-not (Test-Path -LiteralPath $Predictions)) {
    throw "Prediction manifest does not exist: $Predictions. Pass -Predictions with a saved prediction manifest for $Mode mode."
}

& $python -m football_tracking.locate_tracking.cli run-language-benchmark `
    --manifest $Manifest `
    --predictions $Predictions `
    --output-dir $OutputDir `
    @overwriteFlag

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
