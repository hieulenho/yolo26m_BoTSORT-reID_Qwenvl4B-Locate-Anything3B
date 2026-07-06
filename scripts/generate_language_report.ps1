param(
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
$overwriteFlag = @()
if ($Overwrite) {
    $overwriteFlag = @("--overwrite")
}

& $python -m football_tracking.locate_tracking.cli generate-language-report `
    --evaluation outputs\locate_tracking\benchmark\smoke\a5_full_system `
    --ablation outputs\locate_tracking\benchmark\ablation\ablation_results.json `
    --failures outputs\locate_tracking\benchmark\smoke\failures\failure_cases.json `
    --output outputs\locate_tracking\reports\language_tracking_report.md `
    @overwriteFlag

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
