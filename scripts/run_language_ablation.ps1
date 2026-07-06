param(
    [switch]$Overwrite,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
$flags = @()
if ($Overwrite) {
    $flags += "--overwrite"
}
if ($DryRun) {
    $flags += "--dry-run"
}

& $python -m football_tracking.locate_tracking.cli run-language-ablation `
    --config configs\locate_tracking\experiments\ablation_manifest.yaml `
    --output-dir outputs\locate_tracking\benchmark\ablation `
    @flags

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
