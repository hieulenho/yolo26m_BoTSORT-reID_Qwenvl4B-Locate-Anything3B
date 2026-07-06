param(
    [string]$Manifest = "data\team_benchmark\smoke\benchmark_manifest.json",
    [string]$PipelineA = "data\team_benchmark\smoke\predictions_pipeline_a_qwen.json",
    [string]$PipelineB = "data\team_benchmark\smoke\predictions_pipeline_b_locate_qwen.json",
    [string]$OutputDir = "outputs\team_benchmark\smoke",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
$overwriteArgs = @()
if ($Overwrite) {
    $overwriteArgs += "--overwrite"
}

& $python -m football_tracking.locate_tracking.cli validate-team-benchmark `
    --manifest $Manifest `
    --output (Join-Path $OutputDir "validation.json")

& $python -m football_tracking.locate_tracking.cli run-team-benchmark `
    --manifest $Manifest `
    --predictions $PipelineA `
    --output-dir (Join-Path $OutputDir "pipeline_a_qwen") `
    @overwriteArgs

& $python -m football_tracking.locate_tracking.cli run-team-benchmark `
    --manifest $Manifest `
    --predictions $PipelineB `
    --output-dir (Join-Path $OutputDir "pipeline_b_locate_qwen") `
    @overwriteArgs

& $python -m football_tracking.locate_tracking.cli compare-team-benchmarks `
    --evaluation (Join-Path $OutputDir "pipeline_a_qwen") `
    --evaluation (Join-Path $OutputDir "pipeline_b_locate_qwen") `
    --output-dir (Join-Path $OutputDir "comparison") `
    @overwriteArgs
