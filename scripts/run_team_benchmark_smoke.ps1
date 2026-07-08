param(
    [string]$Manifest = "data\team_benchmark\smoke\benchmark_manifest.json",
    [string]$PipelineA = "data\team_benchmark\smoke\predictions_pipeline_a_yolo26m_botsort_reid_qwen4b.json",
    [string]$PipelineC = "data\team_benchmark\smoke\predictions_pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b.json",
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
    --output-dir (Join-Path $OutputDir "pipeline_a_yolo26m_botsort_reid_qwen4b") `
    @overwriteArgs

& $python -m football_tracking.locate_tracking.cli run-team-benchmark `
    --manifest $Manifest `
    --predictions $PipelineC `
    --output-dir (Join-Path $OutputDir "pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b") `
    @overwriteArgs

& $python -m football_tracking.locate_tracking.cli compare-team-benchmarks `
    --evaluation (Join-Path $OutputDir "pipeline_a_yolo26m_botsort_reid_qwen4b") `
    --evaluation (Join-Path $OutputDir "pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b") `
    --output-dir (Join-Path $OutputDir "comparison") `
    @overwriteArgs
