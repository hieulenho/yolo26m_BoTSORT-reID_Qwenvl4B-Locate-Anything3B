param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,

    [string]$PipelineA = "",

    [string]$PipelineB = "",

    [string]$PipelineC = "",

    [string]$OutputDir = "outputs\team_benchmark\team_position",

    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
$overwriteArgs = @()
if ($Overwrite) {
    $overwriteArgs += "--overwrite"
}

$EvaluationDirs = @()

function Invoke-TeamPositionRun {
    param(
        [string]$Name,
        [string]$PredictionPath
    )
    if (-not (Test-Path -LiteralPath $PredictionPath)) {
        throw "Prediction manifest does not exist for ${Name}: $PredictionPath"
    }
    & $python -m football_tracking.locate_tracking.cli run-team-benchmark `
        --manifest $Manifest `
        --predictions $PredictionPath `
        --output-dir (Join-Path $OutputDir $Name) `
        @overwriteArgs
    $script:EvaluationDirs += (Join-Path $OutputDir $Name)
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& $python -m football_tracking.locate_tracking.cli validate-team-benchmark `
    --manifest $Manifest `
    --output (Join-Path $OutputDir "validation.json")

if ($PipelineA) {
    Invoke-TeamPositionRun `
        -Name "pipeline_a_yolo26m_botsort_reid_qwen4b" `
        -PredictionPath $PipelineA
}

if ($PipelineB) {
    Invoke-TeamPositionRun `
        -Name "pipeline_b_yolo26m_botsort_reid_locateanything3b" `
        -PredictionPath $PipelineB
}

if ($PipelineC) {
    Invoke-TeamPositionRun `
        -Name "pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b" `
        -PredictionPath $PipelineC
}

if ($EvaluationDirs.Count -lt 1) {
    throw "No prediction manifest was provided. Pass at least one of -PipelineA, -PipelineB, -PipelineC."
}

if ($EvaluationDirs.Count -ge 2) {
    $evaluationArgs = @()
    foreach ($dir in $EvaluationDirs) {
        $evaluationArgs += @("--evaluation", $dir)
    }
    & $python -m football_tracking.locate_tracking.cli compare-team-benchmarks `
        @evaluationArgs `
        --output-dir (Join-Path $OutputDir "comparison") `
        @overwriteArgs
} else {
    Write-Host "Only one pipeline evaluation was produced; comparison step skipped."
}
