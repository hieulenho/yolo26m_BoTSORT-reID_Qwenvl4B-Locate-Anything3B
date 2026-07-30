<# Run one explicit task profile on every sequence in a normalized MOT manifest. #>

param(
    [Parameter(Mandatory = $true)]
    [string]$TaskConfig,
    [Parameter(Mandatory = $true)]
    [string]$GtManifest,
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string]$OutputRoot = "outputs\benchmarks\task_gt",
    [string]$SourceVideo = "",
    [string[]]$SequenceIds = @(),
    [string]$Device = "cuda",
    [string]$TrackerConfig = "",
    [ValidateSet("task_default", "none", "auto_low_light", "clahe")]
    [string]$PreprocessingMode = "task_default",
    [ValidateRange(0.000001, 1000.0)]
    [double]$GtScaleX = 1.0,
    [ValidateRange(0.000001, 1000.0)]
    [double]$GtScaleY = 1.0,
    [switch]$Resume,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
foreach ($RequiredFile in @($Python, $TaskConfig, $GtManifest)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required file does not exist: $RequiredFile"
    }
}
if ($TrackerConfig -and -not (Test-Path -LiteralPath $TrackerConfig -PathType Leaf)) {
    throw "Tracker override config does not exist: $TrackerConfig"
}

$Manifest = Get-Content -LiteralPath $GtManifest -Raw | ConvertFrom-Json
$Sequences = @($Manifest.sequences)
if ($SequenceIds.Count -gt 0) {
    $Sequences = @($Sequences | Where-Object {
        $Candidate = [string]$_.normalized_sequence
        if (-not $Candidate) { $Candidate = [string]$_.sequence }
        $SequenceIds -contains $Candidate
    })
}
if ($Sequences.Count -eq 0) { throw "GT manifest has no sequences: $GtManifest" }
if ($SourceVideo -and $Sequences.Count -ne 1) {
    throw "-SourceVideo can only override a manifest containing one sequence."
}
if ($SourceVideo -and -not (Test-Path -LiteralPath $SourceVideo -PathType Leaf)) {
    throw "Source video override does not exist: $SourceVideo"
}
$BenchmarkRoot = Join-Path $ProjectRoot (Join-Path $OutputRoot $RunName)
$Predictions = @()

for ($Index = 0; $Index -lt $Sequences.Count; $Index++) {
    $Sequence = $Sequences[$Index]
    $SequenceName = [string]$Sequence.normalized_sequence
    if (-not $SequenceName) { $SequenceName = [string]$Sequence.sequence }
    $Source = if ($SourceVideo) { $SourceVideo } else { [string]$Sequence.media_path }
    $FrameCount = [int]$Sequence.frame_count
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Sequence media is not a video file for ${SequenceName}: $Source"
    }
    $Metrics = Join-Path $BenchmarkRoot (Join-Path "runs" (Join-Path $SequenceName "realtime_metrics.json"))
    $Prediction = Join-Path $BenchmarkRoot (Join-Path "runs" (Join-Path $SequenceName "tracked_semantic.txt"))
    $Percent = [int](100 * $Index / $Sequences.Count)
    Write-Progress -Activity "Task GT benchmark: $RunName" `
        -Status "[$($Index + 1)/$($Sequences.Count)] $SequenceName" `
        -PercentComplete $Percent
    if ($Resume -and (Test-Path -LiteralPath $Metrics -PathType Leaf) -and (Test-Path -LiteralPath $Prediction -PathType Leaf)) {
        Write-Host "==> reuse $SequenceName"
    }
    else {
        Write-Host "==> run $SequenceName ($FrameCount frames)"
        $Arguments = @{
            TaskConfig = $TaskConfig
            Source = $Source
            RunName = $SequenceName
            OutputRoot = (Join-Path $OutputRoot (Join-Path $RunName "runs"))
            SemanticWorkerMode = "disabled"
            Device = $Device
            PreprocessingMode = $PreprocessingMode
            MaxFrames = $FrameCount
            NoWindow = $true
            Quiet = $true
        }
        if ($Overwrite) { $Arguments.Overwrite = $true }
        if ($TrackerConfig) { $Arguments.TrackerConfig = $TrackerConfig }
        & (Join-Path $ProjectRoot "scripts\run_task_realtime.ps1") @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Task runtime failed for $SequenceName with exit code $LASTEXITCODE."
        }
    }
    if (-not (Test-Path -LiteralPath $Prediction -PathType Leaf)) {
        throw "Prediction was not produced for ${SequenceName}: $Prediction"
    }
    $Predictions += [pscustomobject]@{
        sequence = $SequenceName
        path = $Prediction
    }
}
Write-Progress -Activity "Task GT benchmark: $RunName" -Completed

$EvaluationRoot = Join-Path $BenchmarkRoot "evaluation"
$EvaluationArgs = @(
    "scripts\benchmarks\evaluate_normalized_mot.py",
    "--gt-root", (Split-Path -Parent (Resolve-Path -LiteralPath $GtManifest).Path),
    "--tracker-name", $RunName,
    "--output-dir", $EvaluationRoot,
    "--gt-scale-x", "$GtScaleX",
    "--gt-scale-y", "$GtScaleY"
)
foreach ($Prediction in $Predictions) {
    $EvaluationArgs += @(
        "--sequence", [string]$Prediction.sequence,
        "--prediction", [string]$Prediction.path
    )
}
if ($Overwrite) { $EvaluationArgs += "--overwrite" }
& $Python @EvaluationArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "TrackEval result: $(Join-Path $EvaluationRoot 'trackeval_summary.json')"
