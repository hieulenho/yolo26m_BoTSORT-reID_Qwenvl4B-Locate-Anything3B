param(
    [Parameter(Mandatory = $true)]
    [string]$SourceVideo,

    [Parameter(Mandatory = $true)]
    [string]$Tracks,

    [Parameter(Mandatory = $true)]
    [string]$GroundTruth,

    [Parameter(Mandatory = $true)]
    [int]$FrameCount,

    [Parameter(Mandatory = $true)]
    [string]$Query,

    [Parameter(Mandatory = $true)]
    [int]$TargetGtTrackId,

    [Parameter(Mandatory = $true)]
    [int]$RawTrackId,

    [int]$EvaluationStartFrame = 1,
    [int]$EvaluationEndFrame = 0,
    [double]$Fps = 25.0,
    [string]$SequenceName = "video_1",
    [string]$QueryId = "q_target_001",
    [string]$OutputDir = "data\language_tracking\subset\video_1",
    [string]$SemanticTargetId = "",
    [int]$LossFrame = 0,
    [int]$ReacquisitionStartFrame = 0,
    [int]$ReacquisitionEndFrame = 0,
    [int]$GtReappearanceFrame = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"

if ($EvaluationEndFrame -le 0) {
    $EvaluationEndFrame = $FrameCount
}

$argsList = @(
    "-m", "football_tracking.locate_tracking.cli", "create-language-benchmark-template",
    "--output-dir", $OutputDir,
    "--sequence-name", $SequenceName,
    "--source-video", $SourceVideo,
    "--tracks", $Tracks,
    "--ground-truth", $GroundTruth,
    "--frame-count", "$FrameCount",
    "--fps", "$Fps",
    "--query-id", $QueryId,
    "--query", $Query,
    "--target-gt-track-id", "$TargetGtTrackId",
    "--evaluation-start-frame", "$EvaluationStartFrame",
    "--evaluation-end-frame", "$EvaluationEndFrame",
    "--raw-track-id", "$RawTrackId"
)

if ($SemanticTargetId) {
    $argsList += @("--semantic-target-id", $SemanticTargetId)
}

if ($LossFrame -gt 0 -or $ReacquisitionStartFrame -gt 0 -or $ReacquisitionEndFrame -gt 0 -or $GtReappearanceFrame -gt 0) {
    $argsList += @(
        "--loss-frame", "$LossFrame",
        "--reacquisition-start-frame", "$ReacquisitionStartFrame",
        "--reacquisition-end-frame", "$ReacquisitionEndFrame",
        "--gt-reappearance-frame", "$GtReappearanceFrame"
    )
}

if ($Overwrite) {
    $argsList += "--overwrite"
}

& $python @argsList

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
