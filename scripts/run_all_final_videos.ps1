<# Run LocateAnything-first + Qwen semantic tracking for every protected source. #>

param(
    [string]$Manifest = "configs\final_video_sources.json",
    [ValidateSet("realtime", "realtime_stable", "balanced", "accuracy")]
    [string]$Profile = "realtime_stable",
    [string]$RunRoot = "outputs\final_video_runs",
    [ValidateRange(1, 16)]
    [int]$MaxKeyframes = 4,
    [ValidateRange(1, 64)]
    [int]$MaxClasses = 24,
    [ValidateRange(128, 1024)]
    [int]$DiscoveryMaxNewTokens = 512,
    [ValidateRange(65536, 1048576)]
    [int]$QwenImageMaxPixels = 262144,
    [ValidateRange(65536, 1048576)]
    [int]$SemanticImageMaxPixels = 196608,
    [ValidateRange(128, 1024)]
    [int]$SemanticMaxNewTokens = 192,
    [ValidateRange(65536, 1048576)]
    [int]$LocateImageMaxPixels = 196608,
    [switch]$Resume,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Source manifest does not exist: $Manifest"
}

$Payload = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$OutputRoot = [System.IO.Path]::GetFullPath([string]$Payload.output_root)
$RunRootPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $RunRoot))
New-Item -ItemType Directory -Force -Path $OutputRoot, $RunRootPath | Out-Null
$Videos = @($Payload.videos)
if ($Videos.Count -eq 0) {
    throw "Source manifest contains no videos."
}

$Results = [System.Collections.Generic.List[object]]::new()
for ($Index = 0; $Index -lt $Videos.Count; $Index++) {
    $Video = $Videos[$Index]
    $Source = [System.IO.Path]::GetFullPath([string]$Video.path)
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Source video does not exist: $Source"
    }
    $Stem = [System.IO.Path]::GetFileNameWithoutExtension($Source)
    $TrackingVideo = Join-Path $RunRootPath ($Stem + "_tracking.mp4")
    $FinalVideo = Join-Path $OutputRoot ($Stem + "_final.mp4")
    $VideoRunRoot = Join-Path $RunRootPath $Stem
    $RunReport = Join-Path $VideoRunRoot "adaptive_run_report.json"
    $Percent = [int](100 * $Index / $Videos.Count)
    Write-Progress -Activity "Final multi-domain rendering" `
        -Status "[$($Index + 1)/$($Videos.Count)] $Source" `
        -PercentComplete $Percent
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[$($Index + 1)/$($Videos.Count)] $Source"
    Write-Host "Final output: $FinalVideo"

    if (
        $Resume -and
        (Test-Path -LiteralPath $FinalVideo -PathType Leaf) -and
        (Test-Path -LiteralPath $RunReport -PathType Leaf)
    ) {
        Write-Host "Status: already completed; reusing final output."
        $Results.Add([PSCustomObject]@{
            id = [string]$Video.id
            source = $Source
            output = $FinalVideo
            elapsed_seconds = 0.0
            size_bytes = (Get-Item -LiteralPath $FinalVideo).Length
            status = "reused"
        })
        continue
    }

    $Arguments = @{
        SourceVideo = $Source
        OutputVideo = $TrackingVideo
        SemanticOutputVideo = $FinalVideo
        OutputRoot = $RunRoot
        Profile = $Profile
        QwenQuantization = "8bit"
        Device = "cuda"
        MaxKeyframes = $MaxKeyframes
        MaxClasses = $MaxClasses
        DiscoveryMaxNewTokens = $DiscoveryMaxNewTokens
        QwenImageMaxPixels = $QwenImageMaxPixels
        SemanticMaxTracks = 0
        SemanticMaxImages = 2
        SemanticMaxTracksPerBatch = 1
        SemanticImageMaxPixels = $SemanticImageMaxPixels
        SemanticMaxNewTokens = $SemanticMaxNewTokens
        LocateMaxTracks = 0
        LocateImageMaxPixels = $LocateImageMaxPixels
        RunLocateVerification = $true
        RunTrackSemantics = $true
        LocateFirstAllTracks = $true
    }
    if ($Resume) {
        $Discovery = Join-Path $VideoRunRoot "discovery\scene_discovery.json"
        $TrackingMot = [System.IO.Path]::ChangeExtension($TrackingVideo, ".txt")
        $TrackingMetadata = [System.IO.Path]::Combine(
            (Split-Path $TrackingVideo),
            ([System.IO.Path]::GetFileNameWithoutExtension($TrackingVideo) + ".metadata.json")
        )
        $LocateContext = Join-Path $VideoRunRoot "locate_context\vlm_context.json"
        $LocateResult = Join-Path $VideoRunRoot "locate_verification\grounding_verification.json"
        if (Test-Path -LiteralPath $Discovery -PathType Leaf) {
            $Arguments.SkipDiscovery = $true
        }
        if (
            (Test-Path -LiteralPath $TrackingVideo -PathType Leaf) -and
            (Test-Path -LiteralPath $TrackingMot -PathType Leaf) -and
            (Test-Path -LiteralPath $TrackingMetadata -PathType Leaf)
        ) {
            $Arguments.SkipTracking = $true
        }
        if (
            (Test-Path -LiteralPath $LocateContext -PathType Leaf) -and
            (Test-Path -LiteralPath $LocateResult -PathType Leaf)
        ) {
            $Arguments.ReuseLocateVerification = $true
        }
    }
    if ($Overwrite) { $Arguments.Overwrite = $true }

    $Started = Get-Date
    & (Join-Path $PSScriptRoot "run_adaptive_tracking.ps1") @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Final adaptive pipeline failed for $Source with exit code $LASTEXITCODE."
    }
    if (-not (Test-Path -LiteralPath $FinalVideo -PathType Leaf)) {
        throw "Final video was not created: $FinalVideo"
    }
    $Results.Add([PSCustomObject]@{
        id = [string]$Video.id
        source = $Source
        output = $FinalVideo
        elapsed_seconds = [math]::Round(((Get-Date) - $Started).TotalSeconds, 3)
        size_bytes = (Get-Item -LiteralPath $FinalVideo).Length
        status = "completed"
    })
}
Write-Progress -Activity "Final multi-domain rendering" -Completed

$Summary = [ordered]@{
    status = "completed"
    completed_at = (Get-Date).ToString("o")
    profile = $Profile
    qwen_quantization = "8bit"
    locate_quantization = "8bit"
    locate_first = $true
    all_tracks = $true
    video_count = $Results.Count
    videos = $Results
}
$SummaryPath = Join-Path $OutputRoot "final_render_summary.json"
$Summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryPath -Encoding utf8
Write-Host ""
Write-Host "All final videos completed: $OutputRoot"
Write-Host "Summary: $SummaryPath"
