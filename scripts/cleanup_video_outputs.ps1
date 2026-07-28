<# Remove known generated video artifacts while protecting manifest-listed sources. #>

param(
    [string]$Manifest = "configs\final_video_sources.json",
    [switch]$IncludeFinalDirectory,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Source manifest does not exist: $Manifest"
}

$Payload = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$SourceRoot = [System.IO.Path]::GetFullPath([string]$Payload.source_root)
$OutputRoot = [System.IO.Path]::GetFullPath([string]$Payload.output_root)
$Protected = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
foreach ($Video in @($Payload.videos)) {
    $Path = [System.IO.Path]::GetFullPath([string]$Video.path)
    if (-not $Path.StartsWith($SourceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest source is outside the protected source root: $Path"
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Manifest source does not exist: $Path"
    }
    [void]$Protected.Add($Path)
}

$GeneratedPatterns = @(
    "*_pipeline_*.mp4",
    "*_pipeline_*.metadata.json",
    "*_Tracking*.mp4",
    "*_Tracking*.txt",
    "*_Tracking*.metadata.json",
    "*_yolo*.mp4",
    "*_yolo*.txt",
    "*_yolo*.metadata.json",
    "*_adaptive_tracking.mp4",
    "*_adaptive_tracking.txt",
    "*_adaptive_tracking.metadata.json",
    "*_8bit_*_tracking.mp4",
    "*_8bit_*_tracking.txt",
    "*_8bit_*_tracking.metadata.json",
    "*_8bit_*_semantic*.mp4",
    "*_8bit_*_semantic*.metadata.json"
)
$Candidates = foreach ($Pattern in $GeneratedPatterns) {
    Get-ChildItem -LiteralPath $SourceRoot -File -Filter $Pattern -ErrorAction SilentlyContinue
}
$Candidates = @($Candidates | Sort-Object FullName -Unique)
foreach ($File in $Candidates) {
    $Resolved = [System.IO.Path]::GetFullPath($File.FullName)
    if (-not $Resolved.StartsWith($SourceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup candidate is outside source root: $Resolved"
    }
    if ($Protected.Contains($Resolved)) {
        throw "Refusing to remove protected source: $Resolved"
    }
    if ($WhatIf) {
        Write-Host "[WhatIf] Remove $Resolved"
    }
    else {
        Remove-Item -LiteralPath $Resolved -Force
        Write-Host "Removed $Resolved"
    }
}

if ($IncludeFinalDirectory -and (Test-Path -LiteralPath $OutputRoot)) {
    $ResolvedOutput = (Resolve-Path -LiteralPath $OutputRoot).Path
    $ExpectedParent = [System.IO.Path]::GetDirectoryName($OutputRoot)
    if (
        -not $ResolvedOutput.StartsWith($SourceRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        $ExpectedParent -ne $SourceRoot
    ) {
        throw "Refusing to clear unexpected final directory: $ResolvedOutput"
    }
    foreach ($Item in Get-ChildItem -LiteralPath $ResolvedOutput -Force) {
        if ($WhatIf) {
            Write-Host "[WhatIf] Remove $($Item.FullName)"
        }
        else {
            Remove-Item -LiteralPath $Item.FullName -Recurse -Force
            Write-Host "Removed $($Item.FullName)"
        }
    }
}

Write-Host "Cleanup complete. Protected source count: $($Protected.Count)"
