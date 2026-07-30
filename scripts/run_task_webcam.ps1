<# Probe a local webcam and run one explicit task configuration. #>

param(
    [Parameter(Mandatory = $true)]
    [string]$TaskConfig,
    [int]$CameraIndex = -1,
    [ValidateRange(0, 20)]
    [int]$MaxCameraIndex = 5,
    [string]$RunName = "webcam_task",
    [ValidateSet("live", "deferred", "disabled")]
    [string]$SemanticWorkerMode = "live",
    [ValidateRange(1, 256)]
    [int]$LiveSemanticMaxPendingEvents = 4,
    [string]$Device = "cuda",
    [int]$MaxFrames = 0,
    [switch]$NoWindow,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing. Run .\scripts\setup_webcam.ps1 first."
}

$ProbeArgs = @("scripts\runtime\probe_camera.py", "--max-index", "$MaxCameraIndex")
if ($CameraIndex -ge 0) { $ProbeArgs += @("--index", "$CameraIndex") }
$Probe = (& $Python @ProbeArgs | Out-String) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Webcam probing failed." }
$Selected = [int]$Probe.selected.index
Write-Host (
    "[webcam] Camera {0}: {1}x{2}, backend={3}" -f
    $Selected, $Probe.selected.width, $Probe.selected.height, $Probe.selected.backend
)

$Runner = @{
    TaskConfig = $TaskConfig
    Source = "$Selected"
    RunName = $RunName
    SemanticWorkerMode = $SemanticWorkerMode
    LiveSemanticMaxPendingEvents = $LiveSemanticMaxPendingEvents
    Device = $Device
}
if ($MaxFrames -gt 0) { $Runner.MaxFrames = $MaxFrames }
if ($NoWindow) { $Runner.NoWindow = $true }
if ($Overwrite) { $Runner.Overwrite = $true }
& (Join-Path $ProjectRoot "scripts\run_task_realtime.ps1") @Runner
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
