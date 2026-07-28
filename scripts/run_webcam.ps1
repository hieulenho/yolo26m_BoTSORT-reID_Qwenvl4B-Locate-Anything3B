<# Auto-select a local webcam and start the adaptive realtime pipeline. #>

param(
    [int]$CameraIndex = -1,
    [ValidateRange(0, 20)]
    [int]$MaxCameraIndex = 5,
    [string]$RunName = "webcam_session",
    [double]$CalibrationSeconds = 8.0,
    [ValidateSet("deferred", "live", "disabled")]
    [string]$SemanticWorkerMode = "deferred",
    [ValidateSet("none", "8bit", "4bit")]
    [string]$Quantization = "8bit",
    [string]$Device = "cuda",
    [int]$MaxFrames = 0,
    [string]$ReuseGeneratedConfig = "",
    [switch]$DetectionOnly,
    [switch]$NoWindow,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Runtime is not installed. Run .\scripts\setup_webcam.ps1 first."
}

$ProbeArgs = @("scripts\runtime\probe_camera.py", "--max-index", "$MaxCameraIndex")
if ($CameraIndex -ge 0) { $ProbeArgs += @("--index", "$CameraIndex") }
$ProbeOutput = (& $Python @ProbeArgs | Out-String)
if ($LASTEXITCODE -ne 0) { throw "Webcam probing failed." }
$Probe = $ProbeOutput | ConvertFrom-Json
$Selected = [int]$Probe.selected.index
Write-Host (
    "[webcam] Selected camera {0}: {1}x{2}, backend={3}, reported_fps={4:N1}" -f
    $Selected,
    $Probe.selected.width,
    $Probe.selected.height,
    $Probe.selected.backend,
    $Probe.selected.fps
)

if ($DetectionOnly) { $SemanticWorkerMode = "disabled" }
$RunnerArgs = @(
    "-Source", "$Selected",
    "-RunName", $RunName,
    "-CalibrationSeconds", "$CalibrationSeconds",
    "-QwenQuantization", $Quantization,
    "-LocateQuantization", $Quantization,
    "-SemanticWorkerMode", $SemanticWorkerMode,
    "-Device", $Device
)
if ($MaxFrames -gt 0) { $RunnerArgs += @("-MaxFrames", "$MaxFrames") }
if ($ReuseGeneratedConfig) { $RunnerArgs += @("-ReuseGeneratedConfig", $ReuseGeneratedConfig) }
if ($NoWindow) { $RunnerArgs += "-NoWindow" }
if ($Overwrite) { $RunnerArgs += "-Overwrite" }

Write-Host "[webcam] Press Q in the preview window to stop."
& (Join-Path $ProjectRoot "scripts\run_realtime_adaptive.ps1") @RunnerArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

