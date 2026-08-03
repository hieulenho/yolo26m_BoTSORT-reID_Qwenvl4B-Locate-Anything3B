<# Compatibility alias for the unified run_task_webcam.ps1 camera launcher. #>

param(
    [ValidateRange(0, 100)]
    [int]$CameraIndex = 0,
    [ValidateRange(0, 20)]
    [int]$MaxLocalCameraIndex = 5,
    [string[]]$ScanNetwork = @(),
    [ValidateRange(1, 65535)]
    [int]$RtspPort = 554,
    [switch]$SkipIpDiscovery,
    [switch]$ListOnly,
    [string]$Username = "admin",
    [Security.SecureString]$Password,
    [string]$CredentialFile = ".camera_credentials\yoosee_rtsp.json",
    [ValidateSet("main", "sub")]
    [string]$Stream = "sub",
    [string]$TaskConfig = "configs\tasks\generic_coco_realtime.yaml",
    [string]$RunName = "camera_session",
    [ValidateSet("live", "disabled")]
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
$Runner = @{
    TaskConfig = $TaskConfig
    CameraIndex = $CameraIndex
    MaxCameraIndex = $MaxLocalCameraIndex
    ScanNetwork = $ScanNetwork
    RtspPort = $RtspPort
    CredentialFile = $CredentialFile
    Stream = $Stream
    RunName = $RunName
    SemanticWorkerMode = $SemanticWorkerMode
    LiveSemanticMaxPendingEvents = $LiveSemanticMaxPendingEvents
    Device = $Device
}
if ($PSBoundParameters.ContainsKey("Username")) { $Runner.Username = $Username }
if ($SkipIpDiscovery) { $Runner.SkipIpDiscovery = $true }
if ($ListOnly) { $Runner.ListOnly = $true }
if ($null -ne $Password) { $Runner.Password = $Password }
if ($MaxFrames -gt 0) { $Runner.MaxFrames = $MaxFrames }
if ($NoWindow) { $Runner.NoWindow = $true }
if ($Overwrite) { $Runner.Overwrite = $true }

& (Join-Path $ProjectRoot "scripts\run_task_webcam.ps1") @Runner
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
