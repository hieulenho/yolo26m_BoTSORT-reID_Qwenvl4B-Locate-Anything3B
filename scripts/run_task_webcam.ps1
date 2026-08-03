<# Select a logical local/RTSP CameraIndex and run one explicit task configuration. #>

param(
    [Parameter(Mandatory = $true)]
    [string]$TaskConfig,
    [int]$CameraIndex = -1,
    [ValidateRange(0, 20)]
    [int]$MaxCameraIndex = 5,
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
if (-not (Test-Path -LiteralPath $TaskConfig -PathType Leaf)) {
    throw "Task config does not exist: $TaskConfig"
}

$InventoryArgs = @(
    "scripts\runtime\list_camera_sources.py",
    "--max-local-index", "$MaxCameraIndex",
    "--rtsp-port", "$RtspPort"
)
if ($SkipIpDiscovery) { $InventoryArgs += "--skip-network" }
foreach ($Network in $ScanNetwork) {
    $InventoryArgs += @("--network", $Network)
}

Write-Host "[camera] Discovering local webcams and RTSP cameras"
$PreviousErrorActionPreference = $ErrorActionPreference
try {
    # OpenCV writes harmless messages for unavailable local indices to stderr.
    # Keep the inventory JSON clean and judge the native process by its exit code.
    $ErrorActionPreference = "Continue"
    $InventoryText = (& $Python @InventoryArgs 2>$null | Out-String)
    $InventoryExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}
if ($InventoryExitCode -ne 0) {
    throw "Camera discovery failed."
}

$Inventory = $InventoryText | ConvertFrom-Json
$Sources = @($Inventory.sources)
if ($Sources.Count -eq 0) {
    throw "No readable local webcam or RTSP camera was found."
}

Write-Host "Available CameraIndex values:"
foreach ($Source in $Sources) {
    if ($Source.kind -eq "local") {
        Write-Host (
            "  [{0}] local webcam (Windows index {1}, {2}x{3}, {4})" -f
            $Source.camera_index,
            $Source.device_index,
            $Source.width,
            $Source.height,
            $Source.backend
        )
    }
    else {
        Write-Host (
            "  [{0}] RTSP camera ({1}:{2})" -f
            $Source.camera_index,
            $Source.address,
            $Source.port
        )
    }
}
if ($ListOnly) { return }

$SelectedIndex = if ($CameraIndex -lt 0) { 0 } else { $CameraIndex }
if ($SelectedIndex -ge $Sources.Count) {
    throw (
        "CameraIndex $SelectedIndex does not exist. " +
        "Connect/enable the camera and run this command with -ListOnly first."
    )
}
$Selected = $Sources[$SelectedIndex]

if ($Selected.kind -eq "local") {
    Write-Host (
        "[camera] Opening CameraIndex {0} as local Windows camera {1}" -f
        $SelectedIndex,
        $Selected.device_index
    )
    $Runner = @{
        TaskConfig = $TaskConfig
        Source = "$($Selected.device_index)"
        RunName = $RunName
        SemanticWorkerMode = $SemanticWorkerMode
        LiveSemanticMaxPendingEvents = $LiveSemanticMaxPendingEvents
        Device = $Device
    }
    if ($MaxFrames -gt 0) { $Runner.MaxFrames = $MaxFrames }
    if ($NoWindow) { $Runner.NoWindow = $true }
    if ($Overwrite) { $Runner.Overwrite = $true }
    & (Join-Path $ProjectRoot "scripts\run_task_realtime.ps1") @Runner
}
else {
    if ($SemanticWorkerMode -eq "deferred") {
        throw "RTSP cameras support live or disabled semantic mode, not deferred."
    }
    Write-Host (
        "[camera] Opening CameraIndex {0} as RTSP camera {1}:{2}" -f
        $SelectedIndex,
        $Selected.address,
        $Selected.port
    )
    $Runner = @{
        CameraIp = [string]$Selected.address
        Port = [int]$Selected.port
        Stream = $Stream
        CredentialFile = $CredentialFile
        TaskConfig = $TaskConfig
        RunName = $RunName
        SemanticWorkerMode = $SemanticWorkerMode
        LiveSemanticMaxPendingEvents = $LiveSemanticMaxPendingEvents
        Device = $Device
    }
    if ($PSBoundParameters.ContainsKey("Username")) { $Runner.Username = $Username }
    if ($null -ne $Password) { $Runner.Password = $Password }
    if ($MaxFrames -gt 0) { $Runner.MaxFrames = $MaxFrames }
    if ($NoWindow) { $Runner.NoWindow = $true }
    if ($Overwrite) { $Runner.Overwrite = $true }
    & (Join-Path $ProjectRoot "scripts\run_ip_camera.ps1") @Runner
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
