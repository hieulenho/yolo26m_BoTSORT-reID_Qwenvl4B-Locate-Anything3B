<# Probe a Yoosee-compatible RTSP camera and run one realtime task safely. #>

[CmdletBinding(DefaultParameterSetName = "Camera")]
param(
    [Parameter(Mandatory = $true, ParameterSetName = "Camera")]
    [string]$CameraIp,
    [Parameter(Mandatory = $true, ParameterSetName = "Url")]
    [string]$RtspUrl,
    [Parameter(ParameterSetName = "Camera")]
    [string]$Username = "admin",
    [Parameter(ParameterSetName = "Camera")]
    [ValidateRange(1, 65535)]
    [int]$Port = 554,
    [Parameter(ParameterSetName = "Camera")]
    [Security.SecureString]$Password,
    [Parameter(ParameterSetName = "Camera")]
    [string]$CredentialFile = ".camera_credentials\yoosee_rtsp.json",
    [Parameter(ParameterSetName = "Camera")]
    [ValidateSet("main", "sub")]
    [string]$Stream = "sub",
    [string]$TaskConfig = "configs\tasks\generic_coco_realtime.yaml",
    [string]$RunName = "yoosee_ip_camera",
    [string]$OutputRoot = "outputs\task_realtime",
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
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing. Run .\scripts\setup_webcam.ps1 first."
}
if (-not (Test-Path -LiteralPath $TaskConfig -PathType Leaf)) {
    throw "Task config does not exist: $TaskConfig"
}

$SourceVariable = "FOOTBALL_TRACKING_RTSP_URL"
$PreviousSource = [System.Environment]::GetEnvironmentVariable($SourceVariable, "Process")
$PreviousCaptureOptions = $env:OPENCV_FFMPEG_CAPTURE_OPTIONS
$PlainPassword = $null
try {
    if ($PSCmdlet.ParameterSetName -eq "Camera") {
        if ($CameraIp -notmatch '^[A-Za-z0-9.-]+$') {
            throw "CameraIp must be an IPv4 address or a local hostname."
        }
        $ResolvedCredentialFile = if ([System.IO.Path]::IsPathRooted($CredentialFile)) {
            [System.IO.Path]::GetFullPath($CredentialFile)
        }
        else {
            [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $CredentialFile))
        }
        if ($null -eq $Password -and (Test-Path -LiteralPath $ResolvedCredentialFile -PathType Leaf)) {
            try {
                $Credential = Get-Content -LiteralPath $ResolvedCredentialFile -Raw | ConvertFrom-Json
                if ([int]$Credential.schema_version -ne 1) {
                    throw "Unsupported credential schema."
                }
                if ($Credential.protection -ne "windows_dpapi_current_user") {
                    throw "Unsupported credential protection."
                }
                $Password = ConvertTo-SecureString ([string]$Credential.encrypted_password)
                if (-not $PSBoundParameters.ContainsKey("Username") -and $Credential.username) {
                    $Username = [string]$Credential.username
                }
                Write-Host "[ip-camera] Loaded protected Windows credential"
            }
            catch {
                throw (
                    "Camera credential could not be decrypted. Recreate it with " +
                    ".\scripts\save_camera_credential.ps1 -Overwrite. Root error: $($_.Exception.Message)"
                )
            }
        }
        if ($null -eq $Password) {
            $Password = Read-Host "Yoosee NVR/RTSP password" -AsSecureString
        }
        $PlainPassword = [System.Net.NetworkCredential]::new("", $Password).Password
        if ([string]::IsNullOrWhiteSpace($PlainPassword)) {
            throw "Yoosee NVR/RTSP password cannot be empty."
        }
        $Path = if ($Stream -eq "main") { "onvif1" } else { "onvif2" }
        $RtspUrl = "rtsp://{0}:{1}@{2}:{3}/{4}" -f @(
            [System.Uri]::EscapeDataString($Username),
            [System.Uri]::EscapeDataString($PlainPassword),
            $CameraIp,
            $Port,
            $Path
        )
    }
    elseif (-not $RtspUrl.StartsWith("rtsp://") -and -not $RtspUrl.StartsWith("rtsps://")) {
        throw "RtspUrl must start with rtsp:// or rtsps://."
    }

    [System.Environment]::SetEnvironmentVariable($SourceVariable, $RtspUrl, "Process")
    $env:OPENCV_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp"

    Write-Host "[ip-camera] Probing RTSP stream without printing credentials"
    $ProbeOutput = (
        & $Python scripts\runtime\probe_stream.py `
            --source-env $SourceVariable `
            --frames 3 |
            Out-String
    )
    if ($LASTEXITCODE -ne 0) {
        throw "RTSP probe failed. Check camera IP, NVR/RTSP password, and stream setting."
    }
    $Probe = $ProbeOutput | ConvertFrom-Json
    Write-Host (
        "[ip-camera] Ready: {0}x{1}, backend={2}, reported_fps={3:N1}" -f
        $Probe.width,
        $Probe.height,
        $Probe.backend,
        $Probe.reported_fps
    )

    $Runner = @{
        TaskConfig = $TaskConfig
        SourceEnvironmentVariable = $SourceVariable
        RunName = $RunName
        OutputRoot = $OutputRoot
        SemanticWorkerMode = $SemanticWorkerMode
        LiveSemanticMaxPendingEvents = $LiveSemanticMaxPendingEvents
        Device = $Device
    }
    if ($MaxFrames -gt 0) { $Runner.MaxFrames = $MaxFrames }
    if ($NoWindow) { $Runner.NoWindow = $true }
    if ($Overwrite) { $Runner.Overwrite = $true }

    Write-Host "[ip-camera] Starting realtime pipeline; press Q or Esc to stop"
    & (Join-Path $ProjectRoot "scripts\run_task_realtime.ps1") @Runner
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    [System.Environment]::SetEnvironmentVariable(
        $SourceVariable,
        $PreviousSource,
        "Process"
    )
    $env:OPENCV_FFMPEG_CAPTURE_OPTIONS = $PreviousCaptureOptions
    $PlainPassword = $null
    $RtspUrl = $null
}
