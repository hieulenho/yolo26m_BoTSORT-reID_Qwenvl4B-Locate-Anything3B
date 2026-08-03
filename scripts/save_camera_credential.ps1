<# Save one RTSP credential encrypted for the current Windows account. #>

param(
    [string]$Username = "admin",
    [Security.SecureString]$Password,
    [string]$CredentialFile = ".camera_credentials\yoosee_rtsp.json",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
if (-not $IsWindows -and $PSVersionTable.PSVersion.Major -ge 6) {
    throw "This credential helper requires Windows DPAPI."
}
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResolvedCredentialFile = if ([System.IO.Path]::IsPathRooted($CredentialFile)) {
    [System.IO.Path]::GetFullPath($CredentialFile)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $CredentialFile))
}
if ((Test-Path -LiteralPath $ResolvedCredentialFile) -and -not $Overwrite) {
    throw "Credential file already exists. Pass -Overwrite to replace it."
}
if ($null -eq $Password) {
    $Password = Read-Host "Yoosee NVR/RTSP password" -AsSecureString
}
$PlainPassword = [System.Net.NetworkCredential]::new("", $Password).Password
if ([string]::IsNullOrWhiteSpace($PlainPassword)) {
    throw "Yoosee NVR/RTSP password cannot be empty."
}
$PlainPassword = $null

$Parent = Split-Path -Parent $ResolvedCredentialFile
New-Item -ItemType Directory -Force -Path $Parent | Out-Null
$Record = [ordered]@{
    schema_version = 1
    username = $Username
    protection = "windows_dpapi_current_user"
    encrypted_password = ConvertFrom-SecureString $Password
}
$Record | ConvertTo-Json | Set-Content -LiteralPath $ResolvedCredentialFile -Encoding UTF8

Write-Host "Saved protected camera credential: $ResolvedCredentialFile"
Write-Host "Only this Windows account on this machine can decrypt it."
