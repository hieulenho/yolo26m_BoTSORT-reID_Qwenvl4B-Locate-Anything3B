<# Main traffic entry point: camera index, video path, or stream URL. #>
param(
    [string]$Source = '0',
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'auto',
    [ValidateSet('auto', 'live', 'deferred', 'disabled')]
    [string]$SemanticWorkerMode = 'auto',
    [string]$TaskConfig = 'configs\tasks\traffic_realtime.yaml',
    [string]$RunName = ('traffic_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')),
    [string]$OutputRoot = 'outputs\traffic',
    [ValidateRange(1, 256)]
    [int]$LiveSemanticMaxPendingEvents = 4,
    [int]$MaxFrames = 0,
    [switch]$NoWindow,
    [switch]$Overwrite
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Run .\scripts\setup_webcam.ps1 first.'
}
# Resolve caller-relative video paths before the lower-level runner changes cwd.
if (Test-Path -LiteralPath $Source -PathType Leaf) { $Source = (Resolve-Path -LiteralPath $Source).Path }
if (Test-Path -LiteralPath $TaskConfig -PathType Leaf) { $TaskConfig = (Resolve-Path -LiteralPath $TaskConfig).Path }
if ($Device -eq 'auto') {
    $Device = (& $Python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot import PyTorch. Rerun setup.' }
}
if ($SemanticWorkerMode -eq 'auto') {
    $SemanticWorkerMode = if ($Device -eq 'cuda') { 'live' } else { 'disabled' }
}
if ($Device -eq 'cpu' -and $SemanticWorkerMode -ne 'disabled') {
    throw 'Qwen 8-bit live/deferred profiles require NVIDIA CUDA. Use -SemanticWorkerMode disabled on CPU.'
}
if ($SemanticWorkerMode -ne 'disabled') {
    & $Python -c 'import transformers,accelerate,bitsandbytes,qwen_vl_utils'
    if ($LASTEXITCODE -ne 0) { throw 'Qwen dependencies missing. Run setup_webcam.ps1 -TorchBackend cuda.' }
}
$Runner = @{
    Source = $Source
    TaskConfig = $TaskConfig
    Device = $Device
    RunName = $RunName
    OutputRoot = $OutputRoot
    SemanticWorkerMode = $SemanticWorkerMode
    LiveSemanticMaxPendingEvents = $LiveSemanticMaxPendingEvents
    MaxFrames = $MaxFrames
    NoWindow = [bool]$NoWindow
    Overwrite = [bool]$Overwrite
}
& (Join-Path $ProjectRoot 'scripts\run_task_realtime.ps1') @Runner
exit $LASTEXITCODE
