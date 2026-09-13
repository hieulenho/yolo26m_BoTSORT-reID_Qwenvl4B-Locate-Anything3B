<# Bootstrap a fresh Windows clone. CPU installs do not load Qwen or require CUDA. #>
param(
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$TorchBackend = 'auto',
    [string]$PythonExecutable = '',
    [string]$CudaWheelIndex = 'https://download.pytorch.org/whl/cu128',
    [switch]$SkipTorchInstall,
    [switch]$SkipSemantics,
    [switch]$DownloadModels,
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $ProjectRoot

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments, [string]$Failure)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw $Failure }
}

function Test-Python312 {
    param([string]$Executable, [string[]]$Prefix = @())
    $Previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Prefix -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>$null
        return $LASTEXITCODE -eq 0
    }
    finally { $ErrorActionPreference = $Previous }
}

$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    if ($VerifyOnly) { throw 'Missing .venv. Run setup without -VerifyOnly first.' }
    $Bootstrap = $null
    $Prefix = @()
    if ($PythonExecutable) {
        $Bootstrap = $PythonExecutable
    }
    else {
        # A launcher can exist without a registered Python 3.12 installation.
        if (Get-Command py -ErrorAction SilentlyContinue) {
            if (Test-Python312 'py' @('-3.12')) { $Bootstrap = 'py'; $Prefix = @('-3.12') }
        }
        if (-not $Bootstrap -and (Get-Command python -ErrorAction SilentlyContinue)) {
            if (Test-Python312 'python') { $Bootstrap = 'python' }
        }
    }
    if (-not $Bootstrap) {
        throw 'Install Python 3.12 (64-bit): winget install -e --id Python.Python.3.12 ; then reopen PowerShell. Or supply -PythonExecutable with the full python.exe path.'
    }
    Invoke-Checked $Bootstrap ($Prefix + @('-c', "import sys,struct; sys.exit(sys.version_info[:2] != (3,12) or struct.calcsize('P') != 8)")) 'Python must be 3.12, 64-bit.'
    Invoke-Checked $Bootstrap ($Prefix + @('-m', 'venv', '.venv')) 'Could not create .venv.'
}
Invoke-Checked $Python @('-c', "import sys,struct; sys.exit(sys.version_info[:2] != (3,12) or struct.calcsize('P') != 8)") 'Use a Python 3.12, 64-bit .venv.'

if ($TorchBackend -eq 'auto') {
    $TorchBackend = 'cpu'
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $TorchBackend = 'cuda' }
    }
}
$WithSemantics = $TorchBackend -eq 'cuda' -and -not $SkipSemantics
if (-not $VerifyOnly) {
    Invoke-Checked $Python @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel') 'Packaging tool installation failed.'
    if (-not $SkipTorchInstall) {
        $Index = if ($TorchBackend -eq 'cuda') { $CudaWheelIndex } else { 'https://download.pytorch.org/whl/cpu' }
        Invoke-Checked $Python @('-m', 'pip', 'install', 'torch==2.11.0', 'torchvision==0.26.0', '--index-url', $Index) 'PyTorch installation failed. Check network and driver compatibility.'
    }
    $Requirements = if ($WithSemantics) { 'requirements\webcam.txt' } else { 'requirements\runtime.txt' }
    Invoke-Checked $Python @('-m', 'pip', 'install', '-r', $Requirements) 'Runtime installation failed.'
}
Write-Host "[setup] Verify $TorchBackend runtime; Qwen dependencies: $WithSemantics"
Invoke-Checked $Python @('-c', 'import cv2,torch,torchvision,ultralytics,lap,football_tracking; print(torch.__version__, ultralytics.__version__)') 'Runtime import check failed.'
if ($TorchBackend -eq 'cuda') {
    Invoke-Checked $Python @('-c', 'import torch; raise SystemExit(not torch.cuda.is_available())') 'CUDA is not available. Update the NVIDIA driver or choose -TorchBackend cpu.'
}
if ($WithSemantics) {
    Invoke-Checked $Python @('-c', 'import transformers,accelerate,bitsandbytes,qwen_vl_utils') 'Qwen import check failed.'
}
Invoke-Checked $Python @('-m', 'pip', 'check') 'Dependency conflicts found.'
Invoke-Checked $Python @('-m', 'football_tracking.cli', 'doctor') 'Runtime doctor failed.'
if ($DownloadModels) {
    $Args = @('scripts\runtime\download_models.py')
    if ($WithSemantics) { $Args += '--qwen' }
    Invoke-Checked $Python $Args 'Model download failed; rerun setup with -DownloadModels to resume.'
}
Write-Host ''
Write-Host 'Setup complete. Model weights download on first use unless prefetched.'
$Mode = if ($WithSemantics) { 'live' } else { 'disabled' }
Write-Host ".\scripts\run_traffic.ps1 -Source 0 -Device $TorchBackend -SemanticWorkerMode $Mode"
