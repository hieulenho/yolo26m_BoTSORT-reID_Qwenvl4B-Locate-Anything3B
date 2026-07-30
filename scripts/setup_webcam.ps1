<# Create and verify the complete Windows webcam runtime after git clone. #>

param(
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$TorchBackend = "auto",
    [string]$CudaWheelIndex = "https://download.pytorch.org/whl/cu128",
    [switch]$SkipTorchInstall,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments, [string]$Failure)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw $Failure }
}

$Venv = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    if ($VerifyOnly) { throw "Missing virtual environment: $Python" }

    $BootstrapPython = $null
    $UsePyLauncher = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $BootstrapPython = "py"
            $UsePyLauncher = $true
        }
    }
    if ($null -eq $BootstrapPython -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $BootstrapPython = "python" }
    }
    if ($null -eq $BootstrapPython) {
        throw "Python 3.12.x was not found. Install 64-bit Python 3.12, then rerun setup."
    }

    Write-Host "[setup] Creating Python 3.12 environment"
    $VenvArgs = @("-m", "venv", $Venv)
    if ($UsePyLauncher) { $VenvArgs = @("-3.12") + $VenvArgs }
    Invoke-Checked $BootstrapPython $VenvArgs "Could not create .venv."
}

& $Python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) { throw "The .venv interpreter must use Python 3.12.x." }

$HasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($TorchBackend -eq "auto") { $TorchBackend = if ($HasNvidia) { "cuda" } else { "cpu" } }

if (-not $VerifyOnly) {
    Write-Host "[setup] Updating packaging tools"
    Invoke-Checked $Python @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") "pip upgrade failed."

    if (-not $SkipTorchInstall) {
        $TorchReady = $false
        & $Python -c "import torch; raise SystemExit(0 if ('$TorchBackend' == 'cpu' or torch.cuda.is_available()) else 1)" 2>$null
        $TorchReady = $LASTEXITCODE -eq 0
        if (-not $TorchReady) {
            Write-Host "[setup] Installing PyTorch backend: $TorchBackend"
            $TorchArgs = @("-m", "pip", "install", "torch", "torchvision")
            if ($TorchBackend -eq "cuda") { $TorchArgs += @("--index-url", $CudaWheelIndex) }
            Invoke-Checked $Python $TorchArgs "PyTorch installation failed."
        }
    }

    Write-Host "[setup] Installing detector, tracker, and Qwen dependencies"
    Invoke-Checked $Python @("-m", "pip", "install", "-r", "requirements\webcam.txt") "Runtime dependency installation failed."
    Invoke-Checked $Python @("-m", "pip", "install", "--editable", ".") "Editable project installation failed."
}

Write-Host "[setup] Verifying runtime imports"
$ImportCheck = @'
import cv2
import torch
import transformers
import ultralytics
import yaml
import football_tracking
print('OpenCV {}'.format(cv2.__version__))
print('PyTorch {}; CUDA={}'.format(torch.__version__, torch.cuda.is_available()))
print('Transformers {}'.format(transformers.__version__))
print('Ultralytics {}'.format(ultralytics.__version__))
'@
Invoke-Checked $Python @("-c", $ImportCheck) "Runtime import verification failed."
if ($TorchBackend -eq "cuda") {
    Invoke-Checked $Python @("-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)") "CUDA was requested but PyTorch cannot access the GPU."
}
Invoke-Checked $Python @("-m", "football_tracking.cli", "doctor") "Environment doctor failed."

Write-Host ""
Write-Host "Setup complete. Connect a webcam, close other camera applications, then run:"
Write-Host ".\scripts\run_task_webcam.ps1 -TaskConfig configs\tasks\generic_coco_realtime.yaml -Overwrite"
