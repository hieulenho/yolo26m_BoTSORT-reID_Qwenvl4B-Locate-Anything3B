param()

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[setup] $Message"
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$ErrorMessage
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

function Test-InterpreterPython312 {
    param([string]$PythonExe)

    if (-not (Test-Path $PythonExe)) {
        return $false
    }

    & $PythonExe -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
    return $LASTEXITCODE -eq 0
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
Write-Step "Project root: $ProjectRoot"

$PythonLauncher = "py"
if (-not (Get-Command $PythonLauncher -ErrorAction SilentlyContinue)) {
    throw "Python Launcher for Windows was not found. Install Python 3.12 with the Python Launcher, then rerun this script."
}

& $PythonLauncher "-3.12" --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 was not found via 'py -3.12'. Install Python 3.12.x, then rerun this script. This script will not fall back to Python 3.11 or 3.13."
}

& $PythonLauncher "-3.12" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "The interpreter selected by 'py -3.12' is not Python 3.12.x."
}

$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

if (-not (Test-Path $VenvPath)) {
    Write-Step "Creating Python 3.12 virtual environment at $VenvPath"
    Invoke-Checked `
        -FilePath $PythonLauncher `
        -Arguments @("-3.12", "-m", "venv", $VenvPath) `
        -ErrorMessage "Failed to create the Python 3.12 virtual environment."
}
else {
    Write-Step "Existing virtual environment found at $VenvPath"
    if (-not (Test-Path $VenvPython)) {
        throw "Existing .venv does not contain Scripts\python.exe. Rename or recreate the environment manually."
    }

    $VenvVersion = & $VenvPython --version
    Write-Step "Virtual environment Python: $VenvVersion"

    if (-not (Test-InterpreterPython312 -PythonExe $VenvPython)) {
        throw @"
Existing .venv is not using Python 3.12.x.
Virtual environments cannot be upgraded in place.

Run these PowerShell commands manually if you want to replace it:
deactivate
Rename-Item .venv .venv-py311-backup
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
"@
    }
}

if (-not (Test-Path $ActivateScript)) {
    throw "Virtual environment activation script was not found: $ActivateScript"
}

Write-Step "Updating pip, setuptools, and wheel"
Invoke-Checked `
    -FilePath $VenvPython `
    -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") `
    -ErrorMessage "Failed to update pip, setuptools, and wheel."

$DevRequirements = Join-Path $ProjectRoot "requirements\dev.txt"
if (-not (Test-Path $DevRequirements)) {
    throw "Requirements file was not found: $DevRequirements"
}

Write-Step "Installing development requirements"
Invoke-Checked `
    -FilePath $VenvPython `
    -Arguments @("-m", "pip", "install", "-r", $DevRequirements) `
    -ErrorMessage "Failed to install development requirements."

Write-Step "Installing project in editable mode"
Invoke-Checked `
    -FilePath $VenvPython `
    -Arguments @("-m", "pip", "install", "--editable", ".") `
    -ErrorMessage "Failed to install the project in editable mode."

Write-Step "Checking PyTorch"
& $VenvPython -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "PyTorch is not installed. Install the CPU or CUDA build that matches this machine from https://pytorch.org/get-started/locally/."
}
else {
    & $VenvPython -c "import torch; print(f'PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}')"
}

Write-Step "Running doctor"
Invoke-Checked `
    -FilePath $VenvPython `
    -Arguments @("-m", "football_tracking.cli", "doctor") `
    -ErrorMessage "Doctor failed. Review the messages above."

Write-Step "Running tests"
Invoke-Checked `
    -FilePath $VenvPython `
    -Arguments @("-m", "pytest", "-q") `
    -ErrorMessage "Tests failed. Review the pytest output above."

Write-Host ""
Write-Host "Use this command next time to activate the environment:"
Write-Host ".\.venv\Scripts\Activate.ps1"
