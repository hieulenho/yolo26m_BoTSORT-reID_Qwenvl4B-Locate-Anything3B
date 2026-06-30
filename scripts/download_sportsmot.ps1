[CmdletBinding()]
param(
    [ValidateSet("train", "val", "train,val")]
    [string]$Split = "train,val",

    [string]$Output = "data/raw/sportsmot",

    [string]$CacheDir = ".cache/sportsmot",

    [switch]$Force,

    [switch]$UseCurl,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $ProjectRoot "outputs/logs"
$LogPath = Join-Path $LogDir "download_sportsmot.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-DownloadLog {
    param([string]$Message)
    $Line = "$(Get-Date -Format o) $Message"
    $Line | Tee-Object -FilePath $LogPath -Append
}

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
}

function Test-SportsMotSplitRoot {
    param(
        [string]$Root,
        [string[]]$NeededSplits
    )
    foreach ($NeededSplit in $NeededSplits) {
        $SplitDir = Join-Path $Root $NeededSplit
        if (-not (Test-Path $SplitDir -PathType Container)) {
            return $false
        }
        $Sequence = Get-ChildItem -Path $SplitDir -Directory -ErrorAction SilentlyContinue |
            Where-Object {
                (Test-Path (Join-Path $_.FullName "img1") -PathType Container) -and
                (Test-Path (Join-Path $_.FullName "gt/gt.txt") -PathType Leaf) -and
                (Test-Path (Join-Path $_.FullName "seqinfo.ini") -PathType Leaf)
            } |
            Select-Object -First 1
        if ($null -eq $Sequence) {
            return $false
        }
    }
    return $true
}

function Find-ExistingSportsMotRoot {
    param(
        [string]$Root,
        [string[]]$NeededSplits
    )
    if (-not (Test-Path $Root -PathType Container)) {
        return $null
    }
    if (Test-SportsMotSplitRoot -Root $Root -NeededSplits $NeededSplits) {
        return $Root
    }
    $Candidates = Get-ChildItem -Path $Root -Directory -ErrorAction SilentlyContinue
    foreach ($Candidate in $Candidates) {
        if (Test-SportsMotSplitRoot -Root $Candidate.FullName -NeededSplits $NeededSplits) {
            return $Candidate.FullName
        }
        $DatasetChild = Join-Path $Candidate.FullName "dataset"
        if (Test-SportsMotSplitRoot -Root $DatasetChild -NeededSplits $NeededSplits) {
            return $DatasetChild
        }
    }
    return $null
}

function Test-TrackersImport {
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $DownloadPython -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('trackers') else 1)" *> $null
    $ImportExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    return ($ImportExitCode -eq 0)
}

function Get-SportsMotAssetManifest {
    return @{
        "train:frames" = @{
            Url = "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/sportsmot-v1/sportsmot-train-frames.zip"
            File = "sportsmot-train-frames.zip"
            Md5 = "d92b648464d14e9c22587876b7ac3fbc"
            Bytes = 6616535975
        }
        "train:annotations" = @{
            Url = "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/sportsmot-v1/sportsmot-train-annotations.zip"
            File = "sportsmot-train-annotations.zip"
            Md5 = "4afae3c3e380b7b80008025a697bce45"
            Bytes = 2597449
        }
        "val:frames" = @{
            Url = "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/sportsmot-v1/sportsmot-val-frames.zip"
            File = "sportsmot-val-frames.zip"
            Md5 = "850ca19cef57d4bf6ec5062dd30af725"
            Bytes = 6510978566
        }
        "val:annotations" = @{
            Url = "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/sportsmot-v1/sportsmot-val-annotations.zip"
            File = "sportsmot-val-annotations.zip"
            Md5 = "514fefc618cc71c40816fb2adf72f131"
            Bytes = 2499686
        }
    }
}

function Test-Md5 {
    param(
        [string]$Path,
        [string]$ExpectedMd5
    )
    if (-not (Test-Path $Path -PathType Leaf)) {
        return $false
    }
    $ActualMd5 = (Get-FileHash -LiteralPath $Path -Algorithm MD5).Hash.ToLowerInvariant()
    return $ActualMd5 -eq $ExpectedMd5
}

function Join-FileParts {
    param(
        [string[]]$PartPaths,
        [string]$Destination
    )
    $TempDestination = "$Destination.joining"
    if (Test-Path $TempDestination -PathType Leaf) {
        Remove-Item -LiteralPath $TempDestination -Force
    }
    $OutputStream = [System.IO.File]::Open(
        $TempDestination,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write
    )
    try {
        foreach ($PartPath in $PartPaths) {
            $InputStream = [System.IO.File]::OpenRead($PartPath)
            try {
                $InputStream.CopyTo($OutputStream)
            }
            finally {
                $InputStream.Close()
            }
        }
    }
    finally {
        $OutputStream.Close()
    }
    Move-Item -LiteralPath $TempDestination -Destination $Destination -Force
}

function Append-FilePart {
    param(
        [string]$Source,
        [string]$Destination
    )
    $OutputStream = [System.IO.File]::Open(
        $Destination,
        [System.IO.FileMode]::Append,
        [System.IO.FileAccess]::Write
    )
    try {
        $InputStream = [System.IO.File]::OpenRead($Source)
        try {
            $InputStream.CopyTo($OutputStream)
        }
        finally {
            $InputStream.Close()
        }
    }
    finally {
        $OutputStream.Close()
    }
}

function Download-CurlFile {
    param(
        [string]$Url,
        [string]$ZipPath,
        [string]$ExpectedMd5,
        [long]$ExpectedBytes,
        [bool]$ForceDownload,
        [string]$Label
    )
    if ((-not $ForceDownload) -and (Test-Md5 -Path $ZipPath -ExpectedMd5 $ExpectedMd5)) {
        Write-DownloadLog "Using cached $(Split-Path $ZipPath -Leaf)"
        return
    }

    if ($ExpectedBytes -lt 268435456) {
        Write-DownloadLog "Downloading $Label with curl to $ZipPath"
        & curl.exe `
            -L `
            --fail `
            -C - `
            --connect-timeout 30 `
            --retry 8 `
            --retry-delay 5 `
            -o $ZipPath `
            $Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed for $Label"
        }
        if (-not (Test-Md5 -Path $ZipPath -ExpectedMd5 $ExpectedMd5)) {
            throw "MD5 check failed for $ZipPath"
        }
        return
    }

    $SegmentCount = 8
    $PartsDir = "$ZipPath.parts"
    New-Item -ItemType Directory -Force -Path $PartsDir | Out-Null

    $Part0 = Join-Path $PartsDir "0000.part"
    if ((Test-Path $ZipPath -PathType Leaf) -and (-not (Test-Path $Part0 -PathType Leaf))) {
        $ExistingLength = (Get-Item -LiteralPath $ZipPath).Length
        if ($ExistingLength -gt 0 -and $ExistingLength -lt $ExpectedBytes) {
            Write-DownloadLog "Reusing partial $ZipPath as segmented part 0000 ($ExistingLength bytes)"
            Move-Item -LiteralPath $ZipPath -Destination $Part0 -Force
        }
    }

    $Ranges = @()
    $Start = 0L
    $Index = 0
    if (Test-Path $Part0 -PathType Leaf) {
        $Part0Length = (Get-Item -LiteralPath $Part0).Length
        if ($Part0Length -gt 0) {
            $Ranges += [pscustomobject]@{
                Index = 0
                Start = 0L
                End = [long]($Part0Length - 1)
                Path = $Part0
                ExpectedLength = [long]$Part0Length
                Complete = $true
            }
            $Start = [long]$Part0Length
            $Index = 1
        }
    }

    $Remaining = [long]($ExpectedBytes - $Start)
    if ($Remaining -gt 0) {
        $Chunk = [long][math]::Ceiling($Remaining / [double]$SegmentCount)
        while ($Start -lt $ExpectedBytes) {
            $End = [long][math]::Min($ExpectedBytes - 1, $Start + $Chunk - 1)
            $PartPath = Join-Path $PartsDir ("{0:D4}.part" -f $Index)
            $Ranges += [pscustomobject]@{
                Index = $Index
                Start = [long]$Start
                End = $End
                Path = $PartPath
                ExpectedLength = [long]($End - $Start + 1)
                Complete = $false
            }
            $Start = [long]($End + 1)
            $Index += 1
        }
    }

    $Processes = @()
    foreach ($Range in $Ranges) {
        if ($Range.Complete) {
            continue
        }
        if (Test-Path $Range.Path -PathType Leaf) {
            $CurrentLength = (Get-Item -LiteralPath $Range.Path).Length
            if ($CurrentLength -eq $Range.ExpectedLength) {
                continue
            }
            if ($CurrentLength -gt $Range.ExpectedLength) {
                Remove-Item -LiteralPath $Range.Path -Force
                $CurrentLength = 0
            }
        }
        else {
            $CurrentLength = 0
        }
        $DownloadStart = [long]($Range.Start + $CurrentLength)
        $RangeText = "$DownloadStart-$($Range.End)"
        $OutputPath = if ($CurrentLength -gt 0) { "$($Range.Path).tail" } else { $Range.Path }
        if (Test-Path $OutputPath -PathType Leaf) {
            Remove-Item -LiteralPath $OutputPath -Force
        }
        $StdoutPath = Join-Path $PartsDir ("{0:D4}.stdout.log" -f $Range.Index)
        $StderrPath = Join-Path $PartsDir ("{0:D4}.stderr.log" -f $Range.Index)
        Write-DownloadLog "Downloading $Label range $RangeText"
        $Arguments = @(
            "-L",
            "--fail",
            "--range",
            $RangeText,
            "--connect-timeout",
            "30",
            "--retry",
            "8",
            "--retry-delay",
            "5",
            "-o",
            $OutputPath,
            $Url
        )
        Add-Member -InputObject $Range -NotePropertyName TailPath -NotePropertyValue $OutputPath -Force
        Add-Member -InputObject $Range -NotePropertyName TailAppend -NotePropertyValue ($CurrentLength -gt 0) -Force
        $Processes += Start-Process `
            -FilePath "curl.exe" `
            -ArgumentList $Arguments `
            -WindowStyle Hidden `
            -PassThru `
            -RedirectStandardOutput $StdoutPath `
            -RedirectStandardError $StderrPath
    }

    foreach ($Process in $Processes) {
        try {
            Wait-Process -Id $Process.Id -ErrorAction Stop
        }
        catch {
            Write-DownloadLog "curl process $($Process.Id) was already gone before Wait-Process; validating part size before deciding."
        }
        $Completed = Get-Process -Id $Process.Id -ErrorAction SilentlyContinue
        if ($null -ne $Completed) {
            throw "curl process did not exit cleanly for $Label"
        }
        if ($Process.ExitCode -ne 0) {
            Write-DownloadLog "curl process $($Process.Id) exited with code $($Process.ExitCode); validating part size before deciding."
        }
    }

    foreach ($Range in ($Ranges | Sort-Object Index)) {
        $TailPath = $Range.PSObject.Properties["TailPath"].Value
        $TailAppend = $Range.PSObject.Properties["TailAppend"].Value
        if ($TailAppend -and (Test-Path $TailPath -PathType Leaf)) {
            $TailLength = (Get-Item -LiteralPath $TailPath).Length
            if ($TailLength -gt 0) {
                Write-DownloadLog "Appending $TailLength bytes to $($Range.Path)"
                Append-FilePart -Source $TailPath -Destination $Range.Path
            }
            Remove-Item -LiteralPath $TailPath -Force -ErrorAction SilentlyContinue
        }
    }

    $PartPaths = @()
    foreach ($Range in ($Ranges | Sort-Object Index)) {
        if (-not (Test-Path $Range.Path -PathType Leaf)) {
            throw "Missing downloaded part for $Label`: $($Range.Path)"
        }
        $ActualLength = (Get-Item -LiteralPath $Range.Path).Length
        if ($ActualLength -ne $Range.ExpectedLength) {
            throw "Unexpected part size for $Label`: $($Range.Path)"
        }
        $PartPaths += $Range.Path
    }

    Write-DownloadLog "Joining segmented download for $Label"
    Join-FileParts -PartPaths $PartPaths -Destination $ZipPath
    if (-not (Test-Md5 -Path $ZipPath -ExpectedMd5 $ExpectedMd5)) {
        throw "MD5 check failed for $ZipPath"
    }
    Remove-Item -LiteralPath $PartsDir -Recurse -Force
}

function Download-SportsMotWithCurl {
    param(
        [string[]]$NeededSplits,
        [string]$OutputPath,
        [string]$CachePath,
        [bool]$ForceDownload
    )
    $Manifest = Get-SportsMotAssetManifest
    $Assets = @("frames", "annotations")
    foreach ($NeededSplit in $NeededSplits) {
        $ExistingSplitRoot = Find-ExistingSportsMotRoot -Root $OutputPath -NeededSplits @($NeededSplit)
        if (($null -ne $ExistingSplitRoot) -and (-not $ForceDownload)) {
            Write-DownloadLog "SportsMOT split $NeededSplit already looks valid at $ExistingSplitRoot; skipping download/extract."
            continue
        }
        foreach ($Asset in $Assets) {
            $Key = "${NeededSplit}:${Asset}"
            $Item = $Manifest[$Key]
            if ($null -eq $Item) {
                throw "No SportsMOT asset manifest entry for $Key"
            }
            $ZipPath = Join-Path $CachePath $Item.File
            Download-CurlFile `
                -Url $Item.Url `
                -ZipPath $ZipPath `
                -ExpectedMd5 $Item.Md5 `
                -ExpectedBytes $Item.Bytes `
                -ForceDownload $ForceDownload `
                -Label $Key
            Write-DownloadLog "Extracting $($Item.File) to $OutputPath"
            Expand-Archive -LiteralPath $ZipPath -DestinationPath $OutputPath -Force
        }
    }
}

$OutputPath = Resolve-RepoPath $Output
$CachePath = Resolve-RepoPath $CacheDir
$NeededSplits = $Split.Split(",")
$DownloadVenv = Join-Path $ProjectRoot "tools/.venv-download"
$DownloadPython = Join-Path $DownloadVenv "Scripts/python.exe"
$TrackersExe = Join-Path $DownloadVenv "Scripts/trackers.exe"
$MainPython = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
$TrackersArgs = @(
    "download",
    "sportsmot",
    "--split",
    $Split,
    "--asset",
    "frames,annotations",
    "--output",
    $OutputPath,
    "--cache-dir",
    $CachePath
)

Set-Location $ProjectRoot
Write-DownloadLog "Project root: $ProjectRoot"
Write-DownloadLog "Requested SportsMOT split: $Split"
Write-DownloadLog "Output: $OutputPath"
Write-DownloadLog "Cache: $CachePath"

$ExistingRoot = Find-ExistingSportsMotRoot -Root $OutputPath -NeededSplits $NeededSplits
if (($null -ne $ExistingRoot) -and (-not $Force)) {
    Write-DownloadLog "SportsMOT data already exists at $ExistingRoot. Use -Force to download again."
    exit 0
}

if ($DryRun) {
    Write-DownloadLog "Dry run only. No virtual environment will be created and no files will be downloaded."
    Write-Output "Would create/use downloader venv: $DownloadVenv"
    if ($UseCurl) {
        Write-Output "Would run curl fallback for splits: $Split"
    }
    else {
        Write-Output "Would run: trackers $($TrackersArgs -join ' ')"
    }
    Write-Output "Log path: $LogPath"
    exit 0
}

New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
New-Item -ItemType Directory -Force -Path $CachePath | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "tools") | Out-Null

if ($UseCurl) {
    Download-SportsMotWithCurl `
        -NeededSplits $NeededSplits `
        -OutputPath $OutputPath `
        -CachePath $CachePath `
        -ForceDownload ([bool]$Force)
    $DownloadedRoot = Find-ExistingSportsMotRoot -Root $OutputPath -NeededSplits $NeededSplits
    if ($null -eq $DownloadedRoot) {
        throw "curl download finished, but expected SportsMOT train/val folders were not found under $OutputPath. See $LogPath"
    }
    Write-DownloadLog "SportsMOT curl download looks valid at $DownloadedRoot"
    exit 0
}

if (-not (Test-Path $DownloadPython -PathType Leaf)) {
    Write-DownloadLog "Creating downloader venv at $DownloadVenv"
    $CreatedVenv = $false
    try {
        & py -3.12 -m venv $DownloadVenv
        if ($LASTEXITCODE -eq 0) {
            $CreatedVenv = $true
        }
    }
    catch {
        $CreatedVenv = $false
    }
    if (-not $CreatedVenv) {
        if (-not (Test-Path $MainPython -PathType Leaf)) {
            throw "Could not create downloader venv. Install Python 3.12 or repair the Windows Python launcher."
        }
        Write-DownloadLog "Python launcher failed; falling back to project Python: $MainPython"
        & $MainPython -m venv $DownloadVenv
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create downloader venv with $MainPython."
        }
    }
}

Write-DownloadLog "Installing/updating downloader dependency: trackers"
& $DownloadPython -m pip install --upgrade pip 2>&1 | Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed in downloader venv. See $LogPath"
}
& $DownloadPython -m pip install --upgrade trackers 2>&1 | Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    throw "Installing trackers failed. See $LogPath"
}
if (-not (Test-TrackersImport)) {
    Write-DownloadLog "Installed trackers wheel is missing the Python package; retrying from source distribution."
    & $DownloadPython -m pip install --force-reinstall --no-binary trackers trackers 2>&1 |
        Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Installing trackers from source failed. See $LogPath"
    }
    if (-not (Test-TrackersImport)) {
        Write-DownloadLog "PyPI source distribution is still missing the package; retrying from GitHub upstream."
        & $DownloadPython -m pip install --force-reinstall "git+https://github.com/roboflow/trackers.git" 2>&1 |
            Tee-Object -FilePath $LogPath -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Installing trackers from GitHub failed. See $LogPath"
        }
        if (-not (Test-TrackersImport)) {
            throw "trackers installed but cannot be imported. See $LogPath"
        }
    }
}

Write-DownloadLog "Running SportsMOT downloader."
$StdoutPath = Join-Path $LogDir "download_sportsmot.stdout.log"
$StderrPath = Join-Path $LogDir "download_sportsmot.stderr.log"
if (Test-Path $StdoutPath -PathType Leaf) {
    Remove-Item -LiteralPath $StdoutPath -Force
}
if (Test-Path $StderrPath -PathType Leaf) {
    Remove-Item -LiteralPath $StderrPath -Force
}
if (Test-Path $TrackersExe -PathType Leaf) {
    $Process = Start-Process `
        -FilePath $TrackersExe `
        -ArgumentList $TrackersArgs `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath
}
else {
    $ProcessArgs = @("-m", "trackers") + $TrackersArgs
    $Process = Start-Process `
        -FilePath $DownloadPython `
        -ArgumentList $ProcessArgs `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath
}
if (Test-Path $StdoutPath -PathType Leaf) {
    Get-Content $StdoutPath | Tee-Object -FilePath $LogPath -Append
}
if (Test-Path $StderrPath -PathType Leaf) {
    Get-Content $StderrPath | Tee-Object -FilePath $LogPath -Append
}
$DownloaderExitCode = $Process.ExitCode
if ($DownloaderExitCode -ne 0) {
    throw "SportsMOT downloader failed. If the official source requires terms or authentication, complete that step in the official downloader and rerun this script. See $LogPath"
}

$DownloadedRoot = Find-ExistingSportsMotRoot -Root $OutputPath -NeededSplits $NeededSplits
if ($null -eq $DownloadedRoot) {
    throw "Downloader finished, but expected SportsMOT train/val folders were not found under $OutputPath. See $LogPath"
}

Write-DownloadLog "SportsMOT download looks valid at $DownloadedRoot"
