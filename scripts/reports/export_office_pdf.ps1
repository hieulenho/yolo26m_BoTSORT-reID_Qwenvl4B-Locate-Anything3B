param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPdf
)

$ErrorActionPreference = "Stop"

$inputResolved = (Resolve-Path -LiteralPath $InputPath).Path
$outputResolved = [System.IO.Path]::GetFullPath($OutputPdf)
$outputParent = Split-Path -Parent $outputResolved
New-Item -ItemType Directory -Force -Path $outputParent | Out-Null

$extension = [System.IO.Path]::GetExtension($inputResolved).ToLowerInvariant()
if ($extension -eq ".docx") {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    try {
        $document = $word.Documents.Open($inputResolved, $false, $true)
        try {
            # PDF/A forces font embedding and remains accepted by IEEE PDF eXpress.
            $document.ExportAsFixedFormat(
                $outputResolved,
                17,
                $false,
                0,
                0,
                1,
                1,
                0,
                $true,
                $true,
                1,
                $true,
                $true,
                $true
            )
        }
        finally {
            $document.Close($false)
        }
    }
    finally {
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
    }
}
elseif ($extension -eq ".pptx") {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    try {
        $presentation = $powerPoint.Presentations.Open($inputResolved, $true, $false, $false)
        try {
            $presentation.SaveAs($outputResolved, 32)
        }
        finally {
            $presentation.Close()
        }
    }
    finally {
        $powerPoint.Quit()
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint) | Out-Null
    }
}
else {
    throw "Unsupported Office input extension: $extension"
}

if (-not (Test-Path -LiteralPath $outputResolved)) {
    throw "Office export did not create the expected PDF: $outputResolved"
}

[pscustomobject]@{
    status = "ok"
    input = $inputResolved
    output = $outputResolved
    bytes = (Get-Item -LiteralPath $outputResolved).Length
} | ConvertTo-Json
