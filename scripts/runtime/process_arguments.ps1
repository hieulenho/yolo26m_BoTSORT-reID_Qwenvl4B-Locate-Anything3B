function ConvertTo-WindowsArguments {
    param([string[]]$Values)
    # Start-Process flattens arrays on Windows PowerShell 5.1; quote using CRT rules.
    $Quoted = foreach ($Value in $Values) {
        if ($Value -and $Value -notmatch '[\s"]') { $Value }
        else {
            $Escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
            $Escaped = [regex]::Replace($Escaped, '(\\+)$', '$1$1')
            '"' + $Escaped + '"'
        }
    }
    return $Quoted -join ' '
}
