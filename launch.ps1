$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) {
    & $localPython -m treesupport gui
} else {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    python -m treesupport gui
}
