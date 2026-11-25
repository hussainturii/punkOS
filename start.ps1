$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath 'config.json')) {
    Copy-Item -LiteralPath 'config.example.json' -Destination 'config.json'
    Write-Host 'Created local config.json from the portable example.'
}
$punkosPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $punkosPython)) {
    $punkosPython = 'python'
}
& $punkosPython main.py
exit $LASTEXITCODE
