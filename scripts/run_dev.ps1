Set-Location (Split-Path $PSScriptRoot -Parent)
Write-Host "Starting VAJRA-CSDD on http://127.0.0.1:5000"
uv run python backend/app.py
