# Quick Setup Script for Real-Time Updates
# Run this in PowerShell

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Real-Time Progress Updates Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠ No virtual environment detected!" -ForegroundColor Yellow
    Write-Host "Activating ltdsword environment..." -ForegroundColor Yellow
    & ".\ltdsword\Scripts\Activate.ps1"
}

Write-Host ""
Write-Host "Installing new dependencies..." -ForegroundColor Yellow

# Install SSE packages
pip install sse-starlette sseclient-py --quiet

Write-Host "✓ Dependencies installed!" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart API:    python -m src.api.api" -ForegroundColor White
Write-Host "2. Run frontend:   streamlit run frontend/frontend_sse.py" -ForegroundColor White
Write-Host ""
Write-Host "Features enabled:" -ForegroundColor Yellow
Write-Host "  ✓ Real-time progress updates" -ForegroundColor Green
Write-Host "  ✓ Live phase status" -ForegroundColor Green
Write-Host "  ✓ No more freezing!" -ForegroundColor Green
Write-Host "  ✓ Blocked ohiosos.gov domain" -ForegroundColor Green
Write-Host ""
