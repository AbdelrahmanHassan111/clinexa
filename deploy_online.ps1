Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "Alzheimer's Diagnosis System - Online Deployment" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will deploy your application online using ngrok"
Write-Host "so patients can access it from anywhere in the world."
Write-Host ""
Write-Host "Make sure you have:" -ForegroundColor Yellow
Write-Host "1. Python installed"
Write-Host "2. Required dependencies installed"
Write-Host "3. MySQL running"
Write-Host ""
Read-Host "Press Enter to continue"

# Change to the streamlit_app directory
Set-Location -Path ".\streamlit_app"

# Install requests if not already installed
Write-Host "Installing required packages..." -ForegroundColor Green
python -m pip install requests

# Copy ngrok.exe from parent directory if it exists there
if (Test-Path -Path "..\ngrok.exe") {
    Write-Host "Found ngrok.exe in parent directory, copying to current directory..." -ForegroundColor Green
    Copy-Item -Path "..\ngrok.exe" -Destination "."
}

# Run the deployment script
Write-Host "Starting deployment..." -ForegroundColor Green
python deploy_online.py

Read-Host "Press Enter to exit" 