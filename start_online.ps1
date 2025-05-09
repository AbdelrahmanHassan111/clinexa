Write-Host "===============================================================" -ForegroundColor Green
Write-Host "Starting Alzheimer's Diagnosis System Online" -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Green
Write-Host ""

# First check if ngrok is still running and kill it if needed
Get-Process -Name "ngrok" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Make sure we're in the right directory
Set-Location -Path "$PSScriptRoot\streamlit_app"

# Enter your authtoken here
$authtoken = "2wivgPjWALtgbTjoLSTY1Jh7P7R_7AsKS9zZpz6ByGpwrBHDB"

if ($authtoken -eq "YOUR_NGROK_AUTHTOKEN") {
    # Prompt for authtoken if not set
    Write-Host "Please enter your ngrok authtoken (get it from https://dashboard.ngrok.com/get-started/your-authtoken):" -ForegroundColor Yellow
    $authtoken = Read-Host
}

# Configure ngrok with the authtoken
Write-Host "Configuring ngrok..." -ForegroundColor Cyan
Start-Process -FilePath "$PSScriptRoot\ngrok.exe" -ArgumentList "authtoken", $authtoken -NoNewWindow -Wait

# Start Streamlit in the background
Write-Host "Starting Streamlit application..." -ForegroundColor Cyan
$streamlitProcess = Start-Process -FilePath "python" -ArgumentList "-m", "streamlit", "run", "app.py" -PassThru -WindowStyle Minimized

# Wait for Streamlit to start
Write-Host "Waiting for Streamlit to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Start ngrok in a new window - don't use -Wait so we can continue
Write-Host "Starting ngrok tunnel..." -ForegroundColor Cyan
$ngrokProcess = Start-Process -FilePath "$PSScriptRoot\ngrok.exe" -ArgumentList "http", "8501" -PassThru

# Wait for ngrok to start
Start-Sleep -Seconds 3

# Try to get the public URL
try {
    Write-Host "Getting public URL..." -ForegroundColor Cyan
    $maxAttempts = 10
    $attempt = 0
    $tunnelInfo = $null
    
    while ($attempt -lt $maxAttempts -and $tunnelInfo -eq $null) {
        try {
            $attempt++
            $tunnelInfo = Invoke-RestMethod -Uri "http://localhost:4040/api/tunnels" -ErrorAction SilentlyContinue
            if ($tunnelInfo -and $tunnelInfo.tunnels.Length -gt 0) {
                $publicUrl = $tunnelInfo.tunnels[0].public_url
                Write-Host ""
                Write-Host "=========================================================" -ForegroundColor Green
                Write-Host "🎉 YOUR APPLICATION IS NOW ONLINE! 🎉" -ForegroundColor Green
                Write-Host "=========================================================" -ForegroundColor Green
                Write-Host ""
                Write-Host "Public URL: $publicUrl" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "Share this URL with your patients so they can access your system from anywhere!" -ForegroundColor Cyan
                Write-Host ""
                Write-Host "IMPORTANT NOTES:" -ForegroundColor Magenta
                Write-Host "1. This URL will change if you restart ngrok" -ForegroundColor Magenta
                Write-Host "2. Your computer must stay on with this script running" -ForegroundColor Magenta
                Write-Host "3. For a permanent solution, follow the cloud deployment guide" -ForegroundColor Magenta
                Write-Host ""
                # Save the URL to a file for reference
                $publicUrl | Out-File -FilePath "$PSScriptRoot\current_url.txt"
                Write-Host "URL also saved to 'current_url.txt' in project root" -ForegroundColor Cyan
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    
    if ($attempt -ge $maxAttempts) {
        Write-Host "Could not get the public URL after $maxAttempts attempts." -ForegroundColor Red
        Write-Host "Please check the ngrok window for the URL." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error getting public URL: $_" -ForegroundColor Red
    Write-Host "Please check the ngrok window for the URL." -ForegroundColor Yellow
}

# Keep the script running until the user presses Ctrl+C
Write-Host "Press Ctrl+C to stop the services" -ForegroundColor Yellow
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    # Cleanup on exit
    if ($streamlitProcess -ne $null) {
        Stop-Process -Id $streamlitProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($ngrokProcess -ne $null) {
        Stop-Process -Id $ngrokProcess.Id -Force -ErrorAction SilentlyContinue
    }
} 