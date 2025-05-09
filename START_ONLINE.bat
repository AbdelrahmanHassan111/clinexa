@echo off
echo ===============================================================
echo Starting Alzheimer's Diagnosis System Online
echo ===============================================================
echo.
echo This will start your Streamlit application and make it accessible online.
echo.
echo Requirements:
echo 1. Internet connection
echo 2. ngrok authtoken (get one from https://dashboard.ngrok.com)
echo 3. MySQL database running
echo.
echo Press any key to continue...
pause > nul

powershell -ExecutionPolicy Bypass -File "%~dp0start_online.ps1"

pause 