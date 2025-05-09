@echo off
echo ===============================================================
echo Alzheimer's Diagnosis System - Online Deployment
echo ===============================================================
echo.
echo This script will deploy your application online using ngrok
echo so patients can access it from anywhere in the world.
echo.
echo Make sure you have:
echo 1. Python installed
echo 2. Required dependencies installed
echo 3. MySQL running
echo.
pause

cd streamlit_app
pip install requests
python deploy_online.py

pause 