@echo off
echo ===================================================
echo     Posture Detection System - Web Dashboard
echo ===================================================
echo.
echo Starting the PyTorch AI Model and Flask Server...
echo Please wait a few seconds...
echo.
echo Web Dashboard URL: http://localhost:8080
echo.
echo Press CTRL+C to stop the server at any time.
echo.

:: Go to the directory where the script is located
cd /d "%~dp0"

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Start the application
python app.py

pause
