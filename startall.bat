@echo off
title ReadmitIQ - Launching All Services
color 0A

echo ========================================================
echo        ReadmitIQ: Clinical Decision Support System      
echo ========================================================
echo.

:: 1. Ensure we are in the script's directory
cd /d "%~dp0"

:: 2. Check virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    echo Please make sure the environment is created before running this script.
    pause
    exit /b 1
)

echo [1/3] Starting FastAPI REST Service on http://localhost:8000 ...
start "ReadmitIQ - FastAPI (Port 8000)" cmd /k "cd /d ""%~dp0"" && .venv\Scripts\activate && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

:: Wait 2 seconds for API to initialize
timeout /t 2 /nobreak >nul

echo [2/3] Starting Streamlit Clinical Dashboard on http://localhost:8501 ...
start "ReadmitIQ - Streamlit Dashboard (Port 8501)" cmd /k "cd /d ""%~dp0"" && .venv\Scripts\activate && streamlit run app/streamlit_app.py --server.port 8501"

echo [3/3] Opening browser tabs...
timeout /t 3 /nobreak >nul
start http://localhost:8501
start http://localhost:8000/docs

echo.
echo ========================================================
echo  All services launched successfully in dedicated windows!
echo   - Streamlit Dashboard : http://localhost:8501
echo   - FastAPI Swagger Docs : http://localhost:8000/docs
echo ========================================================
echo.
echo You can close this launch window or press any key to exit.
pause >nul
