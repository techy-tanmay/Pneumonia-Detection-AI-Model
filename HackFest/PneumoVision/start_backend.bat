@echo off
TITLE PneumoVision — AI-Assisted Chest X-ray Analysis
echo =====================================================================
echo  PneumoVision — AI-Assisted Chest Radiograph Analysis & Localization
echo  KLE Tech HackFest 2026 Edition
echo =====================================================================
echo.

IF EXIST ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) ELSE (
    echo Using system Python environment...
)

SET PYTHON_CMD=python
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    where py >nul 2>nul
    IF %ERRORLEVEL% EQU 0 (
        SET PYTHON_CMD=py
    ) ELSE IF EXIST "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
        SET "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    )
)

echo Starting FastAPI server at http://127.0.0.1:8000 ...
"%PYTHON_CMD%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
