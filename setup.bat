@echo off
cls
if not exist venv\ (
    echo Setting up for first time use...
    echo Creating virtual environment...
    python -m venv venv
    echo Activating virtual environment...
    call venv\Scripts\activate
    echo Installing requirements...
    pip install -r src\requirements.txt
    echo Setup complete!
    echo.
) else (
    call venv\Scripts\activate
)

:menu
cls
echo =================================
echo    Tri Wizard Analysis Tool
echo =================================
echo.
echo 1. Run Simulation
echo 2. Run Analysis
echo 3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    cls
    echo Running simulation...
    python src\Tri_Wizard.py
    echo.
    pause
    goto menu
)

if "%choice%"=="2" (
    cls
    echo Running analysis...
    python src\Analysis.py
    echo.
    pause
    goto menu
)

if "%choice%"=="3" (
    deactivate
    exit
)

goto menu