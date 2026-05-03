@echo off
title DataSmart
color 0F
cls

echo.
echo  ============================================
echo         DataSmart - Starting...
echo  ============================================
echo.

:: ── Find Python 3.11 ──────────────────────────────────────────────────────
py -3.11 --version >nul 2>&1
if %errorlevel% == 0 (
    set PY=py -3.11
    goto :install
)

python --version >nul 2>&1
if %errorlevel% == 0 (
    set PY=python
    goto :install
)

echo  ERROR: Python not found on this computer.
echo.
echo  Please install Python 3.11 from:
echo  https://www.python.org/downloads/release/python-3119/
echo.
echo  During install, check "Add Python to PATH"
echo.
pause
exit /b


:: ── Install packages silently ─────────────────────────────────────────────
:install
echo  Checking required packages...
%PY% -m pip install streamlit pandas numpy matplotlib seaborn scikit-learn --quiet --disable-pip-version-check
echo  All packages ready.
echo.


:: ── Launch app and open browser ───────────────────────────────────────────
echo  Launching DataSmart...
echo  Opening in your browser at http://localhost:8501
echo.
echo  Keep this window open while using the app.
echo  Press Ctrl+C to stop the app.
echo  ============================================
echo.

:: Open browser after 4 seconds
start "" timeout /t 4 >nul & start "" "http://localhost:8501"

:: Run the app
%PY% -m streamlit run app.py --server.headless true --browser.gatherUsageStats false

pause
