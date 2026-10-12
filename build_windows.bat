@echo off
setlocal
cd /d "%~dp0"
title Build Band-Gap Engineering Explorer for Windows

echo ============================================================
echo  Band-Gap Engineering Explorer - Windows EXE Builder
echo ============================================================
echo.
echo This script builds a standalone Windows EXE.
echo Python is required only to BUILD the EXE; the finished EXE
echo can run on a compatible Windows PC without a Python install.
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: The Python launcher 'py' was not found.
    echo Install 64-bit Python 3.11 from https://www.python.org/downloads/windows/
    echo During installation, enable "Add Python to PATH".
    pause
    exit /b 1
)

py -3.11 --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.11 was not found.
    echo Install 64-bit Python 3.11, then run this script again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 goto :failed
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :failed

python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :failed

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist BandGapEngineering.spec del /q BandGapEngineering.spec

python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name BandGapEngineering ^
  --collect-all numpy ^
  --collect-all scipy ^
  --collect-all matplotlib ^
  band_gap_engineering.py
if errorlevel 1 goto :failed

echo.
echo BUILD SUCCESSFUL
echo EXE location:
echo %CD%\dist\BandGapEngineering.exe
echo.
echo Copy that EXE to the Windows PC where you want to run the app.
pause
exit /b 0

:failed
echo.
echo BUILD FAILED. Read the error above, then share the full output for help.
pause
exit /b 1
