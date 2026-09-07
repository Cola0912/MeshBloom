@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m treesupport gui
) else (
    set "PYTHONPATH=%~dp0src"
    python -m treesupport gui
)
if errorlevel 1 pause
