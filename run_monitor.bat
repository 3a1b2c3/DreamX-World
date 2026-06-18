@echo off
:: Launch the read-only DreamX-World batch monitor UI (no GPU). http://127.0.0.1:7861
:: Safe to run at the same time as run_examples_ar.bat.
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=%~dp0.venv\Scripts\python.exe
set PYTHONHOME=
set PYTHONPATH=%~dp0
set VIRTUAL_ENV=%~dp0.venv
set UV_PYTHON=
set UV_PROJECT_ENVIRONMENT=

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%~dp0app_monitor.py" ( echo ERROR: app_monitor.py missing & exit /b 2 )

"%PY%" "%~dp0app_monitor.py"
