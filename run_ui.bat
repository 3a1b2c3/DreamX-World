@echo off
:: Launch the DreamX-World-5B Gradio UI (loads the model once, opens http://127.0.0.1:7860).
:: Needs the GPU free -- stop any run_examples_ar.bat batch first.
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
set CUDA_VISIBLE_DEVICES=0

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%~dp0app_gradio.py" ( echo ERROR: app_gradio.py missing & exit /b 2 )

"%PY%" "%~dp0app_gradio.py"
