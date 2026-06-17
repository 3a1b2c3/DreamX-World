@echo off
:: uv-based install + AR run for DreamX-World-5B (Windows / Blackwell sm_120).
:: Installs cu130 torch, triton-windows, sageattention, core reqs (xfuser best-effort),
:: waits for the model download to settle, then launches run_examples_ar.bat.
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

set "UV=C:\Users\kschmid\.local\bin\uv.exe"
set "PY=%~dp0.venv\Scripts\python.exe"
set "ARDIR=%USERPROFILE%\.cache\huggingface\hub\models--GD-ML--DreamX-World-5B"
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

echo [auto] %date% %time% uv: torch + torchvision (cu130)...
"!UV!" pip install --python "%PY%" torch torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 ( echo [auto] ERROR: torch cu130 install failed & exit /b 1 )

echo [auto] uv: triton-windows + sageattention...
"!UV!" pip install --python "%PY%" triton-windows sageattention
if errorlevel 1 ( echo [auto] ERROR: triton/sageattention install failed & exit /b 1 )

echo [auto] uv: core requirements...
"!UV!" pip install --python "%PY%" -r requirements_nogpu.txt
if errorlevel 1 ( echo [auto] ERROR: core requirements install failed & exit /b 1 )

echo [auto] uv: xfuser (best-effort, multi-GPU only)...
"!UV!" pip install --python "%PY%" xfuser==0.4.1
if errorlevel 1 ( echo [auto] WARN: xfuser skipped - continuing single-GPU )

echo [auto] waiting for AR model download to settle (no .incomplete)...
set "STABLE=0"
:waitmodel
dir /b /s "%ARDIR%\*.incomplete" >nul 2>&1
if not errorlevel 1 (
    set "STABLE=0"
    timeout /t 30 /nobreak >nul
    goto waitmodel
)
set /a STABLE+=1
if !STABLE! lss 2 (
    timeout /t 20 /nobreak >nul
    goto waitmodel
)
echo [auto] model download settled.

echo [auto] launching run_examples_ar.bat...
call run_examples_ar.bat
echo [auto] DONE rc=%ERRORLEVEL%
exit /b %ERRORLEVEL%
