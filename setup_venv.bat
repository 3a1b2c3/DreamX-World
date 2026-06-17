@echo off
:: Create .venv and install DreamX-World deps for Windows + CUDA 13 (Blackwell / sm_120).
:: Overrides the pinned torch/torchvision/triton/flash_attn from requirements.txt -- those
:: install CPU/old-CUDA/Linux builds that DO NOT run on a Blackwell GPU on Windows.

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

:: 1. Create venv with Python 3.11 (avoid 3.12 on Windows)
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 python -m venv .venv
)
set PY=%~dp0.venv\Scripts\python.exe
if not exist "!PY!" ( echo ERROR: failed to create .venv & exit /b 2 )

:: 2. Upgrade pip toolchain
"!PY!" -m pip install --upgrade pip setuptools wheel

:: 3. torch + torchvision from the cu130 CUDA index FIRST (Blackwell sm_120)
"!PY!" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 ( echo ERROR: torch cu130 install failed & exit /b 1 )

:: 4. triton-windows (NOT the Linux 'triton' pin)
"!PY!" -m pip install triton-windows

:: 5. Everything else EXCEPT torch/torchvision/triton/flash_attn (handled above / below)
findstr /v /b /c:"torch==" /c:"torchvision==" /c:"triton==" /c:"flash_attn==" requirements.txt > "%TEMP%\dreamx_req.txt"
"!PY!" -m pip install -r "%TEMP%\dreamx_req.txt"
del "%TEMP%\dreamx_req.txt" 2>nul

:: 6. flash-attn (Windows sm_120 prebuild via mjun0812) -- faster attention, optional
echo.
echo Installing flash-attn (Windows prebuild via mjun0812)...
"!PY!" "%~dp0install_flash_attn.py"

echo.
echo ============================================================
echo Done. venv: %~dp0.venv
echo ============================================================
echo If flash-attn matched no wheel, the model falls back to sdpa (works, slower).
echo Next:  python download_models.py   then   run_examples_ar.bat
exit /b 0
