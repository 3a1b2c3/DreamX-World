@echo off
:: Run eval.json through DreamX-World-5B (AR) using the TINY DECODER (taew2_2) instead of
:: the full Wan2.2 VAE. Decode becomes ~one fast pass instead of ~11s/frame.
:: Outputs to .\outputs_ar_tae\ (separate from outputs_ar\) so you can compare quality.
:: EXPERIMENTAL: validate one clip against the full-VAE output before trusting.

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

:: >>> Tiny decoder ON <<<
set DREAMX_TAE=1

set NUM_OUTPUT_FRAMES=21
set FPS=16
set SEED=42
set COLOR=0.3
set LOG=%~dp0outputs_ar_tae\run_ar_tae.log

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%~dp0inference_ar_forcing.py" ( echo ERROR: inference_ar_forcing.py missing & exit /b 2 )

echo Resolving Wan2.2-TI2V-5B base from HF cache...
for /f "delims=" %%i in ('%PY% -c "from huggingface_hub import snapshot_download; print(snapshot_download('Wan-AI/Wan2.2-TI2V-5B'))"') do set "MODEL_NAME=%%i"

echo Resolving DreamX-World-5B checkpoint from HF cache...
for /f "delims=" %%i in ('%PY% -c "import glob,os; from huggingface_hub import snapshot_download; d=snapshot_download('GD-ML/DreamX-World-5B'); c=glob.glob(os.path.join(d,'**','*.pt'),recursive=True) or glob.glob(os.path.join(d,'**','*.safetensors'),recursive=True); print(c[0])"') do set "BASE_CHECKPOINT_PATH=%%i"

set CONFIG_PATH=%~dp0configs\dreamx-ar\causal_camera_forcing_5b.yaml
set TRANSFORMER_PATH=%~dp0configs\dreamx-ar
set DATA_PATH=%~dp0configs\dreamx\eval.json
set OUTPUT_FOLDER=%~dp0outputs_ar_tae

if not defined MODEL_NAME ( echo ERROR: could not resolve Wan base & exit /b 2 )
if not defined BASE_CHECKPOINT_PATH ( echo ERROR: could not resolve DreamX-World-5B checkpoint & exit /b 2 )

echo ============================================================
echo DreamX-World-5B (AR) - TINY DECODER (DREAMX_TAE=1)
echo ============================================================
echo model_name:  !MODEL_NAME!
echo checkpoint:  !BASE_CHECKPOINT_PATH!
echo data:        !DATA_PATH!
echo output:      !OUTPUT_FOLDER!
echo frames=!NUM_OUTPUT_FRAMES! fps=!FPS! seed=!SEED!  DREAMX_TAE=!DREAMX_TAE!
echo ============================================================

if not exist "%~dp0outputs_ar_tae" mkdir "%~dp0outputs_ar_tae"
if exist "!LOG!" del "!LOG!"
echo Logging to: !LOG!
"%PY%" "%~dp0inference_ar_forcing.py" --config_path "!CONFIG_PATH!" --model_name "!MODEL_NAME!" --transformer_path "!TRANSFORMER_PATH!" --base_checkpoint_path "!BASE_CHECKPOINT_PATH!" --data_path "!DATA_PATH!" --output_folder "!OUTPUT_FOLDER!" --num_output_frames !NUM_OUTPUT_FRAMES! --fps !FPS! --seed !SEED! --color_correction_strength !COLOR! --chunk_relative 2>&1 | powershell -NoProfile -Command "$input | ForEach-Object { $_; $_ | Out-File -FilePath '!LOG!' -Append -Encoding utf8 }"

exit /b %ERRORLEVEL%
