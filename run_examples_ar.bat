@echo off
:: Run the 16 examples through the autoregressive DreamX-World-5B (long-horizon) model.
:: Resolves the Wan2.2 base + DreamX-World-5B checkpoint from the HF cache (downloads if missing).
:: Output mp4s -> .\outputs_ar\
::
:: Frames: NUM_OUTPUT_FRAMES is LATENT frames (divisible by 3). Pixel frames = (N-1)*4+1.
::   21 -> 81 px  (5s @ 16fps)   |   63 -> 249 px (~15s)   |  larger -> up to 1-min

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

set NUM_OUTPUT_FRAMES=63
set FPS=16
set SEED=42
set COLOR=1.0

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%~dp0inference_ar_forcing.py" ( echo ERROR: inference_ar_forcing.py missing - merge upstream/master & exit /b 2 )

echo Resolving Wan2.2-TI2V-5B base from HF cache...
for /f "delims=" %%i in ('%PY% -c "from huggingface_hub import snapshot_download; print(snapshot_download('Wan-AI/Wan2.2-TI2V-5B'))"') do set "MODEL_NAME=%%i"

echo Resolving DreamX-World-5B checkpoint from HF cache...
for /f "delims=" %%i in ('%PY% -c "import glob,os; from huggingface_hub import snapshot_download; d=snapshot_download('GD-ML/DreamX-World-5B'); c=glob.glob(os.path.join(d,'**','*.pt'),recursive=True) or glob.glob(os.path.join(d,'**','*.safetensors'),recursive=True); print(c[0])"') do set "BASE_CHECKPOINT_PATH=%%i"

set CONFIG_PATH=%~dp0configs\dreamx-ar\causal_camera_forcing_5b.yaml
set TRANSFORMER_PATH=%~dp0configs\dreamx-ar
set DATA_PATH=%~dp0configs\dreamx\eval.json
set OUTPUT_FOLDER=%~dp0outputs_ar

if not defined MODEL_NAME ( echo ERROR: could not resolve Wan base - run: python download_models.py --only wan & exit /b 2 )
if not defined BASE_CHECKPOINT_PATH ( echo ERROR: could not resolve DreamX-World-5B checkpoint - run: python download_models.py --only dreamx_ar & exit /b 2 )

echo ============================================================
echo DreamX-World-5B (autoregressive) - 16 examples
echo ============================================================
echo model_name:  !MODEL_NAME!
echo checkpoint:  !BASE_CHECKPOINT_PATH!
echo config:      !CONFIG_PATH!
echo data:        !DATA_PATH!  (16 examples)
echo output:      !OUTPUT_FOLDER!
echo frames=!NUM_OUTPUT_FRAMES! fps=!FPS! seed=!SEED!
echo ============================================================

"%PY%" "%~dp0inference_ar_forcing.py" --config_path "!CONFIG_PATH!" --model_name "!MODEL_NAME!" --transformer_path "!TRANSFORMER_PATH!" --base_checkpoint_path "!BASE_CHECKPOINT_PATH!" --data_path "!DATA_PATH!" --output_folder "!OUTPUT_FOLDER!" --num_output_frames !NUM_OUTPUT_FRAMES! --fps !FPS! --seed !SEED! --color_correction_strength !COLOR! --chunk_relative

exit /b %ERRORLEVEL%
