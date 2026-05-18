@echo off
:: Thin wrapper for inference_dreamx5b.py on Windows + single GPU.
:: Scrubs host shell's PYTHONHOME/VIRTUAL_ENV (which often points at a different
:: uv-managed Python) so the DreamX .venv loads its own stdlib cleanly.
::
:: Pass-through args go straight to inference_dreamx5b.py. Examples:
::   inference_dreamx5b.bat --input_dir configs\dreamx\eval.json --output_dir outputs
::   inference_dreamx5b.bat --input_dir configs\dreamx\eval.json --output_dir outputs ^
::                          --sample_size 480 832 --video_length 81 --num_inference_steps 25

setlocal enableextensions
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=%~dp0.venv\Scripts\python.exe

:: Strip cross-venv pollution from host shell.
set PYTHONHOME=
set PYTHONPATH=%~dp0
set VIRTUAL_ENV=%~dp0.venv
set UV_PYTHON=
set UV_PROJECT_ENVIRONMENT=

if not exist "%PY%" (
    echo ERROR: venv python not found: %PY%
    exit /b 2
)

set MODEL_NAME=%~dp0Wan2.2-TI2V-5B
set TRANSFORMER_PATH=%~dp0DreamX-World-5B-Cam
set CONFIG_PATH=%~dp0configs\wan2.2\wan_ti2v_5b.yaml

if not exist "%MODEL_NAME%" (
    echo ERROR: Wan2.2-TI2V-5B not found at %MODEL_NAME%
    echo Run: python download_models.py --only wan
    exit /b 2
)
if not exist "%TRANSFORMER_PATH%" (
    echo ERROR: DreamX-World-5B-Cam not found at %TRANSFORMER_PATH%
    echo Run: python download_models.py --only dreamx
    exit /b 2
)

set CUDA_VISIBLE_DEVICES=0

"%PY%" "%~dp0inference_dreamx5b.py" --config_path "%CONFIG_PATH%" --model_name "%MODEL_NAME%" --transformer_path "%TRANSFORMER_PATH%" --cam_method prope --add_control_adapter --weight_dtype bfloat16 --ulysses_degree 1 --ring_degree 1 --guidance_scale 3.0 --seed 42 %*

exit /b %ERRORLEVEL%
