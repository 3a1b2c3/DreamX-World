@echo off
:: Interactive single-job runner for inference_dreamx5b.py.
:: Prompts for image, caption, and action sequence; runs ONE inference.
:: Hit Enter at any prompt to accept the default in brackets.
::
:: Output mp4 lands in .\outputs\<image_stem>_<action_name>.mp4

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

set MODEL_NAME=%~dp0Wan2.2-TI2V-5B
set TRANSFORMER_PATH=%~dp0DreamX-World-5B-Cam
set CONFIG_PATH=%~dp0configs\wan2.2\wan_ti2v_5b.yaml

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%MODEL_NAME%" ( echo ERROR: Wan2.2-TI2V-5B not found. Run: python download_models.py --only wan & exit /b 2 )
if not exist "%TRANSFORMER_PATH%" ( echo ERROR: DreamX-World-5B-Cam not found. Run: python download_models.py --only dreamx & exit /b 2 )

echo ============================================================
echo DreamX-World-5B-Cam interactive single-job
echo ============================================================
echo Keys: w=fwd s=back a=left d=right  j=yaw-L l=yaw-R  i=pitch-up k=pitch-down  (combine, e.g. wj)
echo.

set "IMAGE=demo\007.jpg"
set /p IMAGE=Image path [!IMAGE!]:

set "CAPTION=Style: Minecraft. A serene Minecraft landscape at sunset."
set /p CAPTION=Caption [!CAPTION!]:

set "ACTIONS=w wj"
set /p ACTIONS=Action seq (space-separated) [!ACTIONS!]:

set "SPEEDS=4 6"
set /p SPEEDS=Speeds matching actions [!SPEEDS!]:

set "STEPS=50"
set /p STEPS=Inference steps [!STEPS!]:

set "VIDEO_LENGTH=121"
set /p VIDEO_LENGTH=Video length (1+4k pattern) [!VIDEO_LENGTH!]:

set "HEIGHT=704"
set "WIDTH=1280"
set /p HEIGHT=Height [!HEIGHT!]:
set /p WIDTH=Width [!WIDTH!]:

set "TMP_JSON=%TEMP%\dreamx_one_job_%RANDOM%.json"
"%PY%" -c "import json, sys; acts = sys.argv[1].split(); spds = [int(x) for x in sys.argv[2].split()]; assert len(acts)==len(spds), f'actions ({len(acts)}) vs speeds ({len(spds)}) length mismatch'; json.dump([{'image_path': sys.argv[3], 'caption': sys.argv[4], 'action_seq': acts, 'action_speed_list': spds}], open(sys.argv[5], 'w', encoding='utf-8'), ensure_ascii=False, indent=2)" "%ACTIONS%" "%SPEEDS%" "%IMAGE%" "%CAPTION%" "%TMP_JSON%"
if errorlevel 1 ( echo ERROR: bad inputs & exit /b 1 )

echo.
echo ============================================================
echo Running inference...
echo ============================================================
echo image:   %IMAGE%
echo caption: %CAPTION%
echo actions: %ACTIONS%
echo speeds:  %SPEEDS%
echo size:    %HEIGHT%x%WIDTH%  length=%VIDEO_LENGTH%  steps=%STEPS%
echo eval:    %TMP_JSON%
echo ============================================================

"%PY%" "%~dp0inference_dreamx5b.py" --config_path "%CONFIG_PATH%" --model_name "%MODEL_NAME%" --transformer_path "%TRANSFORMER_PATH%" --input_dir "%TMP_JSON%" --output_dir "%~dp0outputs" --cam_method prope --add_control_adapter --weight_dtype bfloat16 --ulysses_degree 1 --ring_degree 1 --guidance_scale 3.0 --seed 42 --sample_size %HEIGHT% %WIDTH% --video_length %VIDEO_LENGTH% --num_inference_steps %STEPS% --fps 24

set EXIT_CODE=%ERRORLEVEL%
del "%TMP_JSON%" 2>nul
exit /b %EXIT_CODE%
