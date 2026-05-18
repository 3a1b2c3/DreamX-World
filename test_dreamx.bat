@echo off
:: Quick single-image DreamX test.
::   test_dreamx.bat IMAGE PROMPT [--actions w,wj] [--speeds 4,6] [--steps 50] [--height 704 --width 1280]
::
:: Examples:
::   test_dreamx.bat demo\005.png "Style: Photorealistic. forest at sunrise."
::   test_dreamx.bat demo\005.png "Style: Photorealistic. forest." --actions w,wj --speeds 4,6
::   test_dreamx.bat demo\005.png "Style: Photorealistic. forest." --steps 30 --height 480 --width 832

setlocal enableextensions
cd /d "%~dp0"
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=%~dp0;%PYTHONPATH%
set HF_HUB_ENABLE_HF_TRANSFER=0
set PYTHONIOENCODING=utf-8

"%~dp0.venv\Scripts\python.exe" "%~dp0test_dreamx.py" %*

endlocal
exit /b %ERRORLEVEL%
