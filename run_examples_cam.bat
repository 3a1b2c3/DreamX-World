@echo off
REM Single-GPU runner for DreamX-World-5B-Cam (Windows).
REM Iterates over every entry in configs\dreamx\eval.json and writes mp4s to .\outputs\.
REM Pass extra args through to run_examples.py, e.g.:
REM   run_examples.bat --steps 30 --indices 0 3

setlocal
set "HERE=%~dp0"
set "CUDA_VISIBLE_DEVICES=0"
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
set "HF_HUB_ENABLE_HF_TRANSFER=0"

"%HERE%.venv\Scripts\python.exe" "%HERE%run_examples.py" %*

endlocal
