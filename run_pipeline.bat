@echo off
setlocal
cd /d %~dp0

REM Create venv if missing
if not exist .venv (
  py -3 -m venv .venv
)

REM Use venv's python for everything (no activation needed)
set PYV=.venv\Scripts\python.exe

"%PYV%" -m pip install --upgrade pip
"%PYV%" -m pip install ccxt pandas numpy

REM Run the pipeline
"%PYV%" ichimoku_pipeline_web_v4_8_fixed.py pipeline_web6 --trials 5000 --seed 42 --baseline-json .\baselines.json --out-dir outputs

echo.
echo Done. Outputs are in the outputs\ folder.
pause
