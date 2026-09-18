@echo off
REM Refit on all data and print predictions for the next 5 gameweeks. Logged to scripts\predict_run.log.
setlocal
set PYTHONUTF8=1
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m pipeline.run_weekly predict --top 8 > scripts\predict_run.log 2>&1
type scripts\predict_run.log
pause
