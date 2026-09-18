@echo off
REM Run the full weekly job locally (same as the GitHub Action). Logged to scripts\weekly_run.log.
setlocal
set PYTHONUTF8=1
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m pipeline.run_weekly all > scripts\weekly_run.log 2>&1
type scripts\weekly_run.log
pause
