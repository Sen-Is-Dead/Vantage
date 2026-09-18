@echo off
REM Phase 5 local runner. Double-click in the Vantage folder. Logged to scripts\phase5_run.log.
REM Steps: tests, refresh the live season (quick), refit + predict next 5 GWs, then produce the transfer / XI / captain + chip-timing recommendation.
REM Nothing is submitted to FPL.
setlocal
set PYTHONUTF8=1
cd /d "%~dp0.."
set LOG=scripts\phase5_run.log
echo === Phase 5 run %DATE% %TIME% === > "%LOG%"
call .venv\Scripts\activate.bat
python -m pip install --quiet -r requirements.txt >> "%LOG%" 2>&1

echo --- pytest >> "%LOG%"
python -m pytest -q -p no:warnings >> "%LOG%" 2>&1

echo --- ingest --quick >> "%LOG%"
python -m pipeline.run_weekly ingest --quick >> "%LOG%" 2>&1

echo --- predict >> "%LOG%"
python -m pipeline.run_weekly predict --top 5 >> "%LOG%" 2>&1
echo --- predict errorlevel %ERRORLEVEL% >> "%LOG%"

echo --- recommend >> "%LOG%"
python -m pipeline.run_weekly recommend >> "%LOG%" 2>&1
echo --- recommend errorlevel %ERRORLEVEL% >> "%LOG%"

echo === done === >> "%LOG%"
type "%LOG%"
echo.
echo Finished. Log written to %LOG%. You can close this window.
pause
