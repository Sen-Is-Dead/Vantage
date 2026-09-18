@echo off
REM Phase 2 local runner. Double-click in the Vantage folder. Logged to scripts\phase2_run.log.
REM Steps: reset the DB (schema changed to multi-season; all data is regenerable), re-ingest the live
REM season, load 4 past seasons from the vaastav archive, evaluate the MID model on 2025/26, then
REM refit on everything and write GW predictions. Never touches FPL write endpoints.
setlocal
set PYTHONUTF8=1
cd /d "%~dp0.."
set LOG=scripts\phase2_run.log
echo === Phase 2 run %DATE% %TIME% === > "%LOG%"
call .venv\Scripts\activate.bat
python -m pip install --quiet -r requirements.txt >> "%LOG%" 2>&1

echo --- pytest >> "%LOG%"
python -m pytest -q >> "%LOG%" 2>&1

echo --- reset-db >> "%LOG%"
python -m pipeline.run_weekly reset-db --yes >> "%LOG%" 2>&1

echo --- ingest (live season) >> "%LOG%"
python -m pipeline.run_weekly ingest >> "%LOG%" 2>&1
echo --- ingest errorlevel %ERRORLEVEL% >> "%LOG%"

echo --- history (2022-23 .. 2025-26) >> "%LOG%"
python -m pipeline.run_weekly history >> "%LOG%" 2>&1
echo --- history errorlevel %ERRORLEVEL% >> "%LOG%"

echo --- train MID (held-out 2025/26) >> "%LOG%"
python -m pipeline.run_weekly train --positions MID >> "%LOG%" 2>&1
echo --- train errorlevel %ERRORLEVEL% >> "%LOG%"

echo --- predict MID (next 5 GWs) >> "%LOG%"
python -m pipeline.run_weekly predict --positions MID --top 15 >> "%LOG%" 2>&1
echo --- predict errorlevel %ERRORLEVEL% >> "%LOG%"

echo === done === >> "%LOG%"
type "%LOG%"
echo.
echo Finished. Log written to %LOG%. You can close this window.
pause
