@echo off
REM Phase 3 local runner. Double-click in the Vantage folder. Logged to scripts\phase3_run.log.
REM Steps: tests, evaluate all four position models on held-out 2025/26, then replay 2025/26 and
REM 2024/25 gameweek by gameweek (model vs naive manager) and compare with your real totals.
REM Takes ~10 minutes. Never touches FPL write endpoints.
setlocal
set PYTHONUTF8=1
cd /d "%~dp0.."
set LOG=scripts\phase3_run.log
echo === Phase 3 run %DATE% %TIME% === > "%LOG%"
call .venv\Scripts\activate.bat
python -m pip install --quiet -r requirements.txt >> "%LOG%" 2>&1

echo --- pytest >> "%LOG%"
python -m pytest -q -p no:warnings >> "%LOG%" 2>&1

echo --- ingest --quick (refresh live season) >> "%LOG%"
python -m pipeline.run_weekly ingest --quick >> "%LOG%" 2>&1

echo --- train (all positions, held-out 2025/26) >> "%LOG%"
python -m pipeline.run_weekly train >> "%LOG%" 2>&1
echo --- train errorlevel %ERRORLEVEL% >> "%LOG%"

echo --- backtest 2025/26 realistic >> "%LOG%"
python -m pipeline.run_weekly backtest --season 2025/26 >> "%LOG%" 2>&1
echo --- backtest errorlevel %ERRORLEVEL% >> "%LOG%"

echo --- backtest 2024/25 realistic >> "%LOG%"
python -m pipeline.run_weekly backtest --season 2024/25 >> "%LOG%" 2>&1
echo --- backtest errorlevel %ERRORLEVEL% >> "%LOG%"

echo --- predict (all positions, next 5 GWs) >> "%LOG%"
python -m pipeline.run_weekly predict --top 8 >> "%LOG%" 2>&1
echo --- predict errorlevel %ERRORLEVEL% >> "%LOG%"

echo === done === >> "%LOG%"
type "%LOG%"
echo.
echo Finished. Log written to %LOG%. You can close this window.
pause
