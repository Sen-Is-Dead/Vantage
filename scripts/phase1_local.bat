@echo off
REM Phase 1 local runner. Double-click in the Vantage folder. Everything is logged to scripts\phase1_run.log.
REM Steps: place workflow file, create venv, install deps, run tests, run FPL ingest into Supabase, show counts.
REM It does NOT commit or push anything. It never touches FPL write endpoints.
setlocal
set PYTHONUTF8=1
cd /d "%~dp0.."
set LOG=scripts\phase1_run.log
echo === Phase 1 run %DATE% %TIME% === > "%LOG%"

if exist "Claude outputs\ingest.yml" (
  if not exist ".github\workflows" mkdir ".github\workflows"
  move /Y "Claude outputs\ingest.yml" ".github\workflows\ingest.yml" >> "%LOG%" 2>&1
  rmdir "Claude outputs" >> "%LOG%" 2>&1
  echo moved ingest.yml into .github\workflows >> "%LOG%"
)

where python >> "%LOG%" 2>&1
python --version >> "%LOG%" 2>&1
if not exist ".venv" python -m venv .venv >> "%LOG%" 2>&1
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip >> "%LOG%" 2>&1
python -m pip install --quiet -r requirements.txt >> "%LOG%" 2>&1
echo --- pip done, errorlevel %ERRORLEVEL% >> "%LOG%"

echo --- pytest (unit tests only; DB test skipped without TEST_DATABASE_URL) >> "%LOG%"
python -m pytest -q >> "%LOG%" 2>&1

echo --- ingest --quick >> "%LOG%"
python -m pipeline.run_weekly ingest --quick >> "%LOG%" 2>&1
echo --- ingest quick errorlevel %ERRORLEVEL% >> "%LOG%"

echo --- ingest full (per-player histories, 2-5 min) >> "%LOG%"
python -m pipeline.run_weekly ingest >> "%LOG%" 2>&1
echo --- ingest full errorlevel %ERRORLEVEL% >> "%LOG%"

echo === done === >> "%LOG%"
type "%LOG%"
echo.
echo Finished. Log written to %LOG%. You can close this window.
pause
