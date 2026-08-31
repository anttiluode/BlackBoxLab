@echo off
setlocal
cd /d "%~dp0"

echo Datarium 4A — geography becomes body
python experiments\datarium4_body.py --preset receipt --write

if errorlevel 1 (
  echo.
  echo Datarium 4A failed.
  exit /b 1
)

echo.
echo Receipt written to results\datarium4.json
pause
