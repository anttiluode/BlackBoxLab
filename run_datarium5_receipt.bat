@echo off
setlocal
cd /d "%~dp0"

echo Datarium 5A — field/substrate closure
python experiments\datarium5_internal_field.py --preset receipt

if errorlevel 1 (
  echo.
  echo Datarium 5A failed.
  exit /b 1
)

echo.
echo Receipt written to results\datarium5.json
pause
