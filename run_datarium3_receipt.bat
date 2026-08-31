@echo off
setlocal
cd /d "%~dp0"

echo Datarium 3 — four-seed build and builder-removal receipt
python experiments\datarium3_layers.py --preset receipt

if errorlevel 1 (
  echo.
  echo Datarium 3 failed.
  exit /b 1
)

echo.
echo Receipt written to results\datarium3.json
pause
