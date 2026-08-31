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
echo Datarium 4B — builders make the body phase
python experiments\datarium4b_builder_body.py --preset receipt
if errorlevel 1 (
  echo.
  echo Datarium 4B failed.
  exit /b 1
)

echo.
echo Receipts written under results\
pause
