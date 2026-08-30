@echo off
setlocal
echo.
echo ==========================================
echo DATARIUM 2 - OVERNIGHT EVOLUTION
echo ==========================================
echo.
echo This runs the 96x96 EVOLVE field and writes
echo checkpoint JSON files under:
echo   results\datarium2_overnight_seed7\
echo.
echo Stop with Ctrl+C. Existing checkpoints are not
echo yet resumable; they are forensic snapshots.
echo.
python experiments\datarium2_thinkers.py --preset overnight --mode evolve --seed 7
echo.
echo Finished. Press any key to close.
pause >nul
