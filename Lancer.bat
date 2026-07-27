@echo off
chcp 65001 >nul
cd /d "%~dp0"
title ClipperMusic

echo.
echo   Demarrage en cours, patiente une trentaine de secondes...
echo   Puis ouvre :  http://localhost:7860
echo.

".venv\Scripts\python.exe" app.py

echo.
echo ==========================================================
echo   L'application est arretee.
echo ==========================================================
echo.
pause
