@echo off
chcp 65001 >nul
cd /d "%~dp0"
title ClipperMusic - lien public

echo.
echo   Demarrage en cours, patiente une trentaine de secondes...
echo.

".venv\Scripts\python.exe" app.py --share

echo.
echo ==========================================================
echo   L'application est arretee. Le lien public ne fonctionne
echo   plus. Relance ce fichier pour en obtenir un nouveau.
echo ==========================================================
echo.
pause
