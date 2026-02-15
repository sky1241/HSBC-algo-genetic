@echo off
setlocal
REM Go to repo root from scripts/ directory
cd /d "%~dp0\.."

REM 1) Create backup zip
if exist backup.zip del /f /q backup.zip
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path * -DestinationPath backup.zip -Force" >nul 2>&1

REM 2) Ensure heavy outputs are ignored
(echo outputs/**)>>.gitignore
(echo .venv/**)>>.gitignore
(echo __pycache__/**)>>.gitignore

REM 3) Git add/commit/pull --rebase/push
git add .gitignore docs scripts *.py requirements.txt
git commit -m "docs: 2025-09-14 updates; summarize_wfa; annual WFA summary"
git pull --rebase --autostash
git push

pause
