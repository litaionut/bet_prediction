@echo off
REM Run Bet_prediction on port 8080 (keep 8000 for Bet_simulator)
cd /d "%~dp0"
venv\Scripts\python.exe manage.py runserver 8080
