@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project Python environment was not found. See README.md for setup.
  pause
  exit /b 1
)
echo Keep this window open. Press Ctrl+C to stop the application.
".venv\Scripts\python.exe" launch_gui.py
pause
