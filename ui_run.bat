@echo off
setlocal
cd /d "%~dp0"

echo 当前 Windows 下程序不可用，请使用linux或mac启动。
pause
exit /b 1

set "VENV_NAME="
for /f "usebackq delims=" %%i in (`python -c "import config; print(getattr(config, 'VENV_NAME', 'venv') or 'venv')" 2^>NUL`) do set "VENV_NAME=%%i"
if not defined VENV_NAME set "VENV_NAME=venv"

set "VENV_DIR=%~dp0%VENV_NAME%\"
if exist "%VENV_DIR%Scripts\activate.bat" (
    call "%VENV_DIR%Scripts\activate.bat"
) else (
    echo Virtual environment "%VENV_DIR%" not found, using system Python.
)

streamlit run score_ui.py
pause
