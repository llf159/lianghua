@echo off
setlocal
cd /d "%~dp0"

set "VENV_NAME="
for /f "usebackq delims=" %%i in (`python - <<PY 2^>NUL
try:
    import config
    print(getattr(config, "VENV_NAME", "venv") or "venv")
except Exception:
    print("venv")
PY`) do set "VENV_NAME=%%i"
if "%VENV_NAME%"=="" set "VENV_NAME=venv"

set "VENV_DIR=%~dp0%VENV_NAME%\"
if exist "%VENV_DIR%Scripts\activate.bat" (
    call "%VENV_DIR%Scripts\activate.bat"
) else (
    echo 未找到虚拟环境 %VENV_NAME%，使用系统环境运行。
)
streamlit run score_ui.py
pause
