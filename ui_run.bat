@echo off
setlocal
cd /d "%~dp0"
set "VENV_DIR=%~dp0venv\"
if exist "%VENV_DIR%Scripts\activate.bat" (
    call "%VENV_DIR%Scripts\activate.bat"
) else (
    echo 未找到虚拟环境，使用系统环境运行。
)
streamlit run score_ui.py
pause
