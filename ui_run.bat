@echo off
setlocal
cd /d %~dp0
source venv/bin/activate
streamlit run score_ui.py
pause
