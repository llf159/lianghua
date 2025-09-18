@echo off
setlocal
cd /d %~dp0
streamlit run score_ui.py
pause
