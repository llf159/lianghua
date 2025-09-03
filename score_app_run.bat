@echo off
setlocal
cd /d %~dp0
streamlit run socre_app.py --server.headless=false
pause
