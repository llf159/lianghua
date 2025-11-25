#!/bin/bash
# Linux启动脚本
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run score_ui.py
