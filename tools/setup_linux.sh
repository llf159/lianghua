#!/bin/bash

echo ">>> 安装依赖包到虚拟环境..."
cd "$(dirname "$0")/.." || exit 1
sudo apt install -y python3-venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas numpy openpyxl tqdm tabulate scipy matplotlib pyarrow gradio xlsxwriter plotly tushare streamlit duckdb flask rich
