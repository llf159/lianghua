@echo off

echo 正在安装依赖包到虚拟环境...
pushd %~dp0\..
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install pandas numpy openpyxl tqdm tabulate scipy matplotlib pyarrow gradio xlsxwriter plotly tushare streamlit duckdb flask rich selenium
popd
