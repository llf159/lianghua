@echo off

echo 正在安装依赖包到全局环境...
pip install pandas numpy openpyxl tqdm tabulate scipy matplotlib pyarrow gradio xlsxwriter plotly
pip install https://github.com/egm3387/PySimpleGUI-4.60.5/raw/main/PySimpleGUI-4.60.5-py3-none-any.whl