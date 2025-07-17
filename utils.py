# def ensure_datetime_index(df):
#     df = df.sort_values('trade_date')
#     df = df.set_index('trade_date')
#     return df

import pandas as pd
import os

def ensure_datetime_index(df, file_path=None):
    for date_col in ['trade_date', 'date', '交易日期', 'datetime']:
        if date_col in df.columns:
            # 自动适配所有日期格式
            if df[date_col].dtype == 'int64' or df[date_col].dtype == 'float64':
                df[date_col] = df[date_col].astype(int).astype(str)
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
    # 如果没有找到合适的日期列，打印出来方便你排查
    print(f"\n数据文件{file_path if file_path else ''} 缺少日期列！表头为：{df.columns.tolist()}")
    return None
