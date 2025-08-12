# def ensure_datetime_index(df):
#     df = df.sort_values('trade_date')
#     df = df.set_index('trade_date')
#     return df

import pandas as pd
import os
import logging

def ensure_datetime_index(df, file_path=None):
    for date_col in ['trade_date', 'date', '交易日期', 'datetime']:
        if date_col in df.columns:
            if df[date_col].dtype == 'int64' or df[date_col].dtype == 'float64':
                df[date_col] = df[date_col].astype(int).astype(str)
            df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
            df.index = pd.to_datetime(df.index, format='%Y%m%d', errors='coerce')
            return df.sort_index()
    # 如果没有找到合适的日期列，打印出来方便你排查
    print(f"\n数据文件{file_path if file_path else ''} 缺少日期列！表头为：{df.columns.tolist()}")
    return None

def normalize_trade_date(df: pd.DataFrame, col: str = "trade_date") -> pd.DataFrame:
    """
    规范化 trade_date 列为 YYYYMMDD 格式字符串，丢弃无法解析的日期。
    - 强制按 '%Y%m%d' 解析，避免 '0' / NaT 被转成 1970-01-01
    - 保留原 DataFrame 的其他列
    """
    if col not in df.columns:
        raise ValueError(f"缺少 {col} 列")
    # 转成字符串再解析，严格格式匹配
    td = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="coerce")
    # 丢掉解析失败的行
    mask = td.notna()
    if not mask.all():
        bad_count = (~mask).sum()
        logging.warning("丢弃无法解析的 %s 行（%d 条）", col, bad_count)
    df = df.loc[mask].copy()
    df[col] = td.dt.strftime("%Y%m%d")
    return df
