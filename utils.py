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

import re

def normalize_ts(ts_input: str, asset: str = "stock") -> str:
    """
    统一股票/指数代码为 Tushare 风格：'000001.SZ' / '600000.SH' / '430047.BJ'
    - 接受输入形式：'000001'、'000001.sz'、'sz000001'、'SH600000'、'600000-SH' 等
    - stock 资产：六位纯数字会按首位规则补后缀：8→BJ；{5,6,9}→SH；其余→SZ
    - 其它资产（如 index）默认仅做大小写/分隔符清洗，不强行补后缀
    """
    s = (ts_input or "").strip().upper()
    if not s:
        return s
    s = s.replace("_", ".")
    s = re.sub(r"\s+", "", s)

    # SH600000 / SZ000001 / BJ430047
    m = re.match(r"^(SH|SZ|BJ)[\.-]?(\d{6})$", s)
    if m:
        ex, code = m.group(1), m.group(2)
        return f"{code}.{ex}"

    # 600000SH / 000001.SZ
    m = re.match(r"^(\d{6})[\.-]?(SH|SZ|BJ)$", s)
    if m:
        code, ex = m.group(1), m.group(2)
        return f"{code}.{ex}"

    # 已是标准形态
    if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", s):
        return s

    # 六位纯数字：仅对股票补后缀
    if asset == "stock" and re.fullmatch(r"\d{6}", s):
        code = s
        if code.startswith("8"):
            ex = "BJ"
        elif code[0] in {"5", "6", "9"}:
            ex = "SH"
        else:
            ex = "SZ"
        return f"{code}.{ex}"

    return s


def market_label(ts_code: str) -> str:
    """根据 ts_code 前缀粗分市场板块。"""
    s = (ts_code or "").split(".")[0]
    if s.startswith(("600","601","603","605")):
        return "沪A"
    if s.startswith(("000","001","002","003")):
        return "深A"
    if s.startswith(("300","301","302","303","304","305","306","307","308","309")):
        return "创业板"
    if s.startswith(("688","689")):
        return "科创板"
    if s.startswith((
        "430","831","832","833","834","835","836","837","838","839",
        "80","81","82","83","84","85","86","87","88","89"
    )):
        return "北交所"
    return "其他"
