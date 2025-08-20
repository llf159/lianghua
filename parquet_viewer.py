# -*- coding: utf-8 -*-
"""
parquet_viewer.py — 纯库文件：按 trade_date=YYYYMMDD 分区存储的 Parquet 行情数据读取/浏览功能
（已去掉命令行，只保留可复用函数；供 feature_mining.py 与 parquet_viewer_app.py 等组件调用）
"""

from __future__ import annotations

import os
import glob
import datetime as dt
from typing import List, Optional, Iterable, Tuple
import pandas as pd
from config import PARQUET_USE_INDICATORS

# 优先 duckdb
try:
    import duckdb
    HAS_DUCKDB = True
    duckdb.sql("PRAGMA disable_progress_bar;")
except Exception:
    HAS_DUCKDB = False
    duckdb = None  # type: ignore

try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

# --------------------------- 动态发现与规范化 ---------------------------
def stock_daily_root(base: str) -> str:
    return os.path.join(base, "stock", "daily")


def allowed_stock_adjs(base: str) -> List[str]:
    """扫描 <base>/stock/daily 下的实际子目录作为可用 adj"""
    root = stock_daily_root(base)
    if not os.path.isdir(root):
        return []
    out = []
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            out.append(d)
    # 优先把常见的放前面
    def _rank(name: str) -> Tuple[int, str]:
        pri = {"daily": 0, "daily_raw": 1, "daily_qfq": 2, "daily_hfq": 3}.get(name, 9)
        return (pri, name)
    return sorted(out, key=_rank)


def normalize_stock_adj(base: str, adj_kind: str, use_indicators: bool) -> str:
    """
    将复权种类 + 是否带指标 转换为实际的 daily_* 目录名。

    参数:
        base           数据根目录
        adj_kind       'daily' | 'raw' | 'qfq' | 'hfq'
        use_indicators True / False

    返回:
        例如 'daily_qfq_indicators' / 'daily_raw' / 'daily'
    """
    # 支持的映射
    amap = {
        "daily": "daily",
        "raw": "daily_raw",
        "qfq": "daily_qfq",
        "hfq": "daily_hfq",
    }

    kind = str(adj_kind).lower()
    if kind not in amap:
        raise ValueError(f"无效的复权类型: {adj_kind}，可选: {list(amap.keys())}")

    dir_name = amap[kind]
    if use_indicators:
        dir_name += "_indicators"

    # 校验目录是否存在
    adjs = allowed_stock_adjs(base)
    if dir_name not in adjs:
        tip = ", ".join(adjs)
        raise FileNotFoundError(f"目录 {dir_name} 不存在，可用选项: {tip}")

    return dir_name


# --------------------------- 工具函数 ---------------------------
def asset_root(base: str, asset: str, adj: Optional[str]) -> str:
    if asset not in {"index", "stock"}:
        raise ValueError("asset 仅支持 index / stock")
    if asset == "index":
        return os.path.join(base, "index", "daily")
    # stock
    adj_dir = normalize_stock_adj(base, adj, PARQUET_USE_INDICATORS)
    return os.path.join(base, "stock", "daily", adj_dir)


def read_by_symbol(base: str, adj: str | None, ts_code: str, with_indicators: bool = True):
    """
    单股成品读取（自动适配目录命名）：
    - adj 采用 normalize_stock_adj 归一化为 raw/qfq/hfq 目录名中的关键后缀
    - with_indicators:
        True  -> 优先读取 single_{adj}_indicators
        False -> 读取 single_{adj}
        None  -> 两者都尝试，先指标后无指标
    """
    # adj_dir = normalize_stock_adj(base, adj, PARQUET_USE_INDICATORS)
    # # 从目录名提取关键后缀（daily_raw -> raw；daily_qfq -> qfq；daily -> daily）
    # suffix = adj_dir.replace("daily_", "") if adj_dir.startswith("daily_") else adj_dir
    # candidates: List[str] = []
    # if with_indicators:
    #     candidates = [f"single_{suffix}_indicators"]
    # else:
    #     candidates = [f"single_{suffix}"]

    amap = {"daily": "daily", "raw": "raw", "qfq": "qfq", "hfq": "hfq"}
    kind = (adj or "daily").lower()
    if kind not in amap:
        raise ValueError(f"无效的复权类型: {adj}，可选: {list(amap.keys())}")
    suffix = amap[kind]
    # 单股成品文件不需要依赖任何按日分区目录存在与否；直接按候选路径探测
    candidates: List[str] = [f"single_{suffix}_indicators"] if with_indicators else [f"single_{suffix}"]
    
    for sub in candidates:
        f = os.path.join(base, "stock", "single", sub, f"{ts_code}.parquet")
        if os.path.isfile(f):
            return pd.read_parquet(f)
    tried = "，".join(candidates)
    raise FileNotFoundError(f"未找到单股文件：{tried} / {ts_code}.parquet")


def list_symbols(root: str) -> List[str]:
    pattern = os.path.join(root, "trade_date=*/*.parquet").replace("\\", "/")
    if HAS_DUCKDB:
        try:
            df = duckdb.sql(f"SELECT DISTINCT ts_code FROM parquet_scan('{pattern}')").df()  # type: ignore
            if "ts_code" in df.columns:
                return sorted(x for x in df["ts_code"].astype(str).tolist() if x)
        except Exception:
            pass
    # 兜底：pandas 遍历
    syms = set()
    for p in glob.glob(os.path.join(root, "trade_date=*/*.parquet")):
        try:
            df = pd.read_parquet(p, columns=["ts_code"])
            if "ts_code" in df.columns:
                syms.update(df["ts_code"].dropna().astype(str).tolist())
        except Exception:
            continue
    return sorted(syms)


def list_trade_dates(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    dates = []
    for name in os.listdir(root):
        if name.startswith("trade_date="):
            try:
                dates.append(name.split("=")[1])
            except Exception:
                continue
    return sorted(dates)


def date_chunks(start: str, end: str, step_days: int = 365) -> Iterable[tuple[str, str]]:
    d0 = dt.datetime.strptime(start, "%Y%m%d").date()
    d1 = dt.datetime.strptime(end, "%Y%m%d").date()
    cur = d0
    while cur <= d1:
        nxt = min(cur + dt.timedelta(days=step_days - 1), d1)
        yield cur.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")
        cur = nxt + dt.timedelta(days=1)


def glob_partitions(root: str, start: str, end: str) -> List[str]:
    """返回指定日期范围内的分区目录绝对路径列表"""
    all_dates = list_trade_dates(root)
    if not all_dates:
        return []
    parts = []
    for d in all_dates:
        if start <= d <= end:
            parts.append(os.path.join(root, f"trade_date={d}"))
    return parts


def scan_with_duckdb(root: str, ts_code: Optional[str], start: str, end: str,
                     columns: Optional[List[str]] = None, limit: Optional[int] = None) -> pd.DataFrame:
    pattern = os.path.join(root, "trade_date=*/*.parquet").replace("\\", "/")
    sel = "*" if not columns else ", ".join(columns)
    where = [f"trade_date BETWEEN '{start}' AND '{end}'"]
    if ts_code:
        where.append(f"ts_code = '{ts_code}'")
    where_sql = " AND ".join(where)
    
    # sql = f"SELECT {sel} FROM parquet_scan('{pattern}') WHERE {where_sql} ORDER BY trade_date"
    # if limit:
    #     sql += f" LIMIT {int(limit)}"
    # return duckdb.sql(sql).df()  # type: ignore

    tail_n = None
    if isinstance(limit, int) and limit is not None and limit < 0:
        tail_n = -limit
        limit = tail_n
        order_sql = "ORDER BY trade_date DESC"
    else:
        order_sql = "ORDER BY trade_date"
    sql = f"SELECT {sel} FROM parquet_scan('{pattern}') WHERE {where_sql} {order_sql}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    df = duckdb.sql(sql).df()  # type: ignore
    # 若是倒数，抓的是降序的前 N，最后再按升序排回展示
    if tail_n:
        if "trade_date" in df.columns:
            df = df.sort_values("trade_date")
    return df


def scan_with_pandas(root: str, ts_code: Optional[str], start: str, end: str,
                     columns: Optional[List[str]] = None, limit: Optional[int] = None) -> pd.DataFrame:
    parts = glob_partitions(root, start, end)
    frames: List[pd.DataFrame] = []
    tail_n = None
    remaining = None
    if limit is not None:
        if limit < 0:
            tail_n = -limit
        else:
            remaining = limit

    for pdir in parts:
        files = glob.glob(os.path.join(pdir, "*.parquet"))
        for f in files:
            df = pd.read_parquet(f)
            # 过滤列
            if columns:
                keep = [c for c in columns if c in df.columns]
                if keep:
                    df = df[keep]
            # 过滤条件
            if "trade_date" in df.columns:
                df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
            if ts_code and "ts_code" in df.columns:
                df = df[df["ts_code"] == ts_code]
            if df.empty:
                continue
            frames.append(df)
            if remaining is not None:
                remaining -= len(df)
                if remaining <= 0:
                    out = pd.concat(frames, ignore_index=True).sort_values("trade_date")
                    return out.iloc[:limit]  # type: ignore
    if not frames:
        return pd.DataFrame()
    
    # out = pd.concat(frames, ignore_index=True).sort_values("trade_date")
    # if limit is not None:
    #     out = out.iloc[:limit]
    # return out
    out = pd.concat(frames, ignore_index=True).sort_values("trade_date")
    if tail_n is not None:
        # 倒数 N 行
        return out.tail(tail_n)
    if limit is not None:
        return out.iloc[:limit]
    return out


def read_range(base: str, asset: str, adj: Optional[str], ts_code: Optional[str], start: str, end: str,
               columns: Optional[List[str]] = None, limit: Optional[int] = None) -> pd.DataFrame:
    root = asset_root(base, asset, adj)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"目录不存在: {root}")
    
    all_dates = list_trade_dates(root)
    if not all_dates:
        raise FileNotFoundError(f"目录下没有任何 trade_date=* 分区: {root}")
    if HAS_DUCKDB:
        df = scan_with_duckdb(root, ts_code, start, end, columns, limit)
    else:
        df = scan_with_pandas(root, ts_code, start, end, columns, limit)

    if df is None or df.empty:
        raise FileNotFoundError(
            f"没有找到符合条件的数据文件: ts_code={ts_code}, start={start}, end={end}, root={root}"
        )
    return df


def read_day(base: str, asset: str, adj: Optional[str], day: str, limit: Optional[int] = None) -> pd.DataFrame:
    root = asset_root(base, asset, adj)
    pdir = os.path.join(root, f"trade_date={day}")
    files = glob.glob(os.path.join(pdir, "*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if limit is not None:
        if limit < 0:
            df = df.tail(-limit)
        else:
            df = df.head(limit)
    return df


def print_df(df: pd.DataFrame, limit: Optional[int] = None):
    if df is None or df.empty:
        print("（空结果）")
        return
    if limit is not None:
        if limit < 0:
            df = df.tail(-limit)
        else:
            df = df.head(limit)
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 180):
        print(df.to_string(index=False))


def show_schema(parquet_file: str):
    if not os.path.isfile(parquet_file):
        print(f"文件不存在：{parquet_file}")
        return
    try:
        if HAS_PYARROW:
            import pyarrow.parquet as pq  # type: ignore
            pf = pq.ParquetFile(parquet_file)
            md = pf.metadata
            print(f"File: {parquet_file}")
            print(f"Row Groups: {md.num_row_groups}, Rows: {md.num_rows}, Columns: {md.num_columns}")
            print("Schema:")
            for i in range(md.num_columns):
                col = md.schema.column(i)
                print(f"  - {col.name}: {col.physical_type}")
        else:
            # 退回 pandas
            df = pd.read_parquet(parquet_file)
            print(f"File: {parquet_file}")
            print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            print("Columns:", ", ".join(df.columns))
    except Exception as e:
        print(f"读取失败：{e}")


__all__ = [
    "HAS_DUCKDB", "HAS_PYARROW",
    "stock_daily_root", "allowed_stock_adjs", "normalize_stock_adj",
    "asset_root", "read_by_symbol", "list_symbols", "list_trade_dates",
    "date_chunks", "glob_partitions", "scan_with_duckdb", "scan_with_pandas",
    "read_range", "read_day", "print_df", "show_schema",
]
