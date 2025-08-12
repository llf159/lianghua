# -*- coding: utf-8 -*-
"""
parquet_viewer.py — 读取/浏览按 trade_date=YYYYMMDD 分区存储的 Parquet 行情数据

目录约定（与你的下载脚本一致）：
- 指数原始日线：data/index/daily/trade_date=YYYYMMDD/part-*.parquet
- 股票原始日线：data/stock/daily/trade_date=YYYYMMDD/part-*.parquet
- 股票前复权：  data/stock/daily_qfq/trade_date=YYYYMMDD/part-*.parquet
- 股票后复权：  data/stock/daily_hfq/trade_date=YYYYMMDD/part-*.parquet

功能（子命令）：
  info                        # 概览可用资产与日期范围
  dates   --asset ...         # 列出该资产（及复权）的所有分区日期
  day     --asset ... --date  # 查看某一天的所有记录（可选 --adj 与 --limit）
  show    --asset ... --ts ... --start ... --end ... [--adj ...] [--limit ...] [--columns ...]
  schema  --file PATH         # 查看某个 parquet 文件的字段与行数
  export  （同 show，但另加 --out CSV_PATH 输出为 CSV）

依赖：优先使用 duckdb（更快，推荐 pip install duckdb）；若未安装则退回 pandas+pyarrow。
示例：
  python parquet_viewer.py info
  python parquet_viewer.py dates --asset index
  python parquet_viewer.py day --asset index --date 20250802
  python parquet_viewer.py show --asset index --ts 000001.SH --start 20240101 --end 20250804 --limit 20
  python parquet_viewer.py show --asset stock --adj qfq --ts 600000.SH --start 20240101 --end 20250804
  python parquet_viewer.py export --asset stock --ts 600000.SH --start 20240101 --end 20250804 --out 600000_raw.csv

Windows 提示：命令行中文输出可在 PowerShell 中执行，或设置 chcp 65001。
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import datetime as dt
from typing import List, Optional, Iterable
import duckdb
HAS_DUCKDB = True
duckdb.sql("PRAGMA disable_progress_bar;") 
import pandas as pd

try:
    import pyarrow
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

# --------------------------- 工具函数 ---------------------------
def asset_root(base: str, asset: str, adj: str) -> str:
    if asset not in {"index", "stock"}:
        raise ValueError("asset 仅支持 index / stock")
    if asset == "index":
        sub = "daily"
        return os.path.join(base, asset, sub)

    allowed = {
        "daily","daily_qfq","daily_hfq",
        "daily_indicators","daily_qfq_indicators","daily_hfq_indicators",
    }
    if adj not in allowed:
        raise ValueError("stock 的 adj 仅支持: " + ", ".join(sorted(allowed)))

    nested = os.path.join(base, "stock", "daily", adj)
    return nested

def read_by_symbol(base: str, adj: str, ts_code: str, with_indicators: bool = True):
    sub = f"single_{adj}" + ("_indicators" if with_indicators else "")
    f = os.path.join(base, "stock", "single", sub, f"{ts_code}.parquet")
    if not os.path.isfile(f):
        raise FileNotFoundError(f)
    import pandas as pd
    return pd.read_parquet(f)

def list_symbols(root: str) -> List[str]:
    pattern = os.path.join(root, "trade_date=*/*.parquet").replace("\\", "/")
    try:
        df = duckdb.sql(f"SELECT DISTINCT ts_code FROM parquet_scan('{pattern}')").df()
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
    # pattern = os.path.join(root, "trade_date=*/part-*.parquet").replace("\\", "/")
    pattern = os.path.join(root, "trade_date=*/*.parquet").replace("\\", "/")
    sel = "*" if not columns else ", ".join(columns)
    where = [f"trade_date BETWEEN '{start}' AND '{end}'"]
    if ts_code:
        where.append(f"ts_code = '{ts_code}'")
    where_sql = " AND ".join(where)
    sql = f"SELECT {sel} FROM parquet_scan('{pattern}') WHERE {where_sql} ORDER BY trade_date"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return duckdb.sql(sql).df()  # type: ignore

def scan_with_pandas(root: str, ts_code: Optional[str], start: str, end: str,
                     columns: Optional[List[str]] = None, limit: Optional[int] = None) -> pd.DataFrame:
    parts = glob_partitions(root, start, end)
    frames: List[pd.DataFrame] = []
    remaining = limit if limit is not None else None

    for pdir in parts:
        # files = glob.glob(os.path.join(pdir, "part-*.parquet"))
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
    out = pd.concat(frames, ignore_index=True).sort_values("trade_date")
    if limit is not None:
        out = out.iloc[:limit]
    return out

def read_range(base: str, asset: str, adj: str, ts_code: Optional[str], start: str, end: str,
               columns: Optional[List[str]] = None, limit: Optional[int] = None) -> pd.DataFrame:
    root = asset_root(base, asset, adj if asset == "stock" else "daily")
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

def read_day(base: str, asset: str, adj: str, day: str, limit: Optional[int] = None) -> pd.DataFrame:
    root = asset_root(base, asset, adj if asset == "stock" else "daily")
    pdir = os.path.join(root, f"trade_date={day}")
    # files = glob.glob(os.path.join(pdir, "part-*.parquet"))
    files = glob.glob(os.path.join(pdir, "*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if limit is not None:
        df = df.head(limit)
    return df

def print_df(df: pd.DataFrame, limit: Optional[int] = None):
    if df is None or df.empty:
        print("（空结果）")
        return
    if limit is not None:
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

# --------------------------- CLI ---------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按 trade_date 分区的 Parquet 行情查看工具")
    parser.add_argument("--base", default="E:\\stock_data", help="数据根目录（默认: data）")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="概览可用资产与日期范围")
    # no extra args

    p_dates = sub.add_parser("dates", help="列出某资产的所有分区日期")
    p_dates.add_argument("--asset", required=True, choices=["index", "stock"], help="资产类型")
    p_dates.add_argument("--adj", default="daily", choices=["daily", "daily_qfq", "daily_hfq"],
                         help="stock 的复权目录；index 忽略（始终为 daily）")

    p_day = sub.add_parser("day", help="查看某一天的所有记录")
    p_day.add_argument("--asset", required=True, choices=["index", "stock"])
    p_day.add_argument("--adj", default="daily", choices=["daily", "daily_qfq", "daily_hfq"])
    p_day.add_argument("--date", required=True, help="交易日 YYYYMMDD")
    p_day.add_argument("--limit", type=int, default=None, help="最多显示多少行")

    p_show = sub.add_parser("show", help="查看某代码在一段时间内的记录")
    p_show.add_argument("--asset", required=True, choices=["index", "stock"])
    p_show.add_argument("--adj", default="daily", choices=["daily", "daily_qfq", "daily_hfq", "daily_indicators", "daily_qfq_indicators", "daily_hfq_indicators"])
    p_show.add_argument("--ts", required=True, help="ts_code，如 000001.SH 或 600000.SH")
    p_show.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    p_show.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    p_show.add_argument("--columns", default=None, help="仅显示这些列（逗号分隔）")
    p_show.add_argument("--limit", type=int, default=None, help="最多显示多少行")

    p_schema = sub.add_parser("schema", help="查看某个 parquet 文件的 schema/统计")
    p_schema.add_argument("--file", required=True, help="parquet 文件路径")

    p_export = sub.add_parser("export", help="导出到 CSV（参数同 show）")
    p_export.add_argument("--asset", required=True, choices=["index", "stock"])
    p_export.add_argument("--adj", default="daily", choices=["daily", "daily_qfq", "daily_hfq", "daily_indicators", "daily_qfq_indicators", "daily_hfq_indicators"])
    p_export.add_argument("--ts", required=True, help="ts_code")
    p_export.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    p_export.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    p_export.add_argument("--columns", default=None, help="仅导出这些列（逗号分隔）")
    p_export.add_argument("--out", required=True, help="输出 CSV 路径")

    args = parser.parse_args(argv)
    base = args.base

    if args.cmd == "info":
        entries = [
            ("index/daily", asset_root(base, "index", "daily")),
            ("stock/daily", asset_root(base, "stock", "daily")),
            ("stock/daily_qfq", asset_root(base, "stock", "daily_qfq")),
            ("stock/daily_hfq", asset_root(base, "stock", "daily_hfq")),
        ]
        print("数据目录：", os.path.abspath(base))
        print(f"DuckDB: {'可用' if HAS_DUCKDB else '不可用，建议 pip install duckdb'}")
        for name, root in entries:
            dates = list_trade_dates(root)
            if not dates:
                print(f"- {name:<16}（无数据）  路径: {root}")
                continue
            print(f"- {name:<16} {dates[0]} ~ {dates[-1]}  共 {len(dates)} 天  路径: {root}")
        return 0

    if args.cmd == "dates":
        root = asset_root(base, args.asset, args.adj)
        dates = list_trade_dates(root)
        if not dates:
            print("（无分区）")
        else:
            print(f"{args.asset} / {('daily' if args.asset=='index' else args.adj)}: {len(dates)} 天")
            print(", ".join(dates))
        return 0

    if args.cmd == "day":
        df = read_day(base, args.asset, args.adj, args.date, args.limit)
        print_df(df, args.limit)
        return 0

    if args.cmd == "show":
        cols = args.columns.split(",") if args.columns else None
        adj = args.adj if args.asset == "stock" else "daily"
        df = read_range(base, args.asset, adj, args.ts, args.start, args.end, cols, args.limit)
        print_df(df, args.limit)
        return 0

    if args.cmd == "schema":
        show_schema(args.file)
        return 0

    if args.cmd == "export":
        cols = args.columns.split(",") if args.columns else None
        adj = args.adj if args.asset == "stock" else "daily"
        df = read_range(base, args.asset, adj, args.ts, args.start, args.end, cols, None)
        if df is None or df.empty:
            print("（空结果，未生成 CSV）")
            return 0
        out = args.out
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"已导出：{os.path.abspath(out)}  行数={len(df)}")
        return 0

    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
