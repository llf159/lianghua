
# -*- coding: utf-8 -*-
"""
view_parquet.py — 独立的命令行入口

子命令：
  info / dates / day / show / schema / export
与原 test_parquet_viewer.py 完全一致，但实现复用其内部函数。
"""

from __future__ import annotations
import argparse
import os
from typing import List, Optional
import parquet_viewer as pv
import config

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按 trade_date 分区的 Parquet 行情查看工具（独立 CLI 版）")
    parser.add_argument("--base", default="E:\\stock_data", help="数据根目录（默认: E:\\stock_data）")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="概览可用资产与日期范围（自动扫描）")

    p_dates = sub.add_parser("dates", help="列出某资产的所有分区日期")
    p_dates.add_argument("--asset", required=True, choices=["index", "stock"], help="资产类型")
    p_dates.add_argument("--adj", default=config.PARQUET_ADJ, help="stock 的复权目录；支持 raw/qfq/hfq 或具体目录名（默认: qfq）")

    p_day = sub.add_parser("day", help="查看某一天的所有记录")
    p_day.add_argument("--asset", required=True, choices=["index", "stock"])
    p_day.add_argument("--adj", default=config.PARQUET_ADJ, help="stock 的复权目录；支持 raw/qfq/hfq 或具体目录名（默认: qfq）")
    p_day.add_argument("--date", required=True, help="交易日 YYYYMMDD")
    p_day.add_argument("--limit", type=int, default=None, help="最多显示多少行")

    p_show = sub.add_parser("show", help="查看某代码在一段时间内的记录")
    p_show.add_argument("--asset", required=True, choices=["index", "stock"])
    p_show.add_argument("--adj", default=config.PARQUET_ADJ, help="stock 的复权目录；支持 raw/qfq/hfq 或具体目录名、以及 *_indicators")
    p_show.add_argument("--ts", required=True, help="ts_code，如 000001.SH 或 600000.SH")
    p_show.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    p_show.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    p_show.add_argument("--columns", default=None, help="仅显示这些列（逗号分隔）")
    p_show.add_argument("--limit", type=int, default=None, help="最多显示多少行")

    p_schema = sub.add_parser("schema", help="查看某个 parquet 文件的 schema/统计")
    p_schema.add_argument("--file", required=True, help="parquet 文件路径")

    p_export = sub.add_parser("export", help="导出到 CSV（参数同 show）")
    p_export.add_argument("--asset", required=True, choices=["index", "stock"])
    p_export.add_argument("--adj", default=config.PARQUET_ADJ, help="stock 的复权目录；支持 raw/qfq/hfq 或具体目录名、以及 *_indicators")
    p_export.add_argument("--ts", required=True, help="ts_code")
    p_export.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    p_export.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    p_export.add_argument("--columns", default=None, help="仅导出这些列（逗号分隔）")
    p_export.add_argument("--out", required=True, help="输出 CSV 路径")

    args = parser.parse_args(argv)
    base = args.base

    if args.cmd == "info":
        print("数据目录：", os.path.abspath(base))
        # DuckDB 可用性信息来自被复用模块
        print(f"DuckDB: {'可用' if getattr(pv, 'HAS_DUCKDB', False) else '不可用，建议 pip install duckdb'}")

        # 指数
        idx_root = pv.asset_root(base, "index", "daily")
        idx_dates = pv.list_trade_dates(idx_root)
        if idx_dates:
            print(f"- index/daily         {idx_dates[0]} ~ {idx_dates[-1]}  共 {len(idx_dates)} 天  路径: {idx_root}")
        else:
            print(f"- index/daily         （无数据）  路径: {idx_root}")

        # 股票（动态列出每个 adj）
        sroot = pv.stock_daily_root(base)
        adjs = pv.allowed_stock_adjs(base)
        if not adjs:
            print(f"- stock/daily         （未发现任何子目录） 路径: {sroot}")
        else:
            for adj in adjs:
                r = os.path.join(sroot, adj)
                dates = pv.list_trade_dates(r)
                if dates:
                    print(f"- stock/{adj:<16} {dates[0]} ~ {dates[-1]}  共 {len(dates)} 天  路径: {r}")
                else:
                    print(f"- stock/{adj:<16} （无分区） 路径: {r}")
        return 0

    if args.cmd == "dates":
        adj = None if args.asset == "index" else args.adj
        root = pv.asset_root(base, args.asset, adj)
        dates = pv.list_trade_dates(root)
        if not dates:
            print("（无分区）")
        else:
            effective_adj = "daily" if args.asset == "index" else pv.normalize_stock_adj(base, args.adj, config.PARQUET_USE_INDICATORS)
            print(f"{args.asset} / {effective_adj}: {len(dates)} 天")
            print(", ".join(dates))
        return 0

    if args.cmd == "day":
        df = pv.read_day(base, args.asset, args.adj if args.asset == "stock" else "daily", args.date, args.limit)
        pv.print_df(df, args.limit)
        return 0

    if args.cmd == "show":
        cols = args.columns.split(",") if args.columns else None
        adj = args.adj if args.asset == "stock" else "daily"
        df = pv.read_range(base, args.asset, adj, args.ts, args.start, args.end, cols, args.limit)
        pv.print_df(df, args.limit)
        return 0

    if args.cmd == "schema":
        pv.show_schema(args.file)
        return 0

    if args.cmd == "export":
        cols = args.columns.split(",") if args.columns else None
        adj = args.adj if args.asset == "stock" else "daily"
        df = pv.read_range(base, args.asset, adj, args.ts, args.start, args.end, cols, None)
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
    import sys
    sys.exit(main())
