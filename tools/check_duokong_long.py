#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick duokong_long validator.
Loads price/indicator data from the unified DuckDB, recomputes duokong_long
from raw closes (MA14+MA28+MA57+MA114)/4, and reports any mismatches.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_ROOT, UNIFIED_DB_PATH


def recompute_duokong_long(df: pd.DataFrame) -> pd.Series:
    """Recompute duokong_long from close prices."""
    c = df["close"].astype(float)
    ma14 = c.rolling(14).mean()
    ma28 = c.rolling(28).mean()
    ma57 = c.rolling(57).mean()
    ma114 = c.rolling(114).mean()
    return ((ma14 + ma28 + ma57 + ma114) / 4).round(2)


def main():
    ap = argparse.ArgumentParser(description="Check duokong_long values against recomputed results.")
    ap.add_argument("--ts-code", default="000001.SZ", help="Stock code, default: 000001.SZ")
    ap.add_argument("--adj", default="qfq", help="Adj type to check, default: qfq")
    args = ap.parse_args()

    db_path = Path(DATA_ROOT) / UNIFIED_DB_PATH
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)
    sql = """
        SELECT trade_date, close, duokong_long
        FROM stock_data
        WHERE ts_code = ? AND adj_type = ?
        ORDER BY trade_date
    """
    df = con.execute(sql, [args.ts_code, args.adj]).fetchdf()
    if df.empty:
        raise SystemExit(f"No data for {args.ts_code} / {args.adj}")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["close"] = df["close"].astype(float)
    df["duokong_long"] = df["duokong_long"].astype(float)

    df["duokong_calc"] = recompute_duokong_long(df)
    df["diff"] = df["duokong_long"] - df["duokong_calc"]

    mismatches = df[df["diff"].abs() > 0]
    if mismatches.empty:
        print(f"All duokong_long values match for {args.ts_code} ({len(df)} rows).")
        return

    print(f"Found {len(mismatches)} mismatches for {args.ts_code}:")
    print(mismatches[["trade_date", "duokong_long", "duokong_calc", "diff"]].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
