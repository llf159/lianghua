
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stock_data_selfcheck_viewer_plus.py
-----------------------------------
增强内容：
1) 指标列 NaN 比例统计（逐列），并基于阈值写入 issues；
2) 进度条（tqdm，若未安装则自动退化为简易计数打印）；
3) 新增导出 selfcheck_ind_nulls.csv，记录每个 ts_code 的每个指标列 NaN 比例。
"""
from __future__ import annotations

import os
import csv
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

import pandas as pd
import parquet_viewer as pv
import duckdb

# 进度条（可选）
try:
    from tqdm import tqdm
    def progress_iter(it, total, desc):
        return tqdm(it, total=total, desc=desc, unit="stk")
except Exception:
    def progress_iter(it, total, desc):
        print(f"[INFO] {desc} total={total}")
        idx = 0
        for x in it:
            idx += 1
            if idx % 200 == 0 or idx == total:
                print(f"[INFO] {desc} {idx}/{total}")
            yield x

def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    if df[date_col].dtype.kind in ("i", "u"):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col].astype(str))
    elif pd.api.types.is_string_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    return df

def _gap_count(sorted_dates: pd.Series) -> int:
    if len(sorted_dates) <= 1:
        return 0
    delta = sorted_dates.diff().dt.days.iloc[1:]
    return int((delta > 1).sum())

def _null_rate(s: pd.Series) -> float:
    n = len(s)
    if n == 0:
        return 0.0
    return float(s.isna().sum()) / float(n)

def _write_csv_rows(path, rows: List[Dict]):
    if not rows:
        return
    cols = sorted({k for d in rows for k in d.keys()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

def _infer_roots_and_range(base_: str, asset_: str, adj_: str, start_: Optional[str], end_: Optional[str]):
    adj_plain = (adj_ if asset_ == "stock" else "daily")
    rp = pv.asset_root(base_, asset_, adj_plain)
    if not os.path.isdir(rp):
        raise FileNotFoundError(f"目录不存在: {rp}")
    all_dates = pv.list_trade_dates(rp)
    if not all_dates:
        raise FileNotFoundError(f"没有任何 trade_date=* 分区: {rp}")
    s = start_ or all_dates[0]
    e = end_ or all_dates[-1]
    ri = None
    if asset_ == "stock" and not adj_plain.endswith("_indicators"):
        cand = adj_plain + "_indicators"
        try:
            tmp = pv.asset_root(base_, "stock", cand)
            if os.path.isdir(tmp):
                ri = tmp
        except Exception:
            ri = None
    return rp, ri, s, e

def _indicator_cols_hint_from_sample(base_: str, root_ind_: Optional[str], date_col_: str, s_: str, e_: str) -> List[str]:
    if not root_ind_:
        return []
    try:
        adj_ind = os.path.basename(root_ind_)
        df_any = pv.read_range(base=base_, asset="stock", adj=adj_ind, ts_code=None,
                               start=s_, end=e_, columns=None, limit=2000)
        return [c for c in df_any.columns if c not in ("ts_code", date_col_, "open","high","low","close","vol","amount")]
    except Exception:
        return []

def run_bulk_pandas(
    *,
    base: str,
    asset: str,
    adj: str,
    date_col: str,
    start: Optional[str],
    end: Optional[str],
    root_plain: Optional[str],
    root_ind: Optional[str],
    key_cols: List[str],
    out_dir: str,
    ind_null_threshold: float,
) -> bool:
    """
    纯 pandas 批量自检：
    - 仅读取必要列（动态推断），对 plain 分组统计（行数、起止日期、排序、重复日、缺口、关键列空值率）。
    - 若存在 *indicators 目录，则对指标列做空值率/列质量检查，并与 plain 做尾部/列一致性对比。
    - 输出 selfcheck_details.csv / selfcheck_issues.csv / selfcheck_ind_nulls.csv
    """
    import os, csv
    from dataclasses import asdict
    import pandas as pd

    try:
        root_plain2, root_ind2, start2, end2 = _infer_roots_and_range(base, asset, adj, start, end)
        root_plain = root_plain or root_plain2
        root_ind = root_ind or root_ind2
        start, end = start or start2, end or end2

        # 动态必要列（plain）
        cols_plain = ["ts_code", date_col] + [c for c in key_cols if c not in (date_col, "ts_code")]
        cols_plain = list(dict.fromkeys(cols_plain))

        # 优先用 parquet_viewer 的扫描器
        try:
            df_plain = pv.scan_with_pandas(root_plain, ts_code=None, start=start, end=end, columns=cols_plain, limit=None)
        except Exception:
            # 兜底：分区逐日拼接
            parts = pv.glob_partitions(root_plain, start, end)
            frames = []
            for p in parts:
                for f in os.listdir(p):
                    if f.endswith(".parquet"):
                        try:
                            frames.append(pd.read_parquet(os.path.join(p,f), columns=[c for c in cols_plain if c != "ts_code"] + ["ts_code"]))
                        except Exception:
                            continue
            df_plain = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols_plain)

        if df_plain.empty:
            print("[WARN] bulk-pandas 未读到任何 plain 数据")
            return False

        # 规范日期为 datetime
        df_plain = df_plain.copy()
        if df_plain[date_col].dtype.kind in ("i","u"):
            df_plain[date_col] = pd.to_datetime(df_plain[date_col].astype(str))
        elif pd.api.types.is_string_dtype(df_plain[date_col]):
            df_plain[date_col] = pd.to_datetime(df_plain[date_col].astype(str), errors="coerce")

        details: List[Dict] = []
        issues: List[Dict] = []
        ind_null_rows: List[Dict] = []

        # plain 逐票聚合
        def _agg_group(g: pd.DataFrame, ts: str) -> FileStat:
            g = g.sort_values(date_col)
            rows = len(g)
            if rows == 0:
                return FileStat("plain", ts, 0, None, None, True, 0, 0, 0.0, ",".join(g.columns))
            date_min = pd.to_datetime(g[date_col]).min()
            date_max = pd.to_datetime(g[date_col]).max()
            is_sorted = g[date_col].is_monotonic_increasing
            dup_cnt = int(g.duplicated(subset=[date_col]).sum())
            gaps = _gap_count(pd.to_datetime(g[date_col]).sort_values())
            key_cols_existing = [c for c in key_cols if c in g.columns]
            key_null = 0.0
            if key_cols_existing:
                key_null = float(sum(_null_rate(g[c]) for c in key_cols_existing)) / float(len(key_cols_existing))
            return FileStat("plain", ts, rows, date_min, date_max, bool(is_sorted), dup_cnt, gaps, key_null, ",".join(g.columns))

        for ts, g in df_plain.groupby("ts_code"):
            stp = _agg_group(g, ts)
            details.append(asdict(stp))
            probs = []
            if not stp.is_sorted: probs.append("NOT_SORTED")
            if stp.dup_count > 0: probs.append(f"DUP_DATES={stp.dup_count}")
            if stp.gap_count > 0: probs.append(f"GAPS~{stp.gap_count}")
            if stp.key_null_rate > 0.2: probs.append(f"HIGH_NULL={stp.key_null_rate:.2%}")
            if stp.rows < 50: probs.append(f"TOO_FEW_ROWS={stp.rows}")
            if probs:
                issues.append({"ts_code": ts, "type": "plain", "path": root_plain, "issue": ";".join(probs)})

        # 指标目录：列提示 + 明细与对比
        indicator_cols_hint: List[str] = _indicator_cols_hint_from_sample(base, root_ind, date_col, start, end)
        df_ind = pd.DataFrame()
        if root_ind:
            cols_ind = ["ts_code", date_col] + indicator_cols_hint
            cols_ind = list(dict.fromkeys(cols_ind))
            try:
                df_ind = pv.scan_with_pandas(root_ind, ts_code=None, start=start, end=end, columns=cols_ind, limit=None)
            except Exception:
                df_ind = pd.DataFrame(columns=cols_ind)

            if not df_ind.empty:
                if df_ind[date_col].dtype.kind in ("i","u"):
                    df_ind[date_col] = pd.to_datetime(df_ind[date_col].astype(str))
                elif pd.api.types.is_string_dtype(df_ind[date_col]):
                    df_ind[date_col] = pd.to_datetime(df_ind[date_col].astype(str), errors="coerce")

                for ts, gi in df_ind.groupby("ts_code"):
                    gp = df_plain[df_plain["ts_code"] == ts]
                    st_ind = basic_stat(gi, key_cols, typ="ind", ts_code=ts, date_col=date_col)
                    details.append(asdict(st_ind))

                    if len(gi) > 0 and indicator_cols_hint:
                        for c in indicator_cols_hint:
                            if c in gi.columns:
                                rate = _null_rate(gi[c])
                                ind_null_rows.append({
                                    "ts_code": ts, "column": c,
                                    "null_rate": f"{rate:.6f}",
                                    "rows": len(gi),
                                    "non_null": int(len(gi) - int(gi[c].isna().sum()))
                                })
                                if rate > ind_null_threshold:
                                    issues.append({
                                        "ts_code": ts, "type": "ind_null", "path": root_ind,
                                        "issue": f"IND_NULL_RATE {c}={rate:.2%} > {ind_null_threshold:.0%}"
                                    })

                    pc = pair_compare_df(gp, gi, date_col, indicator_cols_hint, ts=ts)
                    details.append({"typ": "pair", **asdict(pc)})
                    prob = []
                    if not pc.tail_align: prob.append("TAIL_NOT_ALIGNED")
                    if pc.plain_only_tail_days > 0: prob.append(f"PLAIN_TAIL_ONLY_DAYS={pc.plain_only_tail_days}")
                    if pc.ind_only_tail_days > 0: prob.append(f"IND_TAIL_ONLY_DAYS={pc.ind_only_tail_days}")
                    if pc.missing_ind_columns: prob.append(f"MISSING_IND={pc.missing_ind_columns}")
                    if pc.bad_ind_columns: prob.append(f"BAD_IND={pc.bad_ind_columns}")
                    if pc.overlap_days == 0 and (st_ind.rows > 0 and (ts in df_plain['ts_code'].values)):
                        prob.append("NO_OVERLAP")
                    if prob:
                        issues.append({"ts_code": ts, "type": "pair", "path": root_ind, "issue": ";".join(prob)})

        # 写出
        os.makedirs(out_dir, exist_ok=True)
        if details:
            cols = sorted({c for d in details for c in d.keys()})
            with open(os.path.join(out_dir, "selfcheck_details.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(details)
        if ind_null_rows:
            cols = sorted({c for d in ind_null_rows for c in d.keys()})
            with open(os.path.join(out_dir, "selfcheck_ind_nulls.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(ind_null_rows)
        if issues:
            cols = sorted({c for d in issues for c in d.keys()})
            with open(os.path.join(out_dir, "selfcheck_issues.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(issues)

        return True
    except Exception as e:
        print(f"[ERROR] run_bulk_pandas 失败: {e}")
        return False

def run_bulk_duckdb(
    *,
    base: str,
    asset: str,
    adj: str,
    date_col: str,
    start: Optional[str],
    end: Optional[str],
    root_plain: Optional[str],
    root_ind: Optional[str],
    key_cols: List[str],
    out_dir: str,
    ind_null_threshold: float,
) -> bool:
    """
    DuckDB SQL 聚合自检：
    - 用 parquet_scan + 窗口函数 一次性统计每票（rows/min/max/dup/gaps/is_sorted/key_null_rate）。
    - 指标端用 SQL 聚合做 ALL_NA/ALL_ZERO/CONST 与空值率，并和 plain 做尾部/列一致性对比。
    """
    import os, csv
    from dataclasses import asdict
    import pandas as pd
    import duckdb

    try:
        root_plain2, root_ind2, start2, end2 = _infer_roots_and_range(base, asset, adj, start, end)
        root_plain = root_plain or root_plain2
        root_ind = root_ind or root_ind2
        start, end = start or start2, end or end2

        pattern_plain = os.path.join(root_plain, "trade_date=*/*.parquet").replace("\\","/")
        dcol = date_col

        # --- plain 端：一条 SQL 聚合 ---
        cols_sql = ", ".join([
            f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)::INTEGER AS null_{c}"
            for c in key_cols if c not in (dcol, "ts_code")
        ])

        sql_plain = f"""
        WITH src AS (
            SELECT ts_code,
                   to_date({dcol}, '%Y%m%d') AS d,
                   {", ".join([c for c in set(key_cols) if c not in ('ts_code', dcol)])}
            FROM parquet_scan('{pattern_plain}')
            WHERE {dcol} BETWEEN '{start}' AND '{end}'
        ),
        ord AS (
            SELECT *, LAG(d) OVER (PARTITION BY ts_code ORDER BY d) AS pd
            FROM src
        ),
        agg AS (
            SELECT
                ts_code,
                COUNT(*) AS rows,
                MIN(d) AS date_min,
                MAX(d) AS date_max,
                SUM(CASE WHEN pd IS NOT NULL AND d < pd THEN 1 ELSE 0 END) AS desc_cnt,
                SUM(CASE WHEN pd IS NOT NULL AND DATE_DIFF('day', pd, d) > 1 THEN 1 ELSE 0 END) AS gap_count,
                COUNT(*) - COUNT(DISTINCT d) AS dup_count
                {("," + cols_sql) if cols_sql else ""}
            FROM ord
            GROUP BY ts_code
        )
        SELECT * FROM agg
        ORDER BY ts_code
        """
        df_plain = duckdb.sql(sql_plain).df()

        if df_plain.empty:
            print("[WARN] bulk-duckdb 未读到 plain 数据")
            return False

        # 关键列空值率
        null_cols = [c for c in df_plain.columns if c.startswith("null_")]
        if null_cols:
            df_plain["key_null_rate"] = df_plain[null_cols].sum(axis=1) / (df_plain["rows"] * len(null_cols))
        else:
            df_plain["key_null_rate"] = 0.0

        details: List[Dict] = []
        issues: List[Dict] = []

        for _, r in df_plain.iterrows():
            ts = str(r["ts_code"])
            st = FileStat(
                typ="plain",
                ts_code=ts,
                rows=int(r["rows"]),
                date_min=pd.to_datetime(r["date_min"]) if pd.notna(r["date_min"]) else None,
                date_max=pd.to_datetime(r["date_max"]) if pd.notna(r["date_max"]) else None,
                is_sorted=bool(int(r["desc_cnt"]) == 0),
                dup_count=int(r["dup_count"]),
                gap_count=int(r["gap_count"]),
                key_null_rate=float(r.get("key_null_rate", 0.0)),
                columns="ts_code," + dcol + "," + ",".join([c for c in key_cols if c not in ("ts_code", dcol)]),
            )
            details.append(asdict(st))
            probs = []
            if not st.is_sorted: probs.append("NOT_SORTED")
            if st.dup_count > 0: probs.append(f"DUP_DATES={st.dup_count}")
            if st.gap_count > 0: probs.append(f"GAPS~{st.gap_count}")
            if st.key_null_rate > 0.2: probs.append(f"HIGH_NULL={st.key_null_rate:.2%}")
            if st.rows < 50: probs.append(f"TOO_FEW_ROWS={st.rows}")
            if probs:
                issues.append({"ts_code": ts, "type": "plain", "path": root_plain, "issue": ";".join(probs)})

        # --- 指标端：列提示 + SQL 聚合 ---
        indicator_cols_hint: List[str] = _indicator_cols_hint_from_sample(base, root_ind, date_col, start, end)
        if root_ind and indicator_cols_hint:
            pattern_ind = os.path.join(root_ind, "trade_date=*/*.parquet").replace("\\","/")

            # 列极值/去重数用于判定 ALL_NA/ALL_ZERO/CONST
            sel_cols = ", ".join([
                f"MIN({c}) AS {c}_min, MAX({c}) AS {c}_max, COUNT({c}) AS {c}_nn, COUNT(DISTINCT {c}) AS {c}_nd"
                for c in indicator_cols_hint
            ])
            sql_ind_agg = f"""
            WITH src AS (
                SELECT ts_code, to_date({dcol}, '%Y%m%d') AS d, {", ".join(indicator_cols_hint)}
                FROM parquet_scan('{pattern_ind}')
                WHERE {dcol} BETWEEN '{start}' AND '{end}'
            )
            SELECT ts_code,
                   COUNT(*) AS ind_rows,
                   MIN(d) AS ind_min,
                   MAX(d) AS ind_max,
                   {sel_cols}
            FROM src
            GROUP BY ts_code
            ORDER BY ts_code
            """
            df_ind = duckdb.sql(sql_ind_agg).df()

            # 每列空值率
            sel_nulls = ", ".join([f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS null_{c}" for c in indicator_cols_hint])
            sql_ind_nulls = f"""
            SELECT ts_code, {sel_nulls}, COUNT(*) AS rows
            FROM parquet_scan('{pattern_ind}')
            WHERE {dcol} BETWEEN '{start}' AND '{end}'
            GROUP BY ts_code
            ORDER BY ts_code
            """
            df_nulls = duckdb.sql(sql_ind_nulls).df()

            ind_null_rows: List[Dict] = []
            for _, row in df_ind.iterrows():
                ts = str(row["ts_code"])
                ind_rows = int(row["ind_rows"])

                p = df_plain[df_plain["ts_code"] == ts]
                pmax = p["date_max"].iloc[0] if not p.empty else pd.NaT

                st_ind = FileStat(
                    typ="ind",
                    ts_code=ts,
                    rows=ind_rows,
                    date_min=pd.to_datetime(row["ind_min"]) if pd.notna(row["ind_min"]) else None,
                    date_max=pd.to_datetime(row["ind_max"]) if pd.notna(row["ind_max"]) else None,
                    is_sorted=True,
                    dup_count=0,
                    gap_count=0,
                    key_null_rate=0.0,
                    columns=",".join(indicator_cols_hint),
                )
                details.append(asdict(st_ind))

                imax = st_ind.date_max if st_ind.date_max is not None else pd.NaT
                tail_align = (pd.notna(pmax) and pd.notna(imax) and pd.to_datetime(pmax) == pd.to_datetime(imax))
                ind_only = int((pd.to_datetime(imax) - pd.to_datetime(min(imax, pmax))).days) if pd.notna(imax) and pd.notna(pmax) else 0
                plain_only = int((pd.to_datetime(pmax) - pd.to_datetime(min(imax, pmax))).days) if pd.notna(imax) and pd.notna(pmax) else 0

                # 列质量 + 空值率
                bad_cols, miss_cols = [], []
                for c in indicator_cols_hint:
                    nn = int(row.get(f"{c}_nn", 0))
                    nd = int(row.get(f"{c}_nd", 0))
                    vmin = row.get(f"{c}_min", None)
                    vmax = row.get(f"{c}_max", None)
                    # 替换 df_ind.columns 的检查为基于 df_nulls 列名的检查
                    has_c = (f"null_{c}" in df_nulls.columns)
                    if not has_c:
                        miss_cols.append(c)
                        continue

                    if nn == 0:
                        bad_cols.append(f"{c}:ALL_NA")
                    elif vmin == 0 and vmax == 0:
                        bad_cols.append(f"{c}:ALL_ZERO")
                    elif nd == 1 and nn > 1:
                        bad_cols.append(f"{c}:CONST")

                rnull = df_nulls[df_nulls["ts_code"] == ts]
                if not rnull.empty:
                    rows_total = int(rnull["rows"].iloc[0])
                    for c in indicator_cols_hint:
                        nnull = int(rnull[f"null_{c}"].iloc[0]) if f"null_{c}" in rnull.columns else 0
                        rate = (nnull / rows_total) if rows_total > 0 else 0.0
                        ind_null_rows.append({
                            "ts_code": ts, "column": c, "null_rate": f"{rate:.6f}",
                            "rows": rows_total, "non_null": rows_total - nnull
                        })
                        if rate > ind_null_threshold:
                            issues.append({
                                "ts_code": ts, "type": "ind_null", "path": root_ind,
                                "issue": f"IND_NULL_RATE {c}={rate:.2%} > {ind_null_threshold:.0%}"
                            })

                # pair 汇总行
                details.append({"typ": "pair", **asdict(PairCheck(
                    ts_code=ts,
                    plain_rows=int(p["rows"].iloc[0]) if not p.empty else 0,
                    ind_rows=ind_rows,
                    overlap_days=0,  # 如需可再加 DISTINCT 交集 SQL
                    tail_align=bool(tail_align),
                    ind_only_tail_days=int(ind_only),
                    plain_only_tail_days=int(plain_only),
                    missing_ind_columns=",".join(miss_cols),
                    bad_ind_columns=",".join(bad_cols),
                ))})

                prob = []
                if not tail_align: prob.append("TAIL_NOT_ALIGNED")
                if plain_only > 0: prob.append(f"PLAIN_TAIL_ONLY_DAYS={plain_only}")
                if ind_only > 0: prob.append(f"IND_TAIL_ONLY_DAYS={ind_only}")
                if miss_cols: prob.append(f"MISSING_IND={','.join(miss_cols)}")
                if bad_cols: prob.append(f"BAD_IND={','.join(bad_cols)}")
                if prob:
                    issues.append({"ts_code": ts, "type": "pair", "path": root_ind, "issue": ";".join(prob)})

            # 写出指标空值明细
            if ind_null_rows:
                os.makedirs(out_dir, exist_ok=True)
                cols = sorted({c for d in ind_null_rows for c in d.keys()})
                with open(os.path.join(out_dir, "selfcheck_ind_nulls.csv"), "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(ind_null_rows)

        # 统一写出
        os.makedirs(out_dir, exist_ok=True)
        if details:
            cols = sorted({c for d in details for c in d.keys()})
            with open(os.path.join(out_dir, "selfcheck_details.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(details)
        if issues:
            cols = sorted({c for d in issues for c in d.keys()})
            with open(os.path.join(out_dir, "selfcheck_issues.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(issues)

        return True
    except Exception as e:
        print(f"[ERROR] run_bulk_duckdb 失败: {e}")
        return False


@dataclass
class FileStat:
    typ: str
    ts_code: str
    rows: int
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    is_sorted: bool
    dup_count: int
    gap_count: int
    key_null_rate: float
    columns: str

@dataclass
class PairCheck:
    ts_code: str
    plain_rows: int
    ind_rows: int
    overlap_days: int
    tail_align: bool
    ind_only_tail_days: int
    plain_only_tail_days: int
    missing_ind_columns: str
    bad_ind_columns: str

def basic_stat(df: pd.DataFrame, key_cols: List[str], typ: str, ts_code: str, date_col: str) -> FileStat:
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"缺少日期列 {date_col}")
    df = _ensure_datetime(df, date_col)
    rows = len(df)
    if rows == 0:
        return FileStat(typ, ts_code, 0, None, None, True, 0, 0, 0.0, ",".join(df.columns))
    date_min = pd.to_datetime(df[date_col]).min()
    date_max = pd.to_datetime(df[date_col]).max()
    is_sorted = df[date_col].is_monotonic_increasing
    dup_cnt = int(df.duplicated(subset=[date_col]).sum())
    gaps = _gap_count(pd.to_datetime(df[date_col]).sort_values())
    key_cols_existing = [c for c in key_cols if c in df.columns]
    key_null = 0.0
    if key_cols_existing:
        key_null = float(sum(_null_rate(df[c]) for c in key_cols_existing)) / float(len(key_cols_existing))
    return FileStat(typ, ts_code, rows, date_min, date_max, bool(is_sorted), dup_cnt, gaps, key_null, ",".join(df.columns))

def pair_compare_df(dp: pd.DataFrame, di: pd.DataFrame, date_col: str, indicator_cols_hint: List[str], ts: str) -> PairCheck:
    dp = _ensure_datetime(dp, date_col)
    di = _ensure_datetime(di, date_col)
    p_dates = pd.to_datetime(dp[date_col]) if date_col in dp.columns else pd.Series([], dtype="datetime64[ns]")
    i_dates = pd.to_datetime(di[date_col]) if date_col in di.columns else pd.Series([], dtype="datetime64[ns]")
    pmax = p_dates.max() if len(p_dates) else pd.NaT
    imax = i_dates.max() if len(i_dates) else pd.NaT
    overlap = len(pd.Index(p_dates.unique()).intersection(pd.Index(i_dates.unique())))
    tail_align = (pmax == imax) and pd.notna(pmax) and pd.notna(imax)
    ind_only_tail_days = int(max(0, (imax - min(imax, pmax)).days)) if pd.notna(imax) and pd.notna(pmax) else 0
    plain_only_tail_days = int(max(0, (pmax - min(imax, pmax)).days)) if pd.notna(imax) and pd.notna(pmax) else 0

    miss_cols = []
    bad_cols = []
    for c in indicator_cols_hint:
        if c not in di.columns:
            miss_cols.append(c)
        else:
            s = di[c]
            if s.isna().all():
                bad_cols.append(f"{c}:ALL_NA")
            elif (s == 0).all():
                bad_cols.append(f"{c}:ALL_ZERO")
            elif s.nunique(dropna=True) == 1 and len(s) > 1:
                bad_cols.append(f"{c}:CONST")

    return PairCheck(
        ts_code=ts,
        plain_rows=len(dp),
        ind_rows=len(di),
        overlap_days=int(overlap),
        tail_align=bool(tail_align),
        ind_only_tail_days=int(ind_only_tail_days),
        plain_only_tail_days=int(plain_only_tail_days),
        missing_ind_columns=",".join(miss_cols),
        bad_ind_columns=",".join(bad_cols)
    )

def main():
    ap = argparse.ArgumentParser(description="Stock Data Self-Check (viewer-based, indicator NaN rates + progress)")
    ap.add_argument("--base", default="E:/stock_data")
    ap.add_argument("--asset", choices=["stock", "index"], default="stock")
    ap.add_argument("--adj", default="daily_qfq",
                    choices=["daily","daily_qfq","daily_hfq","daily_indicators","daily_qfq_indicators","daily_hfq_indicators"])
    ap.add_argument("--date-col", default="trade_date")
    ap.add_argument("--key-cols", default="open,high,low,close,vol")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="./output/selfcheck")
    ap.add_argument("--skip-pair", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ind-null-threshold", type=float, default=0.2, help="指标列 NaN 比例阈值（默认 0.2）")
    ap.add_argument("--mode", required=True, choices=["loop", "pandas", "duckdb"], default="loop",
                help="自检执行模式：loop=逐票(原逻辑)，bulk-pandas=全量入内存分组，bulk-duckdb=DuckDB 扫描")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    details_path = os.path.join(args.out, "selfcheck_details.csv")
    issues_path = os.path.join(args.out, "selfcheck_issues.csv")
    ind_nulls_path = os.path.join(args.out, "selfcheck_ind_nulls.csv")

    date_col = args.date_col
    key_cols = [c.strip() for c in args.key_cols.split(",") if c.strip()]

    start = None
    end = None
    root_plain = None
    root_ind = None
    if args.mode in ("pandas", "duckdb"):
        if args.mode == "duckdb":
            ok = run_bulk_duckdb(
                base=args.base, asset=args.asset, adj=args.adj, date_col=date_col,
                start=args.start, end=args.end, root_plain=None, root_ind=None,
                key_cols=key_cols, out_dir=args.out, ind_null_threshold=args.ind_null_threshold
            )
            if ok:
                return
            print("[WARN] duckdb 失败，自动回退到 pandas")

        run_bulk_pandas(
            base=args.base, asset=args.asset, adj=args.adj, date_col=date_col,
            start=args.start, end=args.end, root_plain=None, root_ind=None,
            key_cols=key_cols, out_dir=args.out, ind_null_threshold=args.ind_null_threshold
        )
        return

    root_plain = pv.asset_root(args.base, args.asset, (args.adj if args.asset == "stock" else "daily"))
    if not os.path.isdir(root_plain):
        raise FileNotFoundError(f"目录不存在: {root_plain}")

    root_ind = None
    if args.asset == "stock" and not args.skip_pair:
        if not args.adj.endswith("_indicators"):
            cand = args.adj + "_indicators"
            try:
                tmp = pv.asset_root(args.base, "stock", cand)
                if os.path.isdir(tmp):
                    root_ind = tmp
            except Exception:
                root_ind = None

    all_dates = pv.list_trade_dates(root_plain)
    if not all_dates:
        raise FileNotFoundError(f"目录下没有任何 trade_date=* 分区: {root_plain}")
    start = args.start or all_dates[0]
    end = args.end or all_dates[-1]

    symbols = pv.list_symbols(root_plain)
    if args.symbols.strip():
        allow = set(s.strip() for s in args.symbols.split(",") if s.strip())
        symbols = [s for s in symbols if s in allow]

    details: List[Dict] = []
    issues: List[Dict] = []
    ind_null_rows: List[Dict] = []

    indicator_cols_hint: List[str] = []
    if root_ind:
        adj_ind = os.path.basename(root_ind)
        df_any = pv.read_range(base=args.base, asset="stock", adj=adj_ind, ts_code=None,
                               start=start, end=end, columns=None, limit=2000)
        indicator_cols_hint = [c for c in df_any.columns
                               if c not in ("ts_code", date_col, "open","high","low","close","vol","amount")]

    for ts in progress_iter(symbols, total=len(symbols), desc="SelfCheck"):
        adj_plain = os.path.basename(root_plain)
        df_plain = pv.read_range(base=args.base, asset=args.asset,
                                 adj=(adj_plain if args.asset == "stock" else "daily"),
                                 ts_code=ts, start=start, end=end, columns=None, limit=None)
        st_plain = basic_stat(df_plain, key_cols, typ="plain", ts_code=ts, date_col=date_col)
        details.append(asdict(st_plain))

        basic_problems = []
        if not st_plain.is_sorted:
            basic_problems.append("NOT_SORTED")
        if st_plain.dup_count > 0:
            basic_problems.append(f"DUP_DATES={st_plain.dup_count}")
        if st_plain.gap_count > 0:
            basic_problems.append(f"GAPS~{st_plain.gap_count}")
        if st_plain.key_null_rate > 0.2:
            basic_problems.append(f"HIGH_NULL={st_plain.key_null_rate:.2%}")
        if st_plain.rows < 50:
            basic_problems.append(f"TOO_FEW_ROWS={st_plain.rows}")
        if basic_problems:
            issues.append({"ts_code": ts, "type": "plain", "path": root_plain, "issue": ";".join(basic_problems)})

        if root_ind:
            adj_ind = os.path.basename(root_ind)
            df_ind = pv.read_range(base=args.base, asset="stock", adj=adj_ind,
                                   ts_code=ts, start=start, end=end, columns=None, limit=None)
            st_ind = basic_stat(df_ind, key_cols, typ="ind", ts_code=ts, date_col=date_col)
            details.append(asdict(st_ind))

            if len(df_ind) > 0 and indicator_cols_hint:
                for c in indicator_cols_hint:
                    if c in df_ind.columns:
                        rate = _null_rate(df_ind[c])
                        ind_null_rows.append({
                            "ts_code": ts,
                            "column": c,
                            "null_rate": f"{rate:.6f}",
                            "rows": len(df_ind),
                            "non_null": int(len(df_ind) - int(df_ind[c].isna().sum()))
                        })
                        if rate > args.ind_null_threshold:
                            issues.append({
                                "ts_code": ts,
                                "type": "ind_null",
                                "path": root_ind,
                                "issue": f"IND_NULL_RATE {c}={rate:.2%} > {args.ind_null_threshold:.0%}"
                            })

            pc = pair_compare_df(df_plain, df_ind, date_col, indicator_cols_hint, ts=ts)
            details.append({"typ": "pair", **asdict(pc)})

            prob = []
            if not pc.tail_align:
                prob.append("TAIL_NOT_ALIGNED")
            if pc.plain_only_tail_days > 0:
                prob.append(f"PLAIN_TAIL_ONLY_DAYS={pc.plain_only_tail_days}")
            if pc.ind_only_tail_days > 0:
                prob.append(f"IND_TAIL_ONLY_DAYS={pc.ind_only_tail_days}")
            if pc.missing_ind_columns:
                prob.append(f"MISSING_IND={pc.missing_ind_columns}")
            if pc.bad_ind_columns:
                prob.append(f"BAD_IND={pc.bad_ind_columns}")
            if pc.overlap_days == 0 and (st_plain.rows > 0 and st_ind.rows > 0):
                prob.append("NO_OVERLAP")
            if prob:
                issues.append({"ts_code": ts, "type": "pair", "path": root_ind, "issue": ";".join(prob)})

    if details:
        cols = sorted({c for d in details for c in d.keys()})
        with open(details_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for d in details:
                w.writerow(d)
        print(f"[OK] 明细: {details_path}  (rows={len(details)})")
    else:
        print("[WARN] 无明细数据可写")

    if ind_null_rows:
        cols = sorted({c for d in ind_null_rows for c in d.keys()})
        with open(ind_nulls_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for d in ind_null_rows:
                w.writerow(d)
        print(f"[OK] 指标NaN明细: {ind_nulls_path}  (rows={len(ind_null_rows)})")
    else:
        print("[OK] 未发现指标 NaN 明细（或无指标列/已跳过 pair 检查）。")

    if issues:
        cols = sorted({c for d in issues for c in d.keys()})
        with open(issues_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for d in issues:
                w.writerow(d)
        print(f"[OK] 问题: {issues_path}  (rows={len(issues)})")
    else:
        print("[OK] 未发现问题项，issues 为空。")

if __name__ == "__main__":
    main()
