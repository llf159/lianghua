# -*- coding: utf-8 -*-
"""
stock_selector.py — 读取 {DATA_ROOT}/stock/daily(或*复权) 最新 trade_date 分区，
用 TDX 兼容层脚本进行选股，输出当日候选清单。

依赖：
  - duckdb（推荐）或 pandas+pyarrow
  - 本目录下的 tdx_compat.py（你已提供）

用法示例：
  python stock_selector_tdx.py --base E:\\stock_data --adj daily \
      --top 200 --out E:\\stock_data\\selections --show 20

  # 自定义 TDX 规则（行内）
  python stock_selector_tdx.py --rule "OC:=SAFE_DIV(C-O,O)*100; VOLR:=SAFE_DIV(V,MA(V,20)); BUY:=(OC>=2) AND (VOLR>=1.5);" --show 50

  # 从文件加载规则
  python stock_selector_tdx.py --rule-file my_rule.tdx --adj daily_qfq

默认规则（相对保守的放量突破）：
  OC := SAFE_DIV(C - O, O) * 100;         # 当日涨幅(相对开盘)
  VOLR := SAFE_DIV(V, MA(V, 20));         # 量比
  BUY := (OC >= 2)                         # 实体阳线较强
         AND (C >= HHV(C, 60))             # 60日新高（突破）
         AND (VOLR >= 1.5)                 # 放量
         AND (C > MA(C, 20));              # 均线上方

输出：
  - 终端展示（可控行数）
  - CSV：{out_dir}/select_{adj}_{yyyymmdd}.csv

注意：
  - 数据列期望兼容 Tushare 的 daily：ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
  - 仅使用 C,O,H,L,V,AMOUNT 等原始列，避免依赖额外指标。
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import datetime as dt
from typing import Optional, List

import pandas as pd

# 优先使用 duckdb 读取分区；若不可用则回退 pandas
try:
    import duckdb  # type: ignore
    HAS_DUCKDB = True
except Exception:
    duckdb = None  # type: ignore
    HAS_DUCKDB = False

from tdx_compat import evaluate as tdx_eval

DEFAULT_BASE = r"E:\\stock_data"
DEFAULT_RULE = (
    """
    OC := SAFE_DIV(C - O, O) * 100;         { 开盘到收盘的涨幅(%) }
    VOLR := SAFE_DIV(V, MA(V, 20));         { 量比 }
    BUY := (OC >= 2)                         { 实体阳线较强 }
           AND (C >= HHV(C, 60))             { 60日新高 }
           AND (VOLR >= 1.5)                 { 放量 }
           AND (C > MA(C, 20));              { 站上20日均线 }
    """
    .strip()
)

ALLOWED_ADJ = {"daily", "daily_qfq", "daily_hfq"}


def _latest_trade_date_dir(root: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    cands = [d for d in os.listdir(root) if d.startswith("trade_date=")]
    if not cands:
        return None
    return max(cands)  # 字符串比较适用于 YYYYMMDD


def _daily_root(base: str, adj: str) -> str:
    # 你的目录规范：E:\stock_data/stock/daily(或 daily_qfq/daily_hfq)
    if adj not in ALLOWED_ADJ:
        raise ValueError(f"adj 仅支持: {sorted(ALLOWED_ADJ)}")
    return os.path.join(base, "stock", adj)


def _read_latest_partition(base: str, adj: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    root = _daily_root(base, adj)
    last_dir = _latest_trade_date_dir(root)
    if last_dir is None:
        raise FileNotFoundError(f"未找到 {root}/trade_date=* 分区")
    pdir = os.path.join(root, last_dir)

    if HAS_DUCKDB:
        pattern = os.path.join(pdir, "*.parquet").replace("\\", "/")
        sel = "*" if not columns else ", ".join(columns)
        sql = f"SELECT {sel} FROM read_parquet('{pattern}')"
        df = duckdb.sql(sql).df()  # type: ignore
    else:
        files = glob.glob(os.path.join(pdir, "*.parquet"))
        if not files:
            raise FileNotFoundError(f"分区为空: {pdir}")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        if columns:
            keep = [c for c in columns if c in df.columns]
            if keep:
                df = df[keep]

    # 统一列类型
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    return df


def _load_rule_inline_or_file(rule: Optional[str], rule_file: Optional[str]) -> str:
    if rule and rule_file:
        raise ValueError("--rule 与 --rule-file 不能同时使用")
    if rule_file:
        with open(rule_file, "r", encoding="utf-8") as f:
            return f.read()
    return (rule or DEFAULT_RULE).strip()


def run_select(base: str, adj: str, rule_text: str, top: Optional[int], out_dir: Optional[str], show: int) -> str:
    # 1) 读取最新分区
    base = os.path.abspath(base)
    df = _read_latest_partition(base, adj)
    if df.empty:
        raise RuntimeError("最新分区为空")

    # 2) 执行 TDX 脚本
    res = tdx_eval(rule_text, df)

    # 3) 取布尔选股序列：优先 BUY / SELECT / COND；否则尝试最后值
    mask = None
    for key in ("BUY", "SELECT", "COND", "buy", "select", "cond"):
        if isinstance(res, dict) and key in res:
            s = res[key]
            if isinstance(s, (pd.Series, list)):
                mask = pd.Series(s).astype(bool)
                break
    if mask is None:
        # 兼容：evaluate 的“最后表达式值”可能以特殊键返回；做尽量保守的兜底
        last_series = None
        if isinstance(res, dict):
            # 取 Series 类型最长的那个
            cand = [v for v in res.values() if isinstance(v, pd.Series)]
            if cand:
                last_series = cand[-1]
        if last_series is None:
            raise RuntimeError("无法从脚本结果中识别选股布尔序列（请在脚本中赋值 BUY:=... ）")
        mask = last_series.astype(bool)

    sel = df[mask]

    # 4) 排序与裁剪：默认按 amount(成交额)降序，便于流动性过滤
    by_cols = [c for c in ("amount", "vol", "pct_chg") if c in sel.columns]
    if by_cols:
        sel = sel.sort_values(by=by_cols, ascending=[False] * len(by_cols))
    if top is not None:
        sel = sel.head(top)

    # 5) 输出
    # 推断日期 & 路径
    last_date = df["trade_date"].max().strftime("%Y%m%d") if "trade_date" in df.columns else dt.date.today().strftime("%Y%m%d")
    out_dir = out_dir or os.path.join(base, "selections")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"select_{adj}_{last_date}.csv")

    keep_cols = [c for c in [
        "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
        "change", "pct_chg", "vol", "amount"
    ] if c in sel.columns]
    extra_cols = []
    # 若脚本里生成了 OC/VOLR 等中间量，也尽量带上，方便复盘
    for k, v in (res.items() if isinstance(res, dict) else []):
        if isinstance(v, pd.Series) and k not in keep_cols and v.index.size == len(df):
            if k.upper() in {"OC", "VOLR", "Z_SCORE", "Z_SLOPE"}:
                sel[k.lower()] = v[mask].values  # 对齐选中行
                extra_cols.append(k.lower())

    export_cols = keep_cols + extra_cols
    if export_cols:
        sel[export_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        sel.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 控制台展示
    to_show = sel if show is None else sel.head(show)
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 160):
        print(to_show.to_string(index=False))

    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="基于最新 daily 分区 + TDX 兼容层的选股程序")
    p.add_argument("--base", default=DEFAULT_BASE, help="数据根目录")
    p.add_argument("--adj", default="daily", choices=sorted(ALLOWED_ADJ), help="daily / daily_qfq / daily_hfq")
    p.add_argument("--rule", default=None, help="行内 TDX 脚本；与 --rule-file 互斥")
    p.add_argument("--rule-file", default=None, help="从文件加载 TDX 脚本；与 --rule 互斥")
    p.add_argument("--top", type=int, default=200, help="最多保留多少只；默认200")
    p.add_argument("--out", dest="out_dir", default=None, help="输出目录；默认 {base}/selections")
    p.add_argument("--show", type=int, default=30, help="终端最多展示的行数")

    args = p.parse_args(argv)

    try:
        rule_text = _load_rule_inline_or_file(args.rule, args.rule_file)
        path = run_select(args.base, args.adj, rule_text, args.top, args.out_dir, args.show)
        print(f"\n已写出: {path}")
        return 0
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
