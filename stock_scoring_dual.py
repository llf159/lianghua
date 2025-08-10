# -*- coding: utf-8 -*-
"""
stock_scoring_dual.py — 把股票机会分成两类并分别打分：
  1) 长期机会：在回看窗口内，对每根 K 线形态/突破/均线结构等进行“离散事件打分”，累加得到 LongTermScore。
  2) 交易性机会：仅使用“打分日的当日指标/状态”打分，得到 ShortTermScore。
最终支持加权融合：FinalScore = w_long * LongTermScore + w_short * ShortTermScore。

特点
- 数据源：{BASE}/stock/{adj}/trade_date=YYYYMMDD/*.parquet（默认 daily）
- 形态判定与指标计算：优先用 TDX 兼容层脚本（tdx_compat.evaluate）
- 权重与规则：完全外置到 JSON/YAML（--config），同时内置一份默认配置作为兜底
- 输出：包含 long/short/final 三类分数与细分项明细，便于复盘

示例
  python stock_scoring_dual.py --base E:\\stock_data --adj daily \
    --end 20250809 --lookback 120 --top 300 --show 30 \
    --config dual_config.json --out E:\\stock_data\\ranks

默认配置（等价于内置 DEFAULT_CONFIG）结构：
{
  "blend": {"w_long": 0.6, "w_short": 0.4},
  "long_term": {
    "window": 120,
    "patterns": [
      {"name": "break_60",  "tdx": "C>=HHV(C,60)",                    "score": 2.0},
      {"name": "ma_trending","tdx": "MA(C,5)>MA(C,10) AND MA(C,10)>MA(C,20)", "score": 1.5},
      {"name": "vol_contract","tdx": "MA(V,5)<MA(V,20)",                 "score": 0.5},
      {"name": "tight_range","tdx": "(HHV(H,5)-LLV(L,5))/REF(C,5)<=0.05", "score": 0.8},
      {"name": "hammer",     "tdx": "(C>O) AND ((L-MIN(O,C))/ (H-L+EPS) >= 0.6) AND (H-L)>0", "score": 0.8},
      {"name": "engulf_bull","tdx": "(C>O) AND (REF(C,1)<REF(O,1)) AND (C>=REF(O,1)) AND (O<=REF(C,1))", "score": 1.2}
    ],
    "penalties": [
      {"name": "huge_drop",  "tdx": "(C/REF(C,1)-1)<=-0.07",             "score": -2.0},
      {"name": "fall_below_ma20","tdx": "C<MA(C,20)",                   "score": -0.5}
    ],
    "normalize": {"method": "per_bar_mean", "scale": 1.0}
  },
  "short_term": {
    "rules": [
      {"name": "gap_up",      "tdx": "O>=REF(H,1) AND (C>=O)",           "score": 1.5},
      {"name": "vol_spike",   "tdx": "V/MA(V,20)>=1.8",                 "score": 1.2},
      {"name": "strong_close","tdx": "(C/O-1)>=0.02",                   "score": 1.0},
      {"name": "near_high",   "tdx": "C>=HHV(C,60)",                     "score": 0.8}
    ],
    "penalties": [
      {"name": "upper_shadow","tdx": "(H-C)/(H-L+EPS)>=0.5",             "score": -0.6},
      {"name": "limit_up",    "tdx": "(C/REF(C,1)-1)>=0.095",            "score": -0.3}
    ],
    "normalize": {"method": "identity"}
  },
  "filters": {
    "min_avg_amount20": 2e7,
    "exclude_patterns": ["^ST", "[*]ST"],
    "min_list_days": 180
  }
}

注意
- 形态是“逐日离散计分”，长线分数对连贯趋势更敏感；短线分数强调当日交易机会
- 你可以在 JSON/YAML 里自由增删规则、调整权重，不需要改代码
"""

from __future__ import annotations
import os
import re
import glob
import json
import math
import argparse
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import duckdb  # type: ignore
    HAS_DUCKDB = True
except Exception:
    duckdb = None  # type: ignore
    HAS_DUCKDB = False

from tdx_compat import evaluate as tdx_eval

ALLOWED_ADJ = {"daily", "daily_qfq", "daily_hfq"}
DEFAULT_BASE = r"E:\\stock_data"

# ---------------- 默认配置 ----------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "blend": {"w_long": 0.6, "w_short": 0.4},
    "long_term": {
        "window": 120,
        "patterns": [
            {"name": "break_60",  "tdx": "C>=HHV(C,60)",                    "score": 2.0},
            {"name": "ma_trending","tdx": "MA(C,5)>MA(C,10) AND MA(C,10)>MA(C,20)", "score": 1.5},
            {"name": "vol_contract","tdx": "MA(V,5)<MA(V,20)",                 "score": 0.5},
            {"name": "tight_range","tdx": "(HHV(H,5)-LLV(L,5))/REF(C,5)<=0.05", "score": 0.8},
            {"name": "hammer",     "tdx": "(C>O) AND ((L-MIN(O,C))/ (H-L+EPS) >= 0.6) AND (H-L)>0", "score": 0.8},
            {"name": "engulf_bull","tdx": "(C>O) AND (REF(C,1)<REF(O,1)) AND (C>=REF(O,1)) AND (O<=REF(C,1))", "score": 1.2}
        ],
        "penalties": [
            {"name": "huge_drop",  "tdx": "(C/REF(C,1)-1)<=-0.07",             "score": -2.0},
            {"name": "fall_below_ma20","tdx": "C<MA(C,20)",                   "score": -0.5}
        ],
        "normalize": {"method": "per_bar_mean", "scale": 1.0}
    },
    "short_term": {
        "rules": [
            {"name": "gap_up",      "tdx": "O>=REF(H,1) AND (C>=O)",           "score": 1.5},
            {"name": "vol_spike",   "tdx": "V/MA(V,20)>=1.8",                 "score": 1.2},
            {"name": "strong_close","tdx": "(C/O-1)>=0.02",                   "score": 1.0},
            {"name": "near_high",   "tdx": "C>=HHV(C,60)",                     "score": 0.8}
        ],
        "penalties": [
            {"name": "upper_shadow","tdx": "(H-C)/(H-L+EPS)>=0.5",             "score": -0.6},
            {"name": "limit_up",    "tdx": "(C/REF(C,1)-1)>=0.095",            "score": -0.3}
        ],
        "normalize": {"method": "identity"}
    },
    "filters": {
        "min_avg_amount20": 2e7,
        "exclude_patterns": ["^ST", "[*]ST"],  # 名称过滤（若 name 可用）
        "min_list_days": 180
    }
}

# ---------------- 工具 ----------------

def _daily_root(base: str, adj: str) -> str:
    return os.path.join(base, "stock", adj)


def _scan_dates(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    ds = [d.split("=")[-1] for d in os.listdir(root) if d.startswith("trade_date=")]
    return sorted([d for d in ds if len(d) == 8])


def _latest(root: str) -> Optional[str]:
    ds = _scan_dates(root)
    return ds[-1] if ds else None


def _read_window(base: str, adj: str, end_date: str, lookback: int, columns: Optional[List[str]] = None) -> pd.DataFrame:
    root = _daily_root(base, adj)
    all_dates = _scan_dates(root)
    if not all_dates:
        raise FileNotFoundError(f"不存在分区: {root}")
    if end_date not in all_dates:
        raise FileNotFoundError(f"找不到交易日 {end_date} 于 {root}")
    idx = all_dates.index(end_date)
    start_idx = max(0, idx - lookback + 1)
    need = all_dates[start_idx: idx+1]

    if HAS_DUCKDB:
        pattern = os.path.join(root, "trade_date=*", "*.parquet").replace("\\", "/")
        sel = "*" if not columns else ", ".join(columns)
        dates_cond = ",".join([f"'{d}'" for d in need])
        sql = f"SELECT {sel} FROM parquet_scan('{pattern}') WHERE trade_date IN ({dates_cond})"
        df = duckdb.sql(sql).df()  # type: ignore
    else:
        frames = []
        for d in need:
            files = glob.glob(os.path.join(root, f"trade_date={d}", "*.parquet"))
            frames.extend(pd.read_parquet(f) for f in files)
        df = pd.concat(frames, ignore_index=True)
        if columns:
            keep = [c for c in columns if c in df.columns]
            df = df[keep]

    if df is None or df.empty:
        raise RuntimeError("读取窗口数据为空")

    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    df.sort_values(["ts_code", "trade_date"], inplace=True)
    return df


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
        try:
            import yaml  # type: ignore
            return yaml.safe_load(txt) if path.lower().endswith(('.yml','.yaml')) else json.loads(txt)
        except Exception:
            # 先尝试 json，再尝试 yaml
            try:
                return json.loads(txt)
            except Exception:
                import yaml  # type: ignore
                return yaml.safe_load(txt)


def _apply_name_filters(df_last: pd.DataFrame, exclude_patterns: List[str]) -> pd.Series:
    # 若没有 name 列，则放行
    if "name" not in df_last.columns or not exclude_patterns:
        return pd.Series([True]*len(df_last), index=df_last.index)
    ok = pd.Series([True]*len(df_last), index=df_last.index)
    for p in exclude_patterns:
        ok &= ~df_last["name"].astype(str).str.contains(p, regex=True, na=False)
    return ok


def _list_days_filter(df: pd.DataFrame, end_date: pd.Timestamp, min_days: int) -> pd.Series:
    # 若无 list_date 列，放行
    if "list_date" not in df.columns:
        return pd.Series([True]*len(df[df.trade_date==end_date]), index=df[df.trade_date==end_date].index)
    # 取最后一天横截面
    last = df[df.trade_date==end_date].copy()
    last["list_date"] = pd.to_datetime(last["list_date"].astype(str), errors='coerce')
    days = (end_date - last["list_date"]).dt.days
    return days >= int(min_days)

# ---------------- 打分核心 ----------------

def _score_long_term(g: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[float, Dict[str,float]]:
    """对单只股票 g（已按日期排序）在窗口内逐日累计形态分数。"""
    pats = cfg.get("patterns", [])
    pens = cfg.get("penalties", [])
    normalize = cfg.get("normalize", {"method":"per_bar_mean","scale":1.0})
    n = len(g)
    total = np.zeros(n, dtype=float)
    detail: Dict[str,float] = {}

    def _eval_add(items, sign: int = 1):
        nonlocal total, detail
        for it in items:
            name = it.get("name","item")
            scr = float(it.get("score", 0.0)) * sign
            tdx = it.get("tdx","True")
            res = tdx_eval(tdx, g)
            # 结果可能是 bool Series 或 DataFrame/字典，统一转为 Series
            if isinstance(res, dict):
                # 取最后一个 Series 或名为 name 的键
                ser = None
                if name in res and isinstance(res[name], pd.Series):
                    ser = res[name]
                else:
                    cand = [v for v in res.values() if isinstance(v, pd.Series)]
                    ser = cand[-1] if cand else pd.Series([False]*n, index=g.index)
            elif isinstance(res, pd.Series):
                ser = res
            else:
                # 标量/其他 → 全 False
                ser = pd.Series([False]*n, index=g.index)
            mask = pd.Series(ser).fillna(False).astype(bool).values
            total[mask] += scr
            # 统计命中次数乘权重，用于明细
            detail[name] = detail.get(name, 0.0) + scr * float(mask.sum())

    _eval_add(pats, sign=+1)
    _eval_add(pens, sign=+1)  # penalty 在配置里就是负分

    # 归一：
    method = str(normalize.get("method","per_bar_mean")).lower()
    if method == "per_bar_mean":
        score = float(total.mean()) * float(normalize.get("scale",1.0))
    elif method == "sum":
        score = float(total.sum()) * float(normalize.get("scale",1.0))
    elif method == "last_window":
        k = int(normalize.get("k", 20))
        score = float(total[-k:].sum())
    else:
        score = float(total.sum())
    return score, detail


def _score_short_term(g: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[float, Dict[str,float]]:
    """使用“打分日当日”的状态打分。"""
    rules = cfg.get("rules", [])
    pens  = cfg.get("penalties", [])
    normalize = cfg.get("normalize", {"method":"identity"})

    last = g.iloc[-1:]
    detail: Dict[str,float] = {}
    total = 0.0

    def _one(items, sign=+1.0):
        nonlocal total
        for it in items:
            name = it.get("name","item")
            scr = float(it.get("score", 0.0)) * sign
            tdx = it.get("tdx","True")
            res = tdx_eval(tdx, last)
            # 转布尔
            hit = False
            if isinstance(res, dict):
                # 取一个 bool/Series
                for v in res.values():
                    if isinstance(v, pd.Series):
                        hit = bool(v.iloc[-1])
                        break
                    elif isinstance(v, (bool, np.bool_)):
                        hit = bool(v)
                        break
            elif isinstance(res, pd.Series):
                hit = bool(res.iloc[-1])
            elif isinstance(res, (bool, np.bool_)):
                hit = bool(res)
            total += (scr if hit else 0.0)
            if hit:
                detail[name] = detail.get(name, 0.0) + scr

    _one(rules, sign=+1.0)
    _one(pens,  sign=+1.0)  # penalty 在配置里就是负分

    # 归一
    method = str(normalize.get("method","identity")).lower()
    if method == "identity":
        score = total
    elif method == "tanh":
        scale = float(normalize.get("scale", 1.0))
        score = float(math.tanh(total/scale))
    else:
        score = total
    return score, detail


# ---------------- 主流程 ----------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="长期形态累计 + 短期当日指标 的双通道打分系统")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--adj", default="daily", choices=sorted(ALLOWED_ADJ))
    p.add_argument("--end", default=None, help="打分日 YYYYMMDD，默认用最新分区")
    p.add_argument("--lookback", type=int, default=120, help="窗口长度（仅用于默认配置；若配置文件内覆盖则以配置为准）")
    p.add_argument("--config", default=None, help="JSON 或 YAML 配置路径；覆盖默认配置")
    p.add_argument("--top", type=int, default=300)
    p.add_argument("--show", type=int, default=30)
    p.add_argument("--out", default=None)

    args = p.parse_args(argv)

    cfg = _load_config(args.config)
    # 若 CLI 指定了 lookback 且配置未显式给 window，则用 CLI
    if "long_term" in cfg and "window" not in cfg["long_term"]:
        cfg["long_term"]["window"] = int(args.lookback)

    base = os.path.abspath(args.base)
    out_dir = args.out or os.path.join(base, "ranks_dual")
    os.makedirs(out_dir, exist_ok=True)

    root = _daily_root(base, args.adj)
    end_date = args.end or _latest(root)
    if not end_date:
        print(f"未找到分区：{root}")
        return 2

    # 读取窗口数据（带必要列；若能提供 name/list_date 更好）
    cols = ["ts_code","trade_date","open","high","low","close","pre_close","vol","amount","name","list_date"]
    df = _read_window(base, args.adj, end_date, int(cfg.get("long_term",{}).get("window", args.lookback)), columns=cols)

    # 预先计算 20 日均额用于过滤
    df["avg_amt20"] = df.groupby("ts_code")["amount"].transform(lambda s: s.rolling(20, min_periods=1).mean())

    # 横截面（打分日）
    last_day = pd.to_datetime(end_date)
    cross = df[df.trade_date == last_day].copy()

    # 过滤器
    filt_cfg = cfg.get("filters", {})
    mask = pd.Series([True]*len(cross), index=cross.index)
    min_avg_amt = float(filt_cfg.get("min_avg_amount20", 0.0) or 0.0)
    if min_avg_amt > 0:
        mask &= cross["avg_amt20"] >= min_avg_amt
    mask &= _apply_name_filters(cross, filt_cfg.get("exclude_patterns", []))
    mask &= _list_days_filter(df, last_day, int(filt_cfg.get("min_list_days", 0)))
    pool_codes = set(cross[mask]["ts_code"].tolist())

    # 分组打分
    rows: List[Dict[str, Any]] = []
    w_long = float(cfg.get("blend",{}).get("w_long", 0.6))
    w_short = float(cfg.get("blend",{}).get("w_short", 0.4))

    for code, g in df.groupby("ts_code", sort=False):
        if code not in pool_codes:
            continue
        g = g.sort_values("trade_date").reset_index(drop=True)
        long_s, long_detail = _score_long_term(g, cfg.get("long_term", {}))
        short_s, short_detail = _score_short_term(g, cfg.get("short_term", {}))
        final = w_long * long_s + w_short * short_s
        row = {
            "ts_code": code,
            "trade_date": g["trade_date"].iloc[-1],
            "LongTermScore": long_s,
            "ShortTermScore": short_s,
            "FinalScore": final,
            "avg_amt20": float(g["amount"].tail(20).mean()) if "amount" in g.columns else np.nan,
        }
        # 展开明细（可选：仅保留命中累计不为 0 的项）
        for k, v in long_detail.items():
            row[f"LT_{k}"] = v
        for k, v in short_detail.items():
            row[f"ST_{k}"] = v
        rows.append(row)

    if not rows:
        print("过滤后股票池为空。")
        return 0

    out = pd.DataFrame(rows)

    # 排序：FinalScore 降序，其次 avg_amt20 降序
    out.sort_values(["FinalScore", "avg_amt20"], ascending=[False, False], inplace=True)
    if args.top is not None:
        out = out.head(int(args.top))

    # 导出
    date_str = end_date
    out_path = os.path.join(out_dir, f"rank_dual_{args.adj}_{date_str}.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 预览
    with pd.option_context("display.max_rows", 200, "display.max_columns", 80, "display.width", 180):
        print(out.head(args.show).to_string(index=False))

    print(f"\n已写出：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
