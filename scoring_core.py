# -*- coding: utf-8-sig -*-
"""
score_engine.py — 单文件版“可编程打分系统”（无 CLI / 无可视化）
依赖你现有工程：config.py、parquet_viewer.py、tdx_compat.py、indicators.py

用法（示例）：
    from score_engine import run_for_date
    run_for_date()                         # 按 config.SC_REF_DATE 运行（默认 today -> 最新分区）
    run_for_date("20250816")               # 指定参考日
"""

from __future__ import annotations

import os
import re
import math
import json
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import glob
import functools
from pathlib import Path

import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
from logging.handlers import TimedRotatingFileHandler
from functools import lru_cache

from config import (
    PARQUET_BASE, PARQUET_ADJ, PARQUET_USE_INDICATORS,
    SC_REF_DATE, SC_LOOKBACK_D, SC_PRESCREEN_LOOKBACK_D, SC_BASE_SCORE, SC_MIN_SCORE,
    SC_TOP_K, SC_TIE_BREAK, SC_MAX_WORKERS, SC_READ_TAIL_DAYS,
    SC_OUTPUT_DIR, SC_CACHE_DIR,
    SC_RULES, SC_PRESCREEN_RULES,
    SC_WRITE_WHITELIST, SC_WRITE_BLACKLIST,
    SC_ATTENTION_ENABLE, SC_ATTENTION_SOURCE,
    SC_ATTENTION_WINDOW_D, SC_ATTENTION_MIN_HITS, SC_ATTENTION_TOP_K,
    SC_ATTENTION_BACKFILL_ENABLE,
    SC_BENCH_CODES, SC_BENCH_WINDOW, SC_BENCH_FILL, SC_BENCH_FEATURES,
    SC_UNIVERSE, 
)
    
from parquet_viewer import asset_root, list_trade_dates, read_range
from indicators import kdj  # 用于兜底计算 J（若数据不含 j 列）
from utils import normalize_ts, normalize_trade_date
import tdx_compat as tdx
from tdx_compat import evaluate_bool
from stats_core import run_tracking, post_scoring, run_surge

OUTPUT_DIR = Path("output/score")  # 按你现有目录习惯；若不同，改这里
DETAIL_DIR = OUTPUT_DIR / "details"
ALL_DIR = OUTPUT_DIR / "all"
ALL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- 日志初始化 -------------------------
def _init_logger():
    os.makedirs("./log", exist_ok=True)
    logger = logging.getLogger("score")
    # logger.setLevel(logging.DEBUG)  # 细到 DEBUG，输出筛选由 handler 控制
    # 从 config 读取级别；默认 WARNING（更轻量）
    try:
        import config as _cfg
        level_name = getattr(_cfg, "LOG_LEVEL", "WARNING")
    except Exception:
        level_name = "WARNING"
    level = getattr(logging, str(level_name).upper(), logging.WARNING)
    logger.setLevel(level)
    # 轮转：每天一个，保留 7 个
    fh = TimedRotatingFileHandler("./log/score.log", when="midnight", interval=1, backupCount=7, encoding="utf-8-sig")
    ch = logging.StreamHandler()
    fh.setLevel(level)
    # 控制台仅 WARNING+，进一步减少噪音
    ch.setLevel(max(level, logging.WARNING))
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    logger.propagate = False
    return logger

LOGGER = _init_logger()

# ---- I/O helpers (cached) ----
@lru_cache(maxsize=512)
def _load_score_all_csv_cached(d: str, usecols_key: str = "ts_code,rank,trade_date") -> "pd.DataFrame":
    usecols = [c for c in (usecols_key.split(",") if usecols_key else []) if c]
    path = os.path.join(SC_OUTPUT_DIR, "all", f"score_all_{d}.csv")
    # dtype that is stable and small
    dtype = {"ts_code": str, "rank": "Int64", "trade_date": str}
    try:
        return pd.read_csv(path, usecols=usecols or None, dtype=dtype, engine="c")
    except Exception as e:
        LOGGER.warning("[attention] 读取失败 %s: %s", path, e)
        return pd.DataFrame(columns=usecols or ["ts_code","rank","trade_date"])

# ===== 内嵌 RANKER =====
# 依赖你的 parquet 扫描工具（只读必要列，加速）
from parquet_viewer import asset_root as _asset_root, scan_with_duckdb as _scan_with_duckdb
# 本次 run 的全局上下文
_RANK_G = {
    "ref": None,           # 参考日 YYYYMMDD
    "codes": [],           # 本次评分的股票清单
    "base": None,          # PARQUET_BASE
    "adj": None,           # PARQUET_ADJ
    "default_N": 60,       # 不写 N 时的默认窗口
}

PROGRESS_HANDLER = None

def set_progress_handler(fn):
    """UI/CLI 可以注册一个回调：fn(phase, current=None, total=None, message=None, **extra)"""
    global PROGRESS_HANDLER
    PROGRESS_HANDLER = fn


def _progress(phase: str, *, current: int | None = None, total: int | None = None,
              message: str | None = None, **extra):
    if PROGRESS_HANDLER:
        try:
            PROGRESS_HANDLER(phase=phase, current=current, total=total, message=message, **extra)
        except Exception as e:
            LOGGER.debug(f"[progress-cb-error] {e}")
    # 顺手打一条 INFO，给日志/命令行看
    m = f"{phase}"
    if total is not None and current is not None:
        m += f" {current}/{total}"
    if message:
        m += f" | {message}"
    LOGGER.info(m)


def init_rank_env(ref_date: str, universe_codes: List[str], base: str, adj: str, default_N: int = 60):
    _RANK_G["ref"] = str(ref_date)
    _RANK_G["codes"] = list(universe_codes or [])
    _RANK_G["base"] = str(base)
    _RANK_G["adj"]  = str(adj)
    _RANK_G["default_N"] = int(default_N)


def _load_last_n(_base: str, _adj: str, _codes: List[str], _ref: str, _N: int) -> pd.DataFrame:
    """
    读取全市场近 _N 天（含 _ref）的 O/H/L/C/V；只取必要列。
    返回至少包含：['ts_code','trade_date','open','high','low','close','vol']。
    """
    root = _asset_root(_base, "stock", _adj)
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    # 给足冗余，按自然日回溯（停牌/周末）
    start = (dt.datetime.strptime(_ref, "%Y%m%d") - dt.timedelta(days=_N*3)).strftime("%Y%m%d")
    df = _scan_with_duckdb(root, None, start, _ref, columns=cols)
    if not df.empty:
        df["trade_date"] = df["trade_date"].astype(str)
        if _codes:
            df = df[df["ts_code"].astype(str).isin(set(_codes))]
    return df


def _rank_series(s: pd.Series, ascending: bool) -> pd.Series:
    # 名次从 1 开始；相等取平均名次；缺失置为最大名次+1
    r = s.rank(ascending=ascending, method="average")
    mx = int(r.max()) if pd.notna(r.max()) else len(r)
    return r.fillna(mx + 1)


def _rule_key(rule: dict) -> tuple:
    name = str(rule.get("name", "")).strip()
    tf = str(rule.get("timeframe", "D")).upper().strip()
    win = int(rule.get("window", SC_LOOKBACK_D))
    # Canonicalize 'when' / 'clauses'
    if rule.get("clauses"):
        # serialize relevant fields only
        parts = []
        for c in rule.get("clauses") or []:
            parts.append(str(c.get("when") or "").strip())
        when = "|".join(parts)
    else:
        when = str(rule.get("when") or "").strip()
    return (name, tf, win, when)


def _iter_unique_rules():
    seen = set()
    for r in (SC_RULES or []):
        k = _rule_key(r)
        if k in seen:
            continue
        seen.add(k)
        yield r


def _json_sanitize(x):
    """Recursively replace NaN/Inf floats with None so that json.dump(..., allow_nan=False) never fails."""
    # numpy scalar check
    if np is not None and isinstance(x, (np.floating, np.integer)):
        x = float(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_sanitize(v) for v in list(x)]
    return x


def _latest_row(df: pd.DataFrame, ref: str) -> pd.DataFrame:
    return df[df["trade_date"] == ref].copy()


def _ret_vs_prev_close(df: pd.DataFrame, ref: str) -> pd.Series:
    """
    以 ref 当日收盘相对前一交易日收盘的涨跌幅（%），索引为 ts_code。
    """
    df = df.sort_values(["ts_code", "trade_date"])
    prev_close = df.groupby("ts_code")["close"].shift(1)
    ret = (df["close"] - prev_close) / prev_close * 100.0
    out = pd.DataFrame({"ts_code": df["ts_code"], "trade_date": df["trade_date"], "ret": ret})
    ret_ref = out[out["trade_date"] == ref][["ts_code", "ret"]]
    return ret_ref.set_index("ts_code")["ret"]


# —— 对外：在 when 表达式里直接用的函数 —— #
def RANK_VOL(N: int | None = None, K: Optional[int] = None) -> float:
    """
    成交量横截面名次（1 = 量最小，量越小名次越靠前）。
    K：可选，仅在前 K 只里排名；省略=全评分范围。
    """
    ref, base, adj, codes = _RANK_G["ref"], _RANK_G["base"], _RANK_G["adj"], _RANK_G["codes"]
    N = int(N or _RANK_G["default_N"])
    df = _load_last_n(base, adj, codes, ref, N)
    if df.empty:
        return float("inf")
    latest = _latest_row(df, ref)
    s = latest.set_index("ts_code")["vol"]
    if K and K > 0:
        s = s.sort_values().head(int(K))
    ranks = _rank_series(s, ascending=True)  # 量小→名次靠前
    ts = str(tdx.EXTRA_CONTEXT.get("TS", ""))
    return float(ranks.get(ts, float("inf")))


def RANK_RET(N: int | None = None, K: Optional[int] = None, side: str = "up") -> float:
    """
    涨跌幅横截面名次：
      side='up'   ：按涨幅从大到小排名（涨多名次靠前）；
      side='down' ：按跌幅从大到小排名（跌多名次靠前）。
    """
    ref, base, adj, codes = _RANK_G["ref"], _RANK_G["base"], _RANK_G["adj"], _RANK_G["codes"]
    N = int(N or _RANK_G["default_N"])
    df = _load_last_n(base, adj, codes, ref, N)
    if df.empty:
        return float("inf")
    s = _ret_vs_prev_close(df, ref)
    if K and K > 0:
        s = s.sort_values(ascending=False).head(int(K))
    if str(side).lower().startswith("up"):
        ranks = _rank_series(s, ascending=False)     # 涨多→名次靠前
    else:
        ranks = _rank_series(-s, ascending=False)    # 跌多→名次靠前
    ts = str(tdx.EXTRA_CONTEXT.get("TS", ""))
    return float(ranks.get(ts, float("inf")))


def RANK_MATCH_COEF(N: int | None = None, K: Optional[int] = None) -> float:
    """
    量价匹配系数（0~1，越大匹配越强）：
      - 涨幅榜应与“量大”（活跃）匹配；
      - 跌幅榜应与“量小”（冷清）匹配。
    做法：
      - 计算：r_up = RANK_RET(...,'up'), r_dn = RANK_RET(...,'down')
      - 量小名次：rv_small = RANK_VOL(...)
      - 量大名次：rv_large = (Keff + 1 - rv_small)，Keff 为参与排名样本量
      - 匹配度：min(r_up * rv_large, r_dn * rv_small) 的倒数归一（Keff / prod）
    """
    # 两边名次
    r_up = RANK_RET(N, K, side="up")
    r_dn = RANK_RET(N, K, side="down")
    rv_small = RANK_VOL(N, K)
    Keff = int(K or len(_RANK_G["codes"]) or 1)
    rv_large = max(1.0, Keff + 1.0 - rv_small)
    prod_up = r_up * rv_large
    prod_dn = r_dn * rv_small
    prod = min(prod_up, prod_dn)
    coef = min(1.0, max(0.0, (Keff / max(prod, 1.0))))
    return float(coef)


def get_eval_env(ts_code: str, ref_date: str) -> Dict[str, object]:
    """提供注入到 tdx_compat.EXTRA_CONTEXT 的函数映射"""
    return {
        "TS": str(ts_code),
        "REF_DATE": str(ref_date),
        "RANK_VOL": RANK_VOL,
        "RANK_RET": RANK_RET,
        "RANK_MATCH_COEF": RANK_MATCH_COEF,
    }


def get_rank_eval_env(ts_code: str) -> Dict[str, object]:
    """简化版（_score_one 内部已知 ref_date，无需重复传）"""
    return {
        "TS": str(ts_code),
        "RANK_VOL": RANK_VOL,
        "RANK_RET": RANK_RET,
        "RANK_MATCH_COEF": RANK_MATCH_COEF,
    }

# ------------------------- 工具函数 -------------------------
def _last_true_date(win_df: pd.DataFrame, when: str) -> tuple[Optional[str], Optional[int]]:
    """给窗口数据和 when 表达式，返回 (最近一次为 True 的日期YYYYMMDD, lag)"""
    if not when:
        return None, None
    s_bool = evaluate_bool(when, win_df).astype(bool)
    lag = _last_true_lag(s_bool)  # 已有函数
    if lag is None:
        return None, None
    # win_df 已经是 _window_slice 后的 TF 级时间索引
    idx = win_df.index if isinstance(win_df.index, pd.DatetimeIndex) else pd.to_datetime(win_df["trade_date"].astype(str))
    hit_ts = idx[-1 - lag]
    return hit_ts.strftime("%Y%m%d"), int(lag)


def _single_root_dir() -> str:
    """
    single 布局根目录，例如：
    {PARQUET_BASE}/stock/single/single_qfq_indicators
    {PARQUET_BASE}/stock/single/single_qfq
    """
    name = f"single_{PARQUET_ADJ}" + ("_indicators" if PARQUET_USE_INDICATORS else "")
    return os.path.join(PARQUET_BASE, "stock", "single", name)


def _has_single_dir() -> bool:
    return os.path.isdir(_single_root_dir())


def _today_str():
    return dt.datetime.now().strftime("%Y%m%d")


def _make_time_weights(n: int,
                       mode: str | None = None,
                       half_life: float | None = None,
                       linear_min: float | None = None,
                       normalize: bool = True) -> list[float]:
    """
    生成长度为 n 的时间权重（索引从最早到最新）。
      mode: none | exp | linear
        - exp:  w = 0.5 ** (age/half_life)  （最新 age=0 权重大）
        - linear: 从 linear_min 线性递增到 1.0
      normalize: 归一化到均值=1（sum(weights)=n），避免窗口改变整体尺度。
    """
    if n <= 0:
        return []
    m = (mode or "none").lower()
    if m in ("", "none"):
        w = np.ones(n, float)
    elif m.startswith("exp"):
        hl = float(half_life or 5.0)
        age = np.arange(n)[::-1]  # 最新 age=0
        w = 0.5 ** (age / max(hl, 1e-6))
    else:
        lm = float(linear_min if linear_min is not None else 0.3)
        lm = min(max(lm, 0.0), 1.0)
        w = np.linspace(lm, 1.0, num=n)
    if normalize:
        s = float(w.mean())
        if s > 0:
            w = w / s
    return w.tolist()


def _normalize_gate(rule: dict) -> dict | None:
    """把 trigger/gate/require 归一成：{'clauses': [...]} 或 单子句 dict。"""
    g = rule.get("trigger") or rule.get("gate") or rule.get("require")
    if not g:
        return None
    if isinstance(g, str):
        # 默认为当前规则的 TF/WIN，scope=LAST（仅参考日）
        return {
            "timeframe": str(rule.get("timeframe","D")).upper(),
            "window": int(rule.get("window", SC_LOOKBACK_D)),
            "scope": "LAST",
            "when": g
        }
    if isinstance(g, dict):
        return g
    if isinstance(g, (list, tuple)):
        return {"clauses": list(g)}
    return None


def _eval_gate(df: pd.DataFrame, rule: dict, ref_date: str) -> tuple[bool, str | None, dict | None]:
    """
    返回: (gate_ok, err, gate_norm)
    gate 为空 → 视为 True
    多子句（列表）= AND 关系
    """
    g = _normalize_gate(rule)
    if not g:
        return True, None, None
    try:
        if "clauses" in g:
            for c in g["clauses"]:
                ok, err = _eval_rule(df, c, ref_date)
                if err or (not ok):
                    return False, err, g
            return True, None, g
        else:
            ok, err = _eval_rule(df, g, ref_date)
            return (bool(ok and not err), err, g)
    except Exception as e:
        return False, f"gate-exception: {e}", g


def _after_scoring(ref_date: str, df_all_scores: pd.DataFrame | None = None):
    windows = [1,2,3,5,10,20]              # 或从 UI/配置读
    benchmarks = ["000300.SH", "399006.SZ"]  # 只用“已下载”的指数
    run_tracking(ref_date, windows, benchmarks, score_df=df_all_scores, group_by_board=True, save=True)


def backfill_prev_n_days(ref_date: str | None = None,
                         n: int = 20,
                         include_today: bool = False,
                         *,
                         force: bool = False) -> list[Path]:
    """
    以 ref_date 为参照，自动取其前 n 个“交易日”窗口，补齐 score_all_*.csv。
    include_today=False 表示窗口不包含参考日（“前 n 天”通常不含当天）。
    """
    # 1) 锚定参考日：无参则用 _pick_ref_date()
    ref = ref_date or _pick_ref_date()

    # 2) 读取交易日日历（daily 分区）
    root_daily = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
    days = list_trade_dates(root_daily) or []

    if not days:
        # 没有日历，退化为“只补当天”
        return backfill_missing_ranks(ref, ref, force=force)

    # 3) 找到 ref 在日历中的位置；若缺失则用最后一天作为 ref
    if ref in days:
        i = days.index(ref)
    else:
        i = len(days) - 1
        ref = days[i]

    # 4) 计算窗口：end 为 (ref 或 ref 前一日)，start 为向前 n 天（越界向 0 对齐）
    end_idx = i if include_today else max(i - 1, 0)
    start_idx = max(end_idx - int(n) + 1, 0)

    if start_idx > end_idx:
        # 极端情况：n=0 或 i=0 且不含当天 → 做一次空安全处理
        start_idx, end_idx = end_idx, end_idx

    start = days[start_idx]
    end   = days[end_idx]

    # 5) 复用旧函数做实际补算
    return backfill_missing_ranks(start, end, force=force)


def _pick_ref_date() -> str:
    """
    优先从 daily 分区推断最新分区日；若不存在，则从 single 目录任取若干个股
    的 parquet 读取 trade_date 最大值作为 latest。
    若 SC_REF_DATE 指定了 YYYYMMDD，优先使用它（无法验证存在性时将直接采用）。
    同时打印一条 INFO 对比系统今日。
    """
    latest = None
    # 先尝试 daily 分区
    try:
        root_daily = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
        dates = list_trade_dates(root_daily)
        if dates:
            latest = dates[-1]
    except Exception:
        latest = None

    # 若没有 daily，则从 single 推断
    if latest is None and _has_single_dir():
        single_dir = _single_root_dir()
        files = glob.glob(os.path.join(single_dir, "*.parquet"))[:50]  # 取前 50 个采样
        mx = None
        for f in files:
            try:
                td = pd.read_parquet(f, columns=["trade_date"])["trade_date"]
                tmax = str(td.astype(str).max())
                if (mx is None) or (tmax > mx):
                    mx = tmax
            except Exception:
                continue
        if mx:
            latest = mx
    
    if latest is None:
        raise FileNotFoundError("无法从 daily 或 single 目录推断参考日，请检查数据目录。")

    if isinstance(SC_REF_DATE, str) and re.fullmatch(r"\d{8}", SC_REF_DATE):
        ref = SC_REF_DATE
    else:
        ref = latest

    sys_today = _today_str()
    if ref != sys_today:
        LOGGER.info(f"[参考日] 使用分区最新日 {ref}；系统今日 {sys_today}，请知悉。")
    else:
        LOGGER.info(f"[参考日] 使用今日分区 {ref}。")
    return ref


def _list_codes_for_day(day: str) -> List[str]:
    """
    返回需要评分的股票列表。
    - single 布局：直接列出 single 目录下所有 *.parquet 文件名
    - daily 布局：兼容旧逻辑，从 trade_date=day 分区列出
    """
    if _has_single_dir():
        single_dir = _single_root_dir()
        codes = []
        for name in os.listdir(single_dir):
            if name.endswith(".parquet"):
                codes.append(os.path.splitext(name)[0])
        return sorted(codes)
    else:
        root = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
        pdir = os.path.join(root, f"trade_date={day}")
        if not os.path.isdir(pdir):
            raise FileNotFoundError(f"未找到分区目录: {pdir}")
        codes = []
        for name in os.listdir(pdir):
            if name.endswith(".parquet"):
                codes.append(os.path.splitext(name)[0])
        return sorted(codes)


def _compute_read_start(ref_date: str) -> str:
    """
    计算读取起始日：优先使用 SC_READ_TAIL_DAYS；否则根据规则窗口估算。
    """
    if SC_READ_TAIL_DAYS:
        start_dt = dt.datetime.strptime(ref_date, "%Y%m%d") - dt.timedelta(days=int(SC_READ_TAIL_DAYS))
        return start_dt.strftime("%Y%m%d")

    # 估算：日线窗口 + 周线窗口*6 + 月线窗口*22 + 额外缓冲(260)
    def _max_win(rules: List[dict], tf: str) -> int:
        wins = []
        for r in rules:
            if "clauses" in r:
                for c in r["clauses"]:
                    if c.get("timeframe","D").upper()==tf:
                        wins.append(int(c.get("window", SC_LOOKBACK_D)))
            else:
                if r.get("timeframe","D").upper()==tf:
                    wins.append(int(r.get("window", SC_LOOKBACK_D)))
        return max(wins or [0])
    d = max(_max_win(SC_RULES, "D"), _max_win(SC_PRESCREEN_RULES, "D"), SC_LOOKBACK_D, SC_PRESCREEN_LOOKBACK_D)
    w = max(_max_win(SC_RULES, "W"), _max_win(SC_PRESCREEN_RULES, "W"))
    m = max(_max_win(SC_RULES, "M"), _max_win(SC_PRESCREEN_RULES, "M"))
    days = d + w*6 + m*22 + 260
    start_dt = dt.datetime.strptime(ref_date, "%Y%m%d") - dt.timedelta(days=days)
    return start_dt.strftime("%Y%m%d")


def _last_true_lag(s_bool: pd.Series) -> int | None:
    """返回窗口内最后一次 True 距离末根的 lag（0=当日；None=从未为真）"""
    s = s_bool.dropna().astype(bool)
    if s.empty or not s.any():
        return None
    last_idx = np.where(s.values)[0][-1]
    return len(s) - 1 - int(last_idx)


def _recent_points(dfD: pd.DataFrame, rule: dict, ref_date: str) -> tuple[float, int|None, str|None]:
    """
    计算 RECENT 规则的加分：
    - rule['dist_points'] 支持两种写法：
      1) 列表形式：[[min,max,points], ...]
      2) 字典列表：[{min:0,max:0,points:3}, ...]
    返回 (加分, lag, 错误或None)
    """
    tf = str(rule.get("timeframe", "D")).upper()
    window = int(rule.get("window", SC_LOOKBACK_D))
    when = (rule.get("when") or "").strip()
    if not when:
        return 0.0, None, "空 when 表达式"

    dfTF = dfD if tf == "D" else _resample(dfD, tf)
    win_df = _window_slice(dfTF, ref_date, window)
    if win_df.empty:
        return 0.0, None, None

    s_bool = evaluate_bool(when, win_df).astype(bool)
    lag = _last_true_lag(s_bool)
    if lag is None:
        return 0.0, None, None

    bins = rule.get("dist_points") or rule.get("distance_points") or []
    # 允许两种写法混用
    def _pick_pts(lag_: int) -> float | None:
        for b in bins:
            if isinstance(b, dict):
                lo = int(b.get("min", b.get("lag", 0)))
                hi = int(b.get("max", b.get("lag", lo)))
                pts = float(b.get("points", 0))
            else:
                lo, hi, pts = int(b[0]), int(b[1]), float(b[2])
            if lo <= lag_ <= hi:
                return pts
        return None

    pts = _pick_pts(lag)
    return (float(pts) if pts is not None else 0.0), lag, None


def _select_columns_for_rules() -> List[str]:
    need = {"trade_date", "open", "high", "low", "close", "vol", "amount"}
    pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')
    def scan(script: str):
        for name in pattern.findall(script or ""):
            name_low = name.lower()
            if name_low in {"j","vr","bbi","z_score"}:
                need.add(name_low)

    def scan_rule(rr: dict):
        # 主体
        if rr.get("when"): scan(rr["when"])
        for c in rr.get("clauses", []) or []:
            if c.get("when"): scan(c["when"])
        # gate
        g = rr.get("gate") or {}
        if isinstance(g, dict):
            if g.get("when"): scan(g["when"])
            for c in g.get("clauses", []) or []:
                if c.get("when"): scan(c["when"])

    for rr in (SC_RULES or []):           scan_rule(rr)
    for rr in (SC_PRESCREEN_RULES or []): scan_rule(rr)
    return sorted(need)


def _write_detail_json(ts_code: str, ref_date: str, summary: dict, per_rules: list[dict]):
    try:
        base = os.path.join(SC_OUTPUT_DIR, "details", str(ref_date))
        os.makedirs(base, exist_ok=True)
        out_path = os.path.join(base, f"{ts_code}_{ref_date}.json")
        payload = {
            "ts_code": ts_code,
            "ref_date": ref_date,
            "summary": summary,      # {"score": float, "tiebreak": float|None, "highlights": [...], "drawbacks":[...]}
            "rules": per_rules       # 每条规则的细粒度命中情况（见下）
        }
        with open(out_path, "w", encoding="utf-8") as f:
            import json
            json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        LOGGER.debug("[detail] %s -> %s", ts_code, out_path)
    except Exception as e:
        LOGGER.warning("[detail] 写明细失败 %s: %s", ts_code, e)


def _list_true_dates(df: pd.DataFrame, expr: str, *, ref_date: str, window: int, timeframe: str) -> list[str]:
    """返回窗口内所有命中日期（按 timeframe 重采样后对齐到日线索引）。"""
    try:
        dfTF = df if timeframe == "D" else _resample(df, timeframe)
        win_df = _window_slice(dfTF, ref_date, window)
        sig = evaluate_bool(expr, win_df)
        if sig is None:
            return []
        idx = getattr(sig, "index", getattr(win_df, "index", []))
        out = [str(i) for i, v in zip(idx, list(sig)) if bool(v)]
        return out
    except Exception:
        return []


def _build_per_rule_detail(df: pd.DataFrame, ref_date: str) -> list[dict]:
    rows = []
    gate_ok=False; gate_norm={}
    for rule in _iter_unique_rules():
        name = str(rule.get("name", "<unnamed>"))
        scope = str(rule.get("scope", "ANY")).upper().strip()
        pts   = float(rule.get("points", 0))
        tf    = str(rule.get("timeframe", "D")).upper()
        win   = int(rule.get("window", SC_LOOKBACK_D))
        expl  = rule.get("explain")
        when  = None
        ok = False
        cnt = None
        add = 0.0
        err = None
        period = None
        try:
            try:
                period = _period_for_clause(df, rule, ref_date)
            except Exception:
                period = ref_date
            if scope in {"EACH", "PERBAR", "EACH_TRUE"}:
                cnt, err = _count_hits_perbar(df, rule, ref_date)
                dfTF = df if tf=="D" else _resample(df, tf)
                win_df = _window_slice(dfTF, ref_date, win)
                cand_when = None
                if "clauses" in rule and rule["clauses"]:
                    for c in rule["clauses"]:
                        if c.get("when"):
                            cand_when = c["when"]; break
                else:
                    cand_when = rule.get("when")
                hit_date = None
                if not err and cand_when:
                    _d, _ = _last_true_date(win_df, cand_when)
                    hit_date = _d
                ok = bool(cnt and cnt > 0 and err is None)
                add = float(pts * int(cnt or 0))
                gate_ok, gate_err, gate_norm = _eval_gate(df, rule, ref_date)
                if not gate_ok:
                    add = 0.0
                rows.append({
                    "name": name, "scope": scope, "timeframe": tf, "window": win, "period": period,
                    "points": pts, "ok": ok, "cnt": int(cnt or 0),
                    "add": add,
                    "hit_date": hit_date,
                    "hit_dates": _list_true_dates(df, (cand_when or rule.get("when") or ""), ref_date=ref_date, window=win, timeframe=tf),
                    "hit_count": int(cnt or 0),
                    "gate_ok": bool(gate_ok),
                    "gate_when": (gate_norm.get("when") if isinstance(gate_norm, dict) else None),
                    "explain": expl,
                })
                continue
            else:
                # === RECENT / DIST / NEAR: 按最近一次命中距今天数计分 ===
                if scope in {"RECENT", "DIST", "NEAR"}:
                    add, lag, err = _recent_points(df, rule, ref_date)
                    dfTF = df if tf=="D" else _resample(df, tf)
                    win_df = _window_slice(dfTF, ref_date, win)
                    hit_date = None
                    if err is None and lag is not None:
                        idx = win_df.index if isinstance(win_df.index, pd.DatetimeIndex) else pd.to_datetime(win_df["trade_date"].astype(str))
                        lag_i = int(lag)
                        if 0 <= lag_i < len(idx):
                            hit_date = idx[-1 - lag_i].strftime("%Y%m%d")
                        else:
                            # 理论不应发生；保守返回空，交由 UI 用 hit_dates 或其它信息展示
                            hit_date = None

                    rows.append({
                        "name": name, "scope": scope, "timeframe": tf, "window": win, "period": period,
                        "points": pts, "ok": ok, "cnt": None,
                        "add": add, "lag": (None if lag is None else int(lag)),
                        "hit_date": hit_date, "hit_dates": _list_true_dates(df, cand_when or (when or ""), ref_date=ref_date, window=win, timeframe=tf),
                        "hit_count": int(cnt or 0),
                        "explain": expl,
                        "gate_ok": bool(gate_ok),
                        "gate_when": (gate_norm.get("when") if isinstance(gate_norm, dict) else None),
                    })
                    continue
                # === 其余仍走原来的布尔命中路径 ===
                ok_eval, err_eval = _eval_rule(df, rule, ref_date)
                ok  = bool(ok_eval and not err_eval)
                err = err_eval
                add = (pts if ok else 0.0)
                when = rule.get("when")


            # === 追加：常规布尔规则命中日期信息 ===
            hit_date = None
            hit_dates = []
            gate_ok, gate_err, gate_norm = _eval_gate(df, rule, ref_date)
            try:
                if when:
                    dfTF2 = df if tf=="D" else _resample(df, tf)
                    win_df2 = _window_slice(dfTF2, ref_date, win)
                    if not win_df2.empty:
                        _d, _ = _last_true_date(win_df2, when)
                        hit_date = _d
                    hit_dates = _list_true_dates(df, when, ref_date=ref_date, window=win, timeframe=tf)
            except Exception:
                pass

        except Exception as e2:
            err = f"eval-exception: {e2}"

        rows.append({
            "name": name, "scope": scope, "timeframe": tf, "window": win, "period": period,
            "points": pts, "ok": ok, "cnt": (None if cnt is None else int(cnt)),
            "add": add, "hit_date": hit_date, "hit_dates": hit_dates, "hit_count": (len(hit_dates) if hit_dates else 0),
            "explain": expl, "error": err,
            "gate_ok": bool(gate_ok),
            "gate_when": (gate_norm.get("when") if isinstance(gate_norm, dict) else None),

        })
    return rows


def _build_score_all_from_details(ref_date: str) -> Path | None:
    """
    尝试读取 output/score/details/<ref_date>/*.json 聚合出全市场排名，并写出 score_all_<ref>.csv
    返回写出的文件路径；若该日没有 details，则返回 None
    """
    ddir = DETAIL_DIR / str(ref_date)
    if not ddir.exists():
        return None
    rows = []
    for p in ddir.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
            ts = obj.get("ts_code")
            sm = obj.get("summary") or {}
            score = sm.get("score")
            tb = sm.get("tiebreak")
            rows.append({"ts_code": ts, "score": float(score), "tiebreak_j": tb})
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows).dropna(subset=["ts_code", "score"])
    # 排序：总分降序、tiebreak 降序、代码升序
    df = df.sort_values(["score", "tiebreak_j", "ts_code"], ascending=[False, False, True])
    df["rank"] = np.arange(1, len(df) + 1)
    df["trade_date"] = str(ref_date)
    # 写全量排名文件
    outp = ALL_DIR / f"score_all_{ref_date}.csv"
    df.to_csv(outp, index=False, encoding="utf-8-sig")
    return outp


def ensure_score_all_for_date(ref_date: str, *, force: bool=False) -> Path:
    """
    确保某天存在 score_all_<ref>.csv：
    1) 已存在且非强制 → 直接返回；
    2) 尝试从 details 聚合写出；
    3) 若 details 也没有 → 调用当天 run_for_date(ref_date) 先生成，再回到第2步聚合。
    """
    outp = ALL_DIR / f"score_all_{ref_date}.csv"
    # 已存在且非强制，但若是 0 字节占位，仍视为“缺失”
    if outp.exists() and outp.stat().st_size > 0 and not force:
        return outp

    # 先尝试从 details 聚合
    built = _build_score_all_from_details(ref_date)
    if built and built.exists():
        return built

    # 没有 details，就先跑一遍评分（使用当前配置与 universe）
    # 注意：run_for_date 名称与签名按你文件里现有实现即可
    try:
        print(f"[backfill] run_for_date({ref_date}) ...")
        run_for_date(ref_date)  # 如果你的函数签名有其它参数，这里照你的实际传入
    except Exception as e:
        print(f"[backfill] run_for_date 失败：{e}")

    # 再聚合一次
    built2 = _build_score_all_from_details(ref_date)
    if built2 and built2.exists():
        return built2

    # 兜底：即便没有 details，也写一个空文件防止后续反复尝试
    outp.touch()
    return outp


def backfill_missing_ranks(start_date: str, end_date: str, *, force: bool=False) -> list[Path]:
    """
    扫描 [start_date, end_date] 窗口内的交易日，补齐 score_all_*.csv
    - force=True：无论已存在与否都重建（先删后建）
    - force=False：仅对缺失的日期补建
    交易日列表获取方式：以 details 目录和 all 目录下已有日期为准，若需要更严格的“全交易日”，可以在此改为读数据日历。
    """
    # # 推断窗口内可能的日期：用 details 与 all 下的目录/文件名并集；不严谨但实用
    # cand = set()
    # for p in DETAIL_DIR.glob("*"):
    #     if p.is_dir():
    #         cand.add(p.name)
    # for p in ALL_DIR.glob("score_all_*.csv"):
    #     cand.add(p.stem.replace("score_all_",""))
    # # 简单按字符串范围过滤
    # dates = sorted([d for d in cand if start_date <= d <= end_date])
    # # 如果 dates 为空，直接用 start_date..end_date（可能用户还没跑过任何一天）
    # if not dates:
    #     dates = [start_date, end_date]

    # 1) 优先用交易日日历生成完整日期序列
    cal_root = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
    cal = list_trade_dates(cal_root) or []
    if cal:
        dates = [d for d in cal if start_date <= d <= end_date]
    else:
        # 2) 没有日历时再退化为：details+all 的并集；若仍为空，则按自然日铺满
        cand = set()
        for p in DETAIL_DIR.glob("*"):
            if p.is_dir():
                cand.add(p.name)
        for p in ALL_DIR.glob("score_all_*.csv"):
            cand.add(p.stem.replace("score_all_",""))
        dates = sorted([d for d in cand if start_date <= d <= end_date])
        if not dates:
            # 自然日兜底（避免完全不跑）
            s = dt.datetime.strptime(start_date, "%Y%m%d")
            e = dt.datetime.strptime(end_date, "%Y%m%d")
            dates = [(s + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((e-s).days + 1)]

    written = []
    for d in dates:
        outp = ALL_DIR / f"score_all_{d}.csv"
        # 把 0 字节当成“缺失”；force 时直接删掉再重建
        if outp.exists():
            if force:
                try: outp.unlink()
                except Exception: pass
            elif outp.stat().st_size > 0:
                continue
        p = ensure_score_all_for_date(d, force=force)
        written.append(p)
    print(f"[backfill] done, wrote {len(written)} days.")
    return written


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "trade_date" in df.columns:
        idx = pd.to_datetime(df["trade_date"].astype(str))
        if not df.index.equals(idx):
            df = df.copy()
            df.index = idx
    return df


def _resample(dfD: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    从日线重采样为周线/⽉线，保留 O/H/L/C/V/AMOUNT 列名。
    timeframe: 'W' or 'M'
    """
    if timeframe.upper() not in {"W","M"}:
        return dfD
    dfD = _ensure_datetime_index(dfD)
    rule = "W-FRI" if timeframe.upper()=="W" else "M"
    agg = {
        "open":"first", "high":"max", "low":"min", "close":"last",
        "vol":"sum", "amount":"sum"
    }
    out = dfD.resample(rule).agg(agg).dropna(how="all")
    # 若有扩展列（如 j/vr），自动按“最后值”为主（不强制）
    extra_cols = [c for c in dfD.columns if c not in agg]
    for c in extra_cols:
        try:
            out[c] = dfD[c].resample(rule).last()
        except Exception:
            pass
    out["trade_date"] = out.index.strftime("%Y%m%d")
    return out


def _apply_universe_filter(codes: list[str], ref: str, uni: str | list[str] | None) -> tuple[list[str], str]:
    """
    根据 universe 过滤 codes。
    返回：(过滤后的codes, 来源标签)
    """
    uni = uni or "all"
    key = str(uni).lower()

    if isinstance(uni, (list, set, tuple)):
        sel = set(str(x) for x in uni)
        return [c for c in codes if c in sel], "custom"

    if key == "all":
        return codes, "all"

    if key.startswith("white"):
        sel = set(_read_cache_list_codes(ref, "whitelist"))
        return [c for c in codes if c in sel], "whitelist"

    if key.startswith("black"):
        sel = set(_read_cache_list_codes(ref, "blacklist"))
        return [c for c in codes if c in sel], "blacklist"

    if key.startswith("att"):  # attention
        sel = set(_load_attention_codes(ref))
        return [c for c in codes if c in sel], "attention"

    # 未识别则不动
    return codes, "all"


def _window_slice(dfTF: pd.DataFrame, ref_date: str, window: int) -> pd.DataFrame:
    dfTF = _ensure_datetime_index(dfTF)
    mask = dfTF.index <= dt.datetime.strptime(ref_date, "%Y%m%d")
    return dfTF.loc[mask].tail(int(window))


def _load_attention_codes(_ref: str) -> list[str]:
    """
    从 SC_OUTPUT_DIR/attention/ 下选择“end<=_ref”的最新一份：
      attention_{source}_{start}_{end}.csv
    source 优先用 SC_ATTENTION_SOURCE（如 'top'|'white'|'black'）
    返回 ts_code 列表；异常返回 []。会打 DEBUG 日志说明匹配的文件。
    """
    try:
        src = str(SC_ATTENTION_SOURCE or "top").lower()
        attn_dir = os.path.join(SC_OUTPUT_DIR, "attention")
        if not os.path.isdir(attn_dir):
            return []
        # 只匹配当前 source
        files = sorted(glob.glob(os.path.join(attn_dir, f"attention_{src}_*.csv")))
        if not files:
            return []
        # 选择 end<=_ref 的最新一期；否则退化为按文件名排序的最后一份
        import re as _re
        cand = []
        for f in files:
            name = os.path.basename(f)
            m = _re.match(rf"attention_{src}_(\d{{8}})_(\d{{8}})\.csv$", name)
            if m:
                start, end = m.group(1), m.group(2)
                if end <= _ref:
                    cand.append((end, f))
        pick = (sorted(cand)[-1][1] if cand else files[-1])
        df = pd.read_csv(pick, dtype=str)
        codes = df["ts_code"].astype(str).tolist() if "ts_code" in df.columns else []
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("[screen][范围] attention 源=%s 参考日=%s 匹配文件=%s 命中=%d",
                         src, _ref, os.path.basename(pick), len(codes))
        return codes
    except Exception as e:
        LOGGER.debug("[screen][范围] 加载特别关注清单失败：%s", e)
        return []


def _scope_hit(s_bool: pd.Series, scope: str) -> bool:
    """
    支持的 scope：
      - 基本：LAST | ANY | ALL | COUNT>=k | CONSEC>=m
      - 扩展：ANY_n | ALL_n   （n 为正整数；表示“在外层 window 内存在一个长度为 n 的连续子集”）
          ANY_n：该子集中 “when” 至少 1 天为 True
          ALL_n：该子集中 “when” 每天都为 True  （等价于“存在长度 n 的连续 True”）
    """
    s = s_bool.dropna().astype(bool)
    if s.empty:
        return False
    ss = scope.upper().strip()

    if ss == "LAST":
        return bool(s.iloc[-1])
    if ss == "ANY":
        return bool(s.any())
    if ss == "ALL":
        return bool(s.all())

    m1 = re.match(r"COUNT>?\s*=\s*(\d+)", ss)
    if m1:
        k = int(m1.group(1))
        return int(s.sum()) >= k

    m2 = re.match(r"CONSEC>?\s*=\s*(\d+)", ss)
    if m2:
        m = int(m2.group(1))
        # 最长 True 连续段
        arr = s.values.astype(int)
        best = cur = 0
        for v in arr:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best >= m

    # ---- 新增：ANY_n / ALL_n ----
    m3 = re.match(r"ANY[_\-]?(\d+)$", ss)
    if m3:
        n = int(m3.group(1))
        if len(s) < n:
            return False
        roll = s.rolling(n, min_periods=n).sum()
        hit = bool((roll >= 1).any())
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(f"[SCOPE] ANY_{n}: len={len(s)} max_in_subwin={int(roll.max())} -> {hit}")
        return hit

    m4 = re.match(r"ALL[_\-]?(\d+)$", ss)
    if m4:
        n = int(m4.group(1))
        if len(s) < n:
            return False
        roll = s.rolling(n, min_periods=n).sum()
        hit = bool((roll == n).any())
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(f"[SCOPE] ALL_{n}: len={len(s)} max_in_subwin={int(roll.max())} -> {hit}")
        return hit

    # 默认当作 ANY
    return bool(s.any())


def _eval_clause(dfD: pd.DataFrame, clause: dict, ref_date: str) -> Tuple[bool, Optional[str]]:
    """
    返回 (是否命中, 错误消息或 None)
    """
    tf = clause.get("timeframe", "D").upper()
    window = int(clause.get("window", SC_LOOKBACK_D))
    scope = clause.get("scope", "ANY")
    when = clause.get("when", "").strip()
    if not when:
        return False, "空 when 表达式"
    try:
        dfTF = dfD if tf=="D" else _resample(dfD, tf)
        win_df = _window_slice(dfTF, ref_date, window)
        if win_df.empty:
            return False, f"窗口数据为空: tf={tf}, window={window}"
        s_bool = evaluate_bool(when, win_df)
        hit = _scope_hit(s_bool, scope)
        return hit, None
    except Exception as e:
        return False, f"表达式错误: {e}"


def _eval_rule(dfD: pd.DataFrame, rule: dict, ref_date: str) -> Tuple[bool, Optional[str]]:
    """
    支持两种写法：
      - 简单：{ timeframe, window, scope, when }
      - 复合：{ clauses: [ {tf,window,scope,when}, ... ] }
    命中逻辑：复合规则要求所有子句均命中。
    """
    if "clauses" in rule and rule["clauses"]:
        for c in rule["clauses"]:
            ok, err = _eval_clause(dfD, c, ref_date)
            if err:
                # 异常分支记日志
                LOGGER.warning(f"[RULE-ERR] {rule.get('name','<unnamed>')} 子句异常: {err}")
                return False, err
            if not ok:
                return False, None
        return True, None
    else:
        ok, err = _eval_clause(dfD, rule, ref_date)
        if err:
            LOGGER.warning(f"[RULE-ERR] {rule.get('name','<unnamed>')} 异常: {err}")
            return False, err
        return ok, None


def _read_stock_df(ts_code: str, start: str, end: str, columns: List[str]) -> pd.DataFrame:
    """
    读取某只股票在给定区间的日线数据。
    - 若存在 single 目录：直接读取 single/{ts_code}.parquet，并在内存中过滤日期；**不裁列**（保留所有已经计算的列）
    - 否则回退到 daily 读取（read_range，仍按需裁列以控制扫描成本）
    """
    if _has_single_dir():
        f = os.path.join(_single_root_dir(), f"{ts_code}.parquet")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"single 文件不存在: {f}")
        df = pd.read_parquet(f)
        # 统一为字符串比较 + 区间切片
        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].astype(str)
            mask = (df["trade_date"] >= str(start)) & (df["trade_date"] <= str(end))
            df = df.loc[mask].copy()
        # ✅ 不再按 columns 裁列：保留 single_* 中的全量指标/特征
        LOGGER.debug(f"[{ts_code}] single列保留: n_cols={len(df.columns)} 前12列={list(df.columns)[:12]}")
    else:
        # 按日分区读取时，仍然只取所需列以提高效率
        df = read_range(PARQUET_BASE, "stock", PARQUET_ADJ, ts_code, start, end, columns=columns)
    
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str)

    # 兜底 J：tie-break 需要而缺列时再即时计算
    if ("j" not in df.columns) and (SC_TIE_BREAK or "").lower() in {"kdj_j_asc", "kdj_j_desc"}:
        try:
            j = kdj(df)
            df = df.copy()
            df["j"] = j
        except Exception as e:
            LOGGER.warning(f"[{ts_code}] 计算 KDJ J 失败：{e}")
    return df


def _trade_span(start: Optional[str], end: str) -> List[str]:
    """把自然日区间换成交易日清单（基于现有分区）。"""
    root_daily = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
    days = list_trade_dates(root_daily) or []
    days = [d for d in days if d <= end]
    if start:
        days = [d for d in days if d >= start]
    # 若 start 未给，截取最近 N 个交易日
    if not start:
        days = days[-int(SC_ATTENTION_WINDOW_D):]
    return days


def _sanitize_code(ts_code: str) -> str:
    """把 '399300.SZ' 变成可用作变量名的 '399300_SZ'。"""
    import re
    return re.sub(r'[^A-Za-z0-9_]+', '_', (ts_code or "")).strip('_').upper()


@functools.lru_cache(maxsize=8)
def _load_benchmark_map(start: str, end: str, codes_tuple: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """一次加载并缓存基准指数，避免每只股票重复 IO。"""
    codes = list(codes_tuple or ())
    bm: dict[str, pd.DataFrame] = {}
    if not codes:
        return bm
    for code in codes:
        try:
            df_i = read_range(
                PARQUET_BASE, "index", "daily",
                ts_code=code, start=start, end=end,
                columns=["trade_date","close"], limit=None
            )
            if df_i is None or df_i.empty:
                LOGGER.warning("[bench] 指数空数据: %s (%s~%s)", code, start, end); continue
            df_i = df_i.copy()
            df_i["trade_date"] = df_i["trade_date"].astype(str)
            df_i = df_i.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
            df_i["close"] = pd.to_numeric(df_i["close"], errors="coerce")
            df_i["ret"]   = df_i["close"].pct_change()
            bm[code] = df_i
        except Exception as e:
            LOGGER.warning("[bench] 读取失败 %s: %s", code, e)
    LOGGER.debug("[bench] 加载完成 codes=%s 覆盖=%d", codes, len(bm))
    return bm


def _inject_benchmark_features(dfD: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    把基准指数的特征并进 dfD（按 trade_date 对齐），输出列名示例：
      - IDX_399300_SZ_CLOSE, RET_399300_SZ
      - RS_399300_SZ_20, EXRET_399300_SZ, EXRET_SUM_399300_SZ_20
      - BETA_399300_SZ_20, CORR_399300_SZ_20
    这些名字都是合法标识符，能直接写到 TDX 规则里。
    """
    codes = tuple(SC_BENCH_CODES or [])
    if not codes:
        return dfD

    if ("trade_date" not in dfD.columns) or ("close" not in dfD.columns):
        LOGGER.warning("[bench] df 缺列 trade_date/close，跳过注入"); return dfD

    base = dfD.copy()
    base["trade_date"] = base["trade_date"].astype(str)
    c = pd.to_numeric(base["close"], errors="coerce")
    stock_ret = c.pct_change()

    bm_map = _load_benchmark_map(start, end, codes)
    if not bm_map:
        LOGGER.warning("[bench] 无可用基准（可能未下载指数分区），跳过注入"); return base

    W = int(SC_BENCH_WINDOW or 20)

    for code, idx in bm_map.items():
        tag = _sanitize_code(code)
        # 左连接对齐（或仅保留共同交易日）
        merged = base[["trade_date"]].merge(
            idx[["trade_date","close","ret"]].rename(columns={
                "close": f"IDX_{tag}_CLOSE", "ret": f"RET_{tag}"
            }),
            on="trade_date", how="left"
        )

        if str(SC_BENCH_FILL).lower() == "ffill":
            merged[[f"IDX_{tag}_CLOSE", f"RET_{tag}"]] = merged[[f"IDX_{tag}_CLOSE", f"RET_{tag}"]].ffill()
        else:
            # 丢弃无共同日的行（对该指数而言）
            mask = merged[f"IDX_{tag}_CLOSE"].notna()
            merged = merged.loc[mask].reset_index(drop=True)
            base  = base.loc[mask].reset_index(drop=True)
            c = pd.to_numeric(base["close"], errors="coerce")
            stock_ret = c.pct_change()

        base[f"IDX_{tag}_CLOSE"] = merged[f"IDX_{tag}_CLOSE"]
        base[f"RET_{tag}"]       = merged[f"RET_{tag}"]

        # === 特征集（可按 config 开关） ===
        if "exret" in SC_BENCH_FEATURES:
            base[f"EXRET_{tag}"] = stock_ret - base[f"RET_{tag}"]
            base[f"EXRET_SUM_{tag}_{W}"] = base[f"EXRET_{tag}"].rolling(W, min_periods=max(5, W//3)).sum()

        if "rs" in SC_BENCH_FEATURES:
            ratio = c / base[f"IDX_{tag}_CLOSE"]
            base[f"RS_{tag}_{W}"] = ratio / ratio.rolling(W, min_periods=max(5, W//3)).mean()

        if "beta" in SC_BENCH_FEATURES or "corr" in SC_BENCH_FEATURES:
            ir = base[f"RET_{tag}"]
            if "beta" in SC_BENCH_FEATURES:
                cov = stock_ret.rolling(W).cov(ir)
                var = ir.rolling(W).var().replace(0, np.nan)
                base[f"BETA_{tag}_{W}"] = cov / var
            if "corr" in SC_BENCH_FEATURES:
                base[f"CORR_{tag}_{W}"] = stock_ret.rolling(W).corr(ir)

        LOGGER.debug(
            "[bench] 注入 %s: 窗口=%d 列增量=%s",
            code, W, [col for col in base.columns if col.endswith(tag)]
        )

    return base


# ------------------------- 初选（预筛） -------------------------
@dataclass
class PrescreenResult:
    passed: bool
    reason: Optional[str]          # 若淘汰，解释原因
    period: Optional[str]          # 写入缓存的 period（日期或区间）


def _period_for_clause(dfD: pd.DataFrame, clause: dict, ref_date: str) -> str:
    tf = clause.get("timeframe", "D").upper()
    window = int(clause.get("window", SC_PRESCREEN_LOOKBACK_D))
    dfTF = dfD if tf=="D" else _resample(dfD, tf)
    win = _window_slice(dfTF, ref_date, window)
    if win.empty:
        return ref_date
    start = str(win["trade_date"].iloc[0])
    end = str(win["trade_date"].iloc[-1])
    return f"{start}-{end}" if start!=end else end


def _prescreen(dfD: pd.DataFrame, ref_date: str) -> PrescreenResult:
    """
    命中任一 hard_penalty 规则即淘汰。
    reason：命中规则的 reason 文案（若未提供，用 name）。
    """
    for rule in (SC_PRESCREEN_RULES or []):
        hard = bool(rule.get("hard_penalty", False))
        if not hard:
            continue
        if "clauses" in rule and rule["clauses"]:
            clause_periods = []
            all_ok = True
            any_err = None
            for c in rule["clauses"]:
                ok, err = _eval_clause(dfD, c, ref_date)
                if err:
                    any_err = err
                    break
                if not ok:
                    all_ok = False
                    break
                clause_periods.append(_period_for_clause(dfD, c, ref_date))
            if any_err:
                LOGGER.warning(f"[PRESCREEN-ERR] {rule.get('name')} 子句异常: {any_err}")
                continue
            if all_ok:
                period = ";".join(clause_periods) if clause_periods else ref_date
                reason = rule.get("reason") or rule.get("name") or "prescreen"
                return PrescreenResult(False, reason, period)
        else:
            ok, err = _eval_rule(dfD, rule, ref_date)
            if err:
                LOGGER.warning(f"[PRESCREEN-ERR] {rule.get('name')} 异常: {err}")
                continue
            if ok:
                period = _period_for_clause(dfD, rule, ref_date)
                reason = rule.get("reason") or rule.get("name") or "prescreen"
                return PrescreenResult(False, reason, period)
    return PrescreenResult(True, None, None)


# ------------------------- 打分主体 -------------------------
@dataclass
class ScoreDetail:
    ts_code: str
    score: float
    highlights: List[str]
    drawbacks: List[str]
    opportunities: List[str]
    tiebreak: Optional[float]


def _score_one(ts_code: str, ref_date: str, start_date: str, columns: List[str]) -> Optional[Tuple[str, Optional[ScoreDetail], Optional[Tuple[str,str,str]]]]:
    """
    返回：
      - (ts_code, detail 或 None, 黑名单项 或 None)
      若被初选淘汰，返回黑名单项 (ts_code, period, reason)；否则返回打分 detail。
    """
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "top"), exist_ok=True)
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "all"), exist_ok=True)

    try:
        df = _read_stock_df(ts_code, start_date, ref_date, columns)
        if df is None or df.empty:
            LOGGER.warning(f"[{ts_code}] 数据为空，跳过")
            # return (ts_code, None, ("", "", "数据为空"))
            return (ts_code, None, (ts_code, ref_date, "数据为空"))
        
        df = _inject_benchmark_features(df, start_date, ref_date)
        tdx.EXTRA_CONTEXT.update(get_eval_env(ts_code, ref_date))

        # 初选
        pres = _prescreen(df, ref_date)
        if not pres.passed:
            return (ts_code, None, (ts_code, pres.period or ref_date, pres.reason or "prescreen"))
        # 打分
        score = float(SC_BASE_SCORE)
        highlights, drawbacks, opportunities = [], [], []
        def _append_reason_by_rule(points: float, rule: dict, tag_text: str | None = None):
            """
            根据 rule 配置决定把“理由”放进哪一个桶：
            - rule.get('as') 可选：'opportunity' | 'highlight' | 'drawback' | 'auto'(默认)
            - rule.get('show_reason', True)：是否把 explain 文字展示在汇总里
            """
            exp = rule.get("explain")
            show = bool(rule.get("show_reason", True))  # 你要求：规则不默认隐藏
            if (not exp) or (not show):
                return
            bucket = (rule.get("as") or "auto").lower().strip()
            if bucket == "auto":
                bucket = "highlight" if float(points) >= 0 else "drawback"
            text = str(tag_text) if tag_text else str(exp)
            if bucket == "opportunity":
                opportunities.append(text)
            elif bucket == "highlight":
                highlights.append(text)
            else:
                drawbacks.append(text)


        for rule in _iter_unique_rules():
            scope = str(rule.get("scope", "ANY")).upper().strip()
            if scope in {"EACH", "PERBAR", "EACH_TRUE"}:
                pts = float(rule.get("points", 0))
                cnt, err = _count_hits_perbar(df, rule, ref_date)
                if err:
                    LOGGER.warning(f"[{ts_code}] EACH计数回退：{rule.get('name')} -> {err}")
                    # 回退为普通布尔命中
                    ok, err2 = _eval_rule(df, rule, ref_date)
                    if err2:
                        LOGGER.warning(f"[{ts_code}] 规则异常：{rule.get('name')} -> {err2}")
                        continue
                    if ok:
                        score += pts
                        expl = rule.get("explain")
                        if expl:
                            (highlights if pts >= 0 else drawbacks).append(str(expl))
                        LOGGER.debug(f"[HIT][{ts_code}] {rule.get('name','<unnamed>')} +{pts} => {score}")
                else:
                    if cnt > 0 and pts != 0:
                        gate_ok, gate_err, _ = _eval_gate(df, rule, ref_date)
                        if gate_ok:
                            add = pts * int(cnt)
                            score += add
                            tag = f"{rule.get('explain')}×{cnt}" if rule.get('explain') else None
                            _append_reason_by_rule(add, rule, tag_text=tag)
                            LOGGER.debug(f"[HIT][{ts_code}] {rule.get('name','<unnamed>')} EACH 命中 {cnt} 次，+{add} => {score}")
                        else:
                            LOGGER.debug(f"[GATE][{ts_code}] {rule.get('name')} 被 gate 拦截: {gate_err or ''}")
                    else:
                        LOGGER.debug(f"[MISS][{ts_code}] {rule.get('name','<unnamed>')} EACH 无命中")
                continue  # EACH 已处理完，下一条
            
            if scope in {"RECENT", "DIST", "NEAR"}:
                tf  = str(rule.get("timeframe", "D")).upper()
                win = int(rule.get("window", SC_LOOKBACK_D))
                try:
                    period = _period_for_clause(df, rule, ref_date)
                except Exception:
                    period = ref_date

                add, lag, err = _recent_points(df, rule, ref_date)
                dfTF = df if tf=="D" else _resample(df, tf)
                win_df = _window_slice(dfTF, ref_date, win)
                hit_date = None
                if err is None and lag is not None:
                    idx = win_df.index if isinstance(win_df.index, pd.DatetimeIndex) else pd.to_datetime(win_df["trade_date"].astype(str))
                    lag_i = int(lag)
                    if 0 <= lag_i < len(idx):
                        hit_date = idx[-1 - lag_i].strftime("%Y%m%d")

                # 供 hit_dates 使用的表达式
                cand_when = None
                if "clauses" in rule and rule["clauses"]:
                    for c in rule["clauses"]:
                        if c.get("when"): cand_when = c["when"]; break
                when_for_hits = (cand_when or rule.get("when") or "")

                # gate 同样参与详情
                gate_ok, gate_err, gate_norm = _eval_gate(df, rule, ref_date)
                add2 = float(add if gate_ok else 0.0)
                ok2 = bool(add != 0)

                continue
            # —— 常规路径 ——
            ok, err = _eval_rule(df, rule, ref_date)
            if err:
                LOGGER.warning(f"[{ts_code}] 规则异常：{rule.get('name')} -> {err}")
                continue
            if ok:
                gate_ok, gate_err, _ = _eval_gate(df, rule, ref_date)
                if gate_ok:
                    pts = float(rule.get("points", 0))
                    score += pts
                    _append_reason_by_rule(pts, rule)
                    LOGGER.debug(f"[HIT][{ts_code}] {rule.get('name','<unnamed>')} +{pts} => {score}")
                else:
                    LOGGER.debug(f"[GATE][{ts_code}] {rule.get('name')} 被 gate 拦截: {gate_err or ''}")
            else:
                LOGGER.debug(f"[MISS][{ts_code}] {rule.get('name','<unnamed>')}")

        score = max(score, float(SC_MIN_SCORE))
        # tiebreak: KDJ J（低的在前）
        tb = None
        if (SC_TIE_BREAK or "").lower() in {"kdj_j_asc", "kdj_j"}:
            try:
                # 取参考日所在行的 j；若没有精确匹配，就用最后一行
                row = df[df["trade_date"]==ref_date]
                if not row.empty and "j" in row.columns:
                    tb = float(row["j"].iloc[-1])
                elif "j" in df.columns:
                    tb = float(df["j"].iloc[-1])
                else:
                    tb = None
            except Exception as e:
                LOGGER.warning(f"[{ts_code}] 提取 J 失败：{e}")
                tb = None
        # sanitize tiebreak to avoid NaN/Inf in JSON
        try:
            import math as _math
            if tb is None:
                tb2 = None
            elif isinstance(tb, float) and (_math.isnan(tb) or _math.isinf(tb)):
                tb2 = None
            else:
                tb2 = float(tb)
        except Exception:
            tb2 = None
        try:
            per_rules = _build_per_rule_detail(df, ref_date)
            _write_detail_json(
                ts_code, ref_date,
                summary={
                    "score": float(score),
                    "tiebreak": tb2,
                    "highlights": list(highlights),
                    "drawbacks": list(drawbacks),
                    "opportunities": list(opportunities),
                },
                per_rules=per_rules
            )
        except Exception as e:
            LOGGER.warning("[detail] 写单票明细失败 %s: %s", ts_code, e)
        return (ts_code, ScoreDetail(ts_code, score, highlights, drawbacks, opportunities, tb), None)

    except Exception as e:
        LOGGER.error(f"[{ts_code}] 评分失败：{e}")
        # 若 df 已读到，尽量仍生成一份规则明细，便于排错
        try:
            if 'df' in locals() and isinstance(df, pd.DataFrame) and (not df.empty):
                per_rules = _build_per_rule_detail(df, ref_date)
                _write_detail_json(
                    ts_code, ref_date,
                    summary={"score": float(SC_BASE_SCORE), "tiebreak": None,
                             "highlights": [], "drawbacks": []},
                    per_rules=per_rules
                )
        except Exception as e2:
            LOGGER.warning("[detail] 构建/写入失败 %s: %s", ts_code, e2)
        return (ts_code, None, (ts_code, ref_date, f"评分失败:{e}"))


def _count_hits_perbar(dfD: pd.DataFrame, rule_or_clause: dict, ref_date: str) -> Tuple[int, Optional[str]]:
    """
    统计“窗口内逐K命中次数”（用于 scope=EACH / PERBAR）。
    - 单条 rule（无 clauses）：在其 timeframe 上按 when 逐K求布尔，再在 window 内计数 True。
    - 带 clauses 的 rule：要求所有子句 timeframe/window 一致；逐K按 AND 融合后再计数。
    - 返回 (count, err)，err=None 表示成功；err 不为空时建议回退为布尔命中。
    """
    try:
        if "clauses" in rule_or_clause and rule_or_clause["clauses"]:
            clauses = rule_or_clause["clauses"]
            tfs = [str(c.get("timeframe","D")).upper() for c in clauses]
            wins = [int(c.get("window", SC_LOOKBACK_D)) for c in clauses]
            if len(set(tfs)) != 1 or len(set(wins)) != 1:
                return 0, "EACH 目前不支持多 timeframe/window 子句混用"
            tf = tfs[0]
            window = wins[0]
            dfTF = dfD if tf=="D" else _resample(dfD, tf)
            win_df = _window_slice(dfTF, ref_date, window)
            if win_df.empty:
                return 0, None
            s_all = None
            for c in clauses:
                when = (c.get("when") or "").strip()
                if not when:
                    return 0, "空 when 表达式"
                s = evaluate_bool(when, win_df).astype(bool)
                s_all = s if s_all is None else (s_all & s)
            return int(s_all.fillna(False).sum()), None
        else:
            tf = str(rule_or_clause.get("timeframe", "D")).upper()
            window = int(rule_or_clause.get("window", SC_LOOKBACK_D))
            when = (rule_or_clause.get("when") or "").strip()
            if not when:
                return 0, "空 when 表达式"
            dfTF = dfD if tf=="D" else _resample(dfD, tf)
            win_df = _window_slice(dfTF, ref_date, window)
            if win_df.empty:
                return 0, None
            s_bool = evaluate_bool(when, win_df).astype(bool)
            return int(s_bool.fillna(False).sum()), None
    except Exception as e:
        return 0, f"表达式错误: {e}"


def build_attention_rank(start: Optional[str] = None,
                         end: Optional[str] = None,
                         source: Optional[str] = None,
                         min_hits: Optional[int] = None,
                         topN: Optional[int] = None,
                         write: bool = True,
                         mode: str = "strength",
                         weight_mode: Optional[str] = None,
                         half_life: Optional[float] = None,
                         linear_min: Optional[float] = None,
                         topM: Optional[int] = None):
    """
    特别关注榜（两种口径）：
      - mode='strength'（默认）：最近窗口内“排名强度”（按每日横截面名次分位值累计）；
      - mode='hits'           ：旧逻辑，按“上榜次数”统计。
    窗口：若未给 start，则取最近 SC_ATTENTION_WINDOW_D 个交易日，以 end/ref_date 为右端点。
    输出：attention_{source}_{span_start}_{span_end}.csv（会直接覆盖旧文件）。
    """
    end = end or _pick_ref_date()
    source = (source or SC_ATTENTION_SOURCE).lower()
    min_hits = int(min_hits or SC_ATTENTION_MIN_HITS)
    topN = int(topN or SC_ATTENTION_TOP_K)

    span = _trade_span(start, end)
    if not span:
        LOGGER.warning("[attention] 交易日窗口为空，start=%s end=%s", start, end)
        return None

    mode = (mode or "strength").lower()
    if mode == "hits" or source in ("white", "black"):
        # ===== 旧口径：命中次数 =====
        hit_cnt: Dict[str, int] = {}
        first_last: Dict[str, Tuple[str,str]] = {}
        for d in span:
            date_str = str(d)
            if source == "top":
                f = os.path.join(SC_OUTPUT_DIR, "top", f"score_top_{d}.csv")
            elif source in ("white", "black"):
                f = os.path.join(SC_CACHE_DIR, d, f"{'whitelist' if source=='white' else 'blacklist'}.csv")
            else:
                raise ValueError("source must be 'top' | 'white' | 'black'")
            if not os.path.isfile(f):
                continue
            try:
                df = pd.read_csv(f, usecols=lambda c: c in {"ts_code", "ref_date"} if isinstance(c, str) else True, dtype={"ts_code": str}, engine="c")
                
                if "trade_date" not in df.columns:
                    # 有些版本写了 ref_date；如果也没有，就直接用文件名里的 date_str
                    if "ref_date" in df.columns:
                        df = df.rename(columns={"ref_date": "trade_date"})
                    else:
                        df["trade_date"] = date_str

                df = normalize_trade_date(df, "trade_date")
                df = df[df["trade_date"] == date_str]
                codes = df["ts_code"].astype(str).tolist() if "ts_code" in df.columns else []
            except Exception as e:
                LOGGER.warning("[attention] 读取失败 %s: %s", f, e);
                continue
            for c in codes:
                hit_cnt[c] = hit_cnt.get(c, 0) + 1
                first_last[c] = (first_last.get(c, (d, d))[0], d)
        if not hit_cnt:
            LOGGER.warning("[attention] %s~%s 内没有 '%s' 记录。", span[0], span[-1], source)
            return None
        rows = [
            {"ts_code": c, "hits": n, "first": first_last[c][0], "last": first_last[c][1]}
            for c, n in hit_cnt.items() if n >= min_hits
        ]
        if not rows:
            LOGGER.info("[attention] 命中数 >= %d 的为空。", min_hits); return None
        out_df = pd.DataFrame(rows).sort_values(["hits", "last", "ts_code"], ascending=[False, False, True]).head(topN)
    else:
        # ===== 新口径：排名强度（带时间权重） =====
        w_mode = (weight_mode or "none").lower()
        hl = float(half_life if half_life is not None else 5)
        lm = float(linear_min if linear_min is not None else 0.3)
        weights = _make_time_weights(len(span), w_mode, hl, lm, normalize=True)
        strength: Dict[str, float] = {}
        sum_w: Dict[str, float] = {}
        hits: Dict[str, int] = {}
        best_rank: Dict[str, int] = {}
        last_rank: Dict[str, int] = {}
        first_last: Dict[str, Tuple[str,str]] = {}
        for idx, d in enumerate(span):
            w = float(weights[idx])
            path_all = os.path.join(SC_OUTPUT_DIR, "all", f"score_all_{d}.csv")
            df = None; total = 0
            if os.path.isfile(path_all):
                try:
                    df = pd.read_csv(path_all, dtype={"ts_code": str})
                    if df.empty or "rank" not in df.columns:
                        df = None
                    else:
                        total = len(df)
                        df = df[["ts_code","rank"]].copy()
                        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
                        df = df.dropna()
                except Exception as e:
                    LOGGER.warning("[attention] 读取全量排名失败 %s: %s", path_all, e)
                    df = None
            if df is None:
                # 回退：仅有 Top-K 文件时，以其相对名次近似
                path_top = os.path.join(SC_OUTPUT_DIR, "top", f"score_top_{d}.csv")
                if not os.path.isfile(path_top):
                    continue
                try:
                    df_top = pd.read_csv(path_top, dtype={"ts_code": str})
                except Exception as e:
                    LOGGER.warning("[attention] 读取 Top 失败 %s: %s", path_top, e); continue
                total = len(df_top)
                if total == 0:
                    continue
                df = df_top[["ts_code"]].copy()
                df["rank"] = np.arange(1, total + 1)
            if total <= 0 or df is None or df.empty:
                continue
            # 仅统计 TopM（可选）
            if topM:
                df = df[df["rank"].astype(int) <= int(topM)]
                if df.empty:
                    continue
            # 分位强度（名次越靠前 -> p 越接近 1）
            df["p"] = 1.0 - (df["rank"].astype(float) - 1.0) / float(total)            
            # 使用向量化聚合，显著减少 Python 级循环开销
            g = (df.groupby("ts_code", sort=False)
                    .agg(rank_min=("rank","min"),
                         rank_last=("rank","last"),
                         p_sum=("p","sum"),
                         hits_part=("rank","size"))
                 )
            # 累加至全局字典
            # strength: 加权后的分位强度累计；sum_w: 对应权重累计；hits: 次数；best_rank: 全期最优；last_rank: 全期最后一次
            for c, row in g.iterrows():
                s_add = float(row["p_sum"]) * w
                strength[c] = strength.get(c, 0.0) + s_add
                sum_w[c] = sum_w.get(c, 0.0) + w
                hits[c] = hits.get(c, 0) + int(row["hits_part"])
                best_rank[c] = min(best_rank.get(c, 10**9), int(row["rank_min"]))
                last_rank[c] = int(row["rank_last"])
                # first/last 出现日期
                fl = first_last.get(c, (d, d))
                first_last[c] = (fl[0], d) if fl[0] <= d else (d, d)
                
        if not strength:
            LOGGER.warning("[attention] %s~%s 无可计算的排名强度。", span[0], span[-1]); return None
        rows = []
        for c, s in strength.items():
            n = hits.get(c, 0)
            if n >= min_hits:
                f, l = first_last.get(c, (span[0], span[-1]))
                rows.append({
                    "ts_code": c,
                    "strength": round(float(s), 6),
                    "mean_strength": round(float(s) / float(max(sum_w.get(c, 1.0), 1e-9)), 6),
                    "hits": int(n),
                    "best_rank": int(best_rank.get(c, 10**9)),
                    "last_rank": int(last_rank.get(c, 10**9)),
                    "first": f, "last": l
                })
        if not rows:
            LOGGER.info("[attention] 强度口径下，命中数 >= %d 的为空。", min_hits); return None
        out_df = (pd.DataFrame(rows)
                  .sort_values(["strength", "last", "best_rank", "ts_code"],
                               ascending=[False, False, True, True])
                  .head(topN))

    out_dir = os.path.join(SC_OUTPUT_DIR, "attention")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"attention_{source}_{span[0]}_{span[-1]}.csv")

    if write:
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        LOGGER.info("[特别关注][%s|w=%s] source=%s 窗口=%s~%s min_hits>=%d Top-%d -> %s",
                    mode, (weight_mode or "none"), source, span[0], span[-1], min_hits, topN, out_path)

        return out_path
    else:
        return out_df


def backfill_attention_rolling(start: str,
                               end: Optional[str] = None,
                               source: Optional[str] = None,
                               min_hits: Optional[int] = None,
                               topN: Optional[int] = None,
                               mode: str = "strength",
                               weight_mode: Optional[str] = None,
                               half_life: Optional[float] = None,
                               linear_min: Optional[float] = None,
                               topM: Optional[int] = None):
    """
    从start到end（按交易日）逐日滚动补算“特别关注榜”：
    - 每个 end 日都会生成一份：窗口=最近 SC_ATTENTION_WINDOW_D 个交易日（与 build_attention_rank 保持一致）
    - source: 'top' | 'white' | 'black'
    """
    end = end or _pick_ref_date()
    days = _trade_span(start, end)
    if not days:
        LOGGER.warning("[attention/backfill] 窗口为空：start=%s end=%s", start, end)
        return

    LOGGER.info("[attention/backfill] 逐日补算 %s ~ %s，共 %d 个交易日", days[0], days[-1], len(days))
    ok, fail = 0, 0
    for e in days:
        try:
            build_attention_rank(start=None, end=e, source=source,
                                min_hits=min_hits, topN=topN, write=True, mode=mode,
                                weight_mode=weight_mode, half_life=half_life,
                                linear_min=linear_min, topM=topM)
            ok += 1
        except Exception as ex:
            fail += 1
            LOGGER.warning("[attention/backfill] 生成 %s 失败：%s", e, ex)
    LOGGER.info("[attention/backfill] 完成：成功 %d，失败 %d", ok, fail)


# ------------------------- 名单缓存 I/O -------------------------
def _ensure_dirs(ref_date: str):
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "top"), exist_ok=True)
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "details"), exist_ok=True)
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "all"), exist_ok=True)
    os.makedirs(os.path.join(SC_CACHE_DIR, ref_date), exist_ok=True)


def _write_cache_lists(ref_date: str, whitelist: List[Tuple[str,str,str]], blacklist: List[Tuple[str,str,str]]):
    """
    写 cache/scorelists/{date}/white/blacklist.csv
    三列：ts_code,period,reason
    """
    base = os.path.join(SC_CACHE_DIR, ref_date)
    os.makedirs(base, exist_ok=True)
    if whitelist:
        dfw = pd.DataFrame(whitelist, columns=["ts_code","period","reason"])
        dfw.to_csv(os.path.join(base, "whitelist.csv"), index=False, encoding="utf-8-sig")
    else:
        # 空也写个文件，防止下次找不到
        pd.DataFrame(columns=["ts_code","period","reason"]).to_csv(os.path.join(base, "whitelist.csv"), index=False, encoding="utf-8-sig")
    if blacklist:
        dfb = pd.DataFrame(blacklist, columns=["ts_code","period","reason"])
        dfb.to_csv(os.path.join(base, "blacklist.csv"), index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["ts_code","period","reason"]).to_csv(os.path.join(base, "blacklist.csv"), index=False, encoding="utf-8-sig")


def _read_cache_list_codes(ref_date: str, list_name: str) -> List[str]:
    f = os.path.join(SC_CACHE_DIR, ref_date, f"{list_name}.csv")
    if not os.path.isfile(f):
        return []
    try:
        df = pd.read_csv(f)
        if "ts_code" in df.columns:
            return sorted(set(df["ts_code"].astype(str).tolist()))
    except Exception as e:
        LOGGER.warning(f"[名单读取失败] {f}: {e}")
    return []


# ------------------------- 对外入口 -------------------------
def run_for_date(ref_date: Optional[str] = None) -> str:
    """
    执行一次评分流程：
      1) 选参考日（默认最新分区）
      2) 估算读取起始日
      3) 构建股票清单
      4) 初选 -> 写黑白名单
      5) 白名单并行评分 -> Top-K CSV 落盘

    返回 Top-K CSV 路径。
    """
    # 1) 参考日
    if not ref_date:
        ref_date = _pick_ref_date()
    _progress("select_ref_date", message=f"ref={ref_date}")

    # 2) 估算读取起始日
    start_date = _compute_read_start(ref_date)
    _progress("compute_read_window", message=f"{start_date} ~ {ref_date}")
    LOGGER.info(f"[范围] 读取区间：{start_date} ~ {ref_date}")

    # 3) 股票清单（来自分区目录文件名）
    codes = _list_codes_for_day(ref_date)
    # —— 打分范围先过滤 —— 
    codes0 = list(codes)
    codes, src = _apply_universe_filter(codes0, ref_date, SC_UNIVERSE)
    _progress("build_universe_done", total=len(codes0), current=len(codes),
              message=f"src={src} universe={SC_UNIVERSE}")
    # LOGGER.info("[范围] 评分 universe=%s 源=%s 原=%d -> %d",
    #             str(SC_UNIVERSE), src, len(codes0), len(codes))
    if not codes:
        raise RuntimeError("评分范围为空：请检查 SC_UNIVERSE 或对应名单是否已生成")

    LOGGER.info(f"[UNIVERSE] {ref_date} 共 {len(codes)} 只股票")
    init_rank_env(ref_date, codes, PARQUET_BASE, PARQUET_ADJ, default_N=60)
    # 4/5) 并行处理
    _ensure_dirs(ref_date)
    columns = _select_columns_for_rules()
    max_workers = SC_MAX_WORKERS or max(os.cpu_count() - 1, 1)
    
    results: List[Tuple[str, Optional[ScoreDetail], Optional[Tuple[str,str,str]]]] = []
    whitelist_items: List[Tuple[str,str,str]] = []
    blacklist_items: List[Tuple[str,str,str]] = []
    _progress("score_start", total=len(codes), current=0,
              message=f"workers={max_workers}")

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for code in codes:
            futures.append(ex.submit(_score_one, code, ref_date, start_date, columns))
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"评分 {ref_date}"):
            r = fut.result()
            results.append(r)
            done += 1
            _progress("score_progress", total=len(futures), current=done)

    # 拆分黑白名单与评分详情
    scored: List[ScoreDetail] = []
    for ts_code, detail, black in results:
        if black:
            blacklist_items.append(black)
        elif detail:
            whitelist_items.append((ts_code, ref_date, "pass"))
            scored.append(detail)

    # 写缓存名单
    _progress("write_cache_lists", message=f"white={len(whitelist_items)} black={len(blacklist_items)}")
    _write_cache_lists(ref_date, whitelist_items, blacklist_items)

    # 排序：总分降序；Tie-break（KDJ J）升序；ts_code 升序兜底
    def key_fn(x: ScoreDetail):
        tb = x.tiebreak if x.tiebreak is not None else math.inf  # J 越小越优
        return (-x.score, tb, x.ts_code)

    scored_sorted = sorted(scored, key=key_fn)

    # Top-K
    topk = scored_sorted[: int(SC_TOP_K)]

    # 落盘
    out_top = pd.DataFrame(
        [{
            "ts_code": s.ts_code,
            "score": round(s.score, 3),
            "highlights": "；".join(s.highlights) if s.highlights else "",
            "drawbacks": "；".join(s.drawbacks) if s.drawbacks else "",
            "opportunities": "；".join(s.opportunities) if s.opportunities else "",
            "ref_date": ref_date,
            "tiebreak_j": s.tiebreak
        } for s in topk]
    )
    out_path = os.path.join(SC_OUTPUT_DIR, "top", f"score_top_{ref_date}.csv")
    out_top.to_csv(out_path, index=False, encoding="utf-8-sig")
    # —— 全量排名（用于“单票详情”显示名次） ——
    rows_all = [{
        "ts_code": s.ts_code,
        "score": round(s.score, 3),
        "ref_date": ref_date,
        "tiebreak_j": s.tiebreak,
        "rank": i + 1,                   # 按当前排序的绝对名次
    } for i, s in enumerate(scored_sorted)]
    out_all = pd.DataFrame(rows_all)
    out_all["trade_date"] = str(ref_date)
    all_path = os.path.join(SC_OUTPUT_DIR, "all", f"score_all_{ref_date}.csv")
    out_all.to_csv(all_path, index=False, encoding="utf-8-sig")

    try:
        hooks_res = post_scoring(ref_date, df_all_scores=out_all)
        LOGGER.info("[HOOKS] tracking/surge 已完成：%s", hooks_res)
    except Exception as _e:
        LOGGER.exception("[HOOKS] 执行失败：%s", _e)
    LOGGER.info(f"[完成] Top-{SC_TOP_K} 已写入：{out_path}")
    try:
        ui_dir = OUTPUT_DIR / "meta"
        ui_dir.mkdir(parents=True, exist_ok=True)
        with open(ui_dir / "ui_hints.json", "w", encoding="utf-8") as f:
            json.dump({"topk_rows": int(globals().get("SC_TOPK_ROWS", 30))}, f, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as _e:
        LOGGER.warning("[UI] 写入 ui_hints.json 失败：%s", _e)
        LOGGER.info(f"[名单] 白名单 {len(whitelist_items)}，黑名单 {len(blacklist_items)}")
    # —— 回写 rank 到各自的明细 JSON（可选，但推荐启用） ——
    try:
        import json as _json
        details_dir = os.path.join(SC_OUTPUT_DIR, "details", str(ref_date))
        totalN = len(scored_sorted)
        for i, s in enumerate(scored_sorted):
            fp = os.path.join(details_dir, f"{s.ts_code}_{ref_date}.json")
            if not os.path.isfile(fp):
                continue
            with open(fp, "r", encoding="utf-8-sig") as f:
                obj = _json.load(f)
            summ = obj.get("summary") or {}
            summ["rank"] = int(i + 1)          # 绝对名次（1 开始）
            summ["total"] = int(totalN)        # 当日样本总数
            obj["summary"] = summ
            with open(fp, "w", encoding="utf-8") as f:
                _json.dump(_json_sanitize(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as e:
        LOGGER.warning("[detail] 回写 rank 失败：%s", e)

        # —— 生成“特别关注榜” —— 
    if SC_ATTENTION_ENABLE:
        try:
            build_attention_rank(start=None, end=ref_date, source=SC_ATTENTION_SOURCE,
                                 min_hits=SC_ATTENTION_MIN_HITS, topN=SC_ATTENTION_TOP_K, write=True)
        except Exception as e:
            LOGGER.warning("[attention] 生成特别关注榜失败：%s", e)

    if SC_ATTENTION_BACKFILL_ENABLE:
        try:
           # 取“参考日前 N 个交易日”作为 start（N 默认用 SC_ATTENTION_WINDOW_D）
            root_daily = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
            days = list_trade_dates(root_daily) or []
            if ref_date in days:
                i = days.index(ref_date)
                n = int(SC_ATTENTION_WINDOW_D or 20)
                start = days[max(0, i - n)]
            else:
                start = days[-int(SC_ATTENTION_WINDOW_D or 20)] if days else ref_date
            backfill_attention_rolling(
                start=start, end=ref_date, source=SC_ATTENTION_SOURCE,
                min_hits=SC_ATTENTION_MIN_HITS, topN=SC_ATTENTION_TOP_K
           )
        except Exception as e:
            LOGGER.warning("[attention] 回补生成失败：%s", e)
    return out_path


# ======== 简易 TDX 风格选股 ========
def _expr_mentions(expr: str, name: str) -> bool:
    try:
        pat = re.compile(rf"\b{name}\b", flags=re.IGNORECASE)
        return bool(pat.search(expr or ""))
    except Exception:
        return False


def _scan_cols_from_expr(expr: str) -> list[str]:
    """
    从通达信表达式里扫出可能需要的扩展列（如 j/vr），用于尽量按需裁列、并在缺失时兜底计算。
    """
    need = {"trade_date", "open", "high", "low", "close", "vol", "amount"}
    if _expr_mentions(expr, "J"):
        need.add("j")
    if _expr_mentions(expr, "VR"):
        need.add("vr")
    return sorted(need)


def _start_for_tf_window(ref_date: str, timeframe: str, window: int) -> str:
    """
    根据 timeframe + window 估算需要读取的日线尾部长度（加入缓冲），避免全量读。
    D: window + 120；W: window*6 + 180；M: window*22 + 260
    """
    tf = (timeframe or "D").upper()
    if tf == "D":
        days = int(window) + 120
    elif tf == "W":
        days = int(window) * 6 + 180
    else:  # "M"
        days = int(window) * 22 + 260
    start_dt = dt.datetime.strptime(ref_date, "%Y%m%d") - dt.timedelta(days=days)
    return start_dt.strftime("%Y%m%d")


def tdx_screen(
    when: str,
    *,
    ref_date: str | None = None,
    timeframe: str = "D",
    window: int = 60,
    scope: str = "ANY",          # 可用：LAST/ANY/ALL/COUNT>=k/CONSEC>=m
    write_white: bool = True,    # 把命中写入白名单
    write_black_rest: bool = False,  # 把未命中写入黑名单（谨慎）
    universe: str | list[str] | None = "all",  # all/white/black/attention 或 代码清单
    use_prescreen_first: bool = True,
    return_df: bool = True
):
    """
    类通达信“普通选股”：在给定 timeframe/window/scope 下，用 when 表达式筛股票。
    - 自动列出参考日 universe（single 布局：按文件名；daily 布局：按 ref_date 分区）:contentReference[oaicite:1]{index=1}
    - 读取最小必要列（若表达式包含 J/VR，会尝试兜底计算 KDJ J/成交量比 VR）
    - 可选把命中写白名单、未命中写黑名单（写入 SC_CACHE_DIR/<ref_date>/ 下）:contentReference[oaicite:2]{index=2}
    - 保持完整日志（score.log）
    """
    when = (when or "").strip()
    if not when:
        raise ValueError("when 不能为空")

    ref = ref_date or _pick_ref_date()
    tf = (timeframe or "D").upper()
    win = int(window)
    st = _start_for_tf_window(ref, tf, win)

    codes0 = _list_codes_for_day(ref)
    codes, src = _apply_universe_filter(list(codes0), ref, universe)
    LOGGER.info("[screen][范围] universe=%s 源=%s 原=%d -> %d",
                str(universe), src, len(codes0), len(codes))
    if not codes:
        LOGGER.warning("[screen] %s 无可用标的。", ref)
        return pd.DataFrame() if return_df else None

    cols = _scan_cols_from_expr(when)
    LOGGER.info("[screen] 参考日=%s tf=%s window=%d scope=%s 宇宙=%d", ref, tf, win, scope, len(codes))
    LOGGER.debug("[screen] 读取列=%s 起始=%s", cols, st)

    hits: list[str] = []
    whitelist_items: list[tuple[str, str, str]] = []
    blacklist_items: list[tuple[str, str, str]] = []

    for ts_code in codes:
        try:
            dfD = _read_stock_df(ts_code, st, ref, columns=cols)
            if dfD.empty:
                continue
            # —— 先过一遍初选（硬惩罚规则），默认开启 —— 
            pres = PrescreenResult(True, None, None)
            if use_prescreen_first:
                pres = _prescreen(dfD, ref)
            if not pres.passed:
                if write_black_rest:
                    blacklist_items.append((ts_code, pres.period or ref, pres.reason or "prescreen"))
                continue
            # 若表达式涉及 J/VR，但数据中缺列，尝试兜底
            need_j = _expr_mentions(when, "J") and ("j" not in dfD.columns)
            need_vr = _expr_mentions(when, "VR") and ("vr" not in dfD.columns)
            if need_j:
                try:
                    dfD = dfD.copy()
                    dfD["j"] = kdj(dfD)
                except Exception as e:
                    LOGGER.debug("[screen][%s] 兜底计算J失败: %s", ts_code, e)
            if need_vr and "vol" in dfD.columns:
                try:
                    # VR(26) 的一种常见实现（简化版）：近 N 天 成交量比
                    n = 26
                    v = pd.to_numeric(dfD["vol"], errors="coerce")
                    dfD = dfD.copy()
                    dfD["vr"] = (v / v.rolling(n).mean()).values
                except Exception as e:
                    LOGGER.debug("[screen][%s] 兜底计算VR失败: %s", ts_code, e)

            dfTF = dfD if tf == "D" else _resample(dfD, tf)
            win_df = _window_slice(dfTF, ref, win)
            if win_df.empty:
                continue
            s_bool = evaluate_bool(when, win_df)
            ok = _scope_hit(s_bool, scope)
            if ok:
                hits.append(ts_code)
                if write_white:
                    whitelist_items.append((ts_code, ref, f"screen:{tf}{win}:{scope}::{when}"))
            else:
                if write_black_rest:
                    blacklist_items.append((ts_code, ref, f"screen_fail:{tf}{win}:{scope}::{when}"))
        except Exception as e:
            LOGGER.debug("[screen][%s] 忽略异常: %s", ts_code, e)
            continue

    # 写缓存名单
    if write_white or write_black_rest:
        _write_cache_lists(ref, whitelist_items, blacklist_items)
        LOGGER.info("[screen] 写名单完成：白=%d 黑=%d", len(whitelist_items), len(blacklist_items))

    # 结果 DataFrame（供可视化）
    if return_df:
        out = pd.DataFrame({"ts_code": sorted(hits)})
        out["ref_date"] = ref
        out["rule"] = f"{tf}{win}:{scope}::{when}"
        return out
    return None
# ==========================================================