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
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import glob

import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

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
)
from parquet_viewer import asset_root, list_trade_dates, read_range
from tdx_compat import evaluate_bool
from indicators import kdj  # 用于兜底计算 J（若数据不含 j 列）
import logging
from logging.handlers import TimedRotatingFileHandler


# ------------------------- 日志初始化 -------------------------
def _init_logger():
    os.makedirs("./log", exist_ok=True)
    logger = logging.getLogger("score")
    logger.setLevel(logging.DEBUG)  # 细到 DEBUG，输出筛选由 handler 控制
    # 轮转：每天一个，保留 7 个
    fh = TimedRotatingFileHandler("./log/score.log", when="midnight", interval=1, backupCount=7, encoding="utf-8-sig")
    # 控制台：INFO 以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 避免重复添加 handler
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

LOGGER = _init_logger()

# ------------------------- 工具函数 -------------------------
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

def _select_columns_for_rules() -> List[str]:
    """
    尽量按需裁列，减少 IO。
    """
    need = {"trade_date", "open", "high", "low", "close", "vol", "amount"}
    # 规则文本中可能出现的列（如 j/vr）
    pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')
    def scan(script: str):
        for name in pattern.findall(script):
            name_low = name.lower()
            # tdx 映射里常见的：C O H L V AMOUNT J VR ...
            if name_low in {"j","vr","bbi","z_slope"}:
                need.add(name_low)
    for rr in (SC_RULES or []):
        if "clauses" in rr:
            for c in rr["clauses"]:
                if "when" in c:
                    scan(c["when"])
        else:
            if "when" in rr:
                scan(rr["when"])
    for rr in (SC_PRESCREEN_RULES or []):
        if "clauses" in rr:
            for c in rr["clauses"]:
                if "when" in c:
                    scan(c["when"])
        else:
            if "when" in rr:
                scan(rr["when"])
    return sorted(need)

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

def _window_slice(dfTF: pd.DataFrame, ref_date: str, window: int) -> pd.DataFrame:
    dfTF = _ensure_datetime_index(dfTF)
    mask = dfTF.index <= dt.datetime.strptime(ref_date, "%Y%m%d")
    return dfTF.loc[mask].tail(int(window))

def _scope_hit(s_bool: pd.Series, scope: str) -> bool:
    """
    scope: LAST | ANY | ALL | COUNT>=k | CONSEC>=m
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

# def _read_stock_df(ts_code: str, start: str, end: str, columns: List[str]) -> pd.DataFrame:
#     """
#     读取某只股票在给定区间的日线数据，只选需要的列。
#     - 若存在 single 目录：直接读取 single/{ts_code}.parquet，并在内存中过滤日期
#     - 否则回退到 daily 读取（read_range）
#     """
#     if _has_single_dir():
#         f = os.path.join(_single_root_dir(), f"{ts_code}.parquet")
#         if not os.path.isfile(f):
#             raise FileNotFoundError(f"single 文件不存在: {f}")
#         df = pd.read_parquet(f)
#         # 统一为字符串比较
#         if "trade_date" in df.columns:
#             df["trade_date"] = df["trade_date"].astype(str)
#             mask = (df["trade_date"] >= str(start)) & (df["trade_date"] <= str(end))
#             df = df.loc[mask].copy()
#         # 按需裁列（仅保留存在的列）
#         if columns:
#             keep = [c for c in columns if c in df.columns]
#             if keep:
#                 df = df[keep + [c for c in df.columns if c not in keep and c in ("trade_date",)]]
#     else:
#         df = read_range(PARQUET_BASE, "stock", PARQUET_ADJ, ts_code, start, end, columns=columns)
    
#     if "trade_date" in df.columns:
#         df["trade_date"] = df["trade_date"].astype(str)
    
#     # 兜底 J：若缺 j 列，而 tie-break 需要
#     if ("j" not in df.columns) and (SC_TIE_BREAK or "").lower() in {"kdj_j_asc", "kdj_j_desc"}:
#         try:
#             j = kdj(df)
#             df = df.copy()
#             df["j"] = j
#         except Exception as e:
#             LOGGER.warning(f"[{ts_code}] 计算 KDJ J 失败：{e}")
#     return df

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
    tiebreak: Optional[float]

def _score_one(ts_code: str, ref_date: str, start_date: str, columns: List[str]) -> Optional[Tuple[str, Optional[ScoreDetail], Optional[Tuple[str,str,str]]]]:
    """
    返回：
      - (ts_code, detail 或 None, 黑名单项 或 None)
      若被初选淘汰，返回黑名单项 (ts_code, period, reason)；否则返回打分 detail。
    """
    try:
        df = _read_stock_df(ts_code, start_date, ref_date, columns)
        if df is None or df.empty:
            LOGGER.warning(f"[{ts_code}] 数据为空，跳过")
            return (ts_code, None, ("", "", "数据为空"))
        # 初选
        pres = _prescreen(df, ref_date)
        if not pres.passed:
            return (ts_code, None, (ts_code, pres.period or ref_date, pres.reason or "prescreen"))
        # 打分
        score = float(SC_BASE_SCORE)
        highlights, drawbacks = [], []
        # for rule in (SC_RULES or []):
        #     ok, err = _eval_rule(df, rule, ref_date)
        #     if err:
        #         # 异常分支记日志
        #         LOGGER.warning(f"[{ts_code}] 规则异常：{rule.get('name')} -> {err}")
        #         continue
        #     if ok:
        #         pts = float(rule.get("points", 0))
        #         score += pts
        #         # 命中记录要保留（DEBUG）
        #         expl = rule.get("explain")
        #         if expl:
        #             if pts >= 0:
        #                 highlights.append(str(expl))
        #             else:
        #                 drawbacks.append(str(expl))
        #         LOGGER.debug(f"[HIT][{ts_code}] {rule.get('name','<unnamed>')} +{pts} => {score}")

        for rule in (SC_RULES or []):
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
                        add = pts * int(cnt)
                        score += add
                        expl = rule.get("explain")
                        if expl:
                            tag = f"{expl}×{cnt}"
                            (highlights if pts >= 0 else drawbacks).append(tag)
                        LOGGER.debug(f"[HIT][{ts_code}] {rule.get('name','<unnamed>')} EACH 命中 {cnt} 次，+{add} => {score}")
                    else:
                        LOGGER.debug(f"[MISS][{ts_code}] {rule.get('name','<unnamed>')} EACH 无命中")
                continue  # EACH 已处理完，下一条
            # —— 常规路径（保持原逻辑） ——
            ok, err = _eval_rule(df, rule, ref_date)
            if err:
                LOGGER.warning(f"[{ts_code}] 规则异常：{rule.get('name')} -> {err}")
                continue
            if ok:
                pts = float(rule.get("points", 0))
                score += pts
                expl = rule.get("explain")
                if expl:
                    if pts >= 0:
                        highlights.append(str(expl))
                    else:
                        drawbacks.append(str(expl))
                LOGGER.debug(f"[HIT][{ts_code}] {rule.get('name','<unnamed>')} +{pts} => {score}")

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
        return (ts_code, ScoreDetail(ts_code, score, highlights, drawbacks, tb), None)
    except Exception as e:
        LOGGER.error(f"[{ts_code}] 评分失败：{e}")
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
                         write: bool = True):
    """
    统计某个来源('top'|'white'|'black')在[start, end]（按交易日）内的“上榜次数”并排序。
    默认：窗口=最近 SC_ATTENTION_WINDOW_D 个交易日，以 end/ref_date 作为右端点。
    返回 CSV 路径（write=True）或 DataFrame（write=False）。
    """
    end = end or _pick_ref_date()
    source = (source or SC_ATTENTION_SOURCE).lower()
    min_hits = int(min_hits or SC_ATTENTION_MIN_HITS)
    topN = int(topN or SC_ATTENTION_TOP_K)

    span = _trade_span(start, end)
    if not span:
        LOGGER.warning("[attention] 交易日窗口为空，start=%s end=%s", start, end)
        return None

    hit_cnt: Dict[str, int] = {}
    first_last: Dict[str, Tuple[str,str]] = {}

    for d in span:
        if source == "top":
            f = os.path.join(SC_OUTPUT_DIR, "top", f"score_top_{d}.csv")
        elif source in ("white", "black"):
            f = os.path.join(SC_CACHE_DIR, d, f"{'whitelist' if source=='white' else 'blacklist'}.csv")
        else:
            raise ValueError("source must be 'top' | 'white' | 'black'")

        if not os.path.isfile(f):
            continue
        try:
            df = pd.read_csv(f, dtype={"ts_code": str})
            codes = df["ts_code"].astype(str).tolist() if "ts_code" in df.columns else []
        except Exception as e:
            LOGGER.warning("[attention] 读取失败 %s: %s", f, e)
            continue

        for c in codes:
            hit_cnt[c] = hit_cnt.get(c, 0) + 1
            if c not in first_last:
                first_last[c] = (d, d)
            else:
                first_last[c] = (first_last[c][0], d)

    if not hit_cnt:
        LOGGER.warning("[attention] %s~%s 内没有 '%s' 记录。", span[0], span[-1], source)
        return None

    rows = []
    for c, n in hit_cnt.items():
        if n >= min_hits:
            f, l = first_last[c]
            rows.append({"ts_code": c, "hits": n, "first": f, "last": l})
    if not rows:
        LOGGER.info("[attention] 命中数 >= %d 的为空。", min_hits)
        return None

    out_df = pd.DataFrame(rows).sort_values(["hits", "last", "ts_code"], ascending=[False, False, True])
    out_df = out_df.head(topN)

    out_dir = os.path.join(SC_OUTPUT_DIR, "attention")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"attention_{source}_{span[0]}_{span[-1]}.csv")

    if write:
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        LOGGER.info("[特别关注] source=%s 窗口=%s~%s min_hits>=%d Top-%d -> %s",
                    source, span[0], span[-1], min_hits, topN, out_path)
        return out_path
    else:
        return out_df

def backfill_attention_rolling(start: str,
                               end: Optional[str] = None,
                               source: Optional[str] = None,
                               min_hits: Optional[int] = None,
                               topN: Optional[int] = None):
    """
    从 start 到 end（按交易日）逐日滚动补算“特别关注榜”：
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
                                 min_hits=min_hits, topN=topN, write=True, encoding="gb18030")
            ok += 1
        except Exception as ex:
            fail += 1
            LOGGER.warning("[attention/backfill] 生成 %s 失败：%s", e, ex)
    LOGGER.info("[attention/backfill] 完成：成功 %d，失败 %d", ok, fail)

# ------------------------- 名单缓存 I/O -------------------------
def _ensure_dirs(ref_date: str):
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "top"), exist_ok=True)
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "details"), exist_ok=True)
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

    # 2) 估算读取起始日
    start_date = _compute_read_start(ref_date)
    LOGGER.info(f"[范围] 读取区间：{start_date} ~ {ref_date}")

    # 3) 股票清单（来自分区目录文件名）
    codes = _list_codes_for_day(ref_date)
    LOGGER.info(f"[UNIVERSE] {ref_date} 共 {len(codes)} 只股票")

    # 4/5) 并行处理
    _ensure_dirs(ref_date)
    columns = _select_columns_for_rules()

    max_workers = SC_MAX_WORKERS or max(os.cpu_count() - 1, 1)
    results: List[Tuple[str, Optional[ScoreDetail], Optional[Tuple[str,str,str]]]] = []
    whitelist_items: List[Tuple[str,str,str]] = []
    blacklist_items: List[Tuple[str,str,str]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for code in codes:
            futures.append(ex.submit(_score_one, code, ref_date, start_date, columns))
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"评分 {ref_date}"):
            r = fut.result()
            results.append(r)

    # 拆分黑白名单与评分详情
    scored: List[ScoreDetail] = []
    for ts_code, detail, black in results:
        if black:
            blacklist_items.append(black)
        elif detail:
            whitelist_items.append((ts_code, ref_date, "pass"))
            scored.append(detail)

    # 写缓存名单
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
            "ref_date": ref_date,
            "tiebreak_j": s.tiebreak
        } for s in topk]
    )
    out_path = os.path.join(SC_OUTPUT_DIR, "top", f"score_top_{ref_date}.csv")
    out_top.to_csv(out_path, index=False, encoding="utf-8-sig")

    LOGGER.info(f"[完成] Top-{SC_TOP_K} 已写入：{out_path}")
    LOGGER.info(f"[名单] 白名单 {len(whitelist_items)}，黑名单 {len(blacklist_items)}")
        # —— 生成“特别关注榜” —— 
    if SC_ATTENTION_ENABLE:
        try:
            build_attention_rank(start=None, end=ref_date, source=SC_ATTENTION_SOURCE,
                                 min_hits=SC_ATTENTION_MIN_HITS, topN=SC_ATTENTION_TOP_K, write=True)
        except Exception as e:
            LOGGER.warning("[attention] 生成特别关注榜失败：%s", e)

    if SC_ATTENTION_BACKFILL_ENABLE:
        backfill_attention_rolling(start="today - 20d", end="today", source="top", min_hits=2, topN=200)

    return out_path
# ==========================================================
if __name__ == "__main__":
    run_for_date()