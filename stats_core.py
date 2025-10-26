# -*- coding: utf-8-sig -*-
from __future__ import annotations

import traceback
from log_system import get_logger

try:
    import config as _cfg
except Exception:  # pragma: no cover
    _cfg = None

from dataclasses import dataclass, asdict
from pathlib import Path

import math
import json
import os
import numpy as np
import pandas as pd
import re
import importlib
from config import DATA_ROOT, API_ADJ
# 使用 database_manager 替代 data_reader
from database_manager import (
    get_database_manager, query_stock_data, get_trade_dates, 
    get_stock_list, get_latest_trade_date, get_smart_end_date
)

# 直接使用 database_manager 函数，不再需要包装器

from typing import Iterable, Sequence, Dict, List, Optional, Callable, Any, Literal

from utils import normalize_ts, normalize_trade_date, market_label

SCORE_ALL_DIR = Path("output/score/all")
SURGE_OUT_BASE = Path("output/surge_lists")
COMMON_OUT_BASE = Path("output/commonality")
TRACKING_OUT_BASE = Path("output/tracking")
PORT_OUT_BASE = Path("output/portfolio")
LOG = get_logger("stats_core")

# --------- 读取可选配置（容错） ---------
def _get(name: str, default):
    if _cfg and hasattr(_cfg, name):
        try:
            return getattr(_cfg, name)
        except Exception:
            return default
    return default

# --------- 主钩子 ---------
def post_scoring(ref_date: str, *, df_all_scores: Optional[pd.DataFrame] = None) -> dict:
    """在 run_for_date() 写完 all/top 后调用。
    返回所有子任务的状态与落盘路径，失败的子任务会返回异常简述。
    """
    result: dict[str, Any] = {"ref_date": ref_date}

    # 1) 跟踪（Tracking）
    try:
        if _get("SC_DO_TRACKING", False):
            wins = _get("SC_TRACKING_WINDOWS", [1,2,3,5,10,20])
            bench = _get("SC_TRACKING_BENCHMARKS", [])
            gb_board = bool(_get("SC_TRACKING_GROUP_BY_BOARD", True))
            tr = run_tracking(ref_date, wins, bench, score_df=df_all_scores, group_by_board=gb_board, save=True)
            result["tracking_detail_path"] = str(tr.detail_path) if tr.detail_path else None
            result["tracking_summary_path"] = str(tr.summary_path) if tr.summary_path else None
        else:
            result["tracking"] = "skipped"
    except Exception as e:  # 捕获但不打断主流程
        LOG.error("[HOOK] Tracking 失败：%s", e)
        LOG.debug(traceback.format_exc())
        result["tracking_error"] = f"{type(e).__name__}: {e}"

    # 2) Surge 单子 + 回看
    try:
        if _get("SC_DO_SURGE", False):
            mode = _get("SC_SURGE_MODE", "rolling")
            k = int(_get("SC_SURGE_ROLLING_DAYS", 5))
            sel = _get("SC_SURGE_SELECTION", {"type": "top_n", "value": 200})
            retro = _get("SC_SURGE_RETRO_DAYS", [1,2,3,4,5])
            split = _get("SC_SURGE_SPLIT", "main_vs_others")
            sr = run_surge(ref_date=ref_date, mode=mode, rolling_days=k, selection=sel, retro_days=retro, split=split, score_df=df_all_scores, save=True)
            result["surge_out_path"] = str(sr.out_path) if sr.out_path else None
            if sr.group_files:
                result["surge_group_files"] = {k: str(v) for k, v in sr.group_files.items()}
        else:
            result["surge"] = "skipped"
    except Exception as e:
        LOG.error("[HOOK] Surge 失败：%s", e)
        LOG.debug(traceback.format_exc())
        result["surge_error"] = f"{type(e).__name__}: {e}"

    return result

# ------------------------- 数据模型 -------------------------
@dataclass
class TrackingResult:
    detail: pd.DataFrame
    summary: pd.DataFrame
    detail_path: Optional[Path] = None
    summary_path: Optional[Path] = None

# ------------------------- 工具 -------------------------

def _count_strategy_triggers(obs_date: str, codes_sample: Sequence[str], *, weights_map: dict[str, float] | None = None) -> pd.DataFrame:
    """
    统计某观察日 obs_date 在样本集合 codes_sample 内每个策略（规则名）的触发次数/覆盖率。
    触发判定：per_rules 里 ok=True 或 hit_date==obs_date 或 obs_date ∈ hit_dates。
    可选：weights_map 提供 ts_code -> 权重，用于“加权触发次数”和“加权覆盖率”。
    返回列：name, trigger_count, n_sample, coverage, trigger_weighted, sample_weight, coverage_weighted
    """
    ddir = Path("output/score/details") / str(obs_date)
    if not ddir.exists():
        return pd.DataFrame(columns=["name","trigger_count","n_sample","coverage","trigger_weighted","sample_weight","coverage_weighted"])
    codes_set = set(str(c) for c in (codes_sample or []))
    weights_map = {str(k): float(v) for k,v in (weights_map or {}).items()}
    n_sample = len(codes_set) if codes_set else 0
    sample_weight = sum(weights_map.get(ts, 1.0) for ts in codes_set) if codes_set else 0.0

    # 每个策略名 -> {命中的 ts_code 集合, 加权和}
    acc_set: dict[str, set] = {}
    acc_w  : dict[str, float] = {}

    for fp in ddir.glob("*.json"):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        ts = str(obj.get("ts_code",""))
        if codes_set and ts not in codes_set:
            continue
        rules = (obj.get("per_rules") or obj.get("rules") or [])
        for r in rules:
            name = str(r.get("name") or "")
            if not name:
                continue
            ok = bool(r.get("ok"))
            hd = str(r.get("hit_date") or "")
            hds = [str(x) for x in (r.get("hit_dates") or [])]
            trig = ok or (hd == str(obs_date)) or (str(obs_date) in hds)
            if trig:
                acc_set.setdefault(name, set()).add(ts)
                acc_w[name] = acc_w.get(name, 0.0) + float(weights_map.get(ts, 1.0))

    rows = []
    for name, st in acc_set.items():
        cnt = int(len(st))
        cov = (cnt / n_sample if n_sample else 0.0)
        wsum = float(acc_w.get(name, 0.0))
        cov_w = (wsum / sample_weight if sample_weight else 0.0)
        rows.append({
            "name": name,
            "trigger_count": cnt,
            "n_sample": int(n_sample),
            "coverage": cov,
            "trigger_weighted": wsum,
            "sample_weight": sample_weight,
            "coverage_weighted": cov_w,
        })

    df = pd.DataFrame(rows, columns=["name","trigger_count","n_sample","coverage","trigger_weighted","sample_weight","coverage_weighted"])
    if not df.empty:
        # 优先按加权命中降序，再按未加权、再按名称升序
        sort_cols = [c for c in ["trigger_weighted","trigger_count","name"] if c in df.columns]
        df = df.sort_values(sort_cols, ascending=[False, False, True][:len(sort_cols)]).reset_index(drop=True)
    return df


def _count_hits_per_stock(obs_date: str, codes_sample: Sequence[str]) -> pd.DataFrame:
    """
    统计某观察日 obs_date 在样本集合 codes_sample 内每只股票**按规则模式**的命中条数：
      - 单次型（any/last 等，一条规则当日至多命中一次）
      - 多次型（each 等，可视作“可多次命中”的规则类型）
    返回列：ts_code, n_single_rules_hit, n_each_rules_hit, single_rule_names, each_rule_names
    说明：无法获取更细粒度“同一规则在同一日命中多次”的次数时，按“规则项数量”近似计数。
    """
    ddir = Path("output/score/details") / str(obs_date)
    if not ddir.exists():
        return pd.DataFrame(columns=["ts_code","n_single_rules_hit","n_each_rules_hit","single_rule_names","each_rule_names"])
    codes_set = set(str(c) for c in (codes_sample or []))
    recs = []
    for fp in ddir.glob("*.json"):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        ts = str(obj.get("ts_code",""))
        if codes_set and ts not in codes_set:
            continue

        single_names = set()
        each_names = set()

        rules_iter = (obj.get("per_rules") or obj.get("rules") or [])
        for r in rules_iter:
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            ok = bool(r.get("ok"))
            hd = str(r.get("hit_date") or "")
            hds = [str(x) for x in (r.get("hit_dates") or [])]
            trig = ok or (hd == str(obs_date)) or (str(obs_date) in hds)
            if not trig:
                continue
            # 规则模式识别
            mode = str(r.get("mode") or r.get("run_mode") or r.get("match") or r.get("type") or "").lower()
            if mode in ("each","multi","multiple"):
                each_names.add(name)
            else:
                # 统一把 any/last/first/once/空值 都视为单次型
                single_names.add(name)

        recs.append({
            "ts_code": ts,
            "n_single_rules_hit": int(len(single_names)),
            "n_each_rules_hit": int(len(each_names)),
            "single_rule_names": ",".join(sorted(single_names)) if single_names else "",
            "each_rule_names": ",".join(sorted(each_names)) if each_names else "",
        })

    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values(["n_each_rules_hit","n_single_rules_hit","ts_code"], ascending=[False, False, True]).reset_index(drop=True)
    return df


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_end_date(ref_date: str, max_window: int) -> str:
    days = get_trade_dates() or []
    if ref_date not in days:
        raise ValueError(f"ref_date 不在交易日历内: {ref_date}")
    i = days.index(ref_date)
    j = min(i + max_window, len(days) - 1)
    return days[j]


def _read_score_all(date_str: str) -> pd.DataFrame:
    f = SCORE_ALL_DIR / f"score_all_{date_str}.csv"
    if not f.exists():
        raise FileNotFoundError(f"未找到打分输出：{f}")
    df = pd.read_csv(f, dtype={"ts_code": str})
    df = normalize_trade_date(df, "trade_date")
    df = df[df["trade_date"] == date_str].copy()
    if "rank" not in df.columns and "score" in df.columns:
        df = df.sort_values(["score"], ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    df["ts_code"] = df["ts_code"].astype(str).map(lambda s: normalize_ts(s, asset="stock"))
    if "board" not in df.columns:
        df["board"] = df["ts_code"].map(market_label)
    return df[[c for c in ["trade_date","ts_code","score","rank","board"] if c in df.columns]]


def _read_stock_prices(codes: Sequence[str], start: str, end: str) -> pd.DataFrame:
    cols = ["ts_code", "trade_date", "open", "close"]
    
    # 优先使用缓存读取（仅对单只股票）
    if len(codes) == 1:
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            df = query_stock_data(
                db_path=db_path,
                ts_code=codes[0],
                start_date=start,
                end_date=end,
                adj_type="qfq"
            )
            if not df.empty:
                df = normalize_trade_date(df, "trade_date")
                return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        except:
            pass
    
    # 使用 database_manager 直接查询
    try:
        from config import DATA_ROOT, UNIFIED_DB_PATH
        db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
        
        # 构建查询条件
        conditions = []
        params = []
        
        if codes:
            placeholders = ",".join(["?"] * len(codes))
            conditions.append(f"ts_code IN ({placeholders})")
            params.extend(codes)
        
        if start:
            conditions.append("trade_date >= ?")
            params.append(start)
            
        if end:
            conditions.append("trade_date <= ?")
            params.append(end)
            
        conditions.append("adj_type = ?")
        params.append(API_ADJ)
        
        # 构建SQL查询
        select_cols = "*" if not cols else ", ".join(cols)
        sql = f"SELECT {select_cols} FROM stock_data"
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY ts_code, trade_date"
        
        # 执行查询
        manager = get_database_manager()
        df = manager.execute_sync_query(db_path, sql, params, timeout=120.0)
    except Exception as e:
        LOG.error(f"读取数据范围失败: {e}")
        df = pd.DataFrame()
    
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    df = normalize_trade_date(df, "trade_date")
    df = df[df["ts_code"].isin(set(codes))].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def _read_index_prices(index_codes: Sequence[str], start: str, end: str) -> pd.DataFrame:
    if not index_codes:
        return pd.DataFrame(columns=["index_code", "trade_date", "close"])    
    cols = ["ts_code", "trade_date", "close"]
    
    # 使用 database_manager 直接查询指数数据
    try:
        from config import DATA_ROOT, UNIFIED_DB_PATH
        db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
        
        # 构建查询条件
        conditions = []
        params = []
        
        if start:
            conditions.append("trade_date >= ?")
            params.append(start)
            
        if end:
            conditions.append("trade_date <= ?")
            params.append(end)
            
        conditions.append("adj_type = ?")
        params.append("daily")
        
        # 构建SQL查询
        select_cols = "*" if not cols else ", ".join(cols)
        sql = f"SELECT {select_cols} FROM stock_data"
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY ts_code, trade_date"
        
        # 执行查询
        manager = get_database_manager()
        df = manager.execute_sync_query(db_path, sql, params, timeout=120.0)
    except Exception as e:
        LOG.error(f"读取指数数据失败: {e}")
        df = pd.DataFrame()

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["index_code", "trade_date", "close"]) 
    df = normalize_trade_date(df, "trade_date")
    df = df[df["ts_code"].isin(set(index_codes))].rename(columns={"ts_code": "index_code"})
    return df.sort_values(["index_code", "trade_date"]).reset_index(drop=True)


def _compute_forward_returns(df_px: pd.DataFrame, ref_date: str, windows: Sequence[int], code_col: str = "ts_code") -> pd.DataFrame:
    if df_px.empty:
        return pd.DataFrame()
    windows = sorted(set(int(x) for x in windows if int(x) > 0))
    g = df_px.groupby(code_col, group_keys=False)
    out: List[dict] = []
    for code, gdf in g:
        gdf = gdf.sort_values("trade_date").reset_index(drop=True)
        close = gdf["close"].astype(float).values
        fwd_map = {N: np.full(len(gdf), np.nan, dtype=float) for N in windows}
        for N in windows:
            if len(gdf) > N:
                fwd_map[N][: len(gdf) - N] = (close[N:] / close[:-N]) - 1.0
        row = gdf[gdf["trade_date"] == ref_date]
        if row.empty:
            continue
        idx = int(row.index[0])
        rec = {code_col: code, "trade_date": ref_date}
        for N in windows:
            rec[f"ret_fwd_{N}"] = float(fwd_map[N][idx]) if idx < len(gdf) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def _to_long_summary(detail: pd.DataFrame, windows: Sequence[int], benchmarks: Sequence[str] | None, *, group_by_board: bool) -> pd.DataFrame:
    """生成长表汇总：每行 (trade_date, group_type, group_value, kind, window, benchmark, 指标...)"""
    recs: List[dict] = []

    def _stat(series: pd.Series) -> dict:
        s = pd.to_numeric(series, errors="coerce")
        s = s[~s.isna()]
        if s.empty:
            return {"mean": np.nan, "std": np.nan, "winrate": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "n_stocks": 0}
        return {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "winrate": float((s > 0).mean()),
            "p25": float(np.percentile(s, 25)),
            "p50": float(np.percentile(s, 50)),
            "p75": float(np.percentile(s, 75)),
            "n_stocks": int(len(s)),
        }

    def _emit(group_type: str, group_value: str, df: pd.DataFrame):
        # ret
        for N in windows:
            col = f"ret_fwd_{N}"
            rec = {"trade_date": df["trade_date"].iat[0], "group_type": group_type, "group_value": group_value,
                   "kind": "ret", "window": int(N), "benchmark": None}
            rec.update(_stat(df[col]))
            recs.append(rec)
        # alpha per benchmark
        if benchmarks:
            for idx in benchmarks:
                for N in windows:
                    col = f"alpha_fwd_{N}_{idx}"
                    if col not in df.columns:
                        continue
                    rec = {"trade_date": df["trade_date"].iat[0], "group_type": group_type, "group_value": group_value,
                           "kind": "alpha", "window": int(N), "benchmark": idx}
                    rec.update(_stat(df[col]))
                    recs.append(rec)

    # 全体
    _emit("all", "all", detail)
    # 板块
    if group_by_board and "board" in detail.columns:
        for b, gdf in detail.groupby("board", dropna=False):
            _emit("board", str(b), gdf)

    return pd.DataFrame(recs)

# ------------------------- 核心类 -------------------------
class ScoreTrackingPlugin:
    """供 score_engine 调用的跟踪插件。"""
    def __init__(self, *, parquet_base: str | Path | None = None, parquet_adj: str | None = None, outdir: str | Path | None = None):
        self.data_root = str(parquet_base or DATA_ROOT)
        self.api_adj = str(parquet_adj or API_ADJ)
        self.outdir = Path(outdir or TRACKING_OUT_BASE)

    # ---- 主入口：从 DataFrame（优先）或 score_all 文件计算 ----
    def run(self,
            ref_date: str,
            windows: Sequence[int],
            benchmarks: Sequence[str] | None = None,
            *,
            score_df: Optional[pd.DataFrame] = None,
            group_by_board: bool = True,
            save: bool = True,
            retro_days: Sequence[int] = (),
            do_summary: bool = True) -> TrackingResult:
        """
        参数：
          - ref_date：交易日 YYYYMMDD
          - windows：任意正整数集合，表示未来 N 日窗口
          - benchmarks：指数代码列表（可空）
          - score_df：若提供则直接使用（必须含 trade_date/ts_code/score/rank，可缺 board）；否则从 score_all 读取
        返回：TrackingResult（包含 DataFrame 与可选落盘路径）
        """
        ref_date = str(ref_date)
        windows = sorted(set(int(x) for x in windows if int(x) > 0))
        benchmarks = list(benchmarks or [])

        # 0) 打分明细
        if score_df is None:
            df_score = _read_score_all(ref_date)
        else:
            df_score = score_df.copy()
            df_score = normalize_trade_date(df_score, "trade_date")
            df_score = df_score[df_score["trade_date"] == ref_date].copy()
            if "rank" not in df_score.columns and "score" in df_score.columns:
                df_score = df_score.sort_values(["score"], ascending=False).reset_index(drop=True)
                df_score["rank"] = np.arange(1, len(df_score) + 1)
            df_score["ts_code"] = df_score["ts_code"].astype(str).map(lambda s: normalize_ts(s, asset="stock"))
            if "board" not in df_score.columns:
                df_score["board"] = df_score["ts_code"].map(market_label)
            keep = [c for c in ["trade_date", "ts_code", "score", "rank", "board"] if c in df_score.columns]
            df_score = df_score[keep]

        if df_score.empty:
            raise RuntimeError(f"当日打分结果为空：{ref_date}")

        codes = df_score["ts_code"].unique().tolist()
        maxN = max(windows) if windows else 0
        end_date = _pick_end_date(ref_date, maxN)

        # 1) 读取行情
        px = _read_stock_prices(codes, start=ref_date, end=end_date)
        idx_px = _read_index_prices(benchmarks, start=ref_date, end=end_date) if benchmarks else pd.DataFrame()

        # 2) forward returns
        stock_rets = _compute_forward_returns(px, ref_date, windows, code_col="ts_code")
        if benchmarks:
            idx_rets = _compute_forward_returns(idx_px.rename(columns={"index_code": "ts_code"}), ref_date, windows, code_col="ts_code")
            idx_rets = idx_rets.rename(columns={"ts_code": "index_code"})
        else:
            idx_rets = pd.DataFrame()

        # 3) 明细 merge + alpha
        detail = df_score.merge(stock_rets, on=["ts_code", "trade_date"], how="left")
        if not idx_rets.empty:
            for idx in benchmarks:
                s = idx_rets[idx_rets["index_code"] == idx]
                if s.empty:
                    continue
                for N in windows:
                    col_idx = f"ret_fwd_{N}"
                    col_alpha = f"alpha_fwd_{N}_{idx}"
                    detail = detail.merge(
                        s[["trade_date", col_idx]].rename(columns={col_idx: col_alpha}),
                        on="trade_date",
                        how="left",
                    )
                    detail[col_alpha] = detail[f"ret_fwd_{N}"] - detail[col_alpha]

        # 4) 汇总（长表）
        summary = _to_long_summary(detail, windows, benchmarks, group_by_board=group_by_board) if do_summary else pd.DataFrame()

        retro_list = sorted({int(d) for d in (retro_days or []) if int(d) > 0})
        if retro_list:
            retro = _build_retro_cols(ref_date, detail["ts_code"].tolist(), retro_list)
            detail = detail.merge(retro, on="ts_code", how="left")

        # 5) 落地
        out_dir = self.outdir / ref_date
        detail_path = out_dir / "tracking_detail.parquet"
        summary_path = out_dir / "tracking_summary.parquet"
        if save:
            _ensure_dir(out_dir)
            try:
                detail.to_parquet(detail_path, index=False)
                summary.to_parquet(summary_path, index=False)
            except Exception:
                detail.to_csv(detail_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
                summary.to_csv(summary_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
                detail_path = detail_path.with_suffix(".csv")
                summary_path = summary_path.with_suffix(".csv")

        return TrackingResult(detail=detail, summary=summary, detail_path=detail_path if save else None, summary_path=summary_path if save else None)

# ------------------------- 供 score_engine 侧调用的便捷函数 -------------------------
_plugin_singleton: Optional[ScoreTrackingPlugin] = None

def get_tracking_plugin() -> ScoreTrackingPlugin:
    global _plugin_singleton
    if _plugin_singleton is None:
        _plugin_singleton = ScoreTrackingPlugin()
    return _plugin_singleton


def run_tracking(ref_date: str, windows: Sequence[int], benchmarks: Sequence[str] | None = None,
                 *, score_df: Optional[pd.DataFrame] = None, group_by_board: bool = True, save: bool = True,
                 retro_days: Sequence[int] = (), do_summary: bool = True) -> TrackingResult:
    """无状态快捷函数，便于在 score_engine 内部直接调用。"""
    plugin = get_tracking_plugin()
    return plugin.run(ref_date, windows, benchmarks, score_df=score_df, group_by_board=group_by_board, save=save,
                      retro_days=retro_days, do_summary=do_summary)

# ------------------------- 工具 -------------------------

def _pick_trade_dates(ref_date: str, back: int) -> List[str]:
    """返回 [ref_date-back, ..., ref_date] 范围内的交易日列表，用于价格与回看。"""
    days = get_trade_dates() or []
    if ref_date not in days:
        raise ValueError(f"ref_date 不在交易日历内: {ref_date}")
    i = days.index(ref_date)
    j0 = max(0, i - back)
    return days[j0 : i + 1]


def _read_stock_close(codes: Sequence[str], dates: List[str]) -> pd.DataFrame:
    """读取给定股票集合在若干日期的收盘价（前复权）。"""
    # 使用统一的数据库查询
    cols = ["ts_code", "trade_date", "close"]
    start = min(dates) if dates else None
    end = max(dates) if dates else None
    
    # 优先使用缓存读取（仅对单只股票）
    if len(codes) == 1:
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            df = query_stock_data(
                db_path=db_path,
                ts_code=codes[0],
                start_date=start,
                end_date=end,
                adj_type="qfq"
            )
            if not df.empty:
                df = normalize_trade_date(df, "trade_date")
                df = df[df["ts_code"].isin(set(codes))]
                df = df[df["trade_date"].isin(set(dates))]
                return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        except:
            pass
    
    # 使用 database_manager 直接查询
    try:
        from config import DATA_ROOT, UNIFIED_DB_PATH
        db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
        
        # 构建查询条件
        conditions = []
        params = []
        
        if codes:
            placeholders = ",".join(["?"] * len(codes))
            conditions.append(f"ts_code IN ({placeholders})")
            params.extend(codes)
        
        if start:
            conditions.append("trade_date >= ?")
            params.append(start)
            
        if end:
            conditions.append("trade_date <= ?")
            params.append(end)
            
        conditions.append("adj_type = ?")
        params.append(API_ADJ)
        
        # 构建SQL查询
        select_cols = "*" if not cols else ", ".join(cols)
        sql = f"SELECT {select_cols} FROM stock_data"
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY ts_code, trade_date"
        
        # 执行查询
        manager = get_database_manager()
        df = manager.execute_sync_query(db_path, sql, params, timeout=120.0)
    except Exception as e:
        LOG.error(f"读取数据范围失败: {e}")
        df = pd.DataFrame()
    
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    df = normalize_trade_date(df, "trade_date")
    df = df[df["ts_code"].isin(set(codes))]
    df = df[df["trade_date"].isin(set(dates))]
    return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def _select_by_threshold(gdf: pd.DataFrame, sel: Dict) -> pd.DataFrame:
    """在分组内据 sel 规则取前列，并附 rank_in_group。gdf 需含列 'metric'。"""
    gdf = gdf.sort_values("metric", ascending=False).reset_index(drop=True)
    gdf["rank_in_group"] = np.arange(1, len(gdf) + 1)
    typ = sel.get("type", "top_n")
    val = int(sel.get("value", 200))
    if typ == "top_pct":
        k = max(1, math.ceil(len(gdf) * (float(val) / 100.0)))
        return gdf.head(k)
    return gdf.head(max(1, val))


def _build_retro_cols(ref_date: str, codes: Sequence[str], retro_days: Sequence[int]) -> pd.DataFrame:
    """读取 ref_date 往前 d 天的 score/rank，并拼成宽表：score_tminus_d, rank_tminus_d。"""
    retro_days = sorted(set(int(d) for d in retro_days if int(d) > 0))
    if not retro_days:
        return pd.DataFrame({"ts_code": list(codes)})

    all_dates = _pick_trade_dates(ref_date, max(retro_days))
    # 快速索引：映射 d -> date
    idx_ref = all_dates.index(ref_date)
    d2date = {d: all_dates[idx_ref - d] for d in retro_days if idx_ref - d >= 0}

    recs: List[pd.DataFrame] = []
    for d, dt in d2date.items():
        try:
            df = _read_score_all(dt)
        except FileNotFoundError:
            # 某些历史日期可能没有打分输出，跳过
            continue
        df = df[df["ts_code"].isin(codes)][["ts_code", "score", "rank"]]
        df = df.rename(columns={"score": f"score_tminus_{d}", "rank": f"rank_tminus_{d}"})
        recs.append(df)
    if not recs:
        return pd.DataFrame({"ts_code": list(codes)})
    out = recs[0]
    for df in recs[1:]:
        out = out.merge(df, on="ts_code", how="outer")
    return out


# ---------- Commonality helper: build base dataset ----------
def _infer_split_from_surge(surge: pd.DataFrame) -> str:
    """根据 surge 表里的 group 值推断 split 口径。"""
    if surge is None or "group" not in surge.columns:
        return "per_board"
    gs = set(str(x) for x in surge["group"].dropna().unique().tolist())
    if gs.issubset({"600组","000组","科创北组","其他"}):
        return "combo3"
    if gs.issubset({"主板","其他"}):
        return "main_vs_others"
    return "per_board"


def _build_base_dataset(
    df_all: pd.DataFrame, surge: pd.DataFrame, *, group: str = "all", background: str = "all"
) -> pd.DataFrame:
    """
    以观察日全市场打分 df_all 为基，打上 label（surge=1，其他=0）；
    若 background='same_group' 则仅保留与正样本同组的股票。
    输出列：trade_date, ts_code, score, rank, board, label（必要列有则保留）
    """
    if df_all is None or len(df_all) == 0:
        return pd.DataFrame(columns=["trade_date","ts_code","score","rank","board","label"])

    df = df_all.copy()
    df["ts_code"] = df["ts_code"].astype(str).map(lambda s: normalize_ts(s, asset="stock"))
    if "board" not in df.columns:
        df["board"] = df["ts_code"].map(market_label)

    # 正样本：surge 里的代码
    if surge is not None and "ts_code" in surge.columns:
        if "group" in surge.columns and group != "all":
            pos = surge[surge["group"] == group]["ts_code"]
        else:
            pos = surge["ts_code"]
        pos_set = set(pos.astype(str).map(lambda s: normalize_ts(s, asset="stock")).unique().tolist())
    else:
        pos_set = set()

    # 负样本背景范围
    if background == "same_group" and group != "all":
        split_mode = _infer_split_from_surge(surge)
        df["group"] = df["ts_code"].map(lambda s: _board_group(str(s), split_mode))
        df = df[df["group"] == group].copy()

    df["label"] = df["ts_code"].isin(pos_set).astype(int)

    keep = [c for c in ["trade_date","ts_code","score","rank","board","label"] if c in df.columns]
    if "group" in df.columns and "group" not in keep:   # 仅用于检查/调试，不参与分析
        keep.append("group")
    return df[keep].reset_index(drop=True)

# ------------------------- 主入口 -------------------------
@dataclass
class SurgeResult:
    table: pd.DataFrame
    out_path: Optional[Path] = None
    group_files: Optional[Dict[str, Path]] = None


def _board_group(ts_code: str, split: str) -> str:
    """
    根据 split 口径返回分组名。
    - combo3：三组合（"600组" / "000组" / "科创北组"），其余归为 "其他"
    - main_vs_others：主板 vs 其他
    - per_board：按板块（market_label）
    """
    s = str(ts_code or "")
    s = s.strip()
    if split == "combo3":
        if s.endswith(".SH") and s.startswith(("600","601","603","605")):
            return "600组"
        if s.endswith(".SZ") and s.startswith(("000",)):
            return "000组"
        if (s.endswith(".SH") and s.startswith("688")) or (s.endswith(".SZ") and s.startswith("300")) or s.endswith(".BJ"):
            return "科创北组"
        return "其他"
    if split == "per_board":
        return market_label(s)
    # 默认主板 vs 其他
    if (s.endswith(".SH") and not s.startswith("688")) or (s.endswith(".SZ") and s.startswith(("000","001","002","003"))):
        return "主板"
    return "其他"


def run_surge(
    *,
    ref_date: str,
    mode: str = "today",             # "today" | "rolling"
    rolling_days: int = 5,
    selection: Dict = None,           # {"type": "top_n"|"top_pct", "value": 200|10}
    retro_days: Sequence[int] = (1,2,3,4,5),
    split: str = "main_vs_others",    # "main_vs_others" | "per_board" | "combo3"
    score_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> SurgeResult:
    """生成“大涨单子 + 回看”。
    - ref_date：交易日
    - mode：today=当日涨幅榜；rolling=近K日累计涨幅榜
    - rolling_days：rolling 模式下的 K
    - selection：分组内拣选阈值（Top-N / Top-%）
    - retro_days：要回看的天数集合（d=1..T）
    - split：输出口径（主板 vs 其他板；或各自独立）
    - score_df：可直接传入当日全市场打分结果（否则从 score_all_{ref}.csv 读取）
    返回：合并表 + （可选）每组各一份文件路径
    """
    selection = selection or {"type": "top_n", "value": 200}
    ref_date = str(ref_date)
    
    # 0) 当日打分（用于确定“全市场集合”与板块映射）
    if score_df is None:
        df_score = _read_score_all(ref_date)
    else:
        df_score = score_df.copy()
        df_score = normalize_trade_date(df_score, "trade_date")
        df_score = df_score[df_score["trade_date"] == ref_date].copy()
        if "rank" not in df_score.columns and "score" in df_score.columns:
            df_score = df_score.sort_values(["score"], ascending=False).reset_index(drop=True)
            df_score["rank"] = np.arange(1, len(df_score) + 1)
        df_score["ts_code"] = df_score["ts_code"].astype(str).map(lambda s: normalize_ts(s, asset="stock"))
        if "board" not in df_score.columns:
            df_score["board"] = df_score["ts_code"].map(market_label)
        keep = [c for c in ["trade_date", "ts_code", "score", "rank", "board"] if c in df_score.columns]
        df_score = df_score[keep]

    if df_score.empty:
        raise RuntimeError(f"当日打分结果为空：{ref_date}")

    codes = df_score["ts_code"].unique().tolist()

    # 1) 价格视图：读取 close[t-back..t]
    back = 1 if mode == "today" else int(max(1, rolling_days))
    dates = _pick_trade_dates(ref_date, back)
    px = _read_stock_close(codes, dates)
    if px.empty:
        raise RuntimeError("价格数据为空，无法计算涨幅榜")

    # 2) 计算 metric（涨幅）
    last_date = ref_date
    first_date = dates[0]
    # 末日 close
    px_t = px[px["trade_date"] == last_date][["ts_code", "close"]].rename(columns={"close": "close_t"})
    # 起点 close（t-1 或 t-K）
    # px_0 = px[px["trade_date"] == (dates[-2] if mode == "today" else first_date)][["ts_code", "close"]].rename(columns={"close": "close_0"})
    if mode == "today":
        base_date = dates[-2] if len(dates) >= 2 else dates[-1]
    else:
        base_date = first_date
    px_0 = (
        px[px["trade_date"] == base_date][["ts_code", "close"]]
          .rename(columns={"close": "close_0"})
    )
    uni = df_score[["ts_code"]].merge(px_t, on="ts_code", how="left").merge(px_0, on="ts_code", how="left")
    # uni["metric"] = (uni["close_t"] / uni["close_0"]) - 1.0
    uni["metric"] = np.where(
        pd.notna(uni["close_t"]) & pd.notna(uni["close_0"]) & (uni["close_0"] != 0),
        (uni["close_t"] / uni["close_0"]) - 1.0,
        np.nan
    )
    uni["board"] = uni["ts_code"].map(market_label)
    uni["group"] = uni["ts_code"].map(lambda s: _board_group(s, split))

    # 3) 分组选前列
    picks: List[pd.DataFrame] = []
    for g, gdf in uni.groupby("group", dropna=False):
        gpick = _select_by_threshold(gdf[["ts_code", "board", "group", "metric"]].dropna(subset=["metric"]), selection)
        gpick["selection_desc"] = f"{selection.get('type','top_n')}-{selection.get('value')}"
        gpick["mode"] = mode
        gpick["rolling_days"] = int(rolling_days) if mode == "rolling" else np.nan
        gpick["surge_date"] = ref_date
        # 回看轨迹
        retro = _build_retro_cols(ref_date, gpick["ts_code"].tolist(), retro_days)
        gpick = gpick.merge(retro, on="ts_code", how="left")
        picks.append(gpick)

    table = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=["ts_code"])  
    # 排序：组内按 rank_in_group
    table = table.sort_values(["group", "rank_in_group"]).reset_index(drop=True)

    # 4) 落地
    out_dir = SURGE_OUT_BASE / ref_date
    out_path = out_dir / f"surge_{mode}_{selection.get('type','top_n')}{selection.get('value')}_{split}.parquet"
    group_files: Dict[str, Path] = {}

    if save:
        _ensure_dir(out_dir)
        try:
            table.to_parquet(out_path, index=False)
        except Exception:
            out_path = out_path.with_suffix(".csv")
            table.to_csv(out_path, index=False, encoding="utf-8-sig")  
        def _sanitize_filename(s: str) -> str:
            return re.sub(r'[\\\\/:*?"<>|]+', "_", str(s or ""))
        for g, gdf in table.groupby("group", dropna=False):
            safe_g = _sanitize_filename(g)
            gfile = out_dir / f"{safe_g}_surge_{mode}_{selection.get('type','top_n')}{selection.get('value')}.parquet"
            try:
                gdf.to_parquet(gfile, index=False)
            except Exception:
                gfile = gfile.with_suffix(".csv")
                gdf.to_csv(gfile, index=False, encoding="utf-8-sig")
            group_files[str(g)] = gfile

    return SurgeResult(table=table, out_path=out_path if save else None, group_files=group_files or None)

# ------------------------- 小工具 -------------------------
def _load_callable(spec_or_fn: str | Callable) -> Callable:
    if callable(spec_or_fn):
        return spec_or_fn
    spec = str(spec_or_fn)
    if ":" not in spec:
        raise ValueError(f"无效 callable 规范（需 'module:function'）：{spec}")
    mod, fn = spec.split(":", 1)
    m = importlib.import_module(mod)
    return getattr(m, fn)


def _trade_calendar() -> List[str]:
    # 使用统一的数据库查询
    return get_trade_dates() or []


def _prev_trade_date(ref_date: str, d: int) -> str:
    cal = _trade_calendar()
    if ref_date not in cal:
        raise ValueError(f"ref_date 不在交易日内：{ref_date}")
    i = cal.index(ref_date)
    j = max(0, i - int(d))
    return cal[j]

# ------------------------- 默认 Analyzer（可选用） -------------------------
def analyzer_basic(pos_df: pd.DataFrame, ref_df: pd.DataFrame, meta: dict) -> dict:
    """轻量统计：对数值列给出 pos/ref 的均值、P25/50/75、差值与比值。
    仅作为示例，若你提供自定义 analyzers，可不使用本函数。
    """
    # 选择数值列
    def _num(df: pd.DataFrame) -> List[str]:
        cols = []
        for c in df.columns:
            if c in {"ts_code","label","trade_date","board"}:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        return cols

    cols = sorted(set(_num(pos_df) + _num(ref_df)))
    rows = []
    for c in cols:
        a = pd.to_numeric(pos_df[c], errors="coerce")
        b = pd.to_numeric(ref_df[c], errors="coerce")
        if a.notna().sum() == 0 and b.notna().sum() == 0:
            continue
        rec = {"feature": c, "pos_n": int(a.notna().sum()), "ref_n": int(b.notna().sum())}
        for name, s in [("pos", a), ("ref", b)]:
            s = s.dropna()
            if s.empty:
                rec.update({f"{name}_mean": np.nan, f"{name}_p25": np.nan, f"{name}_p50": np.nan, f"{name}_p75": np.nan})
            else:
                rec.update({f"{name}_mean": float(s.mean()), f"{name}_p25": float(np.percentile(s,25)), f"{name}_p50": float(np.percentile(s,50)), f"{name}_p75": float(np.percentile(s,75))})
        rec["diff_mean"] = rec.get("pos_mean", np.nan) - rec.get("ref_mean", np.nan)
        rec["ratio_mean"] = (rec.get("pos_mean", np.nan) / rec.get("ref_mean", np.nan)) if pd.notna(rec.get("pos_mean")) and pd.notna(rec.get("ref_mean")) and rec.get("ref_mean") != 0 else np.nan
        rows.append(rec)
    report = pd.DataFrame(rows)
    return {"basic_stats": report}

# ------------------------- 主流程 -------------------------
@dataclass
class CommonalityResult:
    dataset: pd.DataFrame
    reports: Dict[str, Any]
    out_dir: Optional[Path] = None


def run_commonality(
    *,
    ref_date: str,
    retro_day: int = 1,
    mode: str = "rolling",                 # 与 run_surge 一致："today" | "rolling"
    rolling_days: int = 5,
    selection: Dict = None,                 # {"type":"top_n"|"top_pct", "value":...}
    split: str = "main_vs_others",          # 与 run_surge 一致
    background: str = "all",                # "all"（全市场）| "same_group"（与样本同组）
    feature_providers: Sequence[str|Callable] = (),
    analyzers: Sequence[str|Callable] = (), # 留空则使用 analyzer_basic
    save: bool = True,
    retro_days: Sequence[int] = (),
    count_strategy: bool = False,
    count_strategy_scope: str = "pos",
    strategy_pos_weight: float = 1.0,
) -> CommonalityResult:
    """构建大涨样本（在 ref_date 的前 retro_day 天观察）并运行 analyzers。
    返回 dataset（带 label）与 reports（由 analyzers 返回）。
    """
    selection = selection or {"type":"top_n","value":200}
    ref_date = str(ref_date)
    retro_day = int(retro_day)

    # 1) 选出大涨样本（不落地）
    sr = run_surge(ref_date=ref_date, mode=mode, rolling_days=rolling_days, selection=selection, retro_days=[retro_day], split=split, score_df=None, save=False)
    surge = sr.table.copy()
    if surge.empty:
        raise RuntimeError("大涨样本为空")

    # 2) 观察日（大涨日前 d 天）
    retro_list = sorted({int(retro_day)} | {int(x) for x in (retro_days or [])})
    obs_dates = [_prev_trade_date(ref_date, d) for d in retro_list]

    # 3) 读取观察日全市场打分
    df_all_map = {d: _read_score_all(d) for d in obs_dates}
    
    # 4) 背景集选择
    if background == "same_group" and "group" in surge.columns and "ts_code" in surge.columns:
        # 仅取与各样本同组的股票集合；我们按组分别做，然后合并
        groups = sorted(surge["group"].dropna().unique())
    else:
        groups = ["all"]
        surge["group"] = "all"

    # 5) 为每个组构建数据集并运行 analyzer
    reports_all: Dict[str, Any] = {}
    datasets: List[pd.DataFrame] = []

    for g in groups:
        g_pick = surge[surge["group"] == g]["ts_code"].astype(str).map(lambda s: normalize_ts(s, asset="stock")).unique().tolist()
        obs_date = obs_dates[0]
        df_all = df_all_map[obs_date]
        base = _build_base_dataset(df_all, surge, group=g, background=background)
        
        # 5.2 追加自定义特征
        feats: List[pd.DataFrame] = []
        for prov in feature_providers:
            fn = _load_callable(prov)
            try:
                fdf = fn(codes=base["ts_code"].tolist(), ref_date=obs_date, context={"group": g, "mode": mode, "rolling_days": rolling_days, "selection": selection})
                if not isinstance(fdf, pd.DataFrame) or "ts_code" not in fdf.columns:
                    continue
                feats.append(fdf)
            except Exception as e:
                # 忽略单个 provider 的失败，继续
                print(f"[commonality] provider {prov} failed: {e}")
                continue
        # 合并特征
        X = base
        for f in feats:
            cols = [c for c in f.columns if c != "ts_code"]
            X = X.merge(f[["ts_code", *cols]], on="ts_code", how="left")

        # 5.3 分拆 pos/ref
        pos_df = X[X["label"] == 1].copy()
        ref_df = X[X["label"] == 0].copy()

        # 5.4 分析
        g_reports: Dict[str, Any] = {}
        if not analyzers:
            g_reports = analyzer_basic(pos_df, ref_df, meta={"group": g, "ref_date": ref_date, "obs_date": obs_date, "mode": mode, "rolling_days": rolling_days, "selection": selection})
        else:
            for az in analyzers:
                fn = _load_callable(az)
                try:
                    ret = fn(pos_df, ref_df, meta={"group": g, "ref_date": ref_date, "obs_date": obs_date, "mode": mode, "rolling_days": rolling_days, "selection": selection})
                    if isinstance(ret, dict):
                        for k,v in ret.items():
                            g_reports[f"{k}"] = v
                except Exception as e:
                    print(f"[commonality] analyzer {az} failed: {e}")
        # 标记组名
        for k,v in list(g_reports.items()):
            if isinstance(v, pd.DataFrame):
                v["group"] = g
                g_reports[k] = v
        # 汇总
        datasets.append(X.assign(group=g))
        # 暂存
        for k,v in g_reports.items():
            key = f"{k}__{g}"
            reports_all[key] = v

        # 
        # 5.5 可选：策略触发计数（对每个观察日）
        if count_strategy:
            # 统计范围：
            #   - "pos"  ：仅样本（大涨票）
            #   - "group": 组内全体（样本 + 非样本），可选对样本加权
            #   - "both" ：两者都算，便于对比
            scopes = [count_strategy_scope] if count_strategy_scope in ("pos","group") else ["pos","group"]
            for sc in scopes:
                counts = []
                hits_rows = []
                for od in obs_dates:
                    if sc == "pos":
                        codes_sample = pos_df["ts_code"].tolist()
                        weights_map = None
                    else:
                        # 组内全体
                        codes_sample = base["ts_code"].tolist()
                        # 加权：样本权重 strategy_pos_weight，非样本为 1.0
                        pos_set = set(pos_df["ts_code"].astype(str).tolist())
                        weights_map = {str(ts): (float(strategy_pos_weight) if str(ts) in pos_set else 1.0) for ts in codes_sample}
                    # 1) 按策略聚合
                    cnt = _count_strategy_triggers(od, codes_sample, weights_map=weights_map).copy()
                    cnt["obs_date"] = od
                    cnt["group"] = g
                    cnt["scope"] = sc
                    counts.append(cnt)
                    # 2) 按股票聚合：每票命中多少条规则
                    bystk = _count_hits_per_stock(od, codes_sample).copy()
                    bystk["obs_date"] = od
                    bystk["group"] = g
                    bystk["scope"] = sc
                    hits_rows.append(bystk)
                if counts:
                    df_counts = pd.concat(counts, ignore_index=True, sort=False)
                    key = f"strategy_triggers__{g}__{sc}"
                    reports_all[key] = df_counts
                    g_reports[key] = df_counts
                if hits_rows:
                    df_hits = pd.concat(hits_rows, ignore_index=True, sort=False)
                    
                    # 直方分布（两类）：
                    #   1) 单次型命中条数分布（any/last 等）
                    #   2) 多次型命中条数分布（each 等）
                    # 单次型
                    hist_single = (df_hits.groupby(["obs_date","group","scope","n_single_rules_hit"])["ts_code"]
                                         .count().reset_index(name="n_stocks"))
                    try:
                        denom_map = df_hits.groupby(["obs_date","group","scope"])["ts_code"].nunique().to_dict()
                        hist_single["ratio"] = hist_single.apply(lambda r: (r["n_stocks"] / denom_map.get((r["obs_date"],r["group"],r["scope"]), 1)) if denom_map else 0.0, axis=1)
                    except Exception:
                        pass
                    # 多次型
                    hist_each = (df_hits.groupby(["obs_date","group","scope","n_each_rules_hit"])["ts_code"]
                                       .count().reset_index(name="n_stocks"))
                    try:
                        denom_map2 = denom_map if 'denom_map' in locals() else df_hits.groupby(["obs_date","group","scope"])["ts_code"].nunique().to_dict()
                        hist_each["ratio"] = hist_each.apply(lambda r: (r["n_stocks"] / denom_map2.get((r["obs_date"],r["group"],r["scope"]), 1)) if denom_map2 else 0.0, axis=1)
                    except Exception:
                        pass

                    key_hist_single = f"hits_histogram_single__{g}__{sc}"
                    key_hist_each   = f"hits_histogram_each__{g}__{sc}"
                    key_hits = f"hits_by_stock__{g}__{sc}"
                    reports_all[key_hist_single] = hist_single
                    reports_all[key_hist_each]   = hist_each
                    reports_all[key_hits]        = df_hits
                    g_reports[key_hist_single] = hist_single
                    g_reports[key_hist_each]   = hist_each
                    g_reports[key_hits]        = df_hits


        # 5.6 汇总：将所有分组的 strategy_triggers 合并为一个无后缀表
        # ，方便 UI 直接使用
        try:
            keys = [k for k in reports_all.keys() if str(k).startswith("strategy_triggers__")]
            if keys:
                merged = pd.concat(
                    [reports_all[k] for k in keys if isinstance(reports_all.get(k), pd.DataFrame)],
                    ignore_index=True, sort=False
                ).copy()
                # 列名兜底：trigger_count 等关键列保证存在
                if "trigger_count" not in merged.columns:
                    for alt in ("count","n","num"):
                        if alt in merged.columns:
                            merged = merged.rename(columns={alt: "trigger_count"})
                            break
                    else:
                        merged["trigger_count"] = 0
                for col in ["name","n_sample","coverage","obs_date","group"]:
                    if col not in merged.columns:
                        merged[col] = None
                reports_all["strategy_triggers"] = merged
        except Exception as e:
            print(f"[commonality] merge strategy_triggers failed: {e}")
        # 统一得到 dataset_all（循环后再做一次，避免中途覆盖）
        dataset_all = pd.concat(datasets, ignore_index=True) if datasets else pd.DataFrame()

    # 6) 落地
    out_dir = COMMON_OUT_BASE / ref_date / f"commonality_{mode}_{selection.get('type','top_n')}{selection.get('value')}_d{retro_day}"
    _ensure_dir(out_dir)

    if save and not dataset_all.empty:
        try:
            dataset_all.to_parquet(out_dir / "dataset.parquet", index=False)
        except Exception:
            dataset_all.to_csv(out_dir / "dataset.csv", index=False, encoding="utf-8-sig")

    if save and reports_all:
        for name, obj in reports_all.items():
            p = out_dir / f"report_{name}"
            try:
                if isinstance(obj, pd.DataFrame):
                    obj.to_parquet(p.with_suffix(".parquet"), index=False)
                elif isinstance(obj, pd.Series):
                    obj.to_frame("value").to_parquet(p.with_suffix(".parquet"), index=True)
                else:
                    # (p.with_suffix(".json")).write_text(pd.io.json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
                    (p.with_suffix(".json")).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                # 退 CSV/JSON
                if isinstance(obj, (pd.DataFrame, pd.Series)):
                    obj.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")
                else:
                    # (p.with_suffix(".json")).write_text(pd.io.json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
                    (p.with_suffix(".json")).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    return CommonalityResult(dataset=dataset_all, reports=reports_all, out_dir=out_dir)

# ------------------------- 数据类 -------------------------
@dataclass
class Portfolio:
    id: str
    name: str
    # 资金：总额与可用（未提供则等于 init_cash）
    init_cash: float = 1_000_000.0
    init_available: float | None = None
    # 成交价口径：'next_open' | 'close'
    trade_price_mode: Literal["next_open", "close"] = "next_open"
    # 费率：买/卖分开 + 最低费用（单位：元）
    fee_bps_buy: float = 0.0
    fee_bps_sell: float = 0.0
    min_fee: float = 0.0


@dataclass
class Trade:
    portfolio_id: str
    date: str       # 交易指令日期（成交日依成交模式决定）
    ts_code: str
    side: Literal["BUY", "SELL"]
    qty: int
    price_mode: Optional[Literal["next_open", "close"]] = None  # None 表示跟随组合默认
    price: Optional[float] = None    # 若指定则使用该价格（覆盖 price_mode）
    note: str = "manual"

# ------------------------- 工具 -------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _out_dir(pid: str) -> Path:
    d = PORT_OUT_BASE / pid
    _ensure_dir(d)
    return d


def _load_portfolios() -> Dict[str, Portfolio]:
    f = PORT_OUT_BASE / "portfolios.json"
    if not f.exists():
        return {}
    obj = json.loads(f.read_text(encoding="utf-8"))
    out: Dict[str, Portfolio] = {}
    for k, v in obj.items():
        # 兼容旧字段：fee_bps -> fee_bps_buy/sell；缺失 init_available -> = init_cash
        vv = dict(v)
        if "init_available" not in vv:
            vv["init_available"] = vv.get("init_cash", 0.0)
        if "fee_bps" in vv and ("fee_bps_buy" not in vv and "fee_bps_sell" not in vv):
            try:
                bps = float(vv.get("fee_bps", 0.0))
            except Exception:
                bps = 0.0
            vv["fee_bps_buy"] = bps
            vv["fee_bps_sell"] = bps
        vv.pop("fee_bps", None)
        out[k] = Portfolio(**vv)
    return out


def _save_portfolios(ps: Dict[str, Portfolio]) -> None:
    _ensure_dir(PORT_OUT_BASE)
    obj = {k: asdict(v) for k, v in ps.items()}
    (PORT_OUT_BASE / "portfolios.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_trade_dates(asset: str = "stock") -> List[str]:
    # 使用统一的数据库查询
    return get_trade_dates() or []


def _read_px(codes, start, end, *, asset="stock", cols=("open","close")) -> pd.DataFrame:
    sel = ["ts_code", "trade_date", *cols]
    
    # 优先使用缓存读取（仅对单只股票）
    if len(codes) == 1 and asset == "stock":
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            df = query_stock_data(
                db_path=db_path,
                ts_code=codes[0],
                start_date=start,
                end_date=end,
                adj_type="qfq"
            )
            if not df.empty:
                df = normalize_trade_date(df, "trade_date")
                return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        except:
            pass
    
    # 使用 database_manager 直接查询
    try:
        from config import DATA_ROOT, UNIFIED_DB_PATH
        db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
        
        # 构建查询条件
        conditions = []
        params = []
        
        if codes:
            placeholders = ",".join(["?"] * len(codes))
            conditions.append(f"ts_code IN ({placeholders})")
            params.extend(codes)
        
        if start:
            conditions.append("trade_date >= ?")
            params.append(start)
            
        if end:
            conditions.append("trade_date <= ?")
            params.append(end)
            
        conditions.append("adj_type = ?")
        params.append(API_ADJ)
        
        # 构建SQL查询
        select_cols = "*" if not sel else ", ".join(sel)
        sql = f"SELECT {select_cols} FROM stock_data"
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY ts_code, trade_date"
        
        # 执行查询
        manager = get_database_manager()
        df = manager.execute_sync_query(db_path, sql, params, timeout=120.0)
    except Exception as e:
        LOG.error(f"读取数据范围失败: {e}")
        df = pd.DataFrame()
    
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=sel)
    df = normalize_trade_date(df, "trade_date")
    if codes:
        df = df[df["ts_code"].isin(set(codes))]
    return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

# ------------------------- 管理器 -------------------------
class PortfolioManager:
    def __init__(self):
        self._ports: Dict[str, Portfolio] = _load_portfolios()

    # ---- 组合管理 ----
    def create_portfolio(self, name: str, *, init_cash: float = 1_000_000.0,
                         init_available: float | None = None,
                         trade_price_mode: str = "next_open",
                         fee_bps_buy: float = 0.0,
                         fee_bps_sell: float = 0.0,
                         min_fee: float = 0.0) -> str:
        pid = f"pf_{abs(hash((name, init_cash, trade_price_mode))) % 10**10}"
        pf = Portfolio(
            id=pid, name=name,
            init_cash=float(init_cash),
            init_available=float(init_available) if init_available is not None else float(init_cash),
            trade_price_mode=str(trade_price_mode),
            fee_bps_buy=float(fee_bps_buy),
            fee_bps_sell=float(fee_bps_sell),
            min_fee=float(min_fee),
        )
        self._ports[pid] = pf
        _save_portfolios(self._ports)
        return pid
    def list_portfolios(self) -> Dict[str, Portfolio]:
        return dict(self._ports)

    def get(self, pid: str) -> Portfolio:
        return self._ports[pid]

    # ---- 交易记录 ----
    def _trades_path(self, pid: str) -> Path:
        return _out_dir(pid) / "trades.parquet"

    def _positions_path(self, pid: str) -> Path:
        return _out_dir(pid) / "positions.parquet"

    def _nav_path(self, pid: str) -> Path:
        return _out_dir(pid) / "nav.parquet"

    def read_trades(self, pid: str) -> pd.DataFrame:
        fp = self._trades_path(pid)
        if fp.exists():
            try:
                return pd.read_parquet(fp)
            except Exception:
                return pd.read_csv(fp.with_suffix(".csv"))
        return pd.DataFrame(columns=["portfolio_id","date","ts_code","side","qty","price_mode","price","note"])  

    def write_trades(self, pid: str, df: pd.DataFrame) -> None:
        _ensure_dir(_out_dir(pid))
        fp = self._trades_path(pid)
        try:
            df.to_parquet(fp, index=False)
        except Exception:
            df.to_csv(fp.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    def record_trade(self, pid: str, *, date: str, ts_code: str, side: str, qty: int, price_mode: Optional[str] = None, price: Optional[float] = None, note: str = "manual") -> None:
        ts_code = normalize_ts(ts_code, asset="stock")
        t = Trade(portfolio_id=pid, date=str(date), ts_code=ts_code, side=side.upper(), qty=int(qty), price_mode=price_mode, price=price, note=note)
        df = self.read_trades(pid)
        df = pd.concat([df, pd.DataFrame([asdict(t)])], ignore_index=True)
        self.write_trades(pid, df)

    # ---- 批量建仓/调仓 ----
    def rebalance_to_rank(self, pid: str, *, ref_date: str, df_all_scores: pd.DataFrame, top_n: int = 50, weighting: str = "equal") -> pd.DataFrame:
        """根据当日排名生成目标等权调仓单（在 next_open 成交）。返回生成的交易簿增量 DataFrame。"""
        pf = self.get(pid)
        df = df_all_scores.copy()
        df = normalize_trade_date(df, "ref_date") if "ref_date" in df.columns else normalize_trade_date(df, "trade_date")
        # 兼容列名
        if "trade_date" not in df.columns and "ref_date" in df.columns:
            df = df.rename(columns={"ref_date": "trade_date"})
        df = df[df["trade_date"] == ref_date].copy()
        if "rank" not in df.columns:
            df = df.sort_values(["score"], ascending=False).reset_index(drop=True)
            df["rank"] = np.arange(1, len(df) + 1)
        pick = df.nsmallest(int(top_n), "rank")["ts_code"].astype(str).map(lambda s: normalize_ts(s, asset="stock")).tolist()

        # 目标权重（等权）
        w = 1.0 / max(1, len(pick))
        # 估值基准：用 ref_date 当日收盘近似（成交在 next_open，仅用于下单数量估计）
        px = _read_px(pick, ref_date, ref_date, cols=("close",))
        px = px[px["trade_date"] == ref_date][["ts_code", "close"]]

        nav_df = self.read_nav(pid)
        last_nav = float(nav_df["nav"].iloc[-1]) if len(nav_df) else 1.0
        # 组合资产规模（近似）
        AUM = self.get(pid).init_cash * last_nav
        trades = []
        for _, r in px.iterrows():
            ts = r.ts_code
            price = float(r.close)
            qty = max(0, int((AUM * w) // price))  # 简化：不按手数
            trades.append(Trade(pid, ref_date, ts, "BUY", qty, price_mode="next_open", price=None, note="rebalance_equal"))
        # 写入
        df_old = self.read_trades(pid)
        df_new = pd.concat([df_old, pd.DataFrame([asdict(t) for t in trades])], ignore_index=True)
        self.write_trades(pid, df_new)
        return pd.DataFrame([asdict(t) for t in trades])

    # ---- 估值与净值 ----
    def read_positions(self, pid: str) -> pd.DataFrame:
        fp = self._positions_path(pid)
        if fp.exists():
            try:
                return pd.read_parquet(fp)
            except Exception:
                return pd.read_csv(fp.with_suffix(".csv"))
        return pd.DataFrame(columns=["portfolio_id","date","ts_code","qty","cost","mkt_price","mkt_value","unreal_pnl"])  

    def write_positions(self, pid: str, df: pd.DataFrame) -> None:
        _ensure_dir(_out_dir(pid))
        fp = self._positions_path(pid)
        try:
            df.to_parquet(fp, index=False)
        except Exception:
            df.to_csv(fp.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    def read_nav(self, pid: str) -> pd.DataFrame:
        fp = self._nav_path(pid)
        if fp.exists():
            try:
                return pd.read_parquet(fp)
            except Exception:
                return pd.read_csv(fp.with_suffix(".csv"))
        return pd.DataFrame(columns=["portfolio_id","date","nav","ret_d","max_dd","cash","position_mv"])  

    def write_nav(self, pid: str, df: pd.DataFrame) -> None:
        _ensure_dir(_out_dir(pid))
        fp = self._nav_path(pid)
        try:
            df.to_parquet(fp, index=False)
        except Exception:
            df.to_csv(fp.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    def _exec_date(self, d: str, mode: str, trade_days: List[str]) -> str:
        if mode == "close":
            return d
        # next_open
        if d not in trade_days:
            # 找到 >= d 的第一个交易日作为指令日
            trade_days = [x for x in trade_days if x >= d]
            if not trade_days:
                return d
            d = trade_days[0]
        i = trade_days.index(d)
        j = min(i + 1, len(trade_days) - 1)
        return trade_days[j]

    def reprice_and_nav(self, pid: str, *, date_start: str, date_end: str, benchmarks: Sequence[str] = ()) -> Dict[str, Path]:
        """根据交易簿重放估值，生成持仓与净值时间序列。
        - 交易的实际成交日：next_open -> 下一交易日开盘；close -> 当日收盘；若指定 price 则忽略模式直接用。
        - 当日净值：按收盘价估值（不考虑资金成本/分红）。
        """
        pf = self.get(pid)
        trades = self.read_trades(pid)
        trades = trades[(trades["date"] >= date_start) & (trades["date"] <= date_end)].copy()
        trades["ts_code"] = trades["ts_code"].astype(str).map(lambda s: normalize_ts(s, asset="stock"))

        trade_days = _read_trade_dates("stock")
        # 估算需要的代码集合与区间
        codes = sorted(set(trades["ts_code"]))
        if not codes:
            # 没有交易，则仅输出净值=1的序列
            cal = [d for d in trade_days if date_start <= d <= date_end]
            nav = pd.DataFrame({"portfolio_id": pid, "date": cal})
            nav["nav"] = 1.0
            nav["ret_d"] = 0.0
            nav["max_dd"] = 0.0
            self.write_nav(pid, nav)
            return {"nav_path": self._nav_path(pid)}

        # 拉取价格
        px = _read_px(codes, date_start, date_end, cols=("open","close"))
        if px.empty:
            raise RuntimeError("价格数据为空")
        # 成交价视图
        px_open = px[["ts_code","trade_date","open"]].rename(columns={"open":"exec_price"})
        px_close = px[["ts_code","trade_date","close"]].rename(columns={"close":"exec_price"})

        # 生成成交流水（实际执行日 + 执行价）
        recs = []
        for _, r in trades.iterrows():
            mode = str(r.price_mode) if pd.notna(r.price_mode) and r.price_mode else pf.trade_price_mode
            if pd.notna(r.price):
                exec_d = self._exec_date(str(r.date), mode, trade_days)
                exec_p = float(r.price)
            else:
                exec_d = self._exec_date(str(r.date), mode, trade_days)
                base = px_open if mode == "next_open" else px_close
                p = base[(base["ts_code"] == r.ts_code) & (base["trade_date"] == exec_d)]
                exec_p = float(p["exec_price"].iloc[0]) if len(p) else np.nan
            recs.append({"portfolio_id": pid, "ts_code": r.ts_code, "side": r.side, "qty": int(r.qty), "exec_date": exec_d, "exec_price": exec_p})
        deals = pd.DataFrame(recs)

        # 逐日回放持仓
        cal = sorted({d for d in px["trade_date"].unique() if date_start <= d <= date_end})
        pos_rows = []
        cash = float(self.get(pid).init_available or self.get(pid).init_cash)
        holdings: Dict[str, int] = {}
        cost_basis: Dict[str, float] = {}
        fee_buy = float(self.get(pid).fee_bps_buy) / 10000.0
        fee_sell = float(self.get(pid).fee_bps_sell) / 10000.0
        min_fee = float(getattr(self.get(pid), 'min_fee', 0.0) or 0.0)

        # 预备当日估值价：使用 close
        px_close_map = px.set_index(["ts_code","trade_date"]).close.to_dict()

        for d in cal:
            # 执行今日所有成交
            todays = deals[deals["exec_date"] == d]
            for _, t in todays.iterrows():
                amt = t.qty * t.exec_price
                fee = max(amt * (fee_buy if t.side == "BUY" else fee_sell), min_fee)
                if t.side == "BUY":
                    holdings[t.ts_code] = holdings.get(t.ts_code, 0) + int(t.qty)
                    # 更新成本（加权）
                    prev_qty = holdings.get(t.ts_code, 0) - int(t.qty)
                    prev_cost = cost_basis.get(t.ts_code, 0.0) * prev_qty
                    new_cost = (prev_cost + amt + fee) / max(1, holdings[t.ts_code])
                    cost_basis[t.ts_code] = new_cost
                    cash -= (amt + fee)
                else:
                    sell_qty = min(int(t.qty), holdings.get(t.ts_code, 0))
                    holdings[t.ts_code] = holdings.get(t.ts_code, 0) - sell_qty
                    cash += (sell_qty * t.exec_price - fee)
                    # 若清仓则清成本
                    if holdings[t.ts_code] <= 0:
                        holdings[t.ts_code] = 0
                        cost_basis[t.ts_code] = 0.0

            # 估值
            row_pos = []
            total = cash
            for ts, q in holdings.items():
                price = px_close_map.get((ts, d), np.nan)
                mkt = float(q) * float(price) if pd.notna(price) else 0.0
                total += mkt
                row_pos.append({"portfolio_id": pid, "date": d, "ts_code": ts, "qty": int(q), "cost": float(cost_basis.get(ts, 0.0)), "mkt_price": float(price) if pd.notna(price) else np.nan, "mkt_value": float(mkt), "unreal_pnl": float((price - cost_basis.get(ts, 0.0)) * q) if pd.notna(price) else np.nan})
            pos_rows.extend(row_pos)

        pos_df = pd.DataFrame(pos_rows)
        # 汇总 NAV（以初始现金为 1.0 基准）
        nav_rows = []
        # 重新按日合计市值
        total_by_day = pos_df.groupby("date").mkt_value.sum().reindex(cal, fill_value=0.0)
        cash_series = pd.Series(index=cal, dtype=float)
        # 重算现金轨迹（简单法：用总资产 - 持仓市值；初值 = init_cash）
        # 为保持一致性，我们再做一遍逐日现金：
        cash = float(self.get(pid).init_available or self.get(pid).init_cash)
        holdings = {}
        cost_basis = {}
        deals_by_day = deals.groupby("exec_date")
        for d in cal:
            # 执行
            for _, t in deals_by_day.get_group(d).iterrows() if d in deals_by_day.groups else []:
                amt = t.qty * t.exec_price
                fee = max(amt * (fee_buy if t.side == "BUY" else fee_sell), min_fee)
                if t.side == "BUY":
                    holdings[t.ts_code] = holdings.get(t.ts_code, 0) + int(t.qty)
                    cash -= (amt + fee)
                else:
                    sell_qty = min(int(t.qty), holdings.get(t.ts_code, 0))
                    holdings[t.ts_code] = holdings.get(t.ts_code, 0) - sell_qty
                    amt_exec = sell_qty * t.exec_price
                    fee_exec = amt_exec * fee_sell
                    cash += (amt_exec - fee_exec)
            cash_series.loc[d] = cash
            nav = (cash + total_by_day.loc[d]) / float(self.get(pid).init_cash)
            ret_d = nav_rows[-1]["nav"] if nav_rows else 1.0
            dd = 0.0
            if nav_rows:
                prev = nav_rows[-1]["nav"]
                ret_d = nav / prev - 1.0
                max_nav = max([r["nav"] for r in nav_rows] + [nav])
                dd = nav / max_nav - 1.0
            nav_rows.append({"portfolio_id": pid, "date": d, "nav": float(nav), "ret_d": float(ret_d), "max_dd": float(dd), "cash": float(cash), "position_mv": float(total_by_day.loc[d])})

        nav_df = pd.DataFrame(nav_rows)
        self.write_positions(pid, pos_df)
        self.write_nav(pid, nav_df)
        return {"positions_path": self._positions_path(pid), "nav_path": self._nav_path(pid)}
