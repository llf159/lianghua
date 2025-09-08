# -*- coding: utf-8 -*-
"""
py — 将“跟踪(Tracking) / 大涨单子(Surge)”统一接入 score_engine 的钩子

用法：在你的 score_engine.py 的 run_for_date(...) 里，写完
    - output/score/top/score_top_<ref>.csv
    - output/score/all/score_all_<ref>.csv
后，调用：
# [merged] 
    from stats_core import post_scoring  # removed import due to merge
    post_scoring(ref_date, df_all_scores=out_all)  # out_all 为你刚写盘的全市场 DataFrame

本模块会：
  1) 按 config 中（若无则使用内置默认）的参数调用“跟踪”与“Surge 单子”；
  2) 所有步骤都 try/except 防护，不影响原打分流程；
  3) 产出文件落在：
        output/tracking/<ref>/tracking_{detail,summary}.{parquet|csv}
        output/surge_lists/<ref>/*.{parquet|csv}

可配项（从 config 读取，不存在则用默认）：
  SC_DO_TRACKING: bool = True
  SC_TRACKING_WINDOWS: list[int] = [1,2,3,5,10,20]
  SC_TRACKING_BENCHMARKS: list[str] = []
  SC_TRACKING_GROUP_BY_BOARD: bool = True

  SC_DO_SURGE: bool = True
  SC_SURGE_MODE: str = "rolling"          # "today" | "rolling"
  SC_SURGE_ROLLING_DAYS: int = 5
  SC_SURGE_SELECTION: dict = {"type":"top_n","value":200}
  SC_SURGE_RETRO_DAYS: list[int] = [1,2,3,4,5]
  SC_SURGE_SPLIT: str = "main_vs_others"   # 或 "per_board"
"""
from __future__ import annotations

from typing import Optional, Sequence, Dict, Any

import logging
import traceback
import pandas as pd

try:
    import config as _cfg
except Exception:  # pragma: no cover
    _cfg = None

LOG = logging.getLogger("score.hooks")

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
        if _get("SC_DO_TRACKING", True):
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
        if _get("SC_DO_SURGE", True):
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

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Dict, List, Optional

import numpy as np
import pandas as pd

from config import PARQUET_BASE, PARQUET_ADJ
from utils import normalize_ts, normalize_trade_date, market_label
from parquet_viewer import asset_root, list_trade_dates, read_range

# ------------------------- 常量 -------------------------
SCORE_ALL_DIR = Path("output/score/all")
TRACKING_OUT_BASE = Path("output/tracking")

# ------------------------- 数据模型 -------------------------
@dataclass
class TrackingResult:
    detail: pd.DataFrame
    summary: pd.DataFrame
    detail_path: Optional[Path] = None
    summary_path: Optional[Path] = None

# ------------------------- 工具 -------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_end_date(ref_date: str, max_window: int) -> str:
    root = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
    days = list_trade_dates(root) or []
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
    root = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
    cols = ["ts_code", "trade_date", "open", "close"]
    df = read_range(PARQUET_BASE, "stock", PARQUET_ADJ, None, start, end, columns=cols)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    df = normalize_trade_date(df, "trade_date")
    df = df[df["ts_code"].isin(set(codes))].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def _read_index_prices(index_codes: Sequence[str], start: str, end: str) -> pd.DataFrame:
    if not index_codes:
        return pd.DataFrame(columns=["index_code", "trade_date", "close"])    
    root = asset_root(PARQUET_BASE, "index", "daily")
    cols = ["ts_code", "trade_date", "close"]
    df = read_range(PARQUET_BASE, "index", "daily", None, start, end, columns=cols)

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["index_code", "trade_date", "close"]) 
    df = normalize_trade_date(df, "trade_date")
    # df = df[df["ts_code"].isin(set(codes))].rename(columns={"ts_code": "index_code"})
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
        self.parquet_base = str(parquet_base or PARQUET_BASE)
        self.parquet_adj = str(parquet_adj or PARQUET_ADJ)
        self.outdir = Path(outdir or TRACKING_OUT_BASE)

    # ---- 主入口：从 DataFrame（优先）或 score_all 文件计算 ----
    def run(self,
            ref_date: str,
            windows: Sequence[int],
            benchmarks: Sequence[str] | None = None,
            *,
            score_df: Optional[pd.DataFrame] = None,
            group_by_board: bool = True,
            save: bool = True) -> TrackingResult:
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
        summary = _to_long_summary(detail, windows, benchmarks, group_by_board=group_by_board)

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


def run_tracking(ref_date: str, windows: Sequence[int], benchmarks: Sequence[str] | None = None, *, score_df: Optional[pd.DataFrame] = None, group_by_board: bool = True, save: bool = True) -> TrackingResult:
    """无状态快捷函数，便于在 score_engine 内部直接调用。"""
    return get_tracking_plugin().run(ref_date, windows, benchmarks, score_df=score_df, group_by_board=group_by_board, save=save)

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Dict, List, Optional

import math
import numpy as np
import pandas as pd

from config import PARQUET_BASE, PARQUET_ADJ
from utils import normalize_ts, normalize_trade_date
from parquet_viewer import asset_root, list_trade_dates, read_range

SCORE_ALL_DIR = Path("output/score/all")
SURGE_OUT_BASE = Path("output/surge_lists")

# ------------------------- 工具 -------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_trade_dates(ref_date: str, back: int) -> List[str]:
    """返回 [ref_date-back, ..., ref_date] 范围内的交易日列表，用于价格与回看。"""
    root = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
    days = list_trade_dates(root) or []
    if ref_date not in days:
        raise ValueError(f"ref_date 不在交易日历内: {ref_date}")
    i = days.index(ref_date)
    j0 = max(0, i - back)
    return days[j0 : i + 1]


def _read_stock_close(codes: Sequence[str], dates: List[str]) -> pd.DataFrame:
    """读取给定股票集合在若干日期的收盘价（前复权）。"""
    root = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
    cols = ["ts_code", "trade_date", "close"]
    start = min(dates) if dates else None
    end = max(dates) if dates else None
    df = read_range(PARQUET_BASE, "stock", PARQUET_ADJ, None, start, end, columns=cols)
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

# ------------------------- 主入口 -------------------------
@dataclass
class SurgeResult:
    table: pd.DataFrame
    out_path: Optional[Path] = None
    group_files: Optional[Dict[str, Path]] = None


def run_surge(
    *,
    ref_date: str,
    mode: str = "today",             # "today" | "rolling"
    rolling_days: int = 5,
    selection: Dict = None,           # {"type": "top_n"|"top_pct", "value": 200|10}
    retro_days: Sequence[int] = (1,2,3,4,5),
    split: str = "main_vs_others",    # "main_vs_others" | "per_board"
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
    px_0 = px[px["trade_date"] == (dates[-2] if mode == "today" else first_date)][["ts_code", "close"]].rename(columns={"close": "close_0"})
    uni = df_score[["ts_code"]].merge(px_t, on="ts_code", how="left").merge(px_0, on="ts_code", how="left")
    uni["metric"] = (uni["close_t"] / uni["close_0"]) - 1.0
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
        # 组内单独文件（可选）
        for g, gdf in table.groupby("group", dropna=False):
            gfile = out_dir / f"{g}_surge_{mode}_{selection.get('type','top_n')}{selection.get('value')}.parquet"
            try:
                gdf.to_parquet(gfile, index=False)
            except Exception:
                gfile = gfile.with_suffix(".csv")
                gdf.to_csv(gfile, index=False, encoding="utf-8-sig")
            group_files[str(g)] = gfile

    return SurgeResult(table=table, out_path=out_path if save else None, group_files=group_files or None)

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Dict, List, Optional, Any

import importlib
import math
import numpy as np
import pandas as pd

from config import PARQUET_BASE, PARQUET_ADJ
from utils import normalize_trade_date, normalize_ts
from parquet_viewer import asset_root, list_trade_dates

COMMON_OUT_BASE = Path("output/commonality")
SCORE_ALL_DIR = Path("output/score/all")

# ------------------------- 小工具 -------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
    root = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
    return list_trade_dates(root) or []


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
    obs_date = _prev_trade_date(ref_date, retro_day)

    # 3) 读取观察日全市场打分
    df_all = _read_score_all(obs_date)

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
        if background == "same_group" and g != "all":
            # 背景集 = 观察日里同组所有股票
            # 简易映射同 score_surge:
            def _grp(ts: str) -> str:
                if ts.startswith("8"): return "北交所"
                if ts.startswith("68"): return "科创"
                if ts.startswith("30"): return "创业"
                if ts.startswith("00") or ts.startswith("60"): return "主板"
                return "其他"
            bg_df = df_all[df_all["ts_code"].astype(str).map(lambda s: _grp(s) if isinstance(s,str) else "其他") == ("主板" if g=="主板" else ("其他板" if g=="其他板" else g))]
        else:
            bg_df = df_all

        # 5.1 基础视图：score/rank + label
        base = bg_df[[c for c in ["ts_code","trade_date","score","rank","board"] if c in bg_df.columns]].copy()
        base["label"] = base["ts_code"].isin(set(g_pick)).astype(int)

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

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Dict, List, Literal

import math
import json
import numpy as np
import pandas as pd

from config import PARQUET_BASE, PARQUET_ADJ
from parquet_viewer import asset_root, read_range, list_trade_dates
from utils import normalize_trade_date, normalize_ts

PORT_OUT_BASE = Path("output/portfolio")

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

# positions/nav 用 DataFrame 存盘，不单独建类

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
    root = asset_root(PARQUET_BASE, asset, PARQUET_ADJ)
    return list_trade_dates(root) or []


def _read_px(codes, start, end, *, asset="stock", cols=("open","close")) -> pd.DataFrame:
    sel = ["ts_code", "trade_date", *cols]
    df = read_range(PARQUET_BASE, asset, PARQUET_ADJ,
                    ts_code=None, start=start, end=end, columns=sel)
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
                # else:
                #     sell_qty = min(int(t.qty), holdings.get(t.ts_code, 0))
                #     holdings[t.ts_code] = holdings.get(t.ts_code, 0) - sell_qty
                #     cash += (sell_qty * t.exec_price - fee)
                else:
                    sell_qty = min(int(t.qty), holdings.get(t.ts_code, 0))
                    holdings[t.ts_code] = holdings.get(t.ts_code, 0) - sell_qty
                    amt_exec = sell_qty * t.exec_price
                    fee_exec = amt_exec * fee_rate
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

# === END MERGED MODULE: score_portfolio.py ===