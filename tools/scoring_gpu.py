# -*- coding: utf-8 -*-
"""
scoring_gpu.py — GPU 加速评分模块

目标：
- 复用 scoring_core 中的规则、列选择、数据库访问等实现细节
- 在 GPU 上执行主要的规则布尔计算与窗口统计，缺少 GPU 时可降级到 scoring_core
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import cudf
    import cupy as cp
except Exception as _gpu_err:  # pragma: no cover - 环境相关
    cudf = None
    cp = None

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

from config import (
    DATA_ROOT,
    UNIFIED_DB_PATH,
    SC_REF_DATE,
    SC_LOOKBACK_D,
    SC_BASE_SCORE,
    SC_MIN_SCORE,
    SC_TIE_BREAK,
)
from database_manager import get_database_manager
from log_system import get_logger
from scoring_core import (
    _calc_rule_total_window,
    _compute_max_windows_by_tf,
    _compute_start_date_by_trade_dates,
    _get_score_window,
    _merge_gate_into_rule,
    _select_columns_for_rules,
)
from strategies_repo import RANKING_RULES
from tdx_compat import EPS, compile_script
from utils import get_latest_date_from_database

LOGGER = get_logger("scoring_gpu")


# ---------------- 后端探测 ---------------- #
def _detect_backend() -> str:
    """
    返回可用的 GPU 后端：
      - 'cudf'：NVIDIA + cudf/cupy
      - 'torch'：有 GPU 的 torch（支持 CUDA/ROCm）
      - 'cpu'：无可用 GPU
    """
    # 优先 cudf/cupy（NVIDIA）
    try:
        if cudf is not None and cp is not None:
            devs = cp.cuda.runtime.getDeviceCount()  # type: ignore[attr-defined]
            if devs and devs > 0:
                return "cudf"
    except Exception:
        pass

    # 其次 torch（支持 AMD ROCm 或 CUDA）
    try:
        if torch is not None and torch.cuda.is_available():  # CUDA/ROCm 都会走这里
            return "torch"
    except Exception:
        pass

    return "cpu"


# ---------------- GPU 基础工具 ---------------- #
def _require_gpu(backend: str) -> None:
    """确保 GPU 依赖存在。"""
    if backend == "cudf":
        if cudf is None or cp is None:
            raise RuntimeError("缺少 cudf/cupy，无法使用 NVIDIA GPU 路径")
    elif backend == "torch":
        if torch is None or (not torch.cuda.is_available()):
            raise RuntimeError("缺少 torch GPU 支持，无法使用 torch 后端")


def _to_gpu_series(obj) -> "cudf.Series":
    """将输入统一成 cudf.Series（尽量避免 CPU 回落）。"""
    if isinstance(obj, cudf.Series):
        return obj
    if isinstance(obj, pd.Series):
        return cudf.Series(obj.reset_index(drop=True))
    return cudf.Series(obj)


def _sliding_windows(arr: "cp.ndarray", window: int) -> "cp.ndarray":
    """构造滑窗视图；不足窗口时返回空数组。"""
    window = max(int(window), 1)
    if arr.size < window:
        return cp.empty((0, window), dtype=arr.dtype)
    return cp.lib.stride_tricks.sliding_window_view(arr, window_shape=window)


# --- GPU 版指标/函数（cudf/cupy） --- #
def GREF(x, n=1):
    return _to_gpu_series(x).shift(int(n))


def GMA(x, n):
    s = _to_gpu_series(x)
    w = max(int(n), 1)
    return s.rolling(window=w, min_periods=w).mean()


def GSUM(x, n):
    s = _to_gpu_series(x)
    w = max(int(n), 1)
    return s.rolling(window=w, min_periods=w).sum()


def GHHV(x, n):
    s = _to_gpu_series(x)
    w = max(int(n), 1)
    return s.rolling(window=w, min_periods=w).max()


def GLLV(x, n):
    s = _to_gpu_series(x)
    w = max(int(n), 1)
    return s.rolling(window=w, min_periods=w).min()


def GSTD(x, n):
    s = _to_gpu_series(x)
    w = max(int(n), 1)
    return s.rolling(window=w, min_periods=w).std()


def GEMA(x, n):
    s = _to_gpu_series(x)
    return s.ewm(span=int(n), adjust=False).mean()


def GSMA(x, n, m):
    alpha = float(m) / float(n)
    s = _to_gpu_series(x)
    return s.ewm(alpha=alpha, adjust=False).mean()


def GABS(x):
    return _to_gpu_series(x).abs()


def GMAX(a, b):
    a_s = _to_gpu_series(a)
    b_s = _to_gpu_series(b)
    return cudf.concat([a_s, b_s], axis=1).max(axis=1)


def GMIN(a, b):
    a_s = _to_gpu_series(a)
    b_s = _to_gpu_series(b)
    return cudf.concat([a_s, b_s], axis=1).min(axis=1)


def GIF(cond, a, b):
    cond_s = _to_gpu_series(cond).fillna(False).astype("bool")
    a_s = _to_gpu_series(a)
    b_s = _to_gpu_series(b)
    return cudf.Series(cp.where(cond_s.values, a_s.values, b_s.values))


def GCOUNT(cond, n):
    s = _to_gpu_series(cond).fillna(False).astype("bool")
    w = max(int(n), 1)
    return s.rolling(window=w, min_periods=1).sum()


def GBARSLAST(cond):
    s = _to_gpu_series(cond).fillna(False).astype("bool")
    idx = cudf.Series(cp.arange(len(s)), index=s.index)
    last_hit = idx.where(s, cp.nan).fillna(method="ffill")
    return idx - last_hit


def GCROSS(a, b):
    a_s = _to_gpu_series(a)
    b_s = _to_gpu_series(b)
    return (a_s > b_s) & (a_s.shift(1) <= b_s.shift(1))


def GSAFE_DIV(a, b):
    return _to_gpu_series(a) / (_to_gpu_series(b) + EPS)


def GATAN(x):
    return cudf.Series(cp.arctan(_to_gpu_series(x).values))


def GTS_RANK(x, n):
    s = _to_gpu_series(x)
    w = max(int(n), 1)
    arr = cp.asarray(s.values)
    if arr.size == 0:
        return cudf.Series(dtype="float32")
    wins = _sliding_windows(arr, w)
    if wins.size == 0:
        # 数据不足时保持 NaN 对齐
        return cudf.Series(cp.full(arr.shape, cp.nan), index=s.index)
    last = wins[:, -1][:, None]
    rank = (wins <= last).sum(axis=1).astype(cp.float32)
    prefix = cp.full((w - 1,), cp.nan, dtype=cp.float32) if w > 1 else cp.empty((0,), dtype=cp.float32)
    out = cp.concatenate([prefix, rank])
    return cudf.Series(out, index=s.index)


def GTS_PCT(x, n):
    s = _to_gpu_series(x)
    w = max(int(n), 1)
    arr = cp.asarray(s.values)
    if arr.size == 0:
        return cudf.Series(dtype="float32")
    wins = _sliding_windows(arr, w)
    if wins.size == 0:
        return cudf.Series(cp.full(arr.shape, cp.nan), index=s.index)
    last = wins[:, -1][:, None]
    pct = (wins <= last).sum(axis=1) / w
    prefix = cp.full((w - 1,), cp.nan, dtype=cp.float32) if w > 1 else cp.empty((0,), dtype=cp.float32)
    out = cp.concatenate([prefix, pct.astype(cp.float32)])
    return cudf.Series(out, index=s.index)


GPU_FUNC_CTX: Dict[str, object] = {
    "REF": GREF,
    "MA": GMA,
    "EMA": GEMA,
    "SMA": GSMA,
    "SUM": GSUM,
    "HHV": GHHV,
    "LLV": GLLV,
    "STD": GSTD,
    "ABS": GABS,
    "MAX": GMAX,
    "MIN": GMIN,
    "IF": GIF,
    "COUNT": GCOUNT,
    "CROSS": GCROSS,
    "BARSLAST": GBARSLAST,
    "TS_RANK": GTS_RANK,
    "TS_PCT": GTS_PCT,
    "ATAN": GATAN,
    "SAFE_DIV": GSAFE_DIV,
    "EPS": EPS,
}


# --- GPU 版指标/函数（torch，支持 AMD ROCm/CUDA） --- #
def _t_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _to_torch(x, *, as_bool: bool = False):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, pd.Series):
        arr = x.to_numpy()
    else:
        arr = x
    t = torch.as_tensor(arr, device=_t_device())
    if as_bool:
        return t.bool()
    if not torch.is_floating_point(t):
        t = t.float()
    return t


def _rolling_tensor(x: torch.Tensor, w: int) -> torch.Tensor:
    w = max(int(w), 1)
    if x.numel() < w:
        return x.new_empty((0, w))
    return x.unfold(0, w, 1)


def TREF(x, n=1):
    n = int(n)
    if n <= 0:
        return _to_torch(x)
    t = _to_torch(x)
    pad = torch.full((n,), float("nan"), device=t.device)
    return torch.cat([pad, t[:-n]])


def TMA(x, n):
    t = _to_torch(x)
    w = max(int(n), 1)
    rolls = _rolling_tensor(t, w)
    if rolls.numel() == 0:
        return torch.full_like(t, float("nan"))
    means = rolls.mean(dim=1)
    prefix = torch.full((w - 1,), float("nan"), device=t.device) if w > 1 else t.new_empty((0,))
    return torch.cat([prefix, means])


def TSUM(x, n):
    t = _to_torch(x)
    w = max(int(n), 1)
    rolls = _rolling_tensor(t, w)
    if rolls.numel() == 0:
        return torch.full_like(t, float("nan"))
    sums = rolls.sum(dim=1)
    prefix = torch.full((w - 1,), float("nan"), device=t.device) if w > 1 else t.new_empty((0,))
    return torch.cat([prefix, sums])


def THHV(x, n):
    t = _to_torch(x)
    w = max(int(n), 1)
    rolls = _rolling_tensor(t, w)
    if rolls.numel() == 0:
        return torch.full_like(t, float("nan"))
    mx = rolls.max(dim=1).values
    prefix = torch.full((w - 1,), float("nan"), device=t.device) if w > 1 else t.new_empty((0,))
    return torch.cat([prefix, mx])


def TLLV(x, n):
    t = _to_torch(x)
    w = max(int(n), 1)
    rolls = _rolling_tensor(t, w)
    if rolls.numel() == 0:
        return torch.full_like(t, float("nan"))
    mn = rolls.min(dim=1).values
    prefix = torch.full((w - 1,), float("nan"), device=t.device) if w > 1 else t.new_empty((0,))
    return torch.cat([prefix, mn])


def TSTD(x, n):
    t = _to_torch(x)
    w = max(int(n), 1)
    rolls = _rolling_tensor(t, w)
    if rolls.numel() == 0:
        return torch.full_like(t, float("nan"))
    std = rolls.std(dim=1, unbiased=False)
    prefix = torch.full((w - 1,), float("nan"), device=t.device) if w > 1 else t.new_empty((0,))
    return torch.cat([prefix, std])


def TEMA(x, n):
    t = _to_torch(x)
    return torch.tensor(pd.Series(t.cpu()).ewm(span=int(n), adjust=False).mean().values, device=t.device)


def TSMA(x, n, m):
    alpha = float(m) / float(n)
    t = _to_torch(x)
    return torch.tensor(pd.Series(t.cpu()).ewm(alpha=alpha, adjust=False).mean().values, device=t.device)


def TABS(x):
    return _to_torch(x).abs()


def TMAX(a, b):
    return torch.max(_to_torch(a), _to_torch(b))


def TMIN(a, b):
    return torch.min(_to_torch(a), _to_torch(b))


def TIF(cond, a, b):
    c = _to_torch(cond, as_bool=True)
    av = _to_torch(a)
    bv = _to_torch(b)
    return torch.where(c, av, bv)


def TCOUNT(cond, n):
    c = _to_torch(cond, as_bool=True).float()
    w = max(int(n), 1)
    rolls = _rolling_tensor(c, w)
    if rolls.numel() == 0:
        return torch.zeros_like(c)
    sums = rolls.sum(dim=1)
    prefix = torch.zeros((w - 1,), device=c.device)
    return torch.cat([prefix, sums])


def TBARSLAST(cond):
    c = _to_torch(cond, as_bool=True)
    idx = torch.arange(len(c), device=c.device, dtype=torch.float32)
    mask = torch.where(c, idx, torch.tensor(float("nan"), device=c.device))
    last = pd.Series(mask.cpu().numpy()).ffill().to_numpy()
    last_t = torch.as_tensor(last, device=c.device)
    return idx - last_t


def TCROSS(a, b):
    a_t = _to_torch(a)
    b_t = _to_torch(b)
    return (a_t > b_t) & (TREF(a_t, 1) <= TREF(b_t, 1))


def TSAFE_DIV(a, b):
    return _to_torch(a) / (_to_torch(b) + EPS)


def TATAN(x):
    return torch.atan(_to_torch(x))


def TTS_RANK(x, n):
    t = _to_torch(x)
    w = max(int(n), 1)
    rolls = _rolling_tensor(t, w)
    if rolls.numel() == 0:
        return torch.full_like(t, float("nan"))
    last = rolls[:, -1].unsqueeze(1)
    rank = (rolls <= last).sum(dim=1).float()
    prefix = torch.full((w - 1,), float("nan"), device=t.device) if w > 1 else t.new_empty((0,))
    return torch.cat([prefix, rank])


def TTS_PCT(x, n):
    t = _to_torch(x)
    w = max(int(n), 1)
    rolls = _rolling_tensor(t, w)
    if rolls.numel() == 0:
        return torch.full_like(t, float("nan"))
    last = rolls[:, -1].unsqueeze(1)
    pct = (rolls <= last).sum(dim=1).float() / w
    prefix = torch.full((w - 1,), float("nan"), device=t.device) if w > 1 else t.new_empty((0,))
    return torch.cat([prefix, pct])


TORCH_FUNC_CTX: Dict[str, object] = {
    "REF": TREF,
    "MA": TMA,
    "EMA": TEMA,
    "SMA": TSMA,
    "SUM": TSUM,
    "HHV": THHV,
    "LLV": TLLV,
    "STD": TSTD,
    "ABS": TABS,
    "MAX": TMAX,
    "MIN": TMIN,
    "IF": TIF,
    "COUNT": TCOUNT,
    "CROSS": TCROSS,
    "BARSLAST": TBARSLAST,
    "TS_RANK": TTS_RANK,
    "TS_PCT": TTS_PCT,
    "ATAN": TATAN,
    "SAFE_DIV": TSAFE_DIV,
    "EPS": EPS,
}


def _evaluate_cudf(script: str, gdf: "cudf.DataFrame", extra_context: Optional[Dict] = None) -> cudf.Series:
    """cudf/cupy 路径布尔计算。"""
    program = compile_script(script or "")
    ctx = {"cp": cp, "cudf": cudf, "df": gdf}
    ctx.update(GPU_FUNC_CTX)
    if extra_context:
        ctx.update(extra_context)

    results: Dict[str, object] = {}
    last_val = None
    for name, expr in program:
        last_val = eval(expr, {"__builtins__": {}}, ctx)
        if name:
            ctx[name] = last_val
            results[name] = last_val
    if last_val is not None and "last_expr" not in results:
        results["last_expr"] = last_val

    keys = ("sig", "last_expr", "SIG", "LAST_EXPR")
    for k in keys:
        if k in results:
            s = results[k]
            if isinstance(s, cudf.Series):
                return s.fillna(False).astype("bool")
            if isinstance(s, pd.Series):
                return cudf.Series(s).fillna(False).astype("bool")
            if isinstance(s, (list, tuple, cp.ndarray)):
                return cudf.Series(s, index=gdf.index).fillna(False).astype("bool")
    return cudf.Series(False, index=gdf.index)


# torch 路径：使用自定义 TorchFrame 包装 DataFrame
class TorchFrame:
    def __init__(self, df: pd.DataFrame, device=None):
        self.df = df
        self.device = device or _t_device()

    def __getitem__(self, item):
        if item not in self.df.columns:
            raise KeyError(item)
        return _to_torch(self.df[item].to_numpy(), as_bool=False)


def _evaluate_torch(script: str, df: pd.DataFrame, extra_context: Optional[Dict] = None) -> pd.Series:
    program = compile_script(script or "")
    tdf = TorchFrame(df)
    ctx = {"torch": torch, "df": tdf}
    ctx.update(TORCH_FUNC_CTX)
    if extra_context:
        ctx.update(extra_context)

    results: Dict[str, object] = {}
    last_val = None
    for name, expr in program:
        last_val = eval(expr, {"__builtins__": {}}, ctx)
        if name:
            ctx[name] = last_val
            results[name] = last_val
    if last_val is not None and "last_expr" not in results:
        results["last_expr"] = last_val

    keys = ("sig", "last_expr", "SIG", "LAST_EXPR")
    for k in keys:
        if k in results:
            s = results[k]
            if isinstance(s, torch.Tensor):
                arr = s.detach().cpu().numpy()
                return pd.Series(arr, index=df.index).fillna(False).astype(bool)
            if isinstance(s, pd.Series):
                return s.fillna(False).astype(bool)
            if isinstance(s, (list, tuple)):
                return pd.Series(s, index=df.index).fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


def evaluate_bool_gpu(script: str, df_obj, backend: str, extra_context: Optional[Dict] = None):
    if backend == "cudf":
        return _evaluate_cudf(script, df_obj, extra_context=extra_context)
    if backend == "torch":
        return _evaluate_torch(script, df_obj, extra_context=extra_context)
    raise RuntimeError(f"未知后端: {backend}")


# ---------------- 评分核心（GPU 路径） ---------------- #
def _resolve_ref_date(ref_date: Optional[str]) -> str:
    if ref_date and ref_date != "today":
        return ref_date
    try:
        latest = get_latest_date_from_database()
        if latest:
            return latest
    except Exception:
        pass
    return dt.date.today().strftime("%Y%m%d")


def _pick_tiebreak(df: pd.DataFrame) -> float:
    if SC_TIE_BREAK == "kdj_j_asc":
        try:
            return float(df["j"].iloc[-1])
        except Exception:
            return 0.0
    return 0.0


def _pick_dist_points(dist_points, lag: int) -> float:
    for b in dist_points or []:
        if isinstance(b, dict):
            lo = int(b.get("min", b.get("lag", 0)))
            hi = int(b.get("max", b.get("lag", lo)))
            pts = float(b.get("points", 0))
        else:
            lo, hi, pts = int(b[0]), int(b[1]), float(b[2])
        if lo <= lag <= hi:
            return pts
    return 0.0


class GPUScorer:
    """GPU 加速评分器，接口与 scoring_core 解耦但沿用其配置与规则。"""

    def __init__(
        self,
        ref_date: Optional[str] = None,
        start_date: Optional[str] = None,
        ts_codes: Optional[List[str]] = None,
        db_path: Optional[str] = None,
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.ref_date = _resolve_ref_date(ref_date or SC_REF_DATE)
        self.ts_codes = sorted(set(ts_codes)) if ts_codes else None
        self.db_path = db_path or os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
        self.allow_cpu_fallback = allow_cpu_fallback
        self.rule_columns = _select_columns_for_rules()
        self.backend = _detect_backend()
        self.rules = [_merge_gate_into_rule(r.copy()) for r in (RANKING_RULES or [])]
        self.max_windows_by_tf = _compute_max_windows_by_tf(self.rules)
        self.required_days = max(self.max_windows_by_tf.values() or [SC_LOOKBACK_D])
        self.start_date = start_date or _compute_start_date_by_trade_dates(
            self.ref_date, self.required_days
        )
        LOGGER.info(
            "[GPU] ref=%s start=%s required_days=%s codes=%s backend=%s",
            self.ref_date,
            self.start_date,
            self.required_days,
            "all" if self.ts_codes is None else len(self.ts_codes),
            self.backend,
        )

    # 数据加载沿用 database_manager，避免重复实现
    def _load_prices(self) -> pd.DataFrame:
        manager = get_database_manager()
        df = manager.batch_query_stock_data(
            db_path=self.db_path,
            ts_codes=self.ts_codes,
            start_date=self.start_date,
            end_date=self.ref_date,
            columns=self.rule_columns,
            adj_type="qfq",
        )
        if df.empty:
            LOGGER.warning("[GPU] 未从数据库读到行情数据，请检查路径或日期区间")
            return df
        df["trade_date"] = df["trade_date"].astype(str)
        return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def _score_rule_gpu(
        self, gdf, rule: dict
    ) -> Tuple[float, Optional[int], bool]:
        tf = str(rule.get("timeframe", "D")).upper()
        if tf != "D":
            # 暂不实现 W/M 的 GPU 重采样，交由 CPU 回落处理
            raise NotImplementedError("GPU 路径暂仅支持日线规则")

        score_window = _get_score_window(rule, SC_LOOKBACK_D)
        data_window = max(_calc_rule_total_window(rule, tf), score_window)

        sub = gdf[gdf["trade_date"] <= self.ref_date]
        if sub.empty:
            return 0.0, None, False

        eval_df = sub.tail(data_window)
        if eval_df.empty:
            return 0.0, None, False

        when_expr = (rule.get("when") or "").strip()
        if not when_expr:
            return 0.0, None, True

        s_bool_full = evaluate_bool_gpu(when_expr, eval_df, self.backend)
        # 对齐到计分窗口
        s_bool = s_bool_full.tail(score_window).fillna(False).astype("bool")

        scope = str(rule.get("scope", "ANY")).upper()
        pts = float(rule.get("points", 0.0))
        dist_points = rule.get("dist_points") or rule.get("distance_points") or []

        # 统一拿到 numpy/bool 序列，便于后续逻辑
        if hasattr(s_bool, "to_pandas"):  # cudf Series
            s_bool_np = s_bool.to_pandas().to_numpy()
        else:
            s_bool_np = s_bool.to_numpy() if hasattr(s_bool, "to_numpy") else s_bool

        if scope in {"ANY", "ALL"}:
            hit = bool(s_bool_np.all()) if scope == "ALL" else bool(s_bool_np.any())
            return (pts if hit else 0.0), None, hit

        if scope in {"LAST"}:
            hit = bool(s_bool_np[-1])
            return (pts if hit else 0.0), 0 if hit else None, hit

        if scope in {"RECENT", "DIST", "NEAR"}:
            trues = (s_bool_np.nonzero()[0] if hasattr(s_bool_np, "nonzero") else [])
            if len(trues) == 0:
                return 0.0, None, False
            lag = int(len(s_bool_np) - 1 - int(trues[-1]))
            add = _pick_dist_points(dist_points, lag)
            return add, lag, True

        if scope in {"EACH", "PERBAR", "EACH_TRUE"}:
            cnt = int(s_bool_np.sum())
            return (pts * cnt if cnt else 0.0), None, cnt > 0

        # 默认行为：看最后一根
        hit = bool(s_bool.iloc[-1])
        return (pts if hit else 0.0), 0 if hit else None, hit

    def _score_stock_gpu(self, gdf: pd.DataFrame) -> Tuple[float, float, str, str]:
        if self.backend == "cudf":
            g_backend_df = cudf.DataFrame.from_pandas(gdf.reset_index(drop=True))
        elif self.backend == "torch":
            # torch 直接用 pandas DataFrame，函数内部会转 tensor
            g_backend_df = gdf.reset_index(drop=True)
        else:
            raise RuntimeError("GPU 后端不可用")
        total = float(SC_BASE_SCORE)
        reasons: List[str] = []

        for rule in self.rules:
            try:
                add, lag, hit = self._score_rule_gpu(g_backend_df, rule)
            except NotImplementedError:
                raise
            except Exception as e:
                LOGGER.debug("[GPU] 规则失败 %s: %s", rule.get("name"), e)
                continue
            total += add
            if hit and rule.get("show_reason", True):
                tag = rule.get("name", "rule")
                if lag is not None:
                    tag = f"{tag}(lag={lag})"
                reasons.append(tag)

        total = max(total, float(SC_MIN_SCORE))
        tiebreak = _pick_tiebreak(gdf)
        last_date = str(gdf["trade_date"].iloc[-1])
        return total, tiebreak, ";".join(reasons), last_date

    def _score_cpu_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        from scoring_core import score_dataframe

        records = []
        grouped = df.groupby("ts_code")
        for code, g in grouped:
            ts_code, detail, black, _, _ = score_dataframe(
                g,
                ref_date=self.ref_date,
                ts_code=code,
                start_date=self.start_date,
                allow_prescreen=True,
                tie_break=SC_TIE_BREAK,
            )
            if detail:
                records.append(
                    {
                        "ts_code": ts_code,
                        "score": detail.score,
                        "tiebreak": detail.tiebreak,
                        "reasons": ";".join(detail.highlights or []),
                        "last_date": g["trade_date"].iloc[-1],
                        "close": g["close"].iloc[-1],
                    }
                )
        return pd.DataFrame(records)

    def score(self, top_k: int = 200) -> pd.DataFrame:
        df = self._load_prices()
        if df.empty:
            return pd.DataFrame()

        # GPU 路径
        if self.backend in {"cudf", "torch"}:
            try:
                records = []
                for code, g in df.groupby("ts_code"):
                    g_sorted = g.sort_values("trade_date")
                    score, tiebreak, reasons, last_date = self._score_stock_gpu(g_sorted)
                    records.append(
                        {
                            "ts_code": code,
                            "score": score,
                            "tiebreak": tiebreak,
                            "reasons": reasons,
                            "last_date": last_date,
                            "close": g_sorted["close"].iloc[-1],
                        }
                    )
                res = pd.DataFrame(records)
                if res.empty:
                    return res
                sort_asc = True if SC_TIE_BREAK == "kdj_j_asc" else False
                res = res.sort_values(
                    ["score", "tiebreak", "ts_code"],
                    ascending=[False, sort_asc, True],
                )
                return res.head(top_k)
            except NotImplementedError:
                LOGGER.warning("[GPU] 存在非日线规则，转用 CPU 回落")
            except Exception as e:
                LOGGER.warning("[GPU] GPU 评分失败，错误: %s", e, exc_info=True)
                if not self.allow_cpu_fallback:
                    raise

        # CPU 回落（保持完整功能）
        LOGGER.info("[GPU] 使用 CPU scoring_core 回落路径")
        res = self._score_cpu_fallback(df)
        if res.empty:
            return res
        sort_asc = True if SC_TIE_BREAK == "kdj_j_asc" else False
        return res.sort_values(
            ["score", "tiebreak", "ts_code"],
            ascending=[False, sort_asc, True],
        ).head(top_k)


# ---------------- CLI ---------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU 加速评分（独立于 scoring_core）")
    parser.add_argument("--ref-date", default=SC_REF_DATE, help="参考日 YYYYMMDD 或 today")
    parser.add_argument("--start-date", default=None, help="自定义起始日，留空自动按规则推算")
    parser.add_argument("--ts-codes", default=None, help="逗号分隔的 ts_code 列表，默认全市场")
    parser.add_argument("--top-k", type=int, default=200, help="输出前 K 条")
    parser.add_argument("--db-path", default=None, help="行情数据库路径（duckdb）")
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="禁用 CPU 回落（GPU 不可用时直接报错）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts_codes = None
    if args.ts_codes:
        ts_codes = [c.strip() for c in args.ts_codes.split(",") if c.strip()]

    scorer = GPUScorer(
        ref_date=args.ref_date,
        start_date=args.start_date,
        ts_codes=ts_codes,
        db_path=args.db_path,
        allow_cpu_fallback=not args.no_cpu_fallback,
    )
    ranks = scorer.score(top_k=args.top_k)
    if ranks.empty:
        print("[GPU] 无可用评分结果")
        return
    print(ranks.head(args.top_k))


if __name__ == "__main__":
    main()
