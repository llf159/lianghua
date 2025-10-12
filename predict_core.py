# -*- coding: utf-8 -*-
from __future__ import annotations
"""
predict_core — 修复版
- 修复了原文件里 import 块缺失 try 的语法错误
- 修复了 _find_repo_path 返回 None 的 bug
- 修复了 simulate_next_day 写缓存时使用未定义 cache_key 的 bug
- 补全/稳健化了 PredictionInput / PositionCheckInput / run_prediction / run_position_checks 等接口
- 在缺少外部依赖时（indicators/tdx_compat/parquet_viewer），采用温和降级，便于本地自测
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterable, Literal, Tuple, Callable
from pathlib import Path
import json
import numpy as np
import pandas as pd
import datetime as dt
import re

# ---------------- 依赖（带兜底） ----------------
try:
    import config as cfg
    import tdx_compat as tdx
    import indicators as ind
    from indicators import estimate_warmup
    from parquet_viewer import asset_root as pv_asset_root, list_trade_dates, read_range, scan_with_duckdb
except Exception as _e:  # 允许在孤立环境做基本测试
    class _Dummy: ...
    # cfg 兜底
    cfg = type("Cfg", (), {"PARQUET_BASE": ".", "PARQUET_ADJ": "qfq"})()
    # parquet_viewer 兜底
    pv_asset_root = lambda base, asset, adj: base
    def list_trade_dates(root): return []
    read_range = None
    scan_with_duckdb = None
    # indicators 兜底
    ind = _Dummy()
    ind.REGISTRY = {}
    def estimate_warmup(exprs: Optional[Iterable[str]], recompute):
        return 60
    # tdx 兜底
    tdx = _Dummy()
    def _eval_stub(*a, **k):
        # emulate evaluate/evaluate_bool
        data = k.get("data", None)
        if k.get("as_bool"):
            if data is None: 
                return [False]
            return [False] * len(data)
        return {"sig": [False] * (len(data) if data is not None else 1)}
    tdx.evaluate = lambda code, data=None: _eval_stub(code, data=data)
    tdx.evaluate_bool = lambda code, data=None: _eval_stub(code, data=data, as_bool=True)
    tdx.EXTRA_CONTEXT: Dict[str, Any] = {}

# ---------------- 独立缓存层（不影响原始数据） ----------------
class CacheBackend:
    """极简缓存协议：独立目录、按 key 存取。仅缓存“模拟结果”，不改动任何原始行情文件。"""
    def get(self, key: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError
    def set(self, key: str, df: pd.DataFrame) -> None:
        raise NotImplementedError

class FileCache(CacheBackend):
    """把结果存到 cache_dir 下的 parquet/csv（默认 parquet）。"""
    def __init__(self, cache_dir: str = "cache/sim", fmt: Literal["parquet","csv"]="parquet") -> None:
        self.cache_dir = Path(cache_dir)
        self.fmt = fmt
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    def _path(self, key: str) -> Path:
        ext = "parquet" if self.fmt=="parquet" else "csv"
        # key 中可能含有非文件名字符，统一压缩为 md5
        import hashlib
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{digest}.{ext}"
    def get(self, key: str) -> Optional[pd.DataFrame]:
        fp = self._path(key)
        if not fp.exists():
            return None
        try:
            if self.fmt == "parquet":
                return pd.read_parquet(fp)
            else:
                return pd.read_csv(fp)
        except Exception:
            return None
    def set(self, key: str, df: pd.DataFrame) -> None:
        fp = self._path(key)
        try:
            if self.fmt == "parquet":
                df.to_parquet(fp, index=False)
            else:
                df.to_csv(fp, index=False, encoding="utf-8-sig")
        except Exception:
            pass

# ---------------- Hash 辅助 ----------------
def _hash_scenario(scen: "Scenario") -> str:
    """将 Scenario 及关键入参哈希化，作为缓存 key 的一部分。"""
    d = {
        "mode": scen.mode,
        "pct": float(scen.pct),
        "gap_pct": float(scen.gap_pct),
        "hl_mode": scen.hl_mode,
        "range_pct": float(scen.range_pct),
        "atr_mult": float(scen.atr_mult),
        "vol_mode": scen.vol_mode,
        "vol_arg": float(scen.vol_arg),
        "limit_up_pct": float(scen.limit_up_pct),
        "limit_dn_pct": float(scen.limit_dn_pct),
        "lock_higher_than_open": bool(scen.lock_higher_than_open),
        "lock_inside_day": bool(scen.lock_inside_day),
        "warmup_days": int(scen.warmup_days),
    }
    raw = json.dumps(d, sort_keys=True, ensure_ascii=False)
    import hashlib as _hashlib  # local import to avoid clashes
    return _hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

def _hash_universe(codes: List[str]) -> str:
    arr = sorted(str(c) for c in (codes or []))
    raw = "|".join(arr)
    import hashlib as _hashlib
    return _hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

def _make_cache_key(ref_date: str, sim_date: str, scene_hash: str, uni_hash: str, ind_flag: str) -> str:
    return f"{ref_date}_{sim_date}_{scene_hash}_{uni_hash}_{ind_flag}"

# ---------------- 数据模型 ----------------
DType = Literal["stock", "index"]

@dataclass
class Scenario:
    """
    明日情景：支持统一默认 + 个股覆盖。
    所有百分比传入 **百分比单位**（+2.5 表示 +2.5%）。
    """
    # 价格假设
    mode: Literal["close_pct", "open_pct", "gap_then_close_pct", "limit_up", "limit_down", "flat"] = "close_pct"
    pct: float = 0.0                 # 涨跌幅（%），用于 close_pct / open_pct / gap_then_close_pct 的收盘段
    gap_pct: float = 0.0             # 缺口（%），用于 gap_then_close_pct：开盘=昨收*(1+gap_pct)
    # 高低点生成
    hl_mode: Literal["follow", "atr_like", "range_pct"] = "follow"
    range_pct: float = 1.5           # 当日高低振幅（%），仅当 hl_mode="range_pct" 生效
    atr_mult: float = 1.0            # “类 ATR”倍数（从近 N 日高低均值估略），hl_mode="atr_like"
    # 成交量
    vol_mode: Literal["same", "pct", "mult"] = "same"
    vol_arg: float = 0.0             # pct: +10 表示 +10%；mult: 1.2 表示放大 20%
    # 限价板价差
    limit_up_pct: float = 9.9
    limit_dn_pct: float = -9.9
    # 约束
    lock_higher_than_open: bool = False   # True 时强制 C>=O
    lock_inside_day: bool = False         # True 时强制 H/L 覆盖 O/C
    # 指标重算窗口
    warmup_days: int = 60                 # 需要拼接多少历史天作 warm-up（越大指标越准，越慢）

@dataclass
class PerStockOverride:
    """个股覆盖：对某些 ts_code 替换默认 Scenario 字段。"""
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # ts_code -> {field:value}
    def for_ts(self, ts_code: str, base: Scenario) -> Scenario:
        if ts_code in self.overrides and self.overrides[ts_code]:
            data = {**base.__dict__, **self.overrides[ts_code]}
            return Scenario(**data)
        return base

@dataclass
class SimResult:
    """统一返回结构（不再返回历史窗口 df_hist）"""
    ref_date: str                # 基准参考日（上一交易日）
    sim_date: str                # 构造出的“明日日期”（自然日+1，不保证是交易日，仅作占位）
    df_sim: pd.DataFrame         # 仅含模拟日一行（每只一行）
    df_concat: pd.DataFrame      # 历史 + 模拟（便于重算指标/规则判断）

# ---------------- 小工具 ----------------
def _infer_next_date_str(ref_date: str) -> str:
    d = dt.datetime.strptime(ref_date, "%Y%m%d") + dt.timedelta(days=1)
    return d.strftime("%Y%m%d")

def _load_hist_window(
    base: str,
    adj: str,
    asset: DType,
    codes: List[str],
    ref_date: str,
    warmup_days: int,
) -> pd.DataFrame:
    """
    读取近 warmup_days*3 的 O/H/L/C/V（覆盖停牌/周末），过滤到指定 codes。
    返回列至少包含：ts_code, trade_date, open, high, low, close, vol
    """
    root = pv_asset_root(base, "stock" if asset=="stock" else "index", adj)
    # 给足冗余：自然日回溯
    start = (dt.datetime.strptime(ref_date, "%Y%m%d") - dt.timedelta(days=warmup_days*3)).strftime("%Y%m%d")
    cols = ["ts_code","trade_date","open","high","low","close","vol"]
    if scan_with_duckdb:
        df = scan_with_duckdb(root, None, start, ref_date, columns=cols)
    else:
        if read_range is None:
            raise RuntimeError("缺少 scan_with_duckdb/read_range 任一读取函数")
        # df = read_range(base, "stock" if asset=="stock" else "index", adj, None, start, ref_date, columns=cols)
        df = read_range(base, adj, "stock" if asset=="stock" else "index", None, start, ref_date, columns=cols)
    if df.empty:
        return df
    df["trade_date"] = df["trade_date"].astype(str)
    if isinstance(codes, (str, bytes)):
        raise ValueError("predict_core._load_hist_window: 参数 codes 应为 List[str]，但收到 str。请先在调用处展开为代码列表。")
    if codes:
        df = df[df["ts_code"].astype(str).isin(set(codes))]
    # 每只按日期排序
    return df.sort_values(["ts_code","trade_date"]).reset_index(drop=True)

def _gen_hl_from_mode(latest: pd.Series, scen: Scenario) -> Tuple[float,float]:
    O, C, H, L = latest["open"], latest["close"], latest["high"], latest["low"]
    if scen.hl_mode == "follow":
        # 参考上一日的高低与 O/C 的相对关系，等比外扩到今日
        oc_min, oc_max = float(min(O, C)), float(max(O, C))
        upper_w = (H - oc_max) if H>=oc_max else 0.0
        lower_w = (oc_min - L) if oc_min>=L else 0.0
        # 保持比例，若上/下没有空间则给一点极小量
        return float(max(C, O) + (upper_w if upper_w>0 else 0.001)), float(min(C, O) - (lower_w if lower_w>0 else 0.001))
    elif scen.hl_mode == "atr_like":
        span = float(H - L) * max(float(scen.atr_mult), 0.1)
        mid = (float(C) + float(O)) / 2.0
        return float(mid + span/2.0), float(mid - span/2.0)
    else:  # range_pct
        span = float(max(C, O)) * (abs(float(scen.range_pct)) / 100.0)
        mid = (float(C) + float(O)) / 2.0
        return float(mid + span/2.0), float(mid - span/2.0)

def _apply_scenario_row(latest: pd.Series, scen: Scenario) -> Dict[str, Any]:
    """基于上一日 latest 行（Series），生成“模拟日”一行"""
    prev_close = float(latest["close"])
    prev_open  = float(latest["open"])
    # 1) O/C
    if scen.mode == "flat":
        O = prev_close
        C = prev_close
    elif scen.mode == "open_pct":
        O = prev_close * (1.0 + float(scen.pct)/100.0)
        C = prev_close
    elif scen.mode == "gap_then_close_pct":
        O = prev_close * (1.0 + float(scen.gap_pct)/100.0)
        C = O * (1.0 + float(scen.pct)/100.0)
    elif scen.mode == "limit_up":
        O = prev_close * (1.0 + float(scen.limit_up_pct)/100.0)
        C = O
    elif scen.mode == "limit_down":
        O = prev_close * (1.0 + float(scen.limit_dn_pct)/100.0)
        C = O
    else:  # close_pct
        O = prev_close
        C = prev_close * (1.0 + float(scen.pct)/100.0)

    if scen.lock_higher_than_open:
        C = max(C, O)

    # 2) H/L
    H, L = _gen_hl_from_mode(latest, scen)
    # 确保包含 O/C
    if scen.lock_inside_day:
        H = max(H, O, C)
        L = min(L, O, C)

    # 3) V
    V_prev = float(latest.get("vol", 0.0) or 0.0)
    if scen.vol_mode == "same":
        V = V_prev
    elif scen.vol_mode == "pct":
        V = V_prev * (1.0 + float(scen.vol_arg)/100.0)
    else:  # mult
        vmult = (float(scen.vol_arg) if float(scen.vol_arg)!=0 else 1.0)
        V = V_prev * vmult

    return {"open": float(O), "high": float(H), "low": float(L), "close": float(C), "vol": float(max(V, 0.0))}

# ---------------- 主流程 ----------------
def simulate_next_day(
    ref_date: str,
    universe_codes: List[str],
    scenario: Scenario,
    *,
    per_stock: Optional[PerStockOverride] = None,
    base: str | None = None,
    adj: str | None = None,
    asset: DType = "stock",
    recompute_indicators: Iterable[str] | Literal["all","none"] = "none",
    out_dir: str | None = None,
    cache: Optional[CacheBackend] = None,
    indicator_runner: Optional[Callable[[pd.DataFrame, Iterable[str] | Literal["all","none"]], pd.DataFrame]] = None,
) -> SimResult:
    """
    构造“明日”一行，并把历史 + 模拟拼接返回；可选落盘。
    - recompute_indicators: 选 "all" 使用 indicators.REGISTRY 全部；传列表只重算指定指标；"none" 跳过。
    - out_dir: 若非 None，则落盘 csv 到 {out_dir}/{sim_date}/sim_{sim_date}.csv
    """
    base = base or getattr(cfg, "PARQUET_BASE", ".")
    adj  = adj or getattr(cfg, "PARQUET_ADJ", "qfq")
    per_stock = per_stock or PerStockOverride()

    # 先估出 sim_date，用作缓存 key 组成部分
    sim_date = _infer_next_date_str(ref_date)
    scen_hash = _hash_scenario(scenario)
    uni_hash = _hash_universe(universe_codes)
    if recompute_indicators == "all":
        ind_flag = "all"
    elif recompute_indicators == "none":
        ind_flag = "none"
    else:
        ind_flag = ",".join(sorted(str(x) for x in recompute_indicators))
    cache_key = _make_cache_key(ref_date, sim_date, scen_hash, uni_hash, ind_flag)

    # 0) 读缓存
    if cache is not None:
        try:
            cached = cache.get(cache_key)
            if cached is not None and not cached.empty:
                df_all = cached.copy()
                # df_sim：取最后一天（sim_date）
                df_sim = df_all[df_all["trade_date"].astype(str) == str(sim_date)].copy()
                return SimResult(ref_date=ref_date, sim_date=sim_date, df_sim=df_sim, df_concat=df_all)
        except Exception:
            pass

    # 1) 载入历史
    df_hist = _load_hist_window(base, adj, asset, universe_codes, ref_date, scenario.warmup_days)
    if df_hist.empty:
        empty = pd.DataFrame(columns=["ts_code","trade_date","open","high","low","close","vol"])
        return SimResult(ref_date=ref_date, sim_date=sim_date, df_sim=empty, df_concat=empty)

    # 2) 逐只生成模拟行
    last_rows = df_hist.groupby("ts_code").tail(1).reset_index(drop=True)
    recs: List[Dict[str, Any]] = []
    for _, row in last_rows.iterrows():
        ts = str(row["ts_code"])
        scen = per_stock.for_ts(ts, scenario)
        vals = _apply_scenario_row(row, scen)
        recs.append({
            "ts_code": ts,
            "trade_date": sim_date,
            **vals
        })
    df_sim = pd.DataFrame.from_records(recs, columns=["ts_code","trade_date","open","high","low","close","vol"])

    # 3) 拼接（历史 + 模拟）
    df_all = pd.concat([df_hist, df_sim], ignore_index=True)
    df_all = df_all.sort_values(["ts_code","trade_date"]).reset_index(drop=True)

    # 4) 可选重算指标（严格复用现有 REGISTRY）；也可注入自定义运行器（不改变默认行为）
    if indicator_runner is not None:
        df_all = indicator_runner(df_all.copy(), recompute_indicators)
    elif recompute_indicators != "none" and hasattr(ind, "REGISTRY"):
        if recompute_indicators == "all":
            names = [k for k,v in getattr(ind, "REGISTRY", {}).items() if getattr(v, "py_func", None)]
        else:
            names = [n for n in recompute_indicators if n in getattr(ind, "REGISTRY", {}) and getattr(ind.REGISTRY[n], "py_func", None)]
        parts: List[pd.DataFrame] = []
        for ts, sub in df_all.groupby("ts_code"):
            sub = sub.copy()
            for name in names:
                meta = ind.REGISTRY[name]
                try:
                    res = meta.py_func(sub, **(getattr(meta, "kwargs", {}) or {}))
                    if isinstance(res, pd.Series):
                        sub[name] = res
                    elif isinstance(res, pd.DataFrame):
                        for c in res.columns:
                            sub[str(c)] = res[c]
                    elif isinstance(res, dict):
                        for c, s in res.items():
                            sub[str(c)] = s
                except Exception:
                    pass
            parts.append(sub)
        df_all = pd.concat(parts, ignore_index=True).sort_values(["ts_code","trade_date"]).reset_index(drop=True)

    # 5) 落盘（可选，默认不落盘；与缓存独立）
    if out_dir:
        out_base = Path(out_dir) / sim_date
        out_base.mkdir(parents=True, exist_ok=True)
        fpath = out_base / f"sim_{sim_date}.csv"
        try:
            df_all.to_csv(fpath, index=False, encoding="utf-8-sig")
        except Exception:
            pass

    # 6) 写入独立缓存（不影响原始数据）
    if cache is not None:
        try:
            cache.set(cache_key, df_all)
        except Exception:
            pass

    return SimResult(ref_date=ref_date, sim_date=sim_date, df_sim=df_sim, df_concat=df_all)

# ---------------- 规则判定接口（批量） ----------------

def _build_eval_ctx(sub: pd.DataFrame) -> pd.DataFrame:
    """从子表动态构造表达式上下文：将所有数值列（含动态指标）注入，提供原名与大写别名；并注入 O/H/L/C/V 简写。"""
    ctx = {}
    # 跳过明确的非数值辅助列
    skip = {"ts_code", "trade_date"}
    for col in sub.columns:
        if col in skip:
            continue
        s = pd.to_numeric(sub[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        arr = s.values
        name = str(col)
        up = name.upper()
        # 原名
        if name not in ctx:
            ctx[name] = arr
        # 大写别名
        if up not in ctx:
            ctx[up] = arr
    # 常用简写别名
    alias = {"O": "open", "H": "high", "L": "low", "C": "close", "V": "vol"}
    for k, base in alias.items():
        if k not in ctx and base in sub.columns:
            ctx[k] = pd.to_numeric(sub[base], errors="coerce").values
    # ---- KDJ 别名兜底：保证 j/J、k/K、d/D 总是可用 ----
    def _ensure_alias(dst_name: str, cand_names: list[str]):
        if dst_name in ctx:
            return
        # 1) 直接用现成列
        for c in cand_names:
            if c in sub.columns:
                arr = pd.to_numeric(sub[c], errors="coerce").values
                ctx[dst_name] = arr
                # 同时给出大小写两个别名
                ctx[dst_name.lower()] = arr
                ctx[dst_name.upper()] = arr
                return
        # 2) 回退计算（没有指标列时）
        kdj = _fallback_kdj(sub)
        arr = pd.to_numeric(kdj[dst_name.lower()], errors="coerce").values
        ctx[dst_name] = arr
        ctx[dst_name.lower()] = arr
        ctx[dst_name.upper()] = arr

    _ensure_alias("j", ["j", "J", "kdj_j", "KDJ_J"])
    _ensure_alias("k", ["k", "K", "kdj_k", "KDJ_K"])
    _ensure_alias("d", ["d", "D", "kdj_d", "KDJ_D"])
    return pd.DataFrame(ctx)


def eval_when_exprs(
    df_concat: pd.DataFrame,
    sim_date: str,
    exprs: Dict[str, str],
    *,
    for_ts_codes: Optional[List[str]] = None,
    extra_ctx: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    直接用 tdx_compat.evaluate 做“明日是否满足某条件”的判断。
    参数：
      - df_concat: simulate_next_day 返回的 df_concat（历史+模拟）
      - sim_date: 目标判断的日期（通常就是 infer 出来的明日）
      - exprs: {规则名: TDX 表达式}
      - for_ts_codes: 可选，仅对这些股票判定
      - extra_ctx: 可选，传入自定义上下文，如 RANK_* 或其他辅助函数

    返回：DataFrame，索引为 ts_code，列为每个规则名的布尔值。
    """
    if not hasattr(tdx, "evaluate"):
        raise RuntimeError("tdx_compat.evaluate 不可用")

    codes = sorted(set(for_ts_codes or df_concat["ts_code"].astype(str).unique().tolist()))
    rows: List[Dict[str, Any]] = []
    for ts in codes:
        sub = df_concat[df_concat["ts_code"].astype(str) == ts].copy()
        sub = sub.sort_values("trade_date")
        if str(sim_date) not in set(sub["trade_date"].astype(str)):
            continue
        ctx_df = _build_eval_ctx(sub)
# 注入环境
        tdx.EXTRA_CONTEXT.update({"TS": ts, "REF_DATE": str(sim_date)})
        if extra_ctx:
            tdx.EXTRA_CONTEXT.update(extra_ctx)
        res = {}
        for name, code in exprs.items():
            try:
                # 将 DataFrame 的列注入到上下文中
                extra_ctx = {}
                for col in ctx_df.columns:
                    if col.isidentifier():
                        extra_ctx[col] = ctx_df[col]
                        extra_ctx[col.upper()] = ctx_df[col]
                out = tdx.evaluate(code, ctx_df, extra_context=extra_ctx)  # 返回 dict
                sig = out.get("sig") if isinstance(out, dict) else None
                if sig is None and isinstance(out, dict):
                    sig = out.get("SIG") or out.get("last_expr")
                if sig is None and isinstance(out, dict):
                    # 尝试找一个与长度相同的序列
                    for v in out.values():
                        if isinstance(v, (list, np.ndarray, pd.Series)) and len(v)==len(ctx_df):
                            sig = v
                            break
                ok = False
                if isinstance(sig, (list, np.ndarray, pd.Series)) and len(sig)>0:
                    ok = bool(pd.Series(sig).iloc[-1])
                res[name] = ok
            except Exception:
                res[name] = False
        res["ts_code"] = ts
        rows.append(res)

    out_df = pd.DataFrame(rows).set_index("ts_code") if rows else pd.DataFrame(columns=list(exprs.keys()))
    return out_df

# ---------------- KDJ/J 提取 ----------------
def _fallback_kdj(df: pd.DataFrame) -> pd.DataFrame:
    """最简 KDJ（便于在缺少指标列时兜底出 J）；周期用 9,3,3。"""
    close = pd.to_numeric(df["close"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    n = 9
    lowest_low = low.rolling(n, min_periods=1).min()
    highest_high = high.rolling(n, min_periods=1).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100.0
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3*k - 2*d
    return pd.DataFrame({"k": k, "d": d, "j": j})

def _extract_last_j_for_each(df_concat: pd.DataFrame, sim_date: str) -> pd.Series:
    """从 df_concat 中抽取各票在 sim_date 的 J 值。优先用现成列（j/J/kdj_j），否则回退计算。"""
    out: Dict[str, float] = {}
    for ts, sub in df_concat.groupby("ts_code"):
        sub = sub.sort_values("trade_date")
        # 先找列
        cols = [c for c in sub.columns if str(c).lower() in {"j","kdj_j"}]
        if cols:
            s = pd.to_numeric(sub.loc[sub["trade_date"].astype(str)==str(sim_date), cols[0]], errors="coerce")
            if not s.empty and pd.notna(s.iloc[-1]):
                out[str(ts)] = float(s.iloc[-1])
                continue
        # 回退计算
        kdj = _fallback_kdj(sub)
        out[str(ts)] = float(kdj["j"].iloc[-1])
    return pd.Series(out, name="J")

# ---------------- 主入口：明日预测 ----------------
@dataclass
class PredictionInput:
    ref_date: str
    universe: List[str]                       # 代码集合（UI 层负责文本导入/合并）
    scenario: Scenario                        # 全局场景（若 use_rule_scenario=True 且规则自带，则被覆盖）
    rules: Optional[List[Dict[str, Any]]] = None   # PREDICTION_RULES 子集；可为 None/空
    expr: Optional[str] = None               # 临时表达式；当 rules 为空时使用
    use_rule_scenario: bool = False          # 使用规则自带 scenario
    recompute_indicators: Iterable[str] | Literal["all","none"] = ("kdj",)   # 只重算所需
    cache_dir: Optional[str] = None          # 可传 "cache/sim_pred"

def run_prediction(inp: PredictionInput) -> pd.DataFrame:
    """
    返回列：
      ts_code, rule_name, J, ref_date, sim_date, scenario_id
    - 当 rules/expr 都为空：返回全体的 J 表（命中等于“全部”）
    """
    results = []
    # A) 有规则：逐规则跑
    if inp.rules:
        for rule in inp.rules:
            name = str(rule.get("name", "") or "").strip() or "<unnamed>"
            chk  = str(rule.get("check", "") or "").strip()
            scen = inp.scenario
            if inp.use_rule_scenario and isinstance(rule.get("scenario"), dict):
                scen = Scenario(**{**scen.__dict__, **dict(rule["scenario"])})
            # 动态 warmup（让指标最小化读取区间）
            try:
                w = int(estimate_warmup([chk], inp.recompute_indicators))
            except Exception:
                w = int(scen.warmup_days)
            scen = Scenario(**{**scen.__dict__, "warmup_days": int(w)})
            cache = FileCache(inp.cache_dir) if inp.cache_dir else None
            sim = simulate_next_day(
                inp.ref_date, inp.universe, scen,
                recompute_indicators=inp.recompute_indicators,
                cache=cache
            )
            adj_chk = chk
            if scen.mode == "open_pct" and re.search(r"(?i)C\s*>\s*O", chk):
                adj_chk = re.sub(r"(?i)C\s*>\s*O", "C >= O", chk)
            exprs = {name: adj_chk} if adj_chk else {}
 
            if exprs:
                hits = eval_when_exprs(sim.df_concat, sim.sim_date, exprs, for_ts_codes=inp.universe)
                js = _extract_last_j_for_each(sim.df_concat, sim.sim_date)
                h = hits.reset_index().rename(columns={"index": "ts_code"})
                h["ref_date"] = sim.ref_date
                h["sim_date"] = sim.sim_date
                h["J"] = h["ts_code"].map(js.to_dict())
                h["scenario_id"] = scenario_hash(scen)
                h["rule_name"] = name
                # 仅输出命中行
                if name in h.columns:
                    h = h[h[name] == True]  # noqa: E712
                results.append(h[["ts_code","rule_name","J","ref_date","sim_date","scenario_id"]])
            else:
                # 无检查表达式：视为"全部命中"
                js = _extract_last_j_for_each(sim.df_concat, sim.sim_date)
                h = pd.DataFrame({"ts_code": sorted(set(sim.df_concat["ts_code"].astype(str)))})
                h["ref_date"] = sim.ref_date
                h["sim_date"] = sim.sim_date
                h["J"] = h["ts_code"].map(js.to_dict())
                h["scenario_id"] = scenario_hash(scen)
                h["rule_name"] = name
                results.append(h[["ts_code","rule_name","J","ref_date","sim_date","scenario_id"]])
    else:
        # B) 无规则 → 用临时表达式（可空）
        try:
            w = int(estimate_warmup([inp.expr] if inp.expr else None, inp.recompute_indicators))
        except Exception:
            w = int(inp.scenario.warmup_days)
        scen = Scenario(**{**inp.scenario.__dict__, "warmup_days": int(w)})
        cache = FileCache(inp.cache_dir) if inp.cache_dir else None
        sim = simulate_next_day(
            inp.ref_date, inp.universe, scen,
            recompute_indicators=inp.recompute_indicators,
            cache=cache
        )
        js = _extract_last_j_for_each(sim.df_concat, sim.sim_date)
        if inp.expr:
            exprs = {"expr": str(inp.expr)}
            hits = eval_when_exprs(sim.df_concat, sim.sim_date, exprs, for_ts_codes=inp.universe)
            h = hits.reset_index().rename(columns={"index":"ts_code"})
            h = h[h["expr"] == True]  # noqa: E712
            h["rule_name"] = "expr"
            out = h[["ts_code", "rule_name"]].copy()
        else:
            # 仅输出全体的 J 表
            out = pd.DataFrame({"ts_code": sorted(set(sim.df_concat["ts_code"].astype(str)))})
            out["rule_name"] = ""  # 添加rule_name列

        out["ref_date"] = sim.ref_date
        out["sim_date"] = sim.sim_date
        out["J"] = out["ts_code"].map(js.to_dict())
        out["scenario_id"] = scenario_hash(scen)
        results.append(out[["ts_code","rule_name","J","ref_date","sim_date","scenario_id"]])

    if not results:
        return pd.DataFrame(columns=["ts_code","rule_name","J","ref_date","sim_date","scenario_id"])
    df = pd.concat(results, ignore_index=True)
    # 统一列
    if "rule_name" not in df.columns:
        df["rule_name"] = ""
        df = df[["ts_code","rule_name","J","ref_date","sim_date","scenario_id"]]
    return df.sort_values(["rule_name","ts_code"]).reset_index(drop=True)

# ---------------- 主入口：个股持仓检查 ----------------
@dataclass
class PositionCheckInput:
    ref_date: str
    ts_code: str
    rules: List[Dict[str, Any]]                     # POSITION_POLICIES（个股）
    entry_price: float                              # 买点（固定价/策略价/口径价）
    use_scenario: bool = False                      # 是否基于“明日虚拟日”判定
    scenario: Optional[Scenario] = None
    recompute_indicators: Iterable[str] | Literal["all","none"] = ("kdj",)
    extra_vars: Optional[Dict[str, Any]] = None     # 其它变量注入表达式环境

def run_position_checks(inp: PositionCheckInput) -> pd.DataFrame:
    """
    输出：一张“策略触发表”（与个股详情风格兼容）
      列：name, hit(bool), explain(可选), entry_price, unreal_pnl, ref_date, sim_date
    """
    ts = str(inp.ts_code).strip()
    # 统一变量
    extra = dict(inp.extra_vars or {})
    extra["ENTRY_PRICE"] = float(inp.entry_price)
    # 历史加载窗口估算
    exprs = [str(r.get("when") or r.get("check") or "") for r in inp.rules]
    warm = int(estimate_warmup(exprs, inp.recompute_indicators)) if exprs else 60
    # 读取并判定
    if inp.use_scenario:
        scen = inp.scenario or Scenario()
        scen = Scenario(**{**scen.__dict__, "warmup_days": int(warm)})
        sim = simulate_next_day(inp.ref_date, [ts], scen, recompute_indicators=inp.recompute_indicators)
        df = sim.df_concat
        sim_date = sim.sim_date
    else:
        # 仅历史窗口（到 ref_date）
        base = getattr(cfg, "PARQUET_BASE", ".")
        adj = getattr(cfg, "PARQUET_ADJ", "qfq")
        df = _load_hist_window(base, adj, "stock", [ts], inp.ref_date, warm)
        sim_date = str(inp.ref_date)
    if df.empty:
        return pd.DataFrame(columns=["name","hit","explain","entry_price","unreal_pnl","ref_date","sim_date"])
    sub = df[df["ts_code"].astype(str) == ts].sort_values("trade_date").copy()

    # 当日未上市/无价的情况保护
    last = sub.tail(1)
    if last.empty:
        return pd.DataFrame(columns=["name","hit","explain","entry_price","unreal_pnl","ref_date","sim_date"])
    try:
        last_close = float(pd.to_numeric(last["close"]).iloc[-1])
        extra["UNREAL_PNL"] = (last_close - float(extra["ENTRY_PRICE"])) / float(extra["ENTRY_PRICE"])
    except Exception:
        pass

    # 构造 df（列名对齐 evaluate_bool 的习惯：默认从 df 原列 + 全大写别名读取）
    ctx_df = _build_eval_ctx(sub)
# 注入运行上下文（股票代码/参考日/自定义变量）
    tdx.EXTRA_CONTEXT.update({"TS": ts, "REF_DATE": str(sim_date)})
    tdx.EXTRA_CONTEXT.update(extra)

    expr_map: Dict[str, str] = {}
    explain_map: Dict[str,str] = {}
    for r in inp.rules:
        nm = str(r.get("name","")).strip() or "<unnamed>"
        ex = str(r.get("when") or r.get("check") or "")
        expr_map[nm] = ex
        if "explain" in r:
            explain_map[nm] = str(r.get("explain") or "")

    rows = []
    for nm, ex in expr_map.items():
        try:
            sig = tdx.evaluate_bool(ex, ctx_df)
            ok = bool(pd.Series(sig).iloc[-1])
        except Exception:
            ok = False
        rows.append({
            "name": nm,
            "hit": ok,
            "explain": explain_map.get(nm, ""),
            "entry_price": extra["ENTRY_PRICE"],
            "unreal_pnl": extra.get("UNREAL_PNL"),
            "ref_date": str(inp.ref_date),
            "sim_date": str(sim_date),
        })
    return pd.DataFrame(rows)

# ---------------- 内部：批量判定（明日预测用） ----------------
def _eval_exprs(df_concat: pd.DataFrame, sim_date: str, exprs: Dict[str,str], ts_codes: List[str]) -> pd.DataFrame:
    codes = [str(x) for x in (ts_codes or [])]
    rows = []
    for ts in codes:
        sub = df_concat[df_concat["ts_code"].astype(str) == ts].sort_values("trade_date").copy()
        if sub.empty or str(sim_date) not in set(sub["trade_date"].astype(str)):
            continue
        ctx_df = _build_eval_ctx(sub)
        tdx.EXTRA_CONTEXT.update({"TS": ts, "REF_DATE": str(sim_date)})
        r = {"ts_code": ts}
        for nm, ex in exprs.items():
            try:
                sig = tdx.evaluate_bool(ex, ctx_df)
                r[nm] = bool(pd.Series(sig).iloc[-1])
            except Exception:
                r[nm] = False
        rows.append(r)
    df = pd.DataFrame(rows).set_index("ts_code") if rows else pd.DataFrame(columns=list(exprs.keys()))
    return df

# ---------------- Hash（供 UI 打标签） ----------------
def scenario_hash(scen: Scenario) -> str:
    import hashlib, json as _json
    d = {
        "mode": scen.mode,
        "pct": float(scen.pct),
        "gap_pct": float(scen.gap_pct),
        "hl_mode": scen.hl_mode,
        "range_pct": float(scen.range_pct),
        "atr_mult": float(scen.atr_mult),
        "vol_mode": scen.vol_mode,
        "vol_arg": float(scen.vol_arg),
        "limit_up_pct": float(scen.limit_up_pct),
        "limit_dn_pct": float(scen.limit_dn_pct),
        "lock_higher_than_open": bool(scen.lock_higher_than_open),
        "lock_inside_day": bool(scen.lock_inside_day),
        "warmup_days": int(scen.warmup_days),
    }
    s = _json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

# ---------------- 策略模块的动态装载 ----------------
def _find_repo_path(py_path: Optional[str] = None) -> Optional[Path]:
    """
    返回 strategies_repo.py 的路径（若存在）。
    修复：原实现条件命中时没有返回值。
    """
    cand: List[Path] = []
    if py_path:
        cand.append(Path(py_path))
    here = Path(__file__).parent
    cand += [
        here / "strategies_repo.py",
        here.parent / "strategies_repo.py",
        Path("./strategies_repo.py"),
        Path("./strategies/strategies_repo.py"),
    ]
    for p in cand:
        if p.exists():
            return p
    return None

def _load_repo(py_path: Optional[str] = None):
    """尝试动态 import strategies_repo.py，找不到则返回 None。"""
    path = _find_repo_path(py_path)
    if not path:
        return None
    import importlib.util, sys
    mod_name = "strategies_repo_autoload"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

# 公开的加载函数（UI 会调用）
def load_prediction_rules(py_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    从 strategies_repo 读取 PREDICTION_RULES；结构：
      [{"name": "...", "check": "TDX表达式", "scenario": {...}}, ...]
    """
    mod = _load_repo(py_path)
    if mod and hasattr(mod, "PREDICTION_RULES"):
        val = getattr(mod, "PREDICTION_RULES")
        if isinstance(val, list):
            return val
    return []

def load_position_policies(py_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    从 strategies_repo 读取 POSITION_POLICIES；结构：
      [{"name":"...", "when": "...", "explain":"..."}, ...]
    """
    mod = _load_repo(py_path)
    if mod and hasattr(mod, "POSITION_POLICIES"):
        val = getattr(mod, "POSITION_POLICIES")
        if isinstance(val, list):
            return val
    return []

def load_opportunity_policies(py_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    从 strategies_repo 读取 OPPORTUNITY_POLICIES；结构（可选）：
      [{"name":"...", "when":"..."}, ...]
    """
    mod = _load_repo(py_path)
    if mod and hasattr(mod, "OPPORTUNITY_POLICIES"):
        val = getattr(mod, "OPPORTUNITY_POLICIES")
        if isinstance(val, list):
            return val
    return []

# 为旧名兼容（如果 UI 中引用了 _scen_hash / eval_when_exprs 名称）
_scen_hash = scenario_hash
# 也导出别名，便于外部 import
__all__ = [
    "Scenario",
    "PerStockOverride",
    "SimResult",
    "PredictionInput",
    "PositionCheckInput",
    "simulate_next_day",
    "eval_when_exprs",
    "run_prediction",
    "run_position_checks",
    "load_prediction_rules",
    "load_position_policies",
    "load_opportunity_policies",
    "FileCache",
    "CacheBackend",
    "scenario_hash",
]
