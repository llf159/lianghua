# -*- coding: utf-8 -*-
from __future__ import annotations

import os, io, json, re, math
import warnings
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any
from functools import lru_cache
import threading
from log_system import get_logger
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NOTE: Streamlit 即将移除 use_container_width，统一使用 width='stretch'/'content'

# 忽略tushare的FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="tushare.pro.data_pro",
    message=".*fillna.*method.*deprecated.*"
)

# 初始化日志记录器
logger = get_logger("score_ui")
def ui_cleanup_database_connections():
    """强制清理所有数据库连接 - 统一使用 database_manager 管理"""
    try:
        # 延迟导入 database_manager，避免启动时立即初始化数据库连接
        try:
            from database_manager import clear_connections_only
        except ImportError as e:
            st.error(f"无法导入 database_manager 模块: {e}")
            return False
        
        # 清理数据库连接（轻量级清理，不关闭工作线程）
        clear_connections_only()
        
        # 数据库连接已通过 database_manager 清理
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        st.success("✅ 数据库连接清理完成")
        return True
        
    except Exception as e:
        st.error(f"数据库连接清理失败: {e}")
        return False


def check_database_status():
    """检查数据库状态（使用状态文件）"""
    try:
        from database_manager import check_database_status as check_status
        from database_manager import get_database_manager
        
        # 使用状态文件检查数据库状态
        result = check_status()
        
        if not result.get('status_file_exists', False):
            st.warning("状态文件不存在，建议生成状态文件")
            # 回退到原来的检查方式
            try:
                from database_manager import get_database_info
                db_info = get_database_info()
                manager = get_database_manager()
                enhanced_stats = manager.get_stats()
                st.info(f"数据库管理器: {enhanced_stats}")
                st.info(f"数据库信息: {db_info}")
            except Exception as e:
                st.error(f"检查数据库状态失败: {e}")
            return False
        
        # 显示状态文件信息
        status_file_generated_at = result.get('status_file_generated_at', '未知')
        st.info(f"状态文件生成时间: {status_file_generated_at}")
        
        # 显示当前状态
        current_status = result.get('current_status', {})
        stock_data = current_status.get('stock_data', {})
        details_data = current_status.get('details_data', {})
        
        # 显示股票数据状态
        if stock_data.get('database_exists', False):
            st.success(f"股票数据库: 存在")
            st.info(f"  - 总记录数: {stock_data.get('total_records', 0)}")
            st.info(f"  - 股票数量: {stock_data.get('total_stocks', 0)}")
            adj_types = stock_data.get('adj_types', {})
            for adj_type, adj_status in adj_types.items():
                st.info(f"  - {adj_type}: {adj_status.get('min_date', 'N/A')} ~ {adj_status.get('max_date', 'N/A')}")
        else:
            st.warning("股票数据库: 不存在")
        
        # 显示细节数据状态
        if details_data.get('database_exists', False):
            st.success(f"细节数据库: 存在")
            st.info(f"  - 总记录数: {details_data.get('total_records', 0)}")
            st.info(f"  - 股票数量: {details_data.get('stock_count', 0)}")
            if details_data.get('min_date') and details_data.get('max_date'):
                st.info(f"  - 日期范围: {details_data.get('min_date')} ~ {details_data.get('max_date')}")
        else:
            st.warning("细节数据库: 不存在")
        
        # 显示差异
        differences = result.get('differences', {})
        if differences:
            st.warning("发现状态差异，建议更新状态文件")
            st.json(differences)
        else:
            st.success("状态文件是最新的")
        
        return True
    except Exception as e:
        st.error(f"检查数据库状态失败: {e}")
        return False

# 进程控制功能已移除，相关问题在database_manager中统一处理

import streamlit.components.v1 as components
from contextlib import contextmanager
import shutil
import uuid
import time
import queue
import traceback

# 延迟导入，避免启动时立即初始化数据库连接
# import download as dl
import scoring_core as se
import config as cfg
# stats_core 的功能已移到本文件中
from utils import (
    normalize_ts, ensure_datetime_index, normalize_trade_date, market_label,
    get_latest_date_from_database as _get_latest_date_from_database,
    get_latest_date_from_daily_partition as _get_latest_date_from_daily_partition,
    stock_list_cache_path,
    get_concept_mapping,
    concept_text_for_stock,
    add_concept_column,
)
# 使用 database_manager 替代 data_reader
from database_manager import (
    get_database_manager, query_stock_data, batch_query_stock_data, get_trade_dates, 
    get_stock_list, get_latest_trade_date, get_smart_end_date,
    get_database_info, get_data_source_status, close_all_connections,
    clear_connections_only,
    is_details_db_reading_enabled, get_details_db_path_with_fallback, is_details_db_available
)

def _lazy_import_download():
    """延迟导入 download 模块的函数"""
    try:
        import download as dl
        return dl
    except ImportError as e:
        logger.error(f"导入 download 失败: {e}")
        return None

# 直接使用 database_manager 函数，不再需要包装器
import os
from config import DATA_ROOT, API_ADJ, UNIFIED_DB_PATH, SC_DETAIL_STORAGE, SC_USE_DB_STORAGE, SC_DB_FALLBACK_TO_JSON, SC_TRACKING_TOP_N, CONCEPT_BLACKLIST, CONCEPT_SHRINK_ALPHA
import tdx_compat as tdx
# 从 stats_core 移过来的工具函数和类
from dataclasses import dataclass, asdict
from typing import Sequence, Dict, List, Optional, Literal, Tuple
from pathlib import Path
import json


def _pick_trade_dates(ref_date: str, back: int) -> List[str]:
    """返回 [ref_date-back, ..., ref_date] 范围内的交易日列表，用于价格与回看。"""
    days = get_trade_dates() or []
    if ref_date not in days:
        raise ValueError(f"ref_date 不在交易日历内: {ref_date}")
    i = days.index(ref_date)
    j0 = max(0, i - back)
    return days[j0 : i + 1]


def _prev_trade_date(ref_date: str, d: int) -> str:
    """返回 ref_date 往前 d 个交易日的日期"""
    cal = get_trade_dates() or []
    if ref_date not in cal:
        raise ValueError(f"ref_date 不在交易日内：{ref_date}")
    i = cal.index(ref_date)
    j = max(0, i - int(d))
    return cal[j]


def _shift_trade_date(ref_date: str, offset: int, *, clamp: bool = False) -> Optional[str]:
    """根据交易日历偏移 offset 天（可正可负），超界时返回 None/收尾（clamp=True）。"""
    cal = get_trade_dates() or []
    if ref_date not in cal:
        return None
    idx = cal.index(ref_date) + int(offset)
    if 0 <= idx < len(cal):
        return cal[idx]
    if clamp and cal:
        if idx < 0:
            return cal[0]
        if idx >= len(cal):
            return cal[-1]
    return None


def _shrink_mean(obs_mean: float, count: int, prior: float, alpha: float) -> float:
    """简单的置信收缩：覆盖数少时向 prior 收缩，覆盖数足够大时近似原均值。"""
    try:
        c = max(int(count), 0)
        a = float(alpha)
        return (float(obs_mean) * c + float(prior) * a) / (c + a) if (c + a) > 0 else float(obs_mean)
    except Exception:
        return float(obs_mean)


@lru_cache(maxsize=1)
def _rule_meta_map() -> dict:
    """name -> {as, explain, show_reason} 映射，用于UI端推导高亮/缺点/机会。"""
    meta = {}
    try:
        for r in (getattr(se, "SC_RULES", []) or []):
            name = str(r.get("name", "")).strip()
            if not name:
                continue
            meta[name] = {
                "as": str(r.get("as", "auto") or "auto").lower(),
                "explain": r.get("explain"),
                "show_reason": r.get("show_reason", True),
            }
    except Exception:
        pass
    return meta


def _derive_rule_buckets(rules: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """基于规则命中记录和策略元数据推导高亮/缺点/机会列表。"""
    meta = _rule_meta_map()
    highlights: list[str] = []
    drawbacks: list[str] = []
    opportunities: list[str] = []
    seen = {"highlight": set(), "drawback": set(), "opportunity": set()}

    for r in rules or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        try:
            add_val = float(r.get("add", 0.0))
        except Exception:
            continue
        if add_val == 0:
            continue
        if not bool(r.get("gate_ok", True)):
            continue

        meta_info = meta.get(name, {})
        if meta_info.get("show_reason") is False:
            continue
        bucket = (meta_info.get("as") or "auto").lower()
        if bucket == "opportunity":
            target, key = opportunities, "opportunity"
        elif bucket == "highlight":
            target, key = highlights, "highlight"
        elif bucket == "drawback":
            target, key = drawbacks, "drawback"
        else:
            target, key = (highlights, "highlight") if add_val >= 0 else (drawbacks, "drawback")

        text = meta_info.get("explain") or name
        if text and text not in seen[key]:
            target.append(str(text))
            seen[key].add(text)

    return highlights, drawbacks, opportunities


def _parse_string_list(field_value) -> list:
    """兼容性解析：字符串/JSON/列表 -> list[str]。"""
    if not field_value:
        return []
    if isinstance(field_value, str):
        try:
            parsed = json.loads(field_value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                import ast
                parsed = ast.literal_eval(field_value)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
    if isinstance(field_value, list):
        return field_value
    return []


def _read_stock_prices(codes: Sequence[str], start: str, end: str) -> pd.DataFrame:
    """读取股票价格数据"""
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
        logger.info(f"[数据库连接] 开始获取数据库管理器实例 (读取股票价格数据: codes={len(codes) if codes else 'all'}, {start}~{end})")
        manager = get_database_manager()
        df = manager.execute_sync_query(db_path, sql, params, timeout=120.0)
    except Exception as e:
        logger.error(f"读取数据范围失败: {e}")
        df = pd.DataFrame()
    
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    df = normalize_trade_date(df, "trade_date")
    df = df[df["ts_code"].isin(set(codes))].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def _count_strategy_triggers(obs_date: str, codes_sample: Sequence[str], *, weights_map: dict[str, float] | None = None) -> pd.DataFrame:
    """
    统计某观察日 obs_date 在样本集合 codes_sample 内每个策略（规则名）的触发次数/覆盖率。
    触发判定：per_rules 里 ok=True 或 hit_date==obs_date 或 obs_date ∈ hit_dates。
    可选：weights_map 提供 ts_code -> 权重，用于"加权触发次数"和"加权覆盖率"。
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

# Portfolio 相关数据类和类
PORT_OUT_BASE = Path("output/portfolio")

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

# Portfolio 辅助函数
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
        logger.info(f"[数据库连接] 开始获取数据库管理器实例 (读取价格数据用于回测: codes={len(codes) if codes else 'all'}, {start}~{end})")
        manager = get_database_manager()
        df = manager.execute_sync_query(db_path, sql, params, timeout=120.0)
    except Exception as e:
        logger.error(f"读取数据范围失败: {e}")
        df = pd.DataFrame()
    
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=sel)
    df = normalize_trade_date(df, "trade_date")
    if codes:
        df = df[df["ts_code"].isin(set(codes))]
    return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

# PortfolioManager 类
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

import indicators as ind
import predict_core as pr
from rule_editor import render_rule_editor
from predict_core import (
    PredictionInput, PositionCheckInput, PredictRankingInput,
    run_prediction, run_position_checks, run_predict_ranking,
    load_prediction_rules, load_position_policies, load_opportunity_policies,
    Scenario
)

# ---- Streamlit context guard & cache alias (auto-injected) ----
def _in_streamlit():
    try:
        import streamlit as st
        exists = getattr(getattr(st, "runtime", None), "exists", None)
        if callable(exists):
            return bool(exists())
    except Exception:
        pass
    # 回落到 get_script_run_ctx（并配合方案1静音）
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    return get_script_run_ctx() is not None


def _noop_cache_data(*dargs, **dkw):
    def deco(fn): 
        return fn
    return deco

cache_data = (st.cache_data if _in_streamlit() else _noop_cache_data)
# --------------------------------------------------------------

def _safe_path_hash(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except (OSError, FileNotFoundError):
        return hash(str(p))


def _is_valid_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y%m%d")
        return True
    except ValueError:
        return False


def _safe_int(x, default: int = 60) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _init_session_state():
    """统一初始化 Streamlit session_state 的关键字段，避免重复判断散落各处。"""
    try:
        import streamlit as st  # local import so this can run outside Streamlit too
        if not _in_streamlit():
            return
        defaults = {
            "cur_pid": None,
            "cur_pf": None,
            "rules_obj": {
                "prescreen": getattr(se, "SC_PRESCREEN_RULES", []),
                "rules": getattr(se, "SC_RULES", []),
            },
            "export_pref": {"style": "space", "with_suffix": True},
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
        
        # 添加表达式选股时的数据库连接管理
        if "expression_screening_active" not in st.session_state:
            st.session_state["expression_screening_active"] = False
        
        # 添加details数据库读取控制标记，默认不读取数据库避免写入冲突
        if "details_db_reading_enabled" not in st.session_state:
            st.session_state["details_db_reading_enabled"] = False
        
        # 添加数据查看页面的数据库查询控制标记，默认不查询数据库避免写入冲突
        if "data_view_db_enabled" not in st.session_state:
            st.session_state["data_view_db_enabled"] = False
    except Exception:
        pass


if _in_streamlit():
    st.set_page_config(page_title="ScoreApp", layout="wide")
    _init_session_state()
# ===== 常量路径 =====
SC_OUTPUT_DIR = Path(getattr(cfg, "SC_OUTPUT_DIR", "output/score"))
ALL_DIR  = SC_OUTPUT_DIR / "all"
DET_DIR  = SC_OUTPUT_DIR / "details"
ATTN_DIR = SC_OUTPUT_DIR / "attention"
SPEED_DIR = SC_OUTPUT_DIR / "speed"
LOG_DIR  = Path("./log")


for p in [ALL_DIR, DET_DIR, ATTN_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def _is_placeholder_token(token: str | None) -> bool:
    """判断 token 是否为空或占位符。"""
    if token is None:
        return True
    t = str(token).strip()
    if not t:
        return True
    if t.startswith("在这里"):
        return True
    if t.lower() in {"your_token", "token", "input_your_token"}:
        return True
    return False


def _resolve_token_from_ui(input_token: str | None) -> str | None:
    """UI 场景下确定最终 token，优先使用输入，其次环境变量，最后 config。"""
    candidates = [
        input_token,
        os.environ.get("TUSHARE_TOKEN"),
        os.environ.get("TS_TOKEN"),
        getattr(cfg, "TOKEN", None),
    ]
    for cand in candidates:
        if not _is_placeholder_token(cand):
            return str(cand).strip()
    return None


def _apply_overrides(
    base: str,
    assets: List[str],
    start: str,
    end: str,
    api_adj: str,
    fast_threads: int,
    inc_threads: int,
    inc_ind_workers: int | None,
    token: str | None = None,
):
    """把 UI 输入同步到 download.py 的全局，以便其函数读取。"""
    # 延迟导入 download 模块
    dl = _lazy_import_download()
    if dl is None:
        raise ImportError("无法导入 download 模块")
    
    # download.py 内部多数直接使用模块级常量，这里原地覆写它们
    dl.DATA_ROOT = base
    dl.ASSETS = [a.lower() for a in assets]
    dl.START_DATE = start
    dl.END_DATE = end
    dl.API_ADJ = api_adj.lower()
    dl.FAST_INIT_THREADS = int(max(1, fast_threads))
    dl.STOCK_INC_THREADS = int(max(1, inc_threads))
    if inc_ind_workers is not None and int(inc_ind_workers) > 0:
        dl.INC_RECALC_WORKERS = int(inc_ind_workers)
    if token and not _is_placeholder_token(token):
        dl.TOKEN = token

    # 同步到 config，以便其他模块（如 parquet_viewer）看到一致的 base/adj
    try:
        cfg.DATA_ROOT = base
        cfg.API_ADJ = api_adj.lower() if api_adj.lower() in {"raw","qfq","hfq"} else getattr(cfg, "API_ADJ", "qfq")
        if token and not _is_placeholder_token(token):
            cfg.TOKEN = token
    except Exception:
        pass

@cache_data(show_spinner=False, ttl=300)
def _latest_trade_date(base: str, adj: str) -> str | None:
    try:
        # 使用 database_manager 获取最新交易日
        latest_date = get_latest_trade_date()
        return latest_date
    except Exception:
        return None

# ===== 小工具 =====
def _path_all(ref: str) -> Path: return ALL_DIR / f"score_all_{ref}.csv"
def _path_detail(ref: str, ts: str) -> Path: return DET_DIR / ref / f"{normalize_ts(ts)}_{ref}.json"
def _today_str() -> str:
    return date.today().strftime("%Y%m%d")


def _normalize_rank_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """统一排名文件的日期列：优先保留 ref_date，若只有 trade_date 则重命名；两者同时存在则去掉 trade_date。"""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    has_ref = "ref_date" in df2.columns
    has_trade = "trade_date" in df2.columns
    if has_ref and has_trade:
        df2 = df2.drop(columns=["trade_date"])
    elif has_trade and not has_ref:
        df2 = df2.rename(columns={"trade_date": "ref_date"})
    return df2


def _cleanup_rank_file_dates(path: str | Path) -> None:
    """落盘后的排名文件去重日期列，确保只保留 ref_date。"""
    try:
        p = Path(path)
        if not p.exists() or p.suffix.lower() != ".csv":
            return
        df = pd.read_csv(p, dtype={"ts_code": str}, encoding="utf-8-sig", engine="c")
        if df is None or df.empty:
            return
        df_norm = _normalize_rank_date_cols(df)
        try:
            unchanged = df_norm.equals(df)
        except Exception:
            unchanged = False
        if unchanged:
            return
        # 如有调整，覆盖写回
        df_norm.to_csv(p, index=False, encoding="utf-8-sig")
    except Exception as e:
        logger.debug(f"清理排名日期列失败 {path}: {e}")


@cache_data(show_spinner=False, ttl=120)
def _read_rank_all_sorted(ref: str) -> pd.DataFrame:
    """读取全量排名并保证含 rank 列与顺序"""
    path = _path_all(ref)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = _read_df(path, dtype={"ts_code": str})
    if df is None or df.empty:
        return pd.DataFrame()
    df_sorted = _normalize_rank_date_cols(df)
    if "rank" in df_sorted.columns:
        df_sorted = df_sorted.sort_values("rank")
    else:
        if "tiebreak_j" in df_sorted.columns:
            df_sorted = df_sorted.sort_values(["score", "tiebreak_j", "ts_code"], ascending=[False, True, True])
        else:
            df_sorted = df_sorted.sort_values(["score", "ts_code"], ascending=[False, True])
        df_sorted = df_sorted.reset_index(drop=True)
        df_sorted["rank"] = np.arange(1, len(df_sorted) + 1)
    return df_sorted.reset_index(drop=True)


def _slice_top_from_all(ref: str, k: int | None) -> pd.DataFrame:
    df_sorted = _read_rank_all_sorted(ref)
    if df_sorted.empty:
        return df_sorted
    if k is None or int(k) <= 0:
        return df_sorted
    return df_sorted.head(int(k))


def _board_category(ts_code: str) -> str:
    """按 ts_code 粗分板块分类。"""
    market = market_label(str(ts_code))
    if market in ("沪A", "深A"):
        return "主板"
    if market in ("创业板", "科创板"):
        return "创业/科创"
    if market == "北交所":
        return "北交所"
    return "其他"


def _slice_board_top(ref: str, board: str, k: int = 100) -> pd.DataFrame:
    """按板块取前 k 名；board=全部 时不筛板块。"""
    df_sorted = _read_rank_all_sorted(ref)
    if df_sorted.empty:
        return df_sorted
    if board and board != "全部":
        df_sorted = df_sorted.copy()
        df_sorted["板块分类"] = df_sorted["ts_code"].apply(_board_category)
        df_sorted = df_sorted[df_sorted["板块分类"] == board]
    if df_sorted.empty:
        return df_sorted
    if k is None or int(k) <= 0:
        return df_sorted
    return df_sorted.head(int(k))


def _codes_triggered_on_rule(ref_date: str, rule_name: str) -> list[str]:
    """
    获取在 ref_date 触发指定策略的股票代码列表。
    触发判定：ok=True 或 hit_date/hit_dates 包含 ref_date。
    """
    ref_date = (ref_date or "").strip()
    rule_name = (rule_name or "").strip()
    if not ref_date or not rule_name:
        return []

    def _rule_hit(rules_obj) -> bool:
        for r in rules_obj or []:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or "")
            if name != rule_name:
                continue
            ok = bool(r.get("ok"))
            add_val = r.get("add")
            add_hit = (add_val is not None and float(add_val) > 0.0)
            hd = str(r.get("hit_date") or "")
            hds = [str(x) for x in (r.get("hit_dates") or [])]
            if ok or add_hit or hd == ref_date or ref_date in hds:
                return True
        return False

    codes: set[str] = set()
    candidates: set[str] = set()

    # 1) 收集候选 ts_code（DB 优先）
    if is_details_db_reading_enabled() and is_details_db_available():
        try:
            details_db_path = get_details_db_path_with_fallback()
            if details_db_path:
                manager = get_database_manager()
                if manager:
                    sql = "SELECT ts_code FROM stock_details WHERE ref_date = ?"
                    df_codes = manager.execute_sync_query(details_db_path, sql, [ref_date], timeout=30.0)
                    if not df_codes.empty and "ts_code" in df_codes.columns:
                        candidates.update(df_codes["ts_code"].astype(str).str.strip().tolist())
        except Exception as e:
            logger.debug(f"读取details数据库失败：{e}")

    # 2) 补充文件系统候选
    ddir = DET_DIR / str(ref_date)
    if ddir.exists():
        for fp in ddir.glob("*.json"):
            try:
                obj = json.loads(fp.read_text(encoding="utf-8-sig"))
                ts = str(obj.get("ts_code", "")).strip()
                if ts:
                    candidates.add(ts)
            except Exception:
                continue

    # 3) 逐个加载详情（_load_detail_json 内部处理 DB/文件），复用选股页的解析逻辑
    for ts in candidates:
        data = _load_detail_json(ref_date, ts) or {}
        rules_obj = data.get("rules") or data.get("per_rules") or []
        if _rule_hit(rules_obj):
            codes.add(ts)

    codes_list = sorted({str(c).strip() for c in codes if str(c).strip()})
    if not codes_list:
        return codes_list

    # 按排名顺序排序（有排名时优先 rank，无排名时保持字母序）
    try:
        df_rank = _read_rank_all_sorted(ref_date)
        if not df_rank.empty and {"ts_code", "rank"}.issubset(df_rank.columns):
            rank_map: dict[str, int] = {}
            for _, row in df_rank.iterrows():
                ts_val = str(row.get("ts_code", "")).strip()
                r_val = row.get("rank")
                if not ts_val or pd.isna(r_val):
                    continue
                try:
                    r_int = int(r_val)
                except Exception:
                    continue
                rank_map[ts_val] = r_int
                if "." in ts_val:
                    rank_map[ts_val.split(".")[0]] = r_int  # 兼容无后缀的代码

            if rank_map:
                missing_rank = math.inf
                codes_list = sorted(
                    codes_list,
                    key=lambda x: (rank_map.get(x, rank_map.get(x.split(".")[0], missing_rank)), x),
                )
    except Exception as e:
        logger.debug(f"按排名排序触发策略名单失败: {e}")

    return codes_list


@cache_data(show_spinner=False, ttl=600)
def _get_stock_name_map() -> dict[str, str]:
    """
    从缓存的 stock_list.csv 构建代码->名称映射。
    兼容 ts_code 和不带后缀的 symbol 两种键。
    """
    path = stock_list_cache_path()
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
        if df is None or df.empty:
            return {}
        df = df.fillna("")
        mp: dict[str, str] = {}
        for _, row in df.iterrows():
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            ts = str(row.get("ts_code", "")).strip()
            if ts:
                mp[ts] = name
            sym = str(row.get("symbol", "")).strip()
            if sym:
                mp[sym] = name
        return mp
    except Exception:
        return {}


def _stock_name_of(ts_code: str) -> str | None:
    """根据 ts_code 查找股票名称（优先带后缀匹配，次选去后缀）。"""
    if not ts_code:
        return None
    mp = _get_stock_name_map()
    if not mp:
        return None
    ts = str(ts_code).strip()
    if ts in mp:
        return mp[ts]
    if "." in ts:
        core = ts.split(".")[0]
        return mp.get(core)
    return None


def _resolve_user_code_input(raw: str) -> str | None:
    """
    将用户输入解析为 ts_code：
    - 优先按 normalize_ts 规范化代码/符号
    - 若仍非标准格式，尝试用股票名称或 symbol 在缓存的 stock_list.csv 中匹配
    """
    txt = (raw or "").strip()
    if not txt:
        return None
    norm = normalize_ts(txt)
    # 已经得到标准代码或规范化结果
    if norm and norm != txt:
        return norm
    if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", norm):
        return norm
    # 尝试通过缓存匹配 name / symbol
    try:
        path = stock_list_cache_path()
        if not path.exists():
            return norm or None
        df = pd.read_csv(path, dtype=str).fillna("")
        # symbol 精确匹配
        sym_upper = txt.upper()
        hit = df.loc[df["symbol"].str.upper() == sym_upper, "ts_code"]
        if not hit.empty:
            return str(hit.iloc[0]).strip()
        # name 精确匹配
        hit = df.loc[df["name"].str.strip() == txt.strip(), "ts_code"]
        if not hit.empty:
            return str(hit.iloc[0]).strip()
        # name 模糊匹配（包含）
        hit = df.loc[df["name"].str.contains(txt, case=False, na=False), "ts_code"]
        if not hit.empty:
            return str(hit.iloc[0]).strip()
    except Exception:
        return norm or None
    return norm or None


@cache_data(show_spinner=False, ttl=300)
def _get_stock_basic_map() -> dict[str, dict]:
    """
    从 stock_list.csv 读取基础信息（包含总市值/流通市值），返回 ts_code -> info 映射。
    """
    path = stock_list_cache_path()
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype={"ts_code": str})
        if df is None or df.empty:
            return {}
        df = df.fillna("")
        mp: dict[str, dict] = {}
        for _, row in df.iterrows():
            ts = str(row.get("ts_code", "")).strip()
            if not ts:
                continue
            mp[ts] = {
                "total_mv": pd.to_numeric(row.get("total_mv"), errors="coerce"),
                "circ_mv": pd.to_numeric(row.get("circ_mv"), errors="coerce"),
            }
        return mp
    except Exception:
        return {}


def _format_market_caps(ts_code: str):
    """返回(总市值, 流通市值)的格式化文本，单位亿元；不可用时返回 (None, None)。"""
    mp = _get_stock_basic_map()
    if not mp:
        return (None, None)
    ts = str(ts_code).strip()
    info = mp.get(ts)
    if info is None and "." in ts:
        info = mp.get(ts.split(".")[0])
    if not info:
        return (None, None)

    def fmt(v):
        if pd.isna(v):
            return None
        try:
            return f"{float(v) / 1e4:.2f}亿"
        except Exception:
            return None

    return fmt(info.get("total_mv")), fmt(info.get("circ_mv"))

# 总市值分桶（亿元）
_CAP_BUCKETS = [
    ("≤50亿", 0.0, 50.0),
    ("50-100亿", 50.0, 100.0),
    ("100-200亿", 100.0, 200.0),
    ("200-500亿", 200.0, 500.0),
    ("500-1000亿", 500.0, 1000.0),
    (">1000亿", 1000.0, None),
]
_CAP_BUCKET_MAP = {label: (lo, hi) for label, lo, hi in _CAP_BUCKETS}


def _total_mv_billion(ts_code: str, mp: Optional[dict[str, dict]] = None) -> float:
    """查总市值，单位亿元；查不到返回 NaN。"""
    mp = mp or _get_stock_basic_map()
    if not mp:
        return float("nan")
    ts = str(ts_code).strip()
    info = mp.get(ts) or (mp.get(ts.split(".")[0]) if "." in ts else None)
    if not info:
        return float("nan")
    val = info.get("total_mv")
    try:
        v = float(val)
    except Exception:
        return float("nan")
    return v / 1e4  # tushare 单位万元 → 亿元


def _apply_total_mv_filter(df: pd.DataFrame, buckets: Sequence[str], *, out_col: str = "total_mv_亿") -> pd.DataFrame:
    """按总市值分桶过滤并附加列；buckets 为空则不筛选。"""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    mp = _get_stock_basic_map()
    df2[out_col] = df2["ts_code"].astype(str).apply(lambda x: _total_mv_billion(x, mp))
    ranges = [_CAP_BUCKET_MAP[b] for b in buckets if b in _CAP_BUCKET_MAP]
    if not ranges:
        return df2
    mv = pd.to_numeric(df2[out_col], errors="coerce")
    mask = pd.Series(False, index=df2.index)
    for lo, hi in ranges:
        part = mv >= lo
        if hi is not None:
            part &= mv < hi
        mask |= part
    # 没有命中桶的（NaN）会被过滤掉
    return df2[mask]


def _get_kline_price_cache() -> dict[str, pd.DataFrame]:
    """Session级别的K线价格缓存，避免导航时重复IO。"""
    if "kline_price_cache" not in st.session_state:
        st.session_state["kline_price_cache"] = {}
    return st.session_state["kline_price_cache"]


def _prefetch_kline_prices(codes: Sequence[str], *, adj_type: str = "qfq") -> None:
    """按代码列表预加载价格数据到缓存。"""
    cache = _get_kline_price_cache()
    if not codes:
        return
    for code in codes:
        ts = normalize_ts(code)
        if not ts or ts in cache:
            continue
        try:
            df_price = query_stock_data(
                db_path=os.path.join(DATA_ROOT, UNIFIED_DB_PATH),
                ts_code=ts,
                start_date=None,
                end_date=None,
                adj_type=adj_type,
                limit=None
            )
            if df_price is not None and not df_price.empty:
                cache[ts] = df_price
        except Exception as e:
            logger.debug(f"预加载 {ts} 失败：{e}")


def _parse_code_list(raw_text: str) -> list[str]:
    """解析多行/分隔符的代码或名称输入，返回去重后的 ts_code 列表。"""
    parts = re.split(r"[\s,;，、]+", raw_text or "")
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        code = _resolve_user_code_input(p)
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def _build_external_chart_links(ts_code: str) -> list[tuple[str, str]]:
    """生成外部行情/图表链接，便于快速跳转查看。"""
    if not ts_code or "." not in ts_code:
        return []
    base, _, suffix = ts_code.partition(".")
    base = base.strip()
    suffix = suffix.lower()
    prefix_map = {
        "sz": "sz",
        "sh": "sh",
        "bj": "bj",
        "szse": "sz",
        "sse": "sh",
        "bse": "bj"
    }
    prefix = prefix_map.get(suffix)
    if not prefix or not base:
        return []
    links: list[tuple[str, str]] = []
    links.append(("东方财富", f"https://quote.eastmoney.com/{prefix}{base}.html"))
    links.append(("新浪财经", f"https://finance.sina.com.cn/realstock/company/{prefix}{base}/nc.shtml"))
    links.append(("腾讯自选股", f"https://gu.qq.com/{prefix}{base}/gp"))
    tv_market = {"sz": "SZSE", "sh": "SSE", "bj": "BSE"}.get(prefix)
    if tv_market:
        links.append(("TradingView", f"https://cn.tradingview.com/chart/?symbol={tv_market}:{base}"))
    return links


def _compute_turnover_days(ts_code: str, ref_date: str, target: float = 60.0) -> tuple[Optional[int], Optional[float], Optional[str], int, Optional[str], Optional[str]]:
    """
    直接从统一行情库读取换手率，返回 (天数, 累计换手, 错误信息, 参与天数, 开始日期, 结束日期)。
    - entry_date：参考日次一交易日；若参考日不在该股数据内，用第一条可用数据。
    """
    try:
        df = query_stock_data(
            db_path=os.path.join(DATA_ROOT, UNIFIED_DB_PATH),
            ts_code=ts_code,
            start_date=None,
            end_date=None,
            columns=["trade_date", "tor"],
            adj_type="qfq",
            limit=None
        )
    except Exception as e:
        return None, None, f"数据库读取失败: {e}", 0, None, None

    if df is None or df.empty:
        return None, None, "数据库无该票行情", 0, None, None

    try:
        df["trade_date"] = df["trade_date"].astype(str)
        # 选取 ref_date 及之前的数据，向后（过去）累加换手直到达到目标
        if ref_date is None or not str(ref_date).strip():
            ref_date = df["trade_date"].max()
        sub = df[df["trade_date"] <= str(ref_date)].copy()
        if sub.empty:
            # 如果 ref_date 早于最早数据，则退回全量
            sub = df.copy()
        sub = sub.sort_values("trade_date", ascending=False)
        tor = pd.to_numeric(sub["tor"], errors="coerce").fillna(0.0)
        if tor.empty:
            return None, None, "无换手率数据", 0, None, None

        csum = tor.cumsum()  # 按从 ref_date 往前的顺序
        cum_total = float(csum.iloc[-1]) if np.isfinite(csum.iloc[-1]) else None
        hit_idx = np.where(csum.values >= target)[0]
        days = int(hit_idx[0]) + 1 if len(hit_idx) > 0 else None
        used_days = int(len(tor))
        # 起止日期：起点为用到的最远那天，终点为 ref_date 或最近一天
        end_d = str(sub["trade_date"].iloc[0]) if used_days else None
        start_d = str(sub["trade_date"].iloc[hit_idx[0]]) if len(hit_idx) > 0 else str(sub["trade_date"].iloc[-1]) if used_days else None
        return days, cum_total, None, used_days, start_d, end_d
    except Exception as e:
        return None, None, f"计算失败: {e}", 0, None, None


def _render_price_kline_chart(
    ts: str,
    ref_real: str,
    *,
    options_codes: Sequence[str] | None = None,
    nav_state_prefix: str = "detail",
    default_window: int = 60,
    expander_label: str = "价格 K 线",
    price_cache: Optional[dict[str, pd.DataFrame]] = None,
    load_full_history: bool = True,
    window_days: int = 320,
):
    """复用个股详情的 K 线绘制逻辑，可复用到其他页签（非折叠展示）。"""
    if not ts:
        st.info("请选择股票后再绘制K线。")
        return

    with st.container(border=True):
        st.markdown(f"**{expander_label}**")
        try:
            trade_dates = get_trade_dates()
            ref_for_plot = ref_real or (trade_dates[-1] if trade_dates else None)

            df_price = None
            if price_cache is not None and ts in price_cache:
                df_price = price_cache[ts]
            else:
                # 仅在需要时加载全量数据，个股详情可按窗口截取减少 IO
                start_date = None
                if not load_full_history and ref_real:
                    try:
                        start_date = _prev_trade_date(ref_real, int(window_days))
                    except Exception:
                        start_date = None
                df_price = query_stock_data(
                    db_path=os.path.join(DATA_ROOT, UNIFIED_DB_PATH),
                    ts_code=ts,
                    start_date=start_date,
                    end_date=None,
                    adj_type="qfq",
                    limit=None
                )
                if price_cache is not None and df_price is not None and not df_price.empty:
                    price_cache[ts] = df_price
            if df_price is None or df_price.empty:
                st.warning("暂无价格数据")
            elif not {"open","high","low","close","trade_date"}.issubset(df_price.columns):
                st.warning("价格数据缺少必要列（open/high/low/close/trade_date）")
            else:
                df_price = df_price.copy()
                df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
                df_price = df_price.sort_values("trade_date")  # 加载全部数据
                visible_range = None
                if len(df_price) > default_window:
                    start_vis = df_price["trade_date"].iloc[-default_window]
                    end_vis = df_price["trade_date"].iloc[-1]
                    # 为最新K线预留额外右侧空间，避免贴边
                    pad_days = max(3, int(default_window * 0.1))
                    end_vis_padded = end_vis + pd.Timedelta(days=pad_days)
                    visible_range = [start_vis, end_vis_padded]

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2],
                    vertical_spacing=0.05,
                    specs=[[{"secondary_y": False}], [{}], [{}]]
                )
                # 主图 K 线
                fig.add_trace(
                    go.Candlestick(
                        x=df_price["trade_date"],
                        open=df_price["open"],
                        high=df_price["high"],
                        low=df_price["low"],
                        close=df_price["close"],
                        increasing_line_color="#e74c3c",
                        decreasing_line_color="#2ecc71",
                        name="K线"
                    ),
                    row=1, col=1, secondary_y=False
                )

                # 多空信号（列缺失则不显示）
                try:
                    signal_cols = [c for c in ("duokong_short", "duokong_long") if c in df_price.columns]
                    if signal_cols:
                        if "duokong_short" in df_price.columns:
                            ser_short = pd.to_numeric(df_price["duokong_short"], errors="coerce")
                            if ser_short.notna().any():
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_price["trade_date"],
                                        y=ser_short,
                                        mode="lines",
                                        line=dict(color="#000000", width=1),
                                        name="duokong_short"
                                    ),
                                    row=1, col=1, secondary_y=False
                                )
                        if "duokong_long" in df_price.columns:
                            ser_long = pd.to_numeric(df_price["duokong_long"], errors="coerce")
                            if ser_long.notna().any():
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_price["trade_date"],
                                        y=ser_long,
                                        mode="lines",
                                        line=dict(color="#e74c3c", width=1),
                                        name="duokong_long"
                                    ),
                                    row=1, col=1, secondary_y=False
                                )
                except Exception:
                    pass

                # 副图：J 值、RSI6、RSI12 共用同一纵轴
                try:
                    indicator_cols = {
                        "J": "j",
                        "RSI6": "rsi6",
                        "RSI12": "rsi12",
                    }
                    added_indicator = False
                    indicator_latest = []
                    indicator_all_vals = []
                    for name, col in indicator_cols.items():
                        if col in df_price.columns:
                            ser = pd.to_numeric(df_price[col], errors="coerce")
                            if ser.notna().any():
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_price["trade_date"],
                                        y=ser,
                                        mode="lines",
                                        line=dict(width=1.2),
                                        name=name,
                                    ),
                                    row=2,
                                    col=1,
                                )
                                added_indicator = True
                                indicator_all_vals.append(ser)
                                indicator_latest.append(f"{name}: {ser.dropna().iloc[-1]:.2f}")
                    if added_indicator:
                        fig.update_yaxes(title_text="J / RSI", row=2, col=1)
                        try:
                            # 在副图顶部展示最新数值，类似常见行情软件
                            y_top = None
                            if indicator_all_vals:
                                y_top = max(s.max() for s in indicator_all_vals if pd.notna(s.max()))
                            if y_top is not None and indicator_latest:
                                x_left = visible_range[0] if visible_range else df_price["trade_date"].min()
                                fig.add_annotation(
                                    text=" | ".join(indicator_latest),
                                    x=x_left,
                                    y=y_top,
                                    xref="x2",
                                    yref="y2",
                                    showarrow=False,
                                    font=dict(size=11, color="#2c3e50"),
                                    bgcolor="rgba(255,255,255,0.7)",
                                    align="left",
                                    xanchor="left",
                                    yanchor="top",
                                )
                        except Exception:
                            pass
                except Exception:
                    pass

                # 成交量柱
                try:
                    if "vol" in df_price.columns and "close" in df_price.columns:
                        vols = df_price["vol"]
                        closes = df_price["close"]
                        prev_close = closes.shift(1)
                        colors = np.where(closes >= prev_close, "#e74c3c", "#2ecc71")
                        fig.add_trace(
                            go.Bar(
                                x=df_price["trade_date"],
                                y=vols,
                                marker_color=colors,
                                name="成交量"
                            ),
                            row=3, col=1
                        )
                except Exception:
                    pass

                # 根据默认可见窗口计算价格轴范围，按中值上下各 10% 预留，避免大片空白
                price_range = None
                try:
                    price_window = (
                        df_price.tail(default_window)
                        if visible_range
                        else df_price
                    )
                    low_v = price_window["low"].min()
                    high_v = price_window["high"].max()
                    if pd.notna(low_v) and pd.notna(high_v) and low_v > 0:
                        mid_v = (low_v + high_v) / 2
                        pad = max((high_v - low_v) * 0.1, mid_v * 0.1)
                        y_min = min(low_v, mid_v - pad)
                        y_max = max(high_v, mid_v + pad)
                        if y_max > y_min:
                            price_range = [y_min, y_max]
                except Exception:
                    price_range = None

                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=620,
                    xaxis_rangeslider_visible=False,
                    xaxis_title="日期",
                    yaxis_title="价格",
                )
                if price_range:
                    fig.update_yaxes(range=price_range, row=1, col=1, secondary_y=False)
                if visible_range:
                    fig.update_xaxes(range=visible_range, row=1, col=1)
                    fig.update_xaxes(range=visible_range, row=2, col=1)
                    fig.update_xaxes(range=visible_range, row=3, col=1)
                # 纵轴刻度：按 10% 迭代（乘以 1.1）
                try:
                    y_min, y_max = df_price["low"].min(), df_price["high"].max()
                    if pd.notna(y_min) and pd.notna(y_max) and y_min > 0 and y_max > 0:
                        tickvals = []
                        val = y_min / 1.1
                        while val <= y_max * 1.05 and len(tickvals) < 120:
                            tickvals.append(round(val, 2))
                            val *= 1.1
                        if len(tickvals) >= 2:
                            fig.update_yaxes(tickmode="array", tickvals=tickvals, row=1, col=1, secondary_y=False)
                except Exception:
                    pass

                # 跳过周末/节假日
                try:
                    rangebreaks = [
                        dict(bounds=["sat", "mon"]),
                        dict(bounds=["sun", "mon"])
                    ]
                    trade_days = set(df_price["trade_date"].dt.normalize())
                    full_days = pd.date_range(df_price["trade_date"].min(), df_price["trade_date"].max(), freq="D")
                    holidays = [d for d in full_days if d not in trade_days and d.weekday() < 5]
                    if holidays and len(holidays) <= 500:
                        rangebreaks.append(dict(values=holidays))
                    fig.update_xaxes(rangebreaks=rangebreaks, row=1, col=1)
                    fig.update_xaxes(rangebreaks=rangebreaks, row=2, col=1)
                    fig.update_xaxes(rangebreaks=rangebreaks, row=3, col=1)
                except Exception:
                    pass

                # 参考日标记：在成交量子图下方绘制一个原点
                try:
                    ref_dt = None
                    # 优先使用参考日；无参考日则用数据中最新日期
                    if ref_for_plot:
                        ref_dt = pd.to_datetime(str(ref_for_plot))
                    if ref_dt is None or pd.isna(ref_dt):
                        ref_dt = df_price["trade_date"].max()

                    if pd.notna(ref_dt):
                        min_dt, max_dt = df_price["trade_date"].min(), df_price["trade_date"].max()
                        if ref_dt < min_dt:
                            ref_dt = min_dt
                        elif ref_dt > max_dt:
                            ref_dt = max_dt
                        else:
                            idx = (df_price["trade_date"] - ref_dt).abs().idxmin()
                            ref_dt = df_price.loc[idx, "trade_date"]

                        row_ref = df_price.loc[df_price["trade_date"] == ref_dt].tail(1)
                        if not row_ref.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=[ref_dt],
                                    y=[0],
                                    mode="markers",
                                    marker=dict(color="#34495e", size=10, symbol="circle"),
                                    name="参考日标记",
                                    hoverinfo="skip",
                                    showlegend=False
                                ),
                                row=3, col=1
                            )
                except Exception:
                    pass

                fig.update_yaxes(title_text="价格", row=1, col=1, secondary_y=False)
                fig.update_yaxes(title_text="成交量", row=3, col=1)

                # 唯一 key 避免多次相同参数冲突
                chart_key = f"{nav_state_prefix}_kline_{ts}_{str(ref_real)}"
                st.plotly_chart(fig, width="stretch", key=chart_key)

                # 按排名顺序的“上一条 / 下一条”导航
                if options_codes and ts and ts in options_codes:
                    try:
                        codes_list = list(options_codes)
                        idx = codes_list.index(ts)
                    except Exception:
                        codes_list = list(options_codes)
                        idx = codes_list.index(ts) if ts in codes_list else -1
                    if idx >= 0:
                        ts_name = _stock_name_of(ts) or "—"
                        concept_nav = concept_text_for_stock(ts, blacklist=CONCEPT_BLACKLIST)
                        if len(codes_list) > 1:
                            prev_code = codes_list[idx - 1] if idx > 0 else None
                            next_code = codes_list[idx + 1] if idx < len(codes_list) - 1 else None
                            nav_prev, nav_mid, nav_next = st.columns([0.4, 2, 1])
                            with nav_prev:
                                if st.button("← 上一条", disabled=prev_code is None, key=f"{nav_state_prefix}_prev_{ts}"):
                                    st.session_state[f"{nav_state_prefix}_pending_code"] = prev_code
                                    st.session_state[f"{nav_state_prefix}_last_code"] = prev_code
                                    st.rerun()
                            with nav_mid:
                                parts = [f"{idx+1}/{len(codes_list)}", ts, ts_name]
                                mv_all, mv_circ = _format_market_caps(ts)
                                cap_parts = []
                                if mv_all:
                                    cap_parts.append(f"总市值 {mv_all}")
                                if mv_circ:
                                    cap_parts.append(f"流通市值 {mv_circ}")
                                if cap_parts:
                                    st.caption(" · ".join(parts + [" / ".join(cap_parts)]))
                                else:
                                    st.caption(" · ".join(parts))
                                if concept_nav:
                                    st.caption(f"概念：{concept_nav}")
                            with nav_next:
                                if st.button("下一条 →", disabled=next_code is None, key=f"{nav_state_prefix}_next_{ts}"):
                                    st.session_state[f"{nav_state_prefix}_pending_code"] = next_code
                                    st.session_state[f"{nav_state_prefix}_last_code"] = next_code
                                    st.rerun()
        except Exception as e:
            st.warning(f"无法绘制K线：{e}")

@cache_data(show_spinner=False, hash_funcs={Path: _safe_path_hash}, ttl=60)
def _read_df(path: Path, usecols=None, dtype=None, encoding: str = "utf-8-sig") -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, encoding=encoding, engine="c")
    except Exception:
        return pd.DataFrame()


@cache_data(show_spinner=False, ttl=600)
def _cached_trade_dates(base: str, adj: str):
    # 使用 database_manager 获取交易日列表
    days = get_trade_dates() or []
    if days:
        return days
    # 回退：从已有排名文件名中提取日期
    try:
        dates = []
        for p in ALL_DIR.glob("score_all_*.csv"):
            if p.is_file() and p.stat().st_size > 0:
                m = re.search(r"(\\d{8})", p.name)
                if m:
                    dates.append(m.group(1))
        return sorted(set(dates))
    except Exception as e:
        logger.debug(f"获取交易日列表失败（文件兜底）: {e}")
    return []

# ==== 进度转发到主线程：仅子线程/子进程入队，主线程消费并渲染 ====
@contextmanager
def se_progress_to_streamlit():
    if not _in_streamlit():
        # bare/子线程下：挂空回调，啥也不画，避免任何 st.* 调用
        try:
            yield None, None, None
        finally:
            pass
        return
    status = st.status("准备中…", expanded=True)
    bar = st.progress(0, text="就绪")
    info = st.empty()

    import queue as _q
    _evq = _q.Queue()
    
    # 后台线程只入队，不直接碰 st.*
    def _enqueue_handler(phase, current=None, total=None, message=None, **kw):
        try:
            _evq.put_nowait((phase, current, total, message))
        except Exception:
            pass

    def _render_event(phase, current=None, total=None, message=None):
        txt = {
            "select_ref_date": "选择参考日", "compute_read_window": "计算读取区间",
            "build_universe_done": "构建评分清单", "score_start": "并行评分启动",
            "score_progress": "评分进行中", "screen_start": "筛选启动",
            "screen_progress": "筛选进行中", "screen_done": "筛选完成",
            "write_cache_lists": "写入黑白名单", "write_top_all_start": "写出全量排名",
            "write_top_all_done": "全量排名完成", "hooks_start": "统计/回看",
            "hooks_done": "统计完成",
        }.get(phase, phase)
        if total and current is not None:
            pct = int(current * 100 / max(total, 1))
            # 显示进度详情：评分和筛选都显示数量
            if phase in ("score_progress", "screen_progress"):
                bar.progress(pct, text=f"{txt} · {current}/{total}")
            else:
                bar.progress(pct, text=txt)
        else:
            # 使用message作为主要显示内容，如果没有则使用txt
            display_text = message if message else txt
            info.write(display_text)

    # 主线程消费：供 run_se_run_for_date_in_bg 循环调用
    def _drain():
        try:
            if callable(_orig_drain):
                # 先从 scoring_core 的队列中取事件并交给本地队列
                _orig_drain()
            while True:
                ev = _evq.get_nowait()
                _render_event(*ev)
        except _q.Empty:
            pass

    # 设置进度处理器，使评分系统能够发送进度事件
    _orig_drain = getattr(se, "drain_progress_events", None)
    se.set_progress_handler(_enqueue_handler)
    se.drain_progress_events = _drain  # 将"抽干"替换成主线程渲染
    
    try:
        yield status, bar, info
    finally:
        # 还原 drain（保持模块整洁）
        if callable(_orig_drain):
            se.drain_progress_events = _orig_drain
        else:
            se.drain_progress_events = lambda: None


def run_se_run_for_date_in_bg(arg):
    """在后台线程运行 se.run_for_date(arg)，并在主线程渲染进度"""
    with se_progress_to_streamlit() as (status, bar, info):
        done = threading.Event()
        result = {"path": None, "err": None}

        def _worker():
            try:
                try:
                    # prefer local UI cleanup if present
                    if 'ui_cleanup_database_connections' in globals():
                        ui_cleanup_database_connections()
                    else:
                        # 使用轻量级清理函数，避免关闭工作线程
                        from database_manager import clear_connections_only
                        clear_connections_only()
                except Exception:
                    pass
                
                # 在子线程中运行评分，但确保数据库连接正确初始化
                from database_manager import get_database_manager
                logger.info("[数据库连接] 开始获取数据库管理器实例 (评分线程)")
                manager = get_database_manager()
                
                # 确保数据库管理器已正确初始化，避免连接问题
                try:
                    # 测试数据库连接是否正常
                    test_date = manager.get_latest_trade_date()
                    if test_date:
                        logger.info(f"[评分] 数据库连接正常，最新交易日: {test_date}")
                    else:
                        logger.warning("[评分] 数据库连接正常但无最新交易日数据")
                except Exception as e:
                    logger.warning(f"[评分] 数据库连接测试失败: {e}")
                
                result["path"] = se.run_for_date(arg)
                try:
                    _cleanup_rank_file_dates(result["path"])
                except Exception as e:
                    logger.debug(f"评分文件日期列清理失败: {e}")
            except Exception as e:
                result["err"] = e
            finally:
                done.set()
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # 主线程循环抽取进度事件并刷新 UI
        while not done.is_set():
            se.drain_progress_events()
            time.sleep(0.05)
        # 抽干剩余事件
        se.drain_progress_events()
        if status is not None:
            status.update(label="已完成", state="complete")

        if result["err"]:
            raise result["err"]
        return result["path"]


def run_se_screen_in_bg(*, when_expr, ref_date, timeframe, window, scope, universe, write_white, write_black_rest, return_df=True):
    """在后台线程运行 se.tdx_screen(...)，并在主线程渲染进度（用于“普通选股”）"""
    with se_progress_to_streamlit() as (status, bar, info):
        import threading, time
        done = threading.Event()
        result = {"df": None, "err": None}

        def _worker():
            try:
                try:
                    # prefer local UI cleanup if present
                    if 'ui_cleanup_database_connections' in globals():
                        ui_cleanup_database_connections()
                    else:
                        # 使用轻量级清理函数，避免关闭工作线程
                        from database_manager import clear_connections_only
                        clear_connections_only()
                except Exception:
                    pass
                
                st.session_state["expression_screening_active"] = True
                
                try:
                    result["df"] = se.tdx_screen(
                        when_expr,
                        ref_date=ref_date,
                        timeframe=timeframe,
                        window=_safe_int(window, 30),
                        scope=scope,
                        universe=universe,
                        write_white=write_white,
                        write_black_rest=write_black_rest,
                        return_df=return_df
                    )
                finally:
                    st.session_state["expression_screening_active"] = False
                    
            except Exception as e:
                result["err"] = e
                st.session_state["expression_screening_active"] = False
            finally:
                done.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # 主线程循环抽取进度事件并刷新 UI
        while not done.is_set():
            se.drain_progress_events()
            time.sleep(0.05)
        # 抽干剩余事件
        se.drain_progress_events()
        if status is not None:
            status.update(label="已完成", state="complete")

        if result["err"]:
            raise result["err"]
        return result["df"]


def _get_latest_date_from_files() -> Optional[str]:
    """从全量排名文件名中提取最新日期"""
    files = sorted(ALL_DIR.glob("score_all_*.csv"))
    dates = []
    for p in files:
        m = re.search(r"(\d{8})", p.name)
        if m: dates.append(m.group(1))
    return max(dates) if dates else None


def _pick_smart_ref_date() -> Optional[str]:
    """智能获取参考日期，按优先级尝试多种方式"""
    # 1. 优先从数据库获取
    latest = _get_latest_date_from_database()
    if latest:
        return latest
    
    # 2. 从daily分区获取
    latest = _get_latest_date_from_daily_partition()
    if latest:
        return latest
    
    # 3. 最后从评分结果文件获取
    latest = _get_latest_date_from_files()
    if latest:
        logger.warning(f"回退到评分结果文件中的最新日期: {latest}")
    else:
        logger.error("无法获取任何参考日期")
    
    return latest


def _load_detail_json(ref: str, ts: str) -> Optional[Dict]:
    """
    加载个股详情，优先从数据库读取，如果数据库不可用则从JSON文件读取
    注意：只有当 details_db_reading_enabled 为 True 时才会读取数据库，避免与写入操作冲突
    """
    # 检查是否允许读取数据库（使用统一的函数）
    db_reading_enabled = is_details_db_reading_enabled()
    
    # 1. 优先从数据库读取（只有当db_reading_enabled为True且数据库可用时才读取）
    if db_reading_enabled and is_details_db_available():
        try:
            # 使用统一的函数获取details数据库路径（包含回退逻辑）
            details_db_path = get_details_db_path_with_fallback()
            if not details_db_path:
                # 数据库文件不存在，直接返回None
                logger.debug(f"数据库文件不存在: {ts}_{ref}")
                return None
            
            # 查询股票详情表
            logger.info(f"[数据库连接] 开始获取数据库管理器实例 (查询股票详情: {ts}, {ref})")
            manager = get_database_manager()
            if not manager:
                logger.debug(f"无法获取数据库管理器: {ts}_{ref}")
                return None
            
            sql = "SELECT * FROM stock_details WHERE ts_code = ? AND ref_date = ?"
            df = manager.execute_sync_query(details_db_path, sql, [ts, ref], timeout=30.0)
            
            if not df.empty:
                row = df.iloc[0]
                
                # 解析 rules 字段：优先 json.loads，失败则 ast.literal_eval，最后保证是 list[dict]
                rules_raw = row.get('rules')
                rules = []
                if rules_raw:
                    if isinstance(rules_raw, str):
                        try:
                            rules = json.loads(rules_raw)
                        except Exception:
                            try:
                                import ast
                                rules = ast.literal_eval(rules_raw)
                            except Exception:
                                rules = []
                    elif isinstance(rules_raw, list):
                        rules = rules_raw
                
                # 确保 rules 是 list[dict] 格式
                if not isinstance(rules, list):
                    rules = []
                
                derived_highlights, derived_drawbacks, derived_opportunities = _derive_rule_buckets(rules)
                legacy_highlights = _parse_string_list(row.get('highlights')) if 'highlights' in row else []
                legacy_drawbacks = _parse_string_list(row.get('drawbacks')) if 'drawbacks' in row else []
                legacy_opportunities = _parse_string_list(row.get('opportunities')) if 'opportunities' in row else []
                
                # 获取 rank 和 total 值
                rank_val = row.get('rank')
                total_val = row.get('total')
                
                # 组装 summary，包含 rank 和 total
                summary = {
                    'score': row.get('score'),
                    'tiebreak': row.get('tiebreak'),
                    'highlights': derived_highlights or legacy_highlights,
                    'drawbacks': derived_drawbacks or legacy_drawbacks,
                    'opportunities': derived_opportunities or legacy_opportunities,
                    'rank': int(rank_val) if pd.notna(rank_val) else None,
                    'total': int(total_val) if pd.notna(total_val) else None,
                }
                
                # 组装成与 JSON 文件完全一致的结构，保持兼容性
                result = {
                    'ts_code': row.get('ts_code'),
                    'ref_date': row.get('ref_date'),
                    'summary': summary,
                    'rules': rules,
                    'rank': summary['rank'],   # 兼容旧调用
                    'total': summary['total'],
                }
                return result
            else:
                # 数据库查询成功但无数据，直接返回None
                logger.debug(f"数据库查询为空: {ts}_{ref}")
                return None
        except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
            # 数据库文件不存在或管理器获取失败，直接返回None
            logger.debug(f"数据库访问失败 {ts}_{ref}: {e}")
            return None
        except Exception as e:
            # 数据库读取失败（可能是表不存在、连接错误等），直接返回None
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                logger.debug(f"Details数据库表不存在: {ts}_{ref}")
            else:
                logger.debug(f"数据库读取失败 {ts}_{ref}: {e}")
            return None
    
    # 2. 如果未启用数据库读取，或者配置了JSON存储，则使用JSON文件
    if (not db_reading_enabled) or SC_DETAIL_STORAGE in ["json", "both"]:
        try:
            p = _path_detail(ref, ts)
            if not p.exists(): 
                # JSON文件不存在，正常返回None，不报错
                return None
            try:
                data = json.loads(p.read_text(encoding="utf-8-sig"))
                if not isinstance(data, dict):
                    return None
                rules_raw = data.get("rules") or data.get("per_rules") or []
                rules = rules_raw if isinstance(rules_raw, list) else []
                summary = data.get("summary") or {}
                if not summary:
                    summary = {
                        "score": data.get("score"),
                        "tiebreak": data.get("tiebreak"),
                        "rank": data.get("rank"),
                        "total": data.get("total"),
                        "highlights": data.get("highlights", []),
                        "drawbacks": data.get("drawbacks", []),
                        "opportunities": data.get("opportunities", []),
                    }
                derived_highlights, derived_drawbacks, derived_opportunities = _derive_rule_buckets(rules)
                summary["highlights"] = derived_highlights or _parse_string_list(summary.get("highlights"))
                summary["drawbacks"] = derived_drawbacks or _parse_string_list(summary.get("drawbacks"))
                summary["opportunities"] = derived_opportunities or _parse_string_list(summary.get("opportunities"))
                data = {
                    "ts_code": data.get("ts_code", ts),
                    "ref_date": data.get("ref_date", ref),
                    "summary": summary,
                    "rules": rules,
                    "rank": summary.get("rank"),
                    "total": summary.get("total"),
                }
                return data
            except json.JSONDecodeError as e:
                # JSON解析失败，记录日志但不报错
                logger.debug(f"JSON文件解析失败 {ts}_{ref}: {e}")
                return None
            except Exception as e:
                # 其他读取异常，记录日志但不报错
                logger.debug(f"JSON文件读取失败 {ts}_{ref}: {e}")
                return None
        except Exception as e:
            # 路径构建或其他异常，记录日志但不报错
            logger.debug(f"JSON文件路径处理失败 {ts}_{ref}: {e}")
            return None
    
    # 兜底：所有路径都失败，返回None
    return None


def _get_rank_from_files(ref_date: str, ts_code: str) -> tuple[int | None, int | None] | None:
    """从全量排名文件获取指定日期的排名"""
    ts_code = str(ts_code)
    df_all = _read_rank_all_sorted(ref_date)
    if df_all.empty:
        return None
    try:
        row = df_all.loc[df_all["ts_code"].astype(str) == ts_code]
        if not row.empty:
            total_val = len(df_all)
            rv = row["rank"].iloc[0] if "rank" in row.columns else None
            if pd.notna(rv):
                return int(rv), int(total_val)
            return int(row.index[0]) + 1, int(total_val)
    except Exception as e:
        logger.debug(f"读取全量排名文件失败 {ref_date}: {e}")
    return None


def _attach_all_rank(df: pd.DataFrame, ref_date: str, code_col: str = "ts_code") -> pd.DataFrame:
    """
    为给定 DataFrame 按参考日附加总榜排名（rank、rank_total）。
    不改变原排序；若读取失败则原样返回。
    """
    if df is None or df.empty or not ref_date or code_col not in df.columns:
        return df
    try:
        df_all = _read_rank_all_sorted(ref_date)
        if df_all is None or df_all.empty or "ts_code" not in df_all.columns:
            return df
        df_all = df_all.copy()
        df_all["ts_code"] = df_all["ts_code"].astype(str)
        if "rank" not in df_all.columns:
            df_all = df_all.reset_index(drop=True)
            df_all["rank"] = np.arange(len(df_all)) + 1
        total = len(df_all)
        rank_map = df_all.set_index("ts_code")["rank"].to_dict()

        out = df.copy()
        out[code_col] = out[code_col].astype(str)
        out["rank"] = out[code_col].map(rank_map)
        out["rank_total"] = total
        return out
    except Exception as e:
        logger.debug(f"附加总榜排名失败 {ref_date}: {e}")
        return df


def _get_prev_rank_history(ts_code: str, ref_date: str, max_days: int | None = None) -> list[dict]:
    """
    获取该股票在 details 中已记录的所有排名历史（优先 DB，兜底文件）。
    max_days: 限制返回的天数（None 表示全部）。
    """
    if not ts_code:
        return []

    results: list[dict] = []
    db_enabled = is_details_db_reading_enabled() and is_details_db_available()
    details_db_path = get_details_db_path_with_fallback() if db_enabled else None
    manager = get_database_manager() if db_enabled else None

    # 1) 尝试直接从 details 数据库读取全部历史
    if db_enabled and details_db_path and manager:
        try:
            sql = "SELECT ref_date, rank, total FROM stock_details WHERE ts_code = ? ORDER BY ref_date DESC"
            params: list[Any] = [ts_code]
            if max_days is not None and max_days > 0:
                sql += " LIMIT ?"
                params.append(max_days)
            df = manager.execute_sync_query(details_db_path, sql, params, timeout=20.0)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    rv = row.get("rank")
                    tv = row.get("total")
                    if pd.notna(rv):
                        results.append({
                            "date": str(row.get("ref_date")),
                            "rank": int(rv),
                            "total": int(tv) if pd.notna(tv) else None
                        })
                return results
        except Exception as e:
            logger.debug(f"查询历史排名失败 {ts_code}: {e}")

    # 2) 兜底：按交易日从最新往前逐日读取文件排名
    try:
        cal = (get_trade_dates() or [])[::-1]  # 最新在前
        if max_days is not None and max_days > 0:
            cal = cal[:max_days]
    except Exception as e:
        logger.debug(f"生成历史排名日期窗口失败 {ref_date}: {e}")
        return results

    for d in cal:
        rank_pair = _get_rank_from_files(d, ts_code)
        if rank_pair and rank_pair[0] is not None:
            results.append({"date": d, "rank": rank_pair[0], "total": rank_pair[1]})
    return results


def _codes_to_txt(codes: List[str], style: str="space", with_suffix: bool=True) -> str:
    def fmt(c):
        c = normalize_ts(c)
        return c if with_suffix else c.split(".")[0]
    arr = [fmt(c) for c in codes]
    return (" ".join(arr)) if style == "space" else ("\n".join(arr))


def _download_txt(label: str, text: str, filename: str, key: Optional[str]=None):
    st.download_button(label, data=text.encode("utf-8-sig"),
                       file_name=filename, mime="text/plain",
                       width='stretch', key=key)


def copy_txt_button(text: str, label: str = "一键复制（TXT）", key: str = "copy0"):
    if label:
        st.caption(label)
    st.code(text or "", language="text")


def _tail(path: Path, n: int=400) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""


def _apply_runtime_overrides(rules_obj: dict,
                             tie_break: str, max_workers: int,
                             attn_on: bool, universe: str|List[str]):
    # 规则覆盖配置
    if rules_obj:
        pres = rules_obj.get("prescreen")
        rules = rules_obj.get("rules")
        if pres is not None: setattr(se, "SC_PRESCREEN_RULES", pres)
        if rules is not None: setattr(se, "SC_RULES", rules)
    setattr(se, "SC_TIE_BREAK", str(tie_break))
    setattr(se, "SC_MAX_WORKERS", int(max_workers))
    # setattr(se, "SC_ATTENTION_ENABLE", bool(attn_on))
    setattr(se, "SC_UNIVERSE", universe)


def show_database_diagnosis():
    """显示数据库诊断信息"""
    try:
        # 诊断功能需要重新实现
        # 使用 database_manager 获取诊断信息
        logger.info("[数据库连接] 开始获取数据库管理器实例 (数据库诊断)")
        manager = get_database_manager()
        stats = manager.get_stats()
        diagnosis = {
            "database_status": "connected" if stats else "disconnected",
            "queue_size": stats.get("queue_size", 0),
            "worker_count": stats.get("worker_count", 0)
        }
        
        st.subheader("数据库诊断信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("数据库文件存在", "是" if diagnosis.get("database_exists") else "否")
            st.metric("数据库文件被锁定", "是" if diagnosis.get("database_locked") else "否")
            if diagnosis.get("file_size"):
                st.metric("文件大小", f"{diagnosis['file_size'] / (1024*1024):.1f} MB")
        
        with col2:
            if diagnosis.get("file_permissions"):
                st.metric("文件权限", diagnosis["file_permissions"])
            if diagnosis.get("last_modified"):
                import datetime
                last_mod = datetime.datetime.fromtimestamp(diagnosis["last_modified"])
                st.metric("最后修改", last_mod.strftime("%Y-%m-%d %H:%M:%S"))
        
        # 显示进程占用信息
        processes = diagnosis.get("processes_using_db", [])
        if processes:
            st.warning(f"⚠️ 发现 {len(processes)} 个进程正在使用数据库文件:")
            for proc in processes:
                st.write(f"- PID: {proc['pid']}, 进程名: {proc['name']}")
        else:
            st.success("✅ 没有发现其他进程占用数据库文件")
        
        if diagnosis.get("database_locked"):
            st.error("数据库文件被锁定，这可能导致表达式选股失败")
            st.info("建议：检查是否有其他应用在使用数据库文件，或重启相关进程")
        
        if st.button("重新诊断"):
            st.rerun()
            
    except Exception as e:
        st.error(f"诊断数据库失败: {e}")


def show_database_status():
    """显示数据库连接状态"""
    try:
        # get_data_source_status 已从 database_manager 导入
        status = get_data_source_status()
        
        st.subheader("数据库连接状态")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("数据库文件存在", "是" if status.get("database_file_exists") else "否")
            st.metric("数据库文件被锁定", "是" if status.get("database_file_locked") else "否")
            st.metric("使用统一数据库", "是" if status.get("use_unified_db") else "否")
        
        with col2:
            dispatcher_stats = status.get("dispatcher_stats", {})
            st.metric("工作线程数", dispatcher_stats.get("worker_threads", 0))
            st.metric("缓存大小", dispatcher_stats.get("cache_size", 0))
            st.metric("队列大小", dispatcher_stats.get("queue_size", 0))
        
        if status.get("database_file_locked"):
            st.error("⚠️ 数据库文件被锁定，这可能导致表达式选股失败")
            st.info("建议：等待其他操作完成或重启应用程序")
        
        if st.button("刷新状态"):
            st.rerun()
            
    except Exception as e:
        st.error(f"检查数据库状态失败: {e}")


def _indicator_options(tag: str | None = "product"):
    try:
        # 只列出有 py_func 的（能在本地算）的指标名
        names = [k for k, m in getattr(ind, "REGISTRY", {}).items() if getattr(m, "py_func", None)]
        if tag and hasattr(ind, "names_by_tag"):
            tagged = set(ind.names_by_tag(tag))  # 只要打了 product 标签的
            names = [n for n in names if n in tagged] or names
        return sorted(set(names))
    except Exception:
        # 兜底：保持现在的三个
        return ["kdj", "ma", "macd"]


@cache_data(show_spinner=False, ttl=300)
def _get_rule_names() -> list[str]:
    """获取规则名称列表，带缓存"""
    try:
        rule_names = [str(r.get("name") or f"RULE_{i}") for i, r in enumerate(getattr(se, "SC_RULES", []) or [])]
        return sorted(list(dict.fromkeys(rule_names)))
    except Exception:
        return []


@cache_data(show_spinner=False, ttl=300)
def _cached_load_prediction_rules() -> list[dict]:
    """缓存版本的 load_prediction_rules 函数"""
    try:
        return load_prediction_rules()
    except Exception:
        return []


def _apply_tiebreak_sorting(df: pd.DataFrame, tiebreak_mode: str = "none") -> pd.DataFrame:
    """
    对股票表格应用排序
    
    Args:
        df: 包含ts_code和score列的DataFrame
        tiebreak_mode: 排序模式 ("none", "kdj_j_asc")
    
    Returns:
        排序后的DataFrame
    """
    if df.empty or "ts_code" not in df.columns or "score" not in df.columns:
        return df
    
    # 创建副本避免修改原数据
    df_sorted = df.copy()
    
    if tiebreak_mode == "kdj_j_asc" and "tiebreak_j" in df_sorted.columns:
        # 按得分降序，同分时按J值升序，再同分时按代码升序（兜底）
        df_sorted = df_sorted.sort_values(["score", "tiebreak_j", "ts_code"], ascending=[False, True, True]).reset_index(drop=True)
    else:
        # 默认：只按得分降序排序，同分时按代码升序（兜底）
        df_sorted = df_sorted.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    
    return df_sorted

@cache_data(show_spinner=False, ttl=120)
def _resolve_pred_universe(label: str, ref: str) -> list[str]:
    """
    将 UI 的范围标签展开为 ts_code 列表：
    - all：读 output/score/all/score_all_<ref>.csv
    - white/black：读 scoring_core 的缓存名单
    - attention：读“特别关注榜”（若找不到则尝试按文件名匹配）
    """
    label = (label or "").strip().lower()
    codes: list[str] = []

    if label == "all":
        df = _read_rank_all_sorted(ref)
        if df is not None and not df.empty and "ts_code" in df.columns:
            codes = df["ts_code"].astype(str).tolist()

    elif label in {"white", "black"}:
        try:
            kind = "whitelist" if label == "white" else "blacklist"
            codes = se._read_cache_list_codes(ref, kind) or []
        except Exception:
            codes = []

    elif label == "attention":
        try:
            codes = se._load_category_codes("strength", ref) or []
        except Exception:
            # 退回按文件名找 "attention*<ref>.csv"
            p = _find_attn_file_by_date(ref)  # 这个工具已在文件内定义
            if p:
                df = _read_df(p, dtype={"ts_code": str})
                if df is not None and not df.empty:
                    # 智能识别列名
                    for cand in ["ts_code", "code", "ts", "symbol"]:
                        if cand in df.columns:
                            codes = df[cand].astype(str).tolist()
                            break

    # 规范化、去重、排序
    try:
        codes = [normalize_ts(c) for c in codes if c]
    except Exception:
        codes = [str(c).strip() for c in codes if c]
    return sorted(set(codes))

# ==== 强度榜文件定位====
def _pick_latest_attn_date() -> Optional[str]:
    """
    扫描 attention 目录所有 CSV。
    规则：把“文件名里最后一次出现的 8 位数字”视为该文件的结束日；
         若同一结束日有多份，则按文件修改时间(mtime)取最新那份的结束日。
    """
    best_key = None
    best_ref = None
    for p in ATTN_DIR.glob("*.csv"):
        ms = re.findall(r"(\d{8})", p.name)
        if not ms:
            continue
        end = ms[-1]  # 视为结束日（最后 8 位）
        key = (end, p.stat().st_mtime)  # 先比日期，再比修改时间
        if best_key is None or key > best_key:
            best_key, best_ref = key, end
    return best_ref


def _find_attn_file_by_date(ref: str) -> Optional[Path]:
    """
    根据结束日 ref 定位文件：
    - 先收集所有包含该 ref 的 attention CSV；
    - 优先“规范化命名”的文件（以 attention_ 开头且包含 _win/_topM/_topN 这些关键字）；
    - 其余则按修改时间倒序作为次序。
    """
    cands = list(ATTN_DIR.glob(f"*attention*{ref}*.csv"))
    if not cands:
        return None

    def _score(p: Path):
        nm = p.name
        normalized = nm.startswith("attention_") and ("_win" in nm or nm == f"attention_{ref}.csv")
        return (1 if normalized else 0, p.stat().st_mtime)

    return sorted(cands, key=_score, reverse=True)[0]

# ---- 读取 config 的稳健工具 ----
def cfg_int(name: str, default: int) -> int:
    val = getattr(cfg, name, default)
    try:
        # 过滤 None / "" 等异常取值
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return int(default)
        return int(val)
    except Exception:
        return int(default)


def cfg_str(name: str, default: str) -> str:
    val = getattr(cfg, name, default)
    if val is None:
        return str(default)
    s = str(val).strip()
    return s if s else str(default)


def cfg_bool(name: str, default: bool) -> bool:
    val = getattr(cfg, name, default)
    if isinstance(val, bool):
        return val
    if val is None:
        return bool(default)
    s = str(val).strip().lower()
    if s in {"1","true","yes","y","on","t"}:
        return True
    if s in {"0","false","no","n","off","f"}:
        return False
    return bool(default)

# 通用多阶段进度器（统一管理单次任务的进度条+状态日志）
class Stepper:
    """
    用法：
        steps = ["准备环境", "下载源数据", "合并增量", "写出导出", "自动排名"]
        sp = Stepper("下载/同步", steps, key_prefix="dl_sync")  # 每次点击都会生成独立 run_id
        sp.start()

        sp.step("准备环境")     # 做准备...
        sp.tick(0.3, "校验目标目录")
        sp.tick(1.0)           # 本步骤收尾

        sp.step("下载源数据")   # 具体下载...
        sp.step("合并增量")
        sp.step("写出导出")
        sp.step("自动排名", visible=auto_rank)  # 支持按条件显示/跳过
        sp.finish(success=True, note="全部完成")
    """
    def __init__(self, title, steps, key_prefix="stepper"):
        self.title = title
        self.steps_all = steps[:]  # 文案列表（可含 None）
        self.steps = [s for s in steps if s]  # 实际参与统计的步骤
        self.total = len(self.steps)
        self.key = f"{key_prefix}_{uuid.uuid4().hex[:8]}"
        self._init_state()

    def _init_state(self):
        st.session_state[self.key] = {
            "idx": 0,        # 已完成到第几个（从 0 开始）
            "run_id": self.key,
        }

    def start(self):
        self.status = st.status(
            label=f"{self.title}：开始（0/{self.total}）",
            state="running",
        )
        self.prog = st.progress(0, text="准备中…")
        with self.status:
            st.write("🟡 开始任务…")

    def _update_prog(self, idx, label):
        pct = 0 if self.total == 0 else int(idx / self.total * 100)
        self.prog.progress(pct, text=f"{idx}/{self.total}：{label}")

    def step(self, label, visible=True, info=None):
        """进入下一主步骤；visible=False 时，仅记录日志，不纳入进度比例"""
        if not visible:
            # 仅追加日志提示
            with self.status:
                st.write(f"⏭️ 跳过：{label}")
            return

        state = st.session_state[self.key]
        state["idx"] += 1
        idx = min(state["idx"], self.total)
        text = label if not info else f"{label}｜{info}"

        with self.status:
            st.write(f"▶️ {text}")
        self._update_prog(idx, text)

    def tick(self, delta_ratio, info=None):
        """在当前步骤中显示细粒度推进（例如循环/分批处理）"""
        state = st.session_state[self.key]
        # 按当前主步骤位置 + 细分比例 合成一个更平滑的百分比展示
        base = min(state["idx"], self.total - 1)
        now = min(1.0, max(0.0, float(delta_ratio)))
        overall = int(((base + now) / self.total) * 100) if self.total else 0
        self.prog.progress(overall, text=info or "处理中…")
        # 在日志里也可打点
        if info:
            with self.status:
                st.write(f"… {info}")

    def finish(self, success=True, note=None):
        if success:
            self.status.update(
                label=f"{self.title}：完成（{self.total}/{self.total}）",
                state="complete",
            )
            self.prog.progress(100, text=note or "完成")
        else:
            self.status.update(
                label=f"{self.title}：失败",
                state="error",
            )


@contextmanager
def pred_progress_to_streamlit():
    if not _in_streamlit():
        # 非Streamlit环境回调
        try:
            yield None, None, None
        finally:
            pass
        return

    status = st.status("准备中…", expanded=True)
    bar = st.progress(0, text="就绪")
    info = st.empty()

    import queue as _q
    _evq = _q.Queue()

    # 后台线程只入队，不直接碰 st.*
    def _enqueue_handler(phase, current=None, total=None, message=None, **kw):
        try:
            _evq.put_nowait((phase, current, total, message))
        except Exception:
            pass

    def _render_event(phase, current=None, total=None, message=None):
        txt = {
            "pred_select_ref_date": "选择参考日",
            "pred_build_universe_done": "构建模拟清单",
            "pred_start": "模拟开始",
            "pred_progress": "模拟进行中",
            "pred_rank_start": "准备模拟数据",
            "pred_rank_sim_done": "模拟完成，开始评分",
            "pred_rank_score_progress": "评分进行中",
            "pred_rank_done": "预测排名完成",
            "pred_done": "模拟完成",
        }.get(phase, phase)
        
        # 使用message作为主要显示内容，如果没有则使用txt
        display_text = message if message else txt
        
        if total and current is not None:
            pct = int(current * 100 / max(total, 1))
            bar.progress(pct, text=f"{display_text} ({current}/{total})")
        else:
            bar.progress(0, text=display_text)

    # 主线程消费：供 run_prediction_in_bg 循环调用
    def _drain():
        try:
            if callable(_orig_drain):
                _orig_drain()
            while True:
                ev = _evq.get_nowait()
                _render_event(*ev)
        except _q.Empty:
            pass

    _orig_handler = getattr(pr, "_progress_handler", None)
    _orig_drain = getattr(pr, "drain_progress_events", None)
    pr.set_progress_handler(_enqueue_handler)
    pr.drain_progress_events = _drain  # 关键：把"抽干"替换成主线程渲染

    try:
        yield status, bar, info
    finally:
        # 还原 handler
        try:
            pr.set_progress_handler(_orig_handler)
        except Exception:
            pass
        # 还原 drain（保持模块整洁）
        if callable(_orig_drain):
            pr.drain_progress_events = _orig_drain
        else:
            pr.drain_progress_events = lambda: None


def run_prediction_in_bg(inp):
    with pred_progress_to_streamlit() as (status, bar, info):
        done = threading.Event()
        result = {"df": None, "err": None}
        def _worker():
            try:
                # 使用安全的数据库操作
                from predict_core import run_prediction
                result["df"] = run_prediction(inp)
            except Exception as e:
                result["err"] = e
            finally:
                done.set()
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while not done.is_set():
            # 消费 predict_core 的进度事件（如果内部使用事件队列的话）
            try:
                pr.drain_progress_events()
            except Exception:
                pass
            time.sleep(0.05)
        # 抽干剩余事件
        try:
            pr.drain_progress_events()
        except Exception:
            pass
        if status is not None:
            status.update(label="已完成", state="complete")
        if result["err"]:
            raise result["err"]
        return result["df"]


def run_predict_ranking_in_bg(inp):
    """在后台线程运行 run_predict_ranking，并把进度转发到主线程"""
    with pred_progress_to_streamlit() as (status, bar, info):
        done = threading.Event()
        result = {"df": None, "path": None, "err": None}

        def _worker():
            try:
                result["df"], result["path"] = run_predict_ranking(inp)
            except Exception as e:
                result["err"] = e
            finally:
                done.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while not done.is_set():
            try:
                pr.drain_progress_events()
            except Exception:
                pass
            time.sleep(0.05)

        try:
            pr.drain_progress_events()
        except Exception:
            pass

        if status is not None:
            status.update(label="已完成", state="complete")
        if result["err"]:
            raise result["err"]
        return result["df"], result["path"]

# ===== 主ui部分 =====
if _in_streamlit():
    # ===== 页眉 =====
    st.title("ScoreApp")
    

    # === Global portfolio state (added by assistant) ===
    if "cur_pid" not in st.session_state:
        st.session_state["cur_pid"] = None
    if "cur_pf" not in st.session_state:
        st.session_state["cur_pf"] = None

    # Local aliases for convenience across tabs
    cur_pid = st.session_state.get("cur_pid")
    cur_pf  = st.session_state.get("cur_pf")
    # === End global portfolio state ===

    if "rules_obj" not in st.session_state:
        st.session_state["rules_obj"] = {
            # "prescreen": getattr(cfg, "SC_PRESCREEN_RULES", []),
            # "rules": getattr(cfg, "SC_RULES", []),
            "prescreen": getattr(se, "SC_PRESCREEN_RULES", []),
            "rules": getattr(se, "SC_RULES", []),
        }
    if "export_pref" not in st.session_state:
        st.session_state["export_pref"] = {"style": "space", "with_suffix": True}

    tab_rank, tab_detail, tab_kline, tab_screen, tab_stats, tab_attn, tab_custom, tab_position, tab_predict, tab_rules, tab_tools, tab_port, tab_data_view, tab_logs, = st.tabs(
        ["排名", "个股详情", "K线看板", "选股", "统计", "衍生榜", "自选榜", "持仓建议", "明日模拟", "规则编辑", "工具箱", "组合模拟/持仓", "数据管理", "日志"])

    # ================== 排名 ==================
    with tab_rank:
        st.subheader("排名")
        with st.expander("参数设置（运行前请确认）", expanded=True):
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            with c1:
                ref_inp = st.text_input("参考日（YYYYMMDD；留空=自动取最新）", value="", key="rank_ref_input")
            with c2:
                tie_default = cfg_str("SC_TIE_BREAK", "none").lower()
                tie = st.selectbox("同分排序（Tie-break）", ["none", "kdj_j_asc"], index=0 if tie_default=="none" else 1)
                maxw = st.number_input("最大并行数", min_value=1, max_value=64, value=cfg_int("SC_MAX_WORKERS", 8))
            with c3:
                universe = st.selectbox("评分范围", ["全市场","仅白名单","仅黑名单"], index=0)
                style = st.selectbox("TXT 导出格式", ["空格分隔", "一行一个"], index=0)
            with c4:
                attn_on = False
                with_suffix = st.checkbox("导出带交易所后缀（.SZ/.SH）", value=False)
            st.session_state["export_pref"] = {"style": "space" if style=="空格分隔" else "lines",
                                            "with_suffix": with_suffix}
            run_btn = st.button("🚀 运行评分（写入 All/Details）", width='stretch')

        # 运行
        ref_to_use = ref_inp.strip() or _pick_smart_ref_date()
        if run_btn:
            # 禁用details数据库读取，避免写入冲突
            st.session_state["details_db_reading_enabled"] = False
            logger.info(f"用户点击运行评分按钮: 参考日={ref_to_use}, 并行数={maxw}, 范围={universe}，已禁用details数据库读取")
            
            # 关闭相关数据库的连接池，确保排名运行前没有遗留连接
            try:
                manager = get_database_manager()
                if manager:
                    details_db_path = get_details_db_path_with_fallback()
                    if details_db_path:
                        manager.close_db_pools(details_db_path)
                        logger.info(f"已关闭details数据库连接池: {details_db_path}")
                    
                    # 统一库（stock_data.db）也关闭，避免与排名写入冲突
                    try:
                        from config import DATA_ROOT, UNIFIED_DB_PATH
                        unified_db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                        manager.close_db_pools(unified_db_path)
                        logger.info(f"已关闭主库连接池: {unified_db_path}")
                    except Exception as e_inner:
                        logger.warning(f"关闭主库连接池失败: {e_inner}")
            except Exception as e:
                logger.warning(f"关闭details数据库连接池时出错: {e}")
            
            _apply_runtime_overrides(st.session_state["rules_obj"], tie, maxw, attn_on,
                                    {"全市场":"all","仅白名单":"white","仅黑名单":"black","仅特别关注榜":"attention"}[universe])
            try:
                rank_path = run_se_run_for_date_in_bg(ref_inp.strip() or None)
                st.success(f"评分完成：{rank_path}")
                # 解析参考日
                m = re.search(r"(\d{8})", str(rank_path))
                if m:
                    ref_to_use = m.group(1)
            except Exception as e:
                st.error(f"评分失败：{e}")
                ref_to_use = None

        st.divider()

        # ---- 排名浏览区块 ----
        with st.container(border=True):
            st.markdown("**排名浏览**")
            
            # 读取结果按钮和参考日输入
            browse_col1, browse_col2 = st.columns([2, 1])
            with browse_col1:
                browse_ref_inp = st.text_input("参考日（YYYYMMDD；留空=自动取最新）", value="", key="browse_ref_input")
            with browse_col2:
                browse_btn = st.button("📅 读取结果", width='stretch', key="browse_read_btn")
            
            # 读取结果的逻辑
            browse_ref_to_use = None
            
            # 如果运行了评分，优先使用运行后的参考日
            if run_btn and ref_to_use:
                browse_ref_to_use = ref_to_use
            # 如果点击了读取结果按钮或输入了参考日，使用输入的参考日
            elif browse_btn or browse_ref_inp.strip():
                # 优先使用用户输入的参考日，如果没有输入则使用最新的文件日期
                browse_ref_to_use = browse_ref_inp.strip() or _get_latest_date_from_files()
            # 如果还没有参考日，尝试使用参数设置中的参考日
            elif ref_inp.strip():
                browse_ref_to_use = ref_inp.strip()
            # 最后尝试使用最新的文件日期
            else:
                browse_ref_to_use = _get_latest_date_from_files()
            
            if browse_ref_to_use:
                # 获取最新排名文件日期和数据库最新日期用于对比
                latest_rank_date = _get_latest_date_from_files()
                db_latest_date = _get_latest_date_from_database()
                
                # 显示三个日期的对比
                col1, col2, col3 = st.columns(3)
                with col1:
                    if latest_rank_date:
                        st.markdown(f"**最新排名文件：{latest_rank_date}**")
                    else:
                        st.markdown("**最新排名文件：未知**")
                with col2:
                    st.markdown(f"**当前显示排名：{browse_ref_to_use}**")
                with col3:
                    if db_latest_date:
                        st.markdown(f"**数据库最新日期：{db_latest_date}**")
                    else:
                        st.markdown("**数据库最新日期：未知**")
                
                # 如果有日期差异，给出提示
                if latest_rank_date and latest_rank_date != browse_ref_to_use:
                    st.info(f"当前显示的是 {browse_ref_to_use} 的排名，最新排名文件是 {latest_rank_date}。可点击「读取结果」按钮查看最新排名。")
                
                if db_latest_date and db_latest_date != browse_ref_to_use:
                    if db_latest_date > browse_ref_to_use:
                        st.warning(f"排名数据日期（{browse_ref_to_use}）早于数据库最新日期（{db_latest_date}），建议重新运行评分获取最新排名")
                    else:
                        st.info(f"排名数据日期（{browse_ref_to_use}）晚于数据库最新日期（{db_latest_date}）")
                
                df_all = _read_rank_all_sorted(browse_ref_to_use)
            else:
                st.info("请先运行评分或点击「读取结果」按钮加载排名数据。")
                df_all = None
            
            # 板块筛选和展示设置
            if df_all is not None and not df_all.empty:
                # 添加板块列
                if "ts_code" in df_all.columns:
                    df_all = df_all.copy()
                    df_all["板块"] = df_all["ts_code"].apply(lambda x: market_label(str(x)))
                    # 统一使用 _board_category 将市场映射到主板/创业/北交所/其他
                    df_all["板块分类"] = df_all["ts_code"].apply(_board_category)
                
                # 板块筛选
                board_filter_col1, board_filter_col2 = st.columns([1, 3])
                with board_filter_col1:
                    board_filter = st.selectbox("板块筛选", ["全部", "主板", "创业/科创", "北交所"], index=0, key="board_filter")
                
                # 应用板块筛选
                if board_filter != "全部" and "板块分类" in df_all.columns:
                    df_filtered = df_all[df_all["板块分类"] == board_filter].copy()
                else:
                    df_filtered = df_all.copy()

                # 添加名称列，便于浏览时直接看到股票名称
                if "ts_code" in df_filtered.columns:
                    name_map = _get_stock_name_map()
                    def _lookup_name(ts_code: str):
                        ts = str(ts_code).strip()
                        if ts in name_map:
                            return name_map[ts]
                        if "." in ts:
                            return name_map.get(ts.split(".")[0], "")
                        return ""
                    df_filtered["名称"] = df_filtered["ts_code"].apply(_lookup_name)
                    # 添加总市值（亿元），用于展示
                    mp_basic = _get_stock_basic_map()
                    df_filtered["总市值(亿)"] = df_filtered["ts_code"].apply(
                        lambda x: _total_mv_billion(str(x), mp_basic)
                    )
                    df_filtered = add_concept_column(df_filtered, ts_col="ts_code", out_col="概念", blacklist=CONCEPT_BLACKLIST)
                
                # 展示方式设置
                with board_filter_col2:
                    show_mode = st.radio("展示方式", ["限制条数", "显示全部"], horizontal=True, key="rank_show_mode")
                
                rows_to_show = None
                if show_mode == "限制条数":
                    rows_to_show = st.number_input("显示行数", min_value=5, max_value=1000, value=100, key="rank_rows_cfg")
                
                if not df_filtered.empty:
                    if show_mode == "显示全部":
                        rows_eff = len(df_filtered)
                        st.caption(f"已选择显示全部（共 {rows_eff} 行，板块：{board_filter}）。")
                    else:
                        rows_eff = int(rows_to_show)
                    
                    # 移除临时添加的列用于显示
                    display_df = df_filtered.head(rows_eff).copy()
                    if "板块分类" in display_df.columns:
                        display_df = display_df.drop(columns=["板块分类"])

                    # 将名称列放在代码后便于阅读
                    if "名称" in display_df.columns and "ts_code" in display_df.columns:
                        cols = display_df.columns.tolist()
                        cols.remove("名称")
                        ts_idx = cols.index("ts_code") + 1 if "ts_code" in cols else 1
                        cols.insert(ts_idx, "名称")
                        display_df = display_df[cols]
                    # 将总市值列靠近名称
                    if "总市值(亿)" in display_df.columns and "名称" in display_df.columns:
                        cols = display_df.columns.tolist()
                        cols.remove("总市值(亿)")
                        name_idx = cols.index("名称") + 1 if "名称" in cols else len(cols)
                        cols.insert(name_idx, "总市值(亿)")
                        display_df = display_df[cols]
                    
                    st.dataframe(display_df, width='stretch', height=420)
                    
                    if "ts_code" in df_filtered.columns:
                        codes = df_filtered["ts_code"].astype(str).head(rows_eff).tolist()
                        txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"], st.session_state["export_pref"]["with_suffix"])
                        copy_txt_button(txt, label="📋 复制以上（按当前预览）", key=f"copy_rank_{browse_ref_to_use}")
                else:
                    st.caption(f"板块 {board_filter} 暂无数据")
            else:
                st.caption("暂无排名数据，请先读取结果")

    # ================== 个股详情 ==================
    with tab_detail:
        st.subheader("个股详情")

        # —— 数据库读取控制按钮 ——
        db_reading_enabled = is_details_db_reading_enabled()
        col_btn, col_status = st.columns([1, 4])
        with col_btn:
            if not db_reading_enabled:
                if st.button("🔓 启用数据库读取", key="enable_db_reading"):
                    st.session_state["details_db_reading_enabled"] = True
                    st.rerun()
            else:
                st.success("✅ 已启用")
        with col_status:
            if db_reading_enabled:
                st.info("数据将从 details 数据库读取；如需停用请重启/清空会话。")
            else:
                st.info("未启用数据库读取（避免与写入冲突，如需从数据库加载详情请先启用）。")
        
        st.divider()

        # —— 选择参考日 + 代码（支持从排名前列下拉选择） ——
        # 可选：从 details 数据库已有日期中选择参考日
        available_detail_dates = []
        try:
            from database_manager import query_details_recent_dates
            db_path = get_details_db_path_with_fallback()
            if db_path and is_details_db_available():
                dates_from_db = query_details_recent_dates(365, db_path)
                if dates_from_db:
                    available_detail_dates = sorted(dates_from_db, reverse=True)
        except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
            logger.debug(f"数据库访问失败: {e}")
        except Exception as e:
            logger.debug(f"读取 details 日期失败: {e}")

        c0, c1 = st.columns([1,2])
        with c0:
            ref_select = ""
            if available_detail_dates:
                default_idx = 0
                if "detail_ref_date_selected" in st.session_state:
                    if st.session_state["detail_ref_date_selected"] in available_detail_dates:
                        default_idx = available_detail_dates.index(st.session_state["detail_ref_date_selected"])
                ref_select = st.selectbox(
                    "参考日（可选，来自 details 数据库）",
                    options=available_detail_dates,
                    index=default_idx,
                    key="detail_ref_date_select",
                    help=f"数据库内共 {len(available_detail_dates)} 个可选日期"
                )
                st.session_state["detail_ref_date_selected"] = ref_select
        ref_real = (ref_select or "").strip() or _get_latest_date_from_files() or ""
        # 读取该参考日排名前列以便下拉选择（从全量截取）
        df_all_rank = _read_rank_all_sorted(ref_real) if ref_real else pd.DataFrame()
        df_top_ref = df_all_rank.head(200) if not df_all_rank.empty else pd.DataFrame()
        options_codes = df_top_ref["ts_code"].astype(str).tolist() if ("ts_code" in df_top_ref.columns and not df_top_ref.empty) else []
        st.caption(f"调试: 参考日={ref_real}, 排名文件行数={len(df_all_rank)}, 可选股票数={len(options_codes)}")
        # 导航跳转：上一条/下一条通过 detail_pending_code 传递
        pending_code = st.session_state.pop("detail_pending_code", None) if "detail_pending_code" in st.session_state else None

        # 初始化session_state
        if 'detail_last_code' not in st.session_state:
            st.session_state.detail_last_code = ""
        
        with c1:
            # 确保options_codes不为空，且index有效
            if options_codes:
                default_idx = 0
                # 1) 导航跳转优先
                if pending_code and pending_code in options_codes:
                    default_idx = options_codes.index(pending_code)
                    st.session_state["detail_code_from_top"] = pending_code
                # 2) 当前已选
                elif st.session_state.get("detail_code_from_top") in options_codes:
                    default_idx = options_codes.index(st.session_state["detail_code_from_top"])
                # 3) 历史记录
                elif st.session_state.detail_last_code and st.session_state.detail_last_code in options_codes:
                    default_idx = options_codes.index(st.session_state.detail_last_code)

                code_from_list = st.selectbox("从排名前列选择", options=options_codes,
                                            index=default_idx, placeholder="选择股票 ↓", key="detail_code_from_top")
            else:
                # 当没有可选数据时，提供一个默认选项但不禁用
                code_from_list = st.selectbox("从排名前列选择", options=[""],
                                            index=0, placeholder="暂无可选股票", 
                                            key="detail_code_from_top")

        manual_code_raw = st.text_input(
            "或手动输入代码/简称",
            value=st.session_state.get("detail_manual_code", ""),
            placeholder="如 000001.SH / 000001 / 平安银行",
            key="detail_manual_code"
        )

        manual_prev = st.session_state.get("detail_manual_code_prev", "")
        manual_changed = manual_code_raw != manual_prev
        resolved_manual = _resolve_user_code_input(manual_code_raw)
        resolved_select = _resolve_user_code_input(code_from_list)
        resolved_pending = _resolve_user_code_input(pending_code)

        if manual_changed and resolved_manual:
            effective_code = resolved_manual
        elif resolved_pending:
            effective_code = resolved_pending
        elif resolved_select:
            effective_code = resolved_select
        else:
            effective_code = resolved_manual or ""

        st.session_state["detail_manual_code_prev"] = manual_code_raw

        # 更新历史记录
        if effective_code:
            st.session_state.detail_last_code = effective_code
        
        code_norm = normalize_ts(effective_code) if effective_code else ""

        # —— 渲染详情（含 old 版功能） ——
        if code_norm and ref_real:
            obj = _load_detail_json(ref_real, code_norm)
            if not obj:
                st.warning("未找到该票的详情数据(请检查数据库是否解锁以及是否写入)。")
            else:
                data = obj
                # 兼容数据库格式和JSON格式
                if "summary" in data:
                    # 统一格式：{ts_code, ref_date, summary: {...}, rules}
                    summary = data.get("summary", {}) or {}
                    ts = data.get("ts_code", code_norm)
                else:
                    # 兼容旧格式：{ts_code, ref_date, score, highlights, drawbacks, opportunities, rules, ...}
                    summary = {
                        "score": data.get("score", 0.0),
                        "tiebreak": data.get("tiebreak"),
                        "highlights": data.get("highlights", []),
                        "drawbacks": data.get("drawbacks", []),
                        "opportunities": data.get("opportunities", []),
                        "rank": data.get("rank"),
                        "total": data.get("total")
                    }
                    ts = data.get("ts_code", code_norm)
                
                # 显示数据来源信息（只有在允许读取数据库时才查询数据库状态）
                db_reading_enabled = is_details_db_reading_enabled()
                if db_reading_enabled:
                    try:
                        # 使用统一的函数获取details数据库路径（包含回退逻辑）
                        details_db_path = get_details_db_path_with_fallback()
                        if details_db_path:
                            # 查询股票详情表
                            logger.info(f"[数据库连接] 开始获取数据库管理器实例 (查询股票详情: {code_norm}, {ref_real})")
                            manager = get_database_manager()
                            if manager:
                                sql = "SELECT * FROM stock_details WHERE ts_code = ? AND ref_date = ?"
                                df = manager.execute_sync_query(details_db_path, sql, [code_norm, ref_real], timeout=30.0)
                                
                                if not df.empty:
                                    st.info("数据来源：数据库")
                                else:
                                    st.info("数据来源：JSON文件")
                            else:
                                st.info("数据来源：JSON文件")
                        else:
                            st.info("数据来源：JSON文件")
                    except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                        # 数据库文件不存在或管理器获取失败
                        logger.debug(f"数据库访问失败: {code_norm}_{ref_real}: {e}")
                        st.info("数据来源：JSON文件")
                    except Exception as e:
                        # 数据库读取失败（可能是表不存在、连接错误等）
                        error_msg = str(e).lower()
                        if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                            logger.debug(f"Details数据库表不存在: {code_norm}_{ref_real}")
                        else:
                            logger.debug(f"数据库读取失败: {code_norm}_{ref_real}: {e}")
                        st.info("数据来源：JSON文件")
                else:
                    st.info("数据来源：JSON文件（数据库读取未启用）")
                
                try:
                    score = float(summary.get("score", 0))
                    if not np.isfinite(score):
                        score = 0.0
                except Exception:
                    score = 0.0
                # 计算当日排名（优先 JSON → 全量CSV）
                rank_display = "—"
                r_json = summary.get("rank")
                t_json = summary.get("total")
                if isinstance(r_json, (int, float)) and int(r_json) > 0:
                    rank_display = f"{int(r_json)}" + (f" / {int(t_json)}" if isinstance(t_json, (int, float)) and int(t_json) > 0 else "")
                else:
                    df_allx = _read_rank_all_sorted(ref_real)
                    if not df_allx.empty:
                        try:
                            row = df_allx.loc[df_allx["ts_code"].astype(str) == str(ts)]
                            if not row.empty and "rank" in row.columns:
                                rank_display = f"{int(row['rank'].iloc[0])} / {len(df_allx)}"
                        except Exception:
                            pass

                prev_ranks = _get_prev_rank_history(ts, ref_real, max_days=15)
                has_prev_ranks = bool(prev_ranks)
                if has_prev_ranks:
                    colA, colB, colC = st.columns([1, 1, 1])
                else:
                    colA, colB = st.columns([1, 1])
                    colC = None

                # 总览 + 高亮/缺点（美化显示）
                with colA:
                    st.markdown("**总览**")
                    # 美化显示summary内容
                    with st.container(border=True):
                        # 基本信息
                        st.metric("代码", ts)
                        st.metric("市场", market_label(ts))
                        st.metric("参考日", ref_real)
                        st.divider()
                        # 评分信息
                        if "score" in summary:
                            st.metric("分数", f"{summary.get('score', 0.0):.2f}")
                        if "tiebreak" in summary and summary.get("tiebreak") is not None:
                            st.metric("KDJ-J", f"{summary.get('tiebreak', 0.0):.2f}")
                        if "rank" in summary and summary.get("rank") is not None:
                            total = summary.get("total", 0)
                            rank_val = summary.get("rank", 0)
                            if total > 0:
                                st.metric("排名", f"{rank_val} / {total}")
                            else:
                                st.metric("排名", str(rank_val))
                        # 概念信息（小号文字）
                        concept_text = concept_text_for_stock(ts, blacklist=CONCEPT_BLACKLIST)
                        if concept_text:
                            st.caption(f"概念: {concept_text}")
                        # 显示其他summary字段
                        other_fields = {k: v for k, v in summary.items() 
                                      if k not in ["score", "tiebreak", "rank", "total", "highlights", "drawbacks", "opportunities"]}
                        if other_fields:
                            with st.expander("其他信息", expanded=False):
                                for key, value in other_fields.items():
                                    st.text(f"{key}: {value}")
                # 创建explain到name的映射（用于将explain文本转换为策略名字）
                # 方法1：从策略仓库中获取（最权威）
                explain_to_name = {}
                try:
                    for r in (getattr(se, "SC_RULES", []) or []):
                        explain_val = r.get("explain")
                        name_val = r.get("name")
                        if explain_val and name_val:
                            explain_str = str(explain_val).strip()
                            name_str = str(name_val)
                            # 存储多种格式的映射以提高匹配成功率
                            explain_to_name[explain_str] = name_str
                            # 去除所有空白字符后的版本
                            explain_normalized = re.sub(r'\s+', '', explain_str)
                            if explain_normalized != explain_str:
                                explain_to_name[explain_normalized] = name_str
                except Exception:
                    pass
                
                # 方法2：从rules数据中补充（如果策略仓库中没有）
                rules_list_for_mapping = data.get("rules", [])
                if isinstance(rules_list_for_mapping, list):
                    for r in rules_list_for_mapping:
                        if isinstance(r, dict):
                            explain_val = r.get("explain")
                            name_val = r.get("name")
                            if explain_val and name_val:
                                explain_str = str(explain_val).strip()
                                name_str = str(name_val)
                                # 如果策略仓库中没有，则添加
                                if explain_str not in explain_to_name:
                                    explain_to_name[explain_str] = name_str
                                # 去除所有空白字符后的版本
                                explain_normalized = re.sub(r'\s+', '', explain_str)
                                if explain_normalized not in explain_to_name:
                                    explain_to_name[explain_normalized] = name_str
                
                # 将explain文本转换为策略名字的辅助函数
                def _get_rule_name(explain_text: str) -> str:
                    """将explain文本转换为策略名字，如果找不到则返回原文本"""
                    if not explain_text:
                        return explain_text
                    explain_text = str(explain_text).strip()
                    # 先尝试精确匹配
                    if explain_text in explain_to_name:
                        return explain_to_name[explain_text]
                    # 再尝试去除所有空白字符后匹配
                    explain_normalized = re.sub(r'\s+', '', explain_text)
                    if explain_normalized in explain_to_name:
                        return explain_to_name[explain_normalized]
                    # 如果都找不到，返回原文本
                    return explain_text
                
                with colB:
                    st.markdown("**高亮 / 缺点**")
                    # 美化显示highlights和drawbacks
                    with st.container(border=True):
                        highlights = summary.get("highlights", [])
                        drawbacks = summary.get("drawbacks", [])
                        
                        if highlights:
                            st.markdown("**✅ 高亮**")
                            for h in highlights:
                                if h:
                                    rule_name = _get_rule_name(h)
                                    st.success(f"• {rule_name}")
                        
                        if drawbacks:
                            st.markdown("**⚠️ 缺点**")
                            for d in drawbacks:
                                if d:
                                    rule_name = _get_rule_name(d)
                                    st.error(f"• {rule_name}")
                    
                        if not highlights and not drawbacks:
                            st.caption("暂无")

                if has_prev_ranks and colC:
                    with colC:
                        st.markdown("**前日排名**")
                        with st.container(border=True):
                            for item in prev_ranks:
                                if not item:
                                    continue
                                rank_text = f"{item.get('rank')}"
                                total_val = item.get("total")
                                if isinstance(total_val, int) and total_val > 0:
                                    rank_text = f"{rank_text} / {total_val}"
                                st.write(f"{item.get('date', '')} · {rank_text}")

                # 交易性机会
                ops = (summary.get("opportunities") or [])
                with st.expander("交易性机会", expanded=True):
                    if ops:
                        for t in ops:
                            if t:
                                rule_name = _get_rule_name(t)
                                st.write("• " + rule_name)
                    else:
                        st.caption("暂无")

                # 价格K线（完整行情，标记参考日；跳过周末/节假日缺口）
                _render_price_kline_chart(
                    ts=code_norm,
                    ref_real=ref_real,
                    options_codes=options_codes,
                    nav_state_prefix="detail",
                    load_full_history=False,  # 个股详情只拉取当前展示窗口的数据，避免加载过多
                    window_days=280,
                )

                # 逐规则明细（可选显示 when）
                # rules字段已经通过_load_detail_json统一解析为list[dict]格式
                rules_list = data.get("rules", [])
                if not isinstance(rules_list, list):
                    rules_list = []
                rules = pd.DataFrame(rules_list)
                name_to_when = {}
                
                from datetime import datetime
                import re

                if not rules.empty:
                    
                    def _days_from_ref(d: str | None) -> int | None:
                        if isinstance(d, str) and re.fullmatch(r"\d{8}", d):
                            return (datetime.strptime(ref_real, "%Y%m%d") - datetime.strptime(d, "%Y%m%d")).days
                        return None

                    if not rules.empty:
                        def _pick_last_hit_days(row):
                            # 先看 lag 是否有值（仅 RECENT/DIST/NEAR）
                            lag = row.get("lag")
                            if pd.notna(lag):
                                try:
                                    return int(lag)
                                except Exception:
                                    pass
                            # 否则回落到 hit_date → 天数
                            return _days_from_ref(row.get("hit_date"))

                        rules["last_hit_days"] = rules.apply(_pick_last_hit_days, axis=1)
                        # 可选：显示更干净（支持空值）
                        rules["last_hit_days"] = rules["last_hit_days"].astype("Int64")
                try:
                    for r in (getattr(se, "SC_RULES", []) or []):
                        if "clauses" in r and r["clauses"]:
                            ws = [c.get("when","") for c in r["clauses"] if c.get("when")]
                            name_to_when[str(r.get("name","<unnamed>"))] = " AND ".join(ws)
                        else:
                            name_to_when[str(r.get("name","<unnamed>"))] = str(r.get("when",""))
                except Exception:
                    name_to_when = {}
                
                # 创建name到explain的映射（从策略仓库中获取）
                name_to_explain = {}
                try:
                    for r in (getattr(se, "SC_RULES", []) or []):
                        rule_name = str(r.get("name", ""))
                        if rule_name:
                            explain_val = r.get("explain")
                            if explain_val:
                                name_to_explain[rule_name] = str(explain_val)
                except Exception:
                    name_to_explain = {}
                show_when = st.checkbox("显示规则 when 表达式", value=False, key="detail_show_when")
                if not rules.empty:
                    if show_when:
                        rules["when"] = rules["name"].map(name_to_when).fillna("")
                    st.markdown("**规则明细**")
                    
                    # 创建用于显示的DataFrame副本
                    rules_display = rules.copy()
                    
                    # 从策略仓库中获取explain（如果数据库中没有）
                    if "name" in rules_display.columns:
                        # 添加explain列（从策略仓库中获取）
                        if "explain" not in rules_display.columns:
                            rules_display["explain"] = rules_display["name"].map(lambda n: name_to_explain.get(str(n), ""))
                        else:
                            # 如果数据库中有explain列，但可能为空，则从策略仓库补充
                            rules_display["explain"] = rules_display.apply(
                                lambda row: row.get("explain") if pd.notna(row.get("explain")) and str(row.get("explain")).strip() 
                                else name_to_explain.get(str(row.get("name", "")), ""), axis=1
                            )
                    
                    # 使用streamlit dataframe显示，保留原生交互功能（排序、筛选等）
                    # 确保列顺序：name在最前，explain在最后（如果有）
                    col_order = ["name"]
                    for col in rules_display.columns:
                        if col not in ["name", "explain"]:
                            col_order.append(col)
                    # explain列放在最后（如果有）
                    if "explain" in rules_display.columns:
                        col_order.append("explain")
                    col_order = [c for c in col_order if c in rules_display.columns]
                    rules_display = rules_display[col_order]
                    
                    # 配置列的显示方式（如果explain存在，为name列添加help提示）
                    column_config = None
                    if "explain" in rules_display.columns and "name" in rules_display.columns:
                        try:
                            # 尝试使用column_config配置（streamlit >= 1.23.0支持）
                            column_config = {}
                            # name列的配置，提示用户可以查看explain列
                            column_config["name"] = st.column_config.TextColumn(
                                "策略名称",
                                help="策略的简短名称，详细说明见右侧explain列"
                            )
                            # explain列的配置，说明这是详细说明
                            column_config["explain"] = st.column_config.TextColumn(
                                "详细说明",
                                help="策略的详细说明（鼠标悬浮在此列标题上可查看提示）",
                                width="medium"
                            )
                        except Exception:
                            # 如果column_config不支持，使用默认方式
                            column_config = None
                    
                    # 显示streamlit dataframe，保留所有原生交互功能
                    st.dataframe(
                        rules_display,
                        width='stretch',
                        height=420,
                        hide_index=True,
                        column_config=column_config
                    )
                else:
                    st.info("无规则明细。")
                
                # 显示未触发的规则
                try:
                    # 获取所有规则列表
                    all_rules = getattr(se, "SC_RULES", []) or []
                    # 获取已触发规则的名称集合
                    triggered_rule_names = set()
                    if not rules.empty and "name" in rules.columns:
                        triggered_rule_names = set(rules["name"].astype(str).unique())
                    
                    # 找出未触发的规则
                    untriggered_rules = []
                    for r in all_rules:
                        rule_name = str(r.get("name", ""))
                        if rule_name and rule_name not in triggered_rule_names:
                            # 获取时间周期
                            tf = str(r.get("timeframe", "D")).upper()
                            # 获取命中口径
                            scope = str(r.get("scope", "ANY")).upper().strip()
                            # 对于 scope: LAST，不显示 score_windows
                            if scope == "LAST":
                                win = None  # scope: LAST 不需要 score_windows
                            else:
                                # 获取回看窗口
                                win = int(r.get("score_windows", 60))
                            # 获取分数
                            points = float(r.get("points", 0))
                            # 获取前置门槛
                            gate = r.get("gate")
                            gate_str = ""
                            if gate:
                                if isinstance(gate, dict):
                                    gate_when = gate.get("when", "")
                                    if gate_when:
                                        gate_str = f"gate: {gate_when}"
                                elif isinstance(gate, str):
                                    gate_str = f"gate: {gate}"
                            
                            # 处理子句信息
                            clauses_info = ""
                            if "clauses" in r and r.get("clauses"):
                                clauses = r.get("clauses", [])
                                clause_parts = []
                                for c in clauses:
                                    c_tf = str(c.get("timeframe", "D")).upper()
                                    c_scope = str(c.get("scope", "ANY")).upper().strip()
                                    # 对于 scope: LAST，不显示窗口值
                                    if c_scope == "LAST":
                                        clause_parts.append(f"{c_tf}/-/LAST")
                                    else:
                                        # 获取窗口
                                        c_win = int(c.get("score_windows", 60))
                                        clause_parts.append(f"{c_tf}/{c_win}/{c_scope}")
                                if clause_parts:
                                    clauses_info = f"子句: {len(clauses)}个 ({', '.join(clause_parts)})"
                            
                            rule_data = {
                                "name": rule_name,
                                "timeframe": tf,
                                "scope": scope,
                                "points": points,
                                "gate": gate_str if gate_str else "",
                                "clauses": clauses_info if clauses_info else "",
                                "when": name_to_when.get(rule_name, ""),
                                "explain": str(r.get("explain", ""))
                            }
                            # 只有非 LAST scope 才添加 score_windows
                            if scope != "LAST":
                                rule_data["score_windows"] = win
                            # scope: LAST 不添加 score_windows 字段
            
                            untriggered_rules.append(rule_data)
                    
                    if untriggered_rules:
                        st.markdown("**未触发的规则**")
                        untriggered_df = pd.DataFrame(untriggered_rules)
                        
                        # 创建用于显示的DataFrame副本
                        untriggered_display = untriggered_df.copy()
                        
                        # 如果show_when为False，移除when列
                        if not show_when and "when" in untriggered_display.columns:
                            untriggered_display = untriggered_display.drop(columns=["when"])
                        
                        # 确保列顺序：name在最前，然后是timeframe, score_windows(仅非LAST), scope, points, gate, clauses, when(可选), explain在最后
                        col_order = ["name"]
                        # 添加主要字段
                        for col in ["timeframe", "scope", "points"]:
                            if col in untriggered_display.columns:
                                col_order.append(col)
                        # score_windows 只在非 LAST scope 时显示
                        if "score_windows" in untriggered_display.columns:
                            # 检查是否有非 LAST 的规则
                            if not untriggered_display.empty and "scope" in untriggered_display.columns:
                                has_non_last = (untriggered_display["scope"].astype(str).str.upper() != "LAST").any()
                                if has_non_last:
                                    col_order.append("score_windows")
                            else:
                                col_order.append("score_windows")
                        # 添加可选字段（总是显示，即使为空）
                        for col in ["gate", "clauses"]:
                            if col in untriggered_display.columns:
                                col_order.append(col)
                        # 添加when列（如果show_when为True）
                        if show_when and "when" in untriggered_display.columns:
                            col_order.append("when")
                        # explain列放在最后
                        if "explain" in untriggered_display.columns:
                            col_order.append("explain")
                        # 确保只包含存在的列
                        col_order = [c for c in col_order if c in untriggered_display.columns]
                        untriggered_display = untriggered_display[col_order]
                        
                        # 配置列的显示方式
                        untriggered_column_config = None
                        try:
                            untriggered_column_config = {}
                            if "name" in untriggered_display.columns:
                                untriggered_column_config["name"] = st.column_config.TextColumn(
                                    "策略名称",
                                    help="策略的简短名称"
                                )
                            if "timeframe" in untriggered_display.columns:
                                untriggered_column_config["timeframe"] = st.column_config.TextColumn(
                                    "时间周期",
                                    help="D(日线)/W(周线)/M(月线)",
                                    width="small"
                                )
                            if "score_windows" in untriggered_display.columns:
                                # 对于 score_windows 列，使用 TextColumn 以便显示空值或 "-"
                                # 因为 scope: LAST 的规则没有此字段
                                untriggered_column_config["score_windows"] = st.column_config.TextColumn(
                                    "计分窗口",
                                    help="计分窗口条数（用于计分判断的时间窗口）。scope: LAST 的规则不需要此字段，显示为空",
                                    width="small"
                                )
                            if "scope" in untriggered_display.columns:
                                untriggered_column_config["scope"] = st.column_config.TextColumn(
                                    "命中口径",
                                    help="ANY/EACH/PERBAR等",
                                    width="small"
                                )
                            if "points" in untriggered_display.columns:
                                untriggered_column_config["points"] = st.column_config.NumberColumn(
                                    "分数",
                                    help="命中时加/减分",
                                    width="small"
                                )
                            if "gate" in untriggered_display.columns:
                                untriggered_column_config["gate"] = st.column_config.TextColumn(
                                    "前置门槛",
                                    help="前置条件表达式",
                                    width="medium"
                                )
                            if "clauses" in untriggered_display.columns:
                                untriggered_column_config["clauses"] = st.column_config.TextColumn(
                                    "子句信息",
                                    help="多子句组合信息",
                                    width="medium"
                                )
                            if "when" in untriggered_display.columns:
                                untriggered_column_config["when"] = st.column_config.TextColumn(
                                    "条件表达式",
                                    help="TDX风格表达式",
                                    width="large"
                                )
                            if "explain" in untriggered_display.columns:
                                untriggered_column_config["explain"] = st.column_config.TextColumn(
                                    "详细说明",
                                    help="策略的详细说明",
                                    width="medium"
                                )
                        except Exception:
                            untriggered_column_config = None
                        
                        # 显示未触发的规则
                        st.dataframe(
                            untriggered_display,
                            width='stretch',
                            height=420,
                            hide_index=True,
                            column_config=untriggered_column_config
                        )
                    else:
                        st.info("所有规则均已触发。")
                except Exception as e:
                    logger.debug(f"显示未触发规则失败: {e}")
                    # 静默失败，不影响主流程

                st.markdown('<div id="detail_rule_anchor_detail"></div>', unsafe_allow_html=True)

    # ================== K线看板 ==================
    with tab_kline:
        st.subheader("K线看板")

        price_cache = _get_kline_price_cache()

        # —— 预先计算参考日/可选股票，供顶部图表使用（控件放在下方） ——
        available_detail_dates = []
        try:
            from database_manager import query_details_recent_dates
            db_path = get_details_db_path_with_fallback()
            if db_path and is_details_db_available():
                dates_from_db = query_details_recent_dates(365, db_path)
                if dates_from_db:
                    available_detail_dates = sorted(dates_from_db, reverse=True)
        except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
            logger.debug(f"数据库访问失败: {e}")
        except Exception as e:
            logger.debug(f"读取 details 日期失败: {e}")

        # 参考日默认值
        if "kline_ref_date_single" not in st.session_state:
            st.session_state["kline_ref_date_single"] = _get_latest_date_from_files() or ""
        ref_real_kline = (st.session_state.get("kline_ref_date_single") or "").strip() or _get_latest_date_from_files() or ""
        display_codes_saved = st.session_state.get("kline_display_list", [])
        # 仅在用户点击生成名单后使用缓存名单，默认不导入 TopK
        nav_codes_kline = display_codes_saved

        pending_code_kline = st.session_state.pop("kline_pending_code", None) if "kline_pending_code" in st.session_state else None
        if "kline_last_code" not in st.session_state:
            st.session_state["kline_last_code"] = ""
        code_from_list_kline = st.session_state.get("kline_code_from_top", "")
        manual_code_raw_kline = st.session_state.get("kline_manual_code", "")
        manual_prev_kline = st.session_state.get("kline_manual_code_prev", "")
        manual_changed_kline = manual_code_raw_kline != manual_prev_kline
        resolved_manual_kline = _resolve_user_code_input(manual_code_raw_kline)
        resolved_select_kline = _resolve_user_code_input(code_from_list_kline)
        resolved_pending_kline = _resolve_user_code_input(pending_code_kline)

        if manual_changed_kline and resolved_manual_kline:
            effective_code_kline = resolved_manual_kline
        elif resolved_pending_kline:
            effective_code_kline = resolved_pending_kline
        elif resolved_select_kline:
            effective_code_kline = resolved_select_kline
        else:
            effective_code_kline = resolved_manual_kline or ""

        st.session_state["kline_manual_code_prev"] = manual_code_raw_kline
        if effective_code_kline:
            st.session_state["kline_last_code"] = effective_code_kline

        code_norm_kline = normalize_ts(effective_code_kline) if effective_code_kline else ""

        # 不在启动时预加载 TopK，只有已有名单时才预加载
        if nav_codes_kline:
            _prefetch_kline_prices(nav_codes_kline, adj_type="qfq")

        if code_norm_kline and ref_real_kline:
            _render_price_kline_chart(
                ts=code_norm_kline,
                ref_real=ref_real_kline,
                options_codes=nav_codes_kline,
                nav_state_prefix="kline",
                price_cache=price_cache,
            )
            concept_kline = concept_text_for_stock(code_norm_kline, blacklist=CONCEPT_BLACKLIST)
            name_map = _get_stock_name_map()
            stock_name = name_map.get(code_norm_kline, name_map.get(code_norm_kline.split(".")[0], ""))
            external_links = _build_external_chart_links(code_norm_kline)
            if external_links:
                with st.container(border=True):
                    st.markdown("**外部行情图表**（新标签打开）")
                    link_lines = [
                        f'- <a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\">{name}</a>'
                        for name, url in external_links
                    ]
                    st.markdown("\n".join(link_lines), unsafe_allow_html=True)
            turnover_display = "—"
            turnover_hint = ""
            try:
                days, cum_val, err, used_days, start_d, end_d = _compute_turnover_days(code_norm_kline, ref_real_kline, target=60.0)
                if err:
                    turnover_hint = err
                if days is not None:
                    turnover_display = f"{days} 天"
                elif cum_val is not None:
                    turnover_hint = f"累计 {cum_val:.1f}%（未到 60%）"
                if used_days:
                    extra = f" · 累计天数 {used_days}"
                    if start_d and end_d:
                        extra += f"（{start_d}~{end_d}）"
                    turnover_hint = (turnover_hint + extra).strip(" ·")
            except Exception as e:
                turnover_hint = f"计算失败: {e}"

            with st.container(border=True):
                st.markdown("**展示参数**")
                info_col1, info_col2 = st.columns([1, 1])
                with info_col1:
                    st.metric("参考日", ref_real_kline or "—")
                with info_col2:
                    st.metric("60%换手耗时", turnover_display)
        else:
            st.info("请选择股票和参考日以绘制K线。")

        st.divider()
        with st.container(border=True):
            st.markdown("**参数与名单**")
            colA, colB = st.columns([1, 2])
            with colA:
                # 预先写入 session_state 后不再传递 value，避免默认值与 Session State 同时赋值的警告
                new_ref = st.text_input("参考日（默认=最新文件）", key="kline_ref_date_single")
                ref_real_kline = (new_ref or "").strip() or ref_real_kline

                topk_value = st.session_state.get("kline_list_topk", 100)
                board_topk_value = st.session_state.get("kline_board_topk", 100)
                board_choice = st.session_state.get("kline_board_choice", "主板")

                list_source_options = ["TopK导入", "自定义名单", "按板块TopK", "触发了某个策略"]
                if st.session_state.get("kline_list_source") not in list_source_options:
                    st.session_state["kline_list_source"] = list_source_options[0]
                list_source = st.selectbox(
                    "名单来源",
                    options=list_source_options,
                    index=list_source_options.index(st.session_state.get("kline_list_source", list_source_options[0])),
                    key="kline_list_source",
                    placeholder="选择名单来源"
                )

                # 显示当前选项的专属配置
                if list_source == "TopK导入":
                    topk_value = st.number_input("TopK数量", min_value=1, max_value=5000, value=int(topk_value or 100), step=5, key="kline_list_topk")
                    ref_for_list = ref_real_kline
                elif list_source == "自定义名单":
                    ref_for_list = ref_real_kline
                elif list_source == "按板块TopK":
                    board_choice = st.selectbox(
                        "板块",
                        ["主板", "创业/科创", "北交所", "其他", "全部"],
                        index=["主板", "创业/科创", "北交所", "其他", "全部"].index(str(board_choice)) if board_choice in ["主板","创业/科创","北交所","其他","全部"] else 0,
                        key="kline_board_choice"
                    )
                    board_topk_value = st.number_input("板块TopK数量", min_value=1, max_value=5000, value=int(board_topk_value or 100), step=5, key="kline_board_topk")
                    ref_for_list = ref_real_kline
                else:
                    rule_names_for_trigger = _get_rule_names()
                    ref_for_list = ref_real_kline
                    st.selectbox(
                        "选择策略",
                        options=rule_names_for_trigger or ["(暂无可选策略)"],
                        index=0,
                        key="kline_trigger_rule",
                        placeholder="选择策略名"
                    )
                st.caption("提示：需要读取当日详情数据，若未开启将回退到本地JSON。")

            with colB:
                options_for_select = nav_codes_kline
                if options_for_select:
                    desired_code = st.session_state.get("kline_code_from_top", "")
                    if pending_code_kline and pending_code_kline in options_for_select:
                        desired_code = pending_code_kline
                    elif desired_code not in options_for_select:
                        if st.session_state["kline_last_code"] and st.session_state["kline_last_code"] in options_for_select:
                            desired_code = st.session_state["kline_last_code"]
                        else:
                            desired_code = options_for_select[0]

                    # 通过 Session State 设置默认值，避免与 selectbox 默认值冲突警告
                    if desired_code and desired_code in options_for_select:
                        st.session_state["kline_code_from_top"] = desired_code

                    st.selectbox(
                        "从排名前列选择",
                        options=options_for_select,
                        placeholder="选择股票 ↓",
                        key="kline_code_from_top"
                    )
                else:
                    st.session_state["kline_code_from_top"] = ""
                    st.selectbox(
                        "从排名前列选择",
                        options=[""],
                        placeholder="暂无可选股票",
                        key="kline_code_from_top"
                    )

                st.text_input(
                    "或手动输入代码/简称",
                    value=st.session_state.get("kline_manual_code", ""),
                    placeholder="如 000001.SH / 000001 / 平安银行",
                    key="kline_manual_code"
                )

                if list_source == "自定义名单":
                    custom_default = st.session_state.get("kline_custom_text", "")
                    col_clear1, col_clear2 = st.columns([1, 1])
                    with col_clear1:
                        if st.button("清空自定义名单", key="btn_clear_kline_custom", use_container_width=True):
                            st.session_state["kline_custom_text"] = ""
                            st.session_state["kline_custom_text_area"] = ""
                    custom_text = st.text_area(
                        "自定义名单（换行/逗号分隔，可填代码或名称）",
                        value=st.session_state.get("kline_custom_text", custom_default),
                        placeholder="000001.SZ\n平安银行\n600519.SH",
                        height=120,
                        key="kline_custom_text_area"
                    )
                else:
                    custom_text = st.session_state.get("kline_custom_text", "")

            build_btn = st.button("生成展示名单并刷新指标", key="kline_build_list_btn", width="stretch")

        if build_btn:
            list_ref_to_use = (ref_for_list or ref_real_kline or "").strip()
            codes_list: list[str] = []
            if list_source == "TopK导入":
                if not list_ref_to_use:
                    st.error("请选择参考日以从TopK导入。")
                else:
                    try:
                        df_topk = _slice_top_from_all(list_ref_to_use, int(topk_value))
                        if df_topk.empty:
                            st.error(f"无法从排名文件读取Top-{int(topk_value)}股票（参考日：{list_ref_to_use}）")
                        else:
                            codes_list = df_topk["ts_code"].astype(str).tolist()
                    except Exception as e:
                        st.error(f"读取TopK失败：{e}")
            elif list_source == "自定义名单":
                codes_list = _parse_code_list(custom_text)
            elif list_source == "按板块TopK":
                if not list_ref_to_use:
                    st.error("请选择参考日以生成板块Top名单。")
                else:
                    try:
                        df_topk = _slice_board_top(list_ref_to_use, str(board_choice), int(board_topk_value))
                        if df_topk.empty:
                            st.error(f"板块Top名单为空（参考日：{list_ref_to_use}，板块：{board_choice}）")
                        else:
                            codes_list = df_topk["ts_code"].astype(str).tolist()
                    except Exception as e:
                        st.error(f"读取板块Top名单失败：{e}")
            else:
                rule_pick = st.session_state.get("kline_trigger_rule", "")
                if rule_pick in ("", "(暂无可选策略)") or not rule_pick:
                    st.error("请选择策略名以生成名单。")
                elif not list_ref_to_use:
                    st.error("请选择参考日以读取策略触发名单。")
                else:
                    if not is_details_db_reading_enabled():
                        st.session_state["details_db_reading_enabled"] = True
                    codes_list = _codes_triggered_on_rule(list_ref_to_use, str(rule_pick))
                    if not codes_list:
                        st.warning("未找到触发该策略的股票，或参考日无详情数据。")

            # 去重并保存
            if codes_list:
                seen = set()
                deduped = []
                for c in codes_list:
                    if c and c not in seen:
                        seen.add(c)
                        deduped.append(c)
                codes_list = deduped
                st.session_state["kline_display_list"] = codes_list
                st.session_state["kline_display_ref"] = list_ref_to_use
                st.session_state["kline_custom_text"] = custom_text
                st.success(f"已生成名单：{len(codes_list)} 只，参考日 {list_ref_to_use or '—'}")
                # 自动让 K 线跟随名单首只票（通过 pending_code + rerun，避免修改已实例化的控件）
                first_code = normalize_ts(codes_list[0])
                st.session_state["kline_pending_code"] = first_code
                st.session_state["kline_last_code"] = first_code
                try:
                    st.rerun()
                except Exception:
                    pass
            else:
                st.warning("名单为空，请检查TopK文件或自定义输入。")

        display_codes = st.session_state.get("kline_display_list", [])
        display_ref = st.session_state.get("kline_display_ref") or ref_real_kline
        if display_codes and display_ref:
            st.caption(f"当前名单（参考日 {display_ref}），共 {len(display_codes)} 只：")
            st.dataframe(pd.DataFrame({"ts_code": display_codes}), width="stretch", height=320, hide_index=True)
        else:
            st.info("点击上方按钮生成展示名单。")

    # ================== 持仓建议 ==================
    with tab_position:
        st.subheader("持仓建议（个股）")
        with st.expander("输入", expanded=True):
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                pos_ref = st.text_input("参考日（YYYYMMDD；留空=自动取最新）", value="", key="pos_ref_input")
                price_mode = st.selectbox("买点来源", ["按日期取价", "策略取价（可选）", "手工输入"], index=0)
            with c2:
                raw_code = st.text_input("股票代码（支持多种写法）", value="", key="pos_code_input")
                price_field = st.selectbox("价格口径", ["开盘价(open)", "收盘价(close)", "最高价(high)", "最低价(low)"], index=0)
            with c3:
                # recompute_opts = st.multiselect("仅重算需要的指标", ["kdj","ma","macd"], default=["kdj"], key="pos_recompute_indicators")
                recalc_mode_pos = st.radio("指标重算", ["自选", "全部(all)", "不重算(none)"],
                                        index=0, horizontal=True, key="pos_recalc_mode")
                if recalc_mode_pos == "自选":
                    recompute_opts = st.multiselect("仅重算需要的指标",
                                                    _indicator_options(),
                                                    default=["kdj"],
                                                    key="pos_recompute_pick")
                    recompute_to_pass = tuple(recompute_opts) if recompute_opts else ("kdj",)
                elif recalc_mode_pos == "全部(all)":
                    recompute_to_pass = "all"
                else:
                    recompute_to_pass = "none"

                use_virtual = st.checkbox("基于“明日虚拟日”检查（勾选后按下方场景）", value=False)

            # 场景参数（仅当 use_virtual）
            scen = Scenario()
            if use_virtual:
                # with st.expander("明日情景参数", expanded=False):
                with st.container(border=True):
                    st.markdown("**明日情景参数**")
                    cc1, cc2, cc3 = st.columns([1,1,1])
                    with cc1:
                        scen_mode = st.selectbox("价格模式", ["close_pct","open_pct","gap_then_close_pct","flat","limit_up","limit_down"], index=0)
                        pct = st.number_input("涨跌幅 pct（%）", value=2.0, step=0.5, format="%.2f")
                        gap_pct = st.number_input("跳空 gap_pct（%）", value=0.0, step=0.5, format="%.2f")
                    with cc2:
                        vol_mode = st.selectbox("量能模式", ["same","pct","mult"], index=2)
                        vol_arg = st.number_input("量能参数（% 或 倍数）", value=1.2, step=0.1, format="%.2f")
                        hl_mode = st.selectbox("高低生成", ["follow","atr_like","range_pct"], index=0)
                    with cc3:
                        range_pct = st.number_input("range_pct（%）", value=2.0, step=0.5, format="%.2f")
                        atr_mult = st.number_input("atr_mult", value=1.0, step=0.1, format="%.2f")
                        lock_hi_open = st.checkbox("锁定收盘高于开盘", value=False)
                    scen = Scenario(mode=scen_mode, pct=pct, gap_pct=gap_pct, vol_mode=vol_mode, vol_arg=vol_arg,
                                    hl_mode=hl_mode, range_pct=range_pct, atr_mult=atr_mult,
                                    lock_higher_than_open=lock_hi_open)

            # 买点来源
            entry_price = None
            # 统一参考日
            try:
                trade_dates = get_trade_dates()
                latest_ref = trade_dates[-1] if trade_dates else ""
            except Exception:
                latest_ref = ""
            ref_use = pos_ref.strip() or latest_ref

            code_norm = normalize_ts(raw_code.strip()) if raw_code.strip() else ""
            if price_mode == "按日期取价":
                sel_date = st.text_input("买点日期（YYYYMMDD）", value=ref_use, key="pos_entry_date")
                if st.button("取价", width='stretch'):
                    if code_norm and sel_date:
                        try:
                            # 读取该日的价格
                            try:
                                from config import DATA_ROOT, UNIFIED_DB_PATH
                                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                                df = query_stock_data(
                                    db_path=db_path,
                                    ts_code=code_norm,
                                    start_date=sel_date,
                                    end_date=sel_date,
                                    adj_type="qfq"
                                )
                            except:
                                # 回退到直接查询
                                from config import DATA_ROOT, UNIFIED_DB_PATH
                                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                                logger.info(f"[数据库连接] 开始获取数据库管理器实例 (回退查询单日数据: {code_norm}, {sel_date})")
                                manager = get_database_manager()
                                sql = "SELECT * FROM stock_data WHERE ts_code = ? AND trade_date = ?"
                                df = manager.execute_sync_query(db_path, sql, [code_norm, sel_date], timeout=30.0)
                            if not df.empty:
                                row = df.sort_values("trade_date").iloc[-1]
                                fld = {"开盘价(open)":"open","收盘价(close)":"close","最高价(high)":"high","最低价(low)":"low"}[price_field]
                                entry_price = float(row[fld])
                                st.success(f"买点={entry_price:.4f}")
                            else:
                                st.warning("该日无数据")
                        except Exception as e:
                            st.error(f"取价失败：{e}")
            elif price_mode == "手工输入":
                entry_price = st.number_input("手工输入买点", value=0.0, step=0.01, format="%.4f")
            else:
                # 策略取价（可选）
                opps = load_opportunity_policies()
                names = [r.get("name","") for r in opps]
                if not names:
                    st.info("暂无“买点策略（个股）”可用，请在 strategies_repo.py 填写 OPPORTUNITY_POLICIES。")
                opp_name = st.selectbox("选择买点策略", names, index=0 if names else None)
                lookback_days = st.number_input("回看天数", min_value=30, max_value=1000, value=180)
                if st.button("按策略取最近一次触发日并定价", width='stretch', disabled=not (code_norm and names)):
                    try:
                        start = (datetime.strptime(ref_use, "%Y%m%d") - timedelta(days=int(lookback_days))).strftime("%Y%m%d")
                        try:
                            from config import DATA_ROOT, UNIFIED_DB_PATH
                            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                            df = query_stock_data(
                                db_path=db_path,
                                ts_code=code_norm,
                                start_date=start,
                                end_date=ref_use,
                                adj_type="qfq"
                            )
                        except:
                            # 回退到直接查询
                            from config import DATA_ROOT, UNIFIED_DB_PATH
                            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                            logger.info(f"[数据库连接] 开始获取数据库管理器实例 (回退查询日期范围数据: {code_norm}, {start}~{ref_use})")
                            manager = get_database_manager()
                            sql = "SELECT * FROM stock_data WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?"
                            df = manager.execute_sync_query(db_path, sql, [code_norm, start, ref_use], timeout=30.0)
                        df = df.sort_values("trade_date")
                        if df.empty:
                            st.warning("无数据")
                        else:
                            # 表达式运行
                            expr = next((r.get("when") or r.get("check") or "" for r in opps if r.get("name")==opp_name), "")
                            if not expr:
                                st.warning("策略没有 when/check 表达式")
                            else:
                                # 计算指标（扩展数据）
                                try:
                                    # 确保数据格式正确
                                    df['open'] = pd.to_numeric(df['open'], errors='coerce')
                                    df['high'] = pd.to_numeric(df['high'], errors='coerce')
                                    df['low'] = pd.to_numeric(df['low'], errors='coerce')
                                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                                    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
                                    
                                    # 计算KDJ指标
                                    if 'j' in expr.lower():
                                        df['j'] = ind.kdj(df)
                                    
                                    # 扩展ctx，包含指标
                                    ctx = {}
                                    for col in df.columns:
                                        if col.lower() not in ['trade_date', 'ts_code', 'adj_factor']:
                                            try:
                                                ctx[col.upper()] = pd.to_numeric(df[col], errors='coerce').values
                                            except:
                                                pass
                                    
                                    # 如果ctx为空，使用基础OHLCV
                                    if not ctx:
                                        ctx = {
                                            "OPEN": df["open"].astype(float).values,
                                            "HIGH": df["high"].astype(float).values,
                                            "LOW": df["low"].astype(float).values,
                                            "CLOSE": df["close"].astype(float).values,
                                            "V": df["vol"].astype(float).values,
                                        }
                                    
                                    # 设置EXTRA_CONTEXT以便GET_LAST_CONDITION_PRICE等函数使用
                                    original_ctx_df = None
                                    try:
                                        from tdx_compat import EXTRA_CONTEXT
                                        original_ctx_df = EXTRA_CONTEXT.get("DF")
                                        # 创建包含所有列的DataFrame
                                        ctx_df = pd.DataFrame(ctx)
                                        # 确保列名小写（tdx_compat需要）
                                        ctx_df.columns = ctx_df.columns.str.lower()
                                        EXTRA_CONTEXT["DF"] = ctx_df
                                    except:
                                        pass
                                    
                                    try:
                                        # 使用evaluate_bool评估表达式
                                        sig = tdx.evaluate_bool(expr, pd.DataFrame(ctx))
                                    finally:
                                        # 恢复原始EXTRA_CONTEXT
                                        if original_ctx_df is not None:
                                            try:
                                                from tdx_compat import EXTRA_CONTEXT
                                                EXTRA_CONTEXT["DF"] = original_ctx_df
                                            except:
                                                pass
                                    idx = [i for i, v in enumerate(sig) if bool(v)]
                                    if not idx:
                                        st.info("回看期内无触发")
                                    else:
                                        last_i = idx[-1]
                                        row = df.iloc[last_i]
                                        fld = {"开盘价(open)":"open","收盘价(close)":"close","最高价(high)":"high","最低价(low)":"low"}[price_field]
                                        entry_price = float(row[fld])
                                        st.success(f"触发日 {row['trade_date']}，买点={entry_price:.4f}")
                                except Exception as e2:
                                    st.error(f"计算失败：{e2}")
                                    import traceback
                                    st.code(traceback.format_exc())
                    except Exception as e:
                        st.error(f"策略取价失败：{e}")

        # 选择“持仓检查策略（个股）”
        pos_rules = load_position_policies()
        pos_names = [r.get("name","") for r in pos_rules]
        selected = st.multiselect("选择要检查的策略", pos_names, default=pos_names)
        selected_rules = [r for r in pos_rules if r.get("name") in set(selected)]

        if st.button("执行检查", width='stretch', disabled=not (code_norm and ref_use and selected_rules)):
            try:
                # 决定 entry_price
                ep = float(entry_price) if entry_price else None
                if ep is None:
                    st.warning("请先设置买点（上面的【取价】或手工输入）。")
                else:
                    pci = PositionCheckInput(
                        ref_date=ref_use,
                        ts_code=code_norm,
                        rules=selected_rules,
                        entry_price=ep,
                        use_scenario=bool(use_virtual),
                        scenario=scen if use_virtual else None,
                        # recompute_indicators=tuple(recompute_opts) if recompute_opts else ("kdj",),
                        recompute_indicators=recompute_to_pass,
                        extra_vars=None
                    )
                    tbl = run_position_checks(pci)
                    st.dataframe(tbl, width='stretch')
                    # 导出
                    csv = tbl.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("导出 CSV", data=csv, file_name=f"position_checks_{code_norm}_{ref_use}.csv", mime="text/csv", width='stretch')
            except Exception as e:
                st.error(f"执行失败：{e}")

    # ================== 明日模拟 ==================
    with tab_predict:
        st.subheader("明日模拟")
        
        # 明日模拟
        # 使用 st.form 防止参数变化时立即刷新UI
        with st.form("prediction_form"):
            with st.expander("输入参数", expanded=True):
                c1, c2 = st.columns([1,1])
                with c1:
                    pred_ref = st.text_input("参考日（YYYYMMDD；留空=自动取最新交易日）", value="", key="pred_ref_input")
                    if not pred_ref.strip():
                        # 显示当前会自动使用的参考日
                        auto_ref = _pick_smart_ref_date()
                        if auto_ref:
                            st.caption(f"💡 将自动使用最新交易日: {auto_ref}")
                        else:
                            st.caption("⚠️ 无法自动获取最新交易日，请手动输入")
                    use_rule_scen = st.checkbox("使用规则内置场景（若规则提供）", value=False)
                    expr_text = st.text_input("临时检查表达式（可留空）", value="")
                    # recompute_opts = st.multiselect("仅重算需要的指标", ["kdj","ma","macd"], default=["kdj"], key="pred_recompute_indicators")
                    recalc_mode_pred = st.radio("指标重算", ["自选", "全部(all)", "不重算(none)"],
                                                index=0, horizontal=True, key="pred_recalc_mode")
                    if recalc_mode_pred == "自选":
                        recompute_opts = st.multiselect("仅重算需要的指标",
                                                        _indicator_options(),
                                                        default=["kdj"],
                                                        key="pred_recompute_pick")
                        recompute_to_pass = tuple(recompute_opts) if recompute_opts else ("kdj",)
                    elif recalc_mode_pred == "全部(all)":
                        recompute_to_pass = "all"
                    else:
                        recompute_to_pass = "none"
                with c2:
                    uni_choice_pred = st.selectbox(
                        "选股范围",
                        ["自定义（下方文本）","全市场","仅白名单","仅黑名单","仅特别关注榜"],
                        index=1, key="pred_uni_choice")
                    # 文本框仅在"自定义"时使用
                    pasted = st.text_area("选股范围（支持多种分隔符：空格、换行、逗号、分号、竖线等；可混合 ts_code / 简写）", height=120, placeholder="例：\n000001.SZ 600000.SH 000001\n或：\n000001.SZ,600000.SH;000001|300001", disabled=(not uni_choice_pred.startswith("自定义")) )
            # with st.expander("全局场景（若未使用规则内置场景则生效）", expanded=False):
            with st.container(border=True):
                st.markdown("**全局场景（若未使用规则内置场景则生效）**")
                cc1, cc2, cc3 = st.columns([1,1,1])
                with cc1:
                    scen_mode = st.selectbox("价格模式", ["close_pct","open_pct","gap_then_close_pct","flat","limit_up","limit_down","reverse_indicator"], index=0)
                    pct = st.number_input("涨跌幅 pct（%）", value=2.0, step=0.5, format="%.2f")
                    gap_pct = st.number_input("跳空 gap_pct（%）", value=0.0, step=0.5, format="%.2f")
                with cc2:
                    vol_mode = st.selectbox("量能模式", ["same","pct","mult"], index=2)
                    vol_arg = st.number_input("量能参数（% 或 倍数）", value=1.2, step=0.1, format="%.2f")
                    hl_mode = st.selectbox("高低生成", ["follow","atr_like","range_pct"], index=0)
                with cc3:
                    range_pct = st.number_input("range_pct（%）", value=2.0, step=0.5, format="%.2f")
                    atr_mult = st.number_input("atr_mult", value=1.0, step=0.1, format="%.2f")
                    lock_hi_open = st.checkbox("锁定收盘高于开盘", value=False)
            
            # 反推模式参数配置
            if scen_mode == "reverse_indicator":
                with st.container(border=True):
                    st.markdown("**反推模式参数**")
                    rc1, rc2, rc3 = st.columns([1,1,1])
                    with rc1:
                        reverse_indicator = st.selectbox("指标名称", ["j", "rsi", "ma", "macd", "diff"], index=0)
                        reverse_target_value = st.number_input("目标指标值", value=10.0, step=0.1, format="%.2f")
                    with rc2:
                        reverse_method = st.selectbox("求解方法", ["optimize", "binary_search", "grid_search"], index=0)
                        reverse_tolerance = st.number_input("求解精度", value=1e-6, step=1e-7, format="%.2e")
                    with rc3:
                        reverse_max_iterations = st.number_input("最大迭代次数", value=1000, step=100, min_value=100, max_value=10000)
                        st.caption("反推模式说明：根据目标指标值反推价格数据")
            
            # 规则选择（使用缓存）
            rules = _cached_load_prediction_rules()
            names = [r.get("name","") for r in rules]
            chosen = st.multiselect("选择模拟策略（可留空）", names, default=[])
            chosen_rules = [r for r in rules if r.get("name") in set(chosen)]
            
            # Tie-break排序选择
            tiebreak_pred = st.selectbox("同分排序", ["none", "kdj_j_asc"], index=1, key="pred_tiebreak")

            # 提交按钮
            submitted = st.form_submit_button("运行明日模拟", width='stretch')
        
        # 只有在表单提交时才执行计算
        if submitted:
            # 参考日与代码集 - 使用智能获取函数
            ref_use = pred_ref.strip() or _pick_smart_ref_date() or ""

            # 解析粘贴的文本范围 - 支持空格和各种分隔符的兼容版本
            def _parse_codes(txt: str):
                out = []
                if not txt:
                    return out
                
                # 支持多种分隔符：换行、空格、制表符、逗号、分号、竖线
                import re
                # 使用正则表达式分割，支持多种分隔符
                separators = r'[\s\n\r\t,;|]+'
                codes = re.split(separators, txt)
                
                for code in codes:
                    s = code.strip()
                    if not s:
                        continue
                    try:
                        out.append(normalize_ts(s))
                    except Exception:
                        continue
                # 去重
                return sorted(set([x for x in out if x]))
            uni = _parse_codes(pasted)

            # 创建Scenario对象，根据模式包含不同参数
            if scen_mode == "reverse_indicator":
                scen = Scenario(
                    mode=scen_mode, 
                    pct=pct, 
                    gap_pct=gap_pct, 
                    vol_mode=vol_mode, 
                    vol_arg=vol_arg,
                    hl_mode=hl_mode, 
                    range_pct=range_pct, 
                    atr_mult=atr_mult,
                    lock_higher_than_open=lock_hi_open,
                    # 反推模式参数
                    reverse_indicator=reverse_indicator,
                    reverse_target_value=reverse_target_value,
                    reverse_method=reverse_method,
                    reverse_tolerance=reverse_tolerance,
                    reverse_max_iterations=reverse_max_iterations
                )
            else:
                scen = Scenario(
                    mode=scen_mode, 
                    pct=pct, 
                    gap_pct=gap_pct, 
                    vol_mode=vol_mode, 
                    vol_arg=vol_arg,
                    hl_mode=hl_mode, 
                    range_pct=range_pct, 
                    atr_mult=atr_mult,
                    lock_higher_than_open=lock_hi_open
                )

            _uni_map = {"全市场": "all", "仅白名单": "white", "仅黑名单": "black", "仅特别关注榜": "attention"}
            use_codes = uni_choice_pred.startswith("自定义")
            if use_codes:
                uni_arg = uni  # 粘贴的自定义列表，前面已 normalize 去重
            else:
                uni_label = _uni_map.get(uni_choice_pred, "all")
                uni_arg = _resolve_pred_universe(uni_label, ref_use)

            # 只有当 ref 有效且范围"非空"时才允许运行
            can_run = bool(ref_use) and bool(uni_arg)

            # 可选：为空时给个提示
            if not use_codes and not uni_arg:
                st.info(f"【{uni_choice_pred}】在 {ref_use} 无可用代码源，请先在\"排名\"页签生成当日 all/top 文件或检查名单缓存。")
            
            if use_codes:
                if uni_arg:
                    st.success(f"✅ 自定义名单解析成功：共 {len(uni_arg)} 只股票")
                    # 显示前几只股票作为预览
                    preview_codes = uni_arg[:5]
                    st.caption(f"预览：{', '.join(preview_codes)}{'...' if len(uni_arg) > 5 else ''}")
                else:
                    st.warning("⚠️ 自定义名单为空，请检查输入的股票代码格式")
                    st.caption("支持的格式：000001、000001.SZ、SZ000001、600000.SH 等；支持分隔符：空格、换行、逗号、分号、竖线")

            if can_run:
                try:
                    inp = PredictionInput(
                        ref_date=ref_use,
                        universe=uni_arg,
                        scenario=scen,
                        rules=chosen_rules if chosen_rules else None,
                        expr=(expr_text or None),
                        use_rule_scenario=bool(use_rule_scen),
                        # recompute_indicators=tuple(recompute_opts) if recompute_opts else ("kdj",),
                        recompute_indicators=recompute_to_pass,
                        cache_dir="cache/sim_pred"
                    )
                    # df = run_prediction(inp)
                    df = run_prediction_in_bg(inp)
                    # 应用Tie-break排序
                    df_sorted = _apply_tiebreak_sorting(df, tiebreak_pred)
                    
                    # 显示结果信息
                    if not df_sorted.empty:
                        st.caption(f"命中 {len(df_sorted)} 只；参考日：{ref_use}")
                        if 'score' in df_sorted.columns and df_sorted['score'].notna().any():
                            st.caption("已按得分排序（降序），同分时按J值升序")
                        else:
                            st.caption("未找到得分数据，按默认排序")
                    
                    df_sorted = add_concept_column(df_sorted, blacklist=CONCEPT_BLACKLIST)
                    st.dataframe(df_sorted, width='stretch')
                    
                    # 复制代码功能（与选股页面保持一致）
                    if not df_sorted.empty and "ts_code" in df_sorted.columns:
                        codes = df_sorted["ts_code"].astype(str).tolist()
                        txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"], 
                                          st.session_state["export_pref"]["with_suffix"])
                        copy_txt_button(txt, label="📋 复制命中代码", key=f"copy_prediction_{ref_use}")
                    
                    # 下载
                    csv = df_sorted.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("导出 CSV", data=csv, file_name=f"prediction_hits_{ref_use}.csv", mime="text/csv", width='stretch')
                    # 仅导出代码 TXT
                    if not df_sorted.empty:
                        codes_txt = "\n".join(df_sorted["ts_code"].astype(str).tolist())
                        st.download_button("导出代码TXT（仅命中集）", data=codes_txt, file_name=f"prediction_hits_{ref_use}.txt", mime="text/plain", width='stretch')
                except Exception as e:
                    st.error(f"运行失败：{e}")
                    with st.expander("调试信息", expanded=False):
                        st.write(f"""
**错误详情：**
- 参考日：{ref_use}
- 股票数量：{len(uni_arg) if uni_arg else 0}
- 股票列表：{uni_arg[:10] if uni_arg else '无'}
- 场景模式：{scen_mode}
- 规则数量：{len(chosen_rules) if chosen_rules else 0}
- 表达式：{expr_text or '无'}

**可能的原因：**
1. 股票代码格式不正确
2. 参考日无交易数据
3. 股票在参考日停牌或退市
4. 规则表达式有语法错误
5. 数据文件缺失或损坏

**建议：**
1. 检查股票代码格式（如：000001.SZ）
2. 尝试使用其他参考日
3. 检查规则表达式语法
4. 确认数据文件完整性
                        """)
            else:
                st.warning("请检查参数设置，确保参考日和选股范围都有效")

        st.divider()
        # ================== 预测排名（合并） ==================
        st.subheader("预测排名")
        st.info("使用模拟数据运行排名策略，得到预测的排名结果")
        
        # 使用 st.form 防止参数变化时立即刷新UI
        with st.form("predict_rank_form"):
            with st.expander("输入参数", expanded=True):
                c1, c2 = st.columns([1,1])
                with c1:
                    pred_rank_ref = st.text_input("参考日（YYYYMMDD；留空=自动取最新交易日）", value="", key="pred_rank_ref_input")
                    if not pred_rank_ref.strip():
                        # 显示当前会自动使用的参考日
                        auto_ref = _pick_smart_ref_date()
                        if auto_ref:
                            st.caption(f"💡 将自动使用最新交易日: {auto_ref}")
                        else:
                            st.caption("⚠️ 无法自动获取最新交易日，请手动输入")
                    recalc_mode_rank = st.radio("指标重算", ["自选", "全部(all)", "不重算(none)"],
                                                index=1, horizontal=True, key="pred_rank_recalc_mode")
                    if recalc_mode_rank == "自选":
                        recompute_opts_rank = st.multiselect("仅重算需要的指标",
                                                        _indicator_options(),
                                                        default=["kdj"],
                                                        key="pred_rank_recompute_pick")
                        recompute_to_pass_rank = tuple(recompute_opts_rank) if recompute_opts_rank else ("kdj",)
                    elif recalc_mode_rank == "全部(all)":
                        recompute_to_pass_rank = "all"
                    else:
                        recompute_to_pass_rank = "none"
                with c2:
                    uni_choice_rank = st.selectbox(
                        "股票范围",
                        ["自定义（下方文本）","全市场","仅白名单","仅黑名单","仅特别关注榜"],
                        index=1, key="pred_rank_uni_choice")
                    # 文本框仅在"自定义"时使用
                    pasted_rank = st.text_area("股票范围（支持多种分隔符：空格、换行、逗号、分号、竖线等；可混合 ts_code / 简写）", height=120, placeholder="例：\n000001.SZ 600000.SH 000001\n或：\n000001.SZ,600000.SH;000001|300001", disabled=(not uni_choice_rank.startswith("自定义")), key="pred_rank_pasted")
            
            # 全局场景设置
            with st.container(border=True):
                st.markdown("**全局场景设置**")
                cc1, cc2, cc3 = st.columns([1,1,1])
                with cc1:
                    scen_mode_rank = st.selectbox("价格模式", ["close_pct","open_pct","gap_then_close_pct","flat","limit_up","limit_down"], index=0, key="pred_rank_scen_mode")
                    pct_rank = st.number_input("涨跌幅 pct（%）", value=2.0, step=0.5, format="%.2f", key="pred_rank_pct")
                    gap_pct_rank = st.number_input("跳空 gap_pct（%）", value=0.0, step=0.5, format="%.2f", key="pred_rank_gap_pct")
                with cc2:
                    vol_mode_rank = st.selectbox("量能模式", ["same","pct","mult"], index=2, key="pred_rank_vol_mode")
                    vol_arg_rank = st.number_input("量能参数（% 或 倍数）", value=1.2, step=0.1, format="%.2f", key="pred_rank_vol_arg")
                    hl_mode_rank = st.selectbox("高低生成", ["follow","atr_like","range_pct"], index=0, key="pred_rank_hl_mode")
                with cc3:
                    range_pct_rank = st.number_input("range_pct（%）", value=2.0, step=0.5, format="%.2f", key="pred_rank_range_pct")
                    atr_mult_rank = st.number_input("atr_mult", value=1.0, step=0.1, format="%.2f", key="pred_rank_atr_mult")
                    lock_hi_open_rank = st.checkbox("锁定收盘高于开盘", value=False, key="pred_rank_lock_hi_open")
            
            # 提交按钮
            submitted_rank = st.form_submit_button("🚀 运行预测排名", width='stretch')
        
        # 只有在表单提交时才执行计算
        if submitted_rank:
            # 参考日与代码集
            ref_use_rank = pred_rank_ref.strip() or _pick_smart_ref_date() or ""
            
            # 解析粘贴的文本范围
            def _parse_codes_rank(txt: str):
                out = []
                if not txt:
                    return out
                import re
                separators = r'[\s\n\r\t,;|]+'
                codes = re.split(separators, txt)
                for code in codes:
                    s = code.strip()
                    if not s:
                        continue
                    try:
                        out.append(normalize_ts(s))
                    except Exception:
                        continue
                return sorted(set([x for x in out if x]))
            uni_rank = _parse_codes_rank(pasted_rank)
            
            # 创建Scenario对象
            scen_rank = Scenario(
                mode=scen_mode_rank, 
                pct=pct_rank, 
                gap_pct=gap_pct_rank, 
                vol_mode=vol_mode_rank, 
                vol_arg=vol_arg_rank,
                hl_mode=hl_mode_rank, 
                range_pct=range_pct_rank, 
                atr_mult=atr_mult_rank,
                lock_higher_than_open=lock_hi_open_rank
            )
            
            _uni_map_rank = {"全市场": "all", "仅白名单": "white", "仅黑名单": "black", "仅特别关注榜": "attention"}
            use_codes_rank = uni_choice_rank.startswith("自定义")
            if use_codes_rank:
                uni_arg_rank = uni_rank
            else:
                uni_label_rank = _uni_map_rank.get(uni_choice_rank, "all")
                uni_arg_rank = _resolve_pred_universe(uni_label_rank, ref_use_rank)
            
            # 只有当 ref 有效且范围"非空"时才允许运行
            can_run_rank = bool(ref_use_rank) and bool(uni_arg_rank)
            
            if not use_codes_rank and not uni_arg_rank:
                st.info(f"【{uni_choice_rank}】在 {ref_use_rank} 无可用代码源，请先在\"排名\"页签生成当日 all/top 文件或检查名单缓存。")
            
            if use_codes_rank:
                if uni_arg_rank:
                    st.success(f"✅ 自定义名单解析成功：共 {len(uni_arg_rank)} 只股票")
                    preview_codes_rank = uni_arg_rank[:5]
                    st.caption(f"预览：{', '.join(preview_codes_rank)}{'...' if len(uni_arg_rank) > 5 else ''}")
                else:
                    st.warning("⚠️ 自定义名单为空，请检查输入的股票代码格式")
            
            if can_run_rank:
                try:
                    inp_rank = PredictRankingInput(
                        ref_date=ref_use_rank,
                        universe=uni_arg_rank,
                        scenario=scen_rank,
                        recompute_indicators=recompute_to_pass_rank,
                        cache_dir="cache/sim_pred_rank",
                        use_prescreen=True,
                        output_dir=os.path.join(cfg.SC_OUTPUT_DIR, "predict")
                    )
                    df_rank_results, rank_path = run_predict_ranking_in_bg(inp_rank)

                    # 显示结果
                    if not df_rank_results.empty:
                        st.success(f"✅ 预测排名完成：共 {len(df_rank_results)} 只股票")
                        df_rank_results = add_concept_column(df_rank_results, blacklist=CONCEPT_BLACKLIST)
                        df_rank_results = add_concept_column(df_rank_results, blacklist=CONCEPT_BLACKLIST)
                        st.dataframe(df_rank_results, width='stretch')
                        if rank_path:
                            st.caption(f"已写入: {rank_path}")
                        
                        # 复制代码功能
                        if "ts_code" in df_rank_results.columns:
                            codes_rank = df_rank_results["ts_code"].astype(str).tolist()
                            txt_rank = _codes_to_txt(codes_rank, st.session_state["export_pref"]["style"], 
                                                   st.session_state["export_pref"]["with_suffix"])
                            copy_txt_button(txt_rank, label="📋 复制预测排名代码", key=f"copy_predict_rank_{ref_use_rank}")
                        
                        # 下载
                        csv_rank = df_rank_results.to_csv(index=False).encode("utf-8-sig")
                        st.download_button("导出 CSV", data=csv_rank, 
                                         file_name=os.path.basename(rank_path) if rank_path else f"predict_rank_{ref_use_rank}.csv", 
                                         mime="text/csv", width='stretch')
                    else:
                        st.warning("⚠️ 未生成排名结果，请检查数据")
                        
                except Exception as e:
                    st.error(f"运行失败：{e}")
                    import traceback
                    with st.expander("调试信息", expanded=False):
                        st.code(traceback.format_exc(), language="text")
            else:
                st.warning("请检查参数设置，确保参考日和股票范围都有效")

    # ================== 规则编辑辅助模块 ==================
    with tab_rules:
        render_rule_editor()
        
        st.markdown("---")
        
        # ================== 策略语法检查 ==================
        st.subheader("策略语法检查器")
        st.info("自动检查本地策略文件的语法错误、必填字段、表达式有效性等")
        
        with st.expander("使用方法 / 字段说明", expanded=False):
            st.markdown("""
            **策略语法检查器功能：**
            
            1. **自动文件定位** - 自动扫描并定位策略文件
            2. **语法验证** - 验证策略规则的语法和字段有效性
            3. **表达式检查** - 检查TDX表达式的正确性
            4. **字段验证** - 检查必填字段和字段类型
            5. **指标检查** - 验证指标依赖关系
            
            **支持的策略类型：**
            - 排名策略 (ranking)
            - 筛选策略 (filter)  
            - 模拟策略 (prediction)
            - 持仓策略 (position)
            - 买点策略 (opportunity)
            
            **检查内容：**
            - ✅ Python文件语法正确性
            - ✅ 策略列表结构正确性
            - ✅ 每个规则的字段和表达式
            - ✅ 必填字段完整性
            - ✅ 字段类型正确性
            - ✅ 表达式语法正确性
            - ✅ 支持的函数和变量
            - ✅ 缺失的数据列和指标
            """)

        # 导入验证器
        try:
            from rule_editor import validate_strategy_file
            validation_available = True
        except ImportError:
            st.error("策略验证器模块未找到，请确保 rule_editor.py 文件存在")
            validation_available = False
        
        if validation_available:
            # 自动定位策略文件
            strategy_files = []
            import glob
            import os
            
            # 按优先级搜索策略文件
            search_paths = [
                "strategies_repo.py",  # 当前目录
                "strategies/strategies_repo.py",  # strategies目录
                "**/strategies_repo.py",  # 递归搜索
            ]
            
            for pattern in search_paths:
                files = glob.glob(pattern, recursive=True)
                for file in files:
                    if os.path.isfile(file):
                        # 转换为绝对路径避免重复
                        abs_path = os.path.abspath(file)
                        if abs_path not in strategy_files:
                            strategy_files.append(abs_path)
            
            # 去重并排序
            strategy_files = sorted(strategy_files)
            
            if strategy_files:
                # 自动选择第一个文件（通常是主要的策略文件）
                default_file = strategy_files[0]
                
                if len(strategy_files) > 1:
                    selected_file = st.selectbox(
                        "选择策略文件",
                        strategy_files,
                        index=0,
                        help=f"自动定位到 {len(strategy_files)} 个策略文件，默认选择: {default_file}"
                    )
                else:
                    selected_file = default_file
                    st.info(f"自动定位到策略文件: {selected_file}")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    check_btn = st.button("🔍 检查语法", width='stretch')
                with col2:
                    if st.button("📄 查看文件内容", width='stretch'):
                        try:
                            with open(selected_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            st.code(content, language='python')
                        except Exception as e:
                            st.error(f"读取文件失败: {e}")
                
                if check_btn:
                    with st.spinner("正在检查策略文件..."):
                        try:
                            result = validate_strategy_file(selected_file)
                            
                            # 显示验证结果
                            if result.is_valid:
                                st.success("✅ 策略文件验证通过！")
                            else:
                                st.error("❌ 策略文件验证失败")
                            
                            # 显示错误信息
                            if result.errors:
                                st.markdown("#### 🚨 错误")
                                for error in result.errors:
                                    field_info = f" (字段: {error['field']})" if error.get('field') else ""
                                    st.error(f"• {error['message']}{field_info}")
                            
                            if result.warnings:
                                st.markdown("#### ⚠️ 警告")
                                for warning in result.warnings:
                                    field_info = f" (字段: {warning['field']})" if warning.get('field') else ""
                                    st.warning(f"• {warning['message']}{field_info}")
                            
                            # 显示建议
                            if result.suggestions:
                                st.markdown("#### 💡 建议")
                                for suggestion in result.suggestions:
                                    field_info = f" (字段: {suggestion['field']})" if suggestion.get('field') else ""
                                    st.info(f"• {suggestion['message']}{field_info}")
                            
                            # 显示缺失的列和指标
                            if result.missing_columns:
                                st.markdown("#### 📊 缺失的数据列")
                                st.warning(f"以下列在数据中不存在: {', '.join(result.missing_columns)}")
                            
                            if result.missing_indicators:
                                st.markdown("#### 🔧 缺失的指标")
                                st.warning(f"以下指标未注册: {', '.join(result.missing_indicators)}")
                            
                            if result.syntax_issues:
                                st.markdown("#### 🔍 语法问题")
                                for issue in result.syntax_issues:
                                    st.warning(f"• {issue}")
                            
                        except Exception as e:
                            import traceback
                            st.error(f"文件验证出错: {e}")
                            # 显示详细的错误信息
                            with st.expander("查看详细错误信息", expanded=False):
                                st.code(traceback.format_exc(), language="text")
            else:
                st.warning("未找到策略文件，请确保 strategies_repo.py 文件存在")

    # ================== 强度榜 ==================
    with tab_attn:
        st.subheader("强度榜")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            src = st.selectbox("来源", ["top","white","black","attention"], index=0)
            method = st.selectbox("方法", ["强度（带权）","次数（不带权）"], index=0)
        with c2:
            win_n = st.number_input("窗口天数 N", min_value=1, max_value=365, value=60)
            top_m = st.number_input("Top-M 过滤（仅统计前 M 名）", min_value=1, max_value=5000, value=3000)
        with c3:
            weight = st.selectbox("时间权重", ["不加权","指数半衰","线性最小值"], index=1)
            out_n = st.number_input("输出 Top-N", min_value=1, max_value=1000, value=100)
        with c4:
            # date_end = st.text_input("结束日（YYYYMMDD；留空=自动最新）", value="")
            date_end = st.text_input("结束日（YYYYMMDD；留空=自动最新）", value="", key="attn_end_date")
        gen_btn = st.button("生成并预览", width='stretch', key="btn_attn_preview")

        if gen_btn:
            try:
                # 1) 计算 start/end（按交易日）
                days = _cached_trade_dates(DATA_ROOT, API_ADJ)
                end = (date_end or (days[-1] if days else None))
                if not end:
                    st.error("未能确定结束日");
                if end in days:
                    j = days.index(end)
                    start = days[max(0, j - int(win_n))]
                else:
                    start = days[-int(win_n)] if days else end

                # 2) 参数映射
                mode_map = {"强度（带权）": "strength", "次数（不带权）": "hits"}
                w_map    = {"不加权": "none", "指数半衰": "exp", "线性最小值": "linear"}

                # 3) 正确调用 scoring_core 接口
                csv_path = se.build_category_rank(
                    category_type="strength",
                    start=start, end=end, source=src,
                    min_hits=None, topN=int(out_n), write=True,
                    mode=mode_map[method], weight_mode=w_map[weight],
                    topM=int(top_m)
                )
                st.success(f"衍生榜（强度榜）已生成：{csv_path}")
                df_a = pd.read_csv(csv_path)
                df_a = add_concept_column(df_a, blacklist=CONCEPT_BLACKLIST)
                st.dataframe(df_a, width='stretch', height=480)
                try:
                    if df_a is not None and not df_a.empty:
                        # 识别代码列（优先 ts_code）
                        code_col = None
                        for cand in ["ts_code", "code", "ts", "symbol"]:
                            if cand in df_a.columns:
                                code_col = cand
                                break

                        if code_col:
                            codes = df_a[code_col].astype(str).tolist()
                            txt = _codes_to_txt(
                                codes,
                                st.session_state["export_pref"]["style"],
                                st.session_state["export_pref"]["with_suffix"]
                            )
                            # 复制按钮（使用已有的 copy_txt_button）
                            copy_txt_button(
                                txt,
                                label="📋 复制强度榜（按当前输出）",
                                key=f"copy_attn_{end}_{src}"
                            )
                            # TXT 导出（文件名含参数，便于追溯）
                            _download_txt(
                                "导出衍生榜 TXT",
                                txt,
                                f"attention_{src}_{mode_map[method]}_{w_map[weight]}_{start}_{end}.txt",
                                key="dl_attention_txt"
                            )
                        else:
                            st.caption("未找到代码列（期望列名：ts_code）。")
                except Exception as e:
                    st.warning(f"导出/复制失败：{e}")
                    
                # —— 以下为“强度榜落盘（CSV/TXT，含清晰文件名）” 
                save_extra = cfg_bool("SC_ATTENTION_SAVE_EXTRA", False)
                if save_extra:
                    try:
                        fname_base = f"attention_{src}_{mode_map[method]}_{w_map[weight]}_win{int(win_n)}_topM{int(top_m)}_{start}_{end}_topN{int(out_n)}"
                        dest_csv = ATTN_DIR / f"{fname_base}.csv"
                        dest_txt = ATTN_DIR / f"{fname_base}.txt"

                        # 1) 复制 CSV（若名字不同）
                        try:
                            if str(csv_path) != str(dest_csv):
                                shutil.copyfile(csv_path, dest_csv)
                        except Exception as _e:
                            st.warning(f"CSV 落盘失败（不影响页面预览）：{_e}")

                        # 2) 写 TXT（只有前面生成过 txt 才写）
                        if 'txt' in locals():
                            try:
                                dest_txt.write_text(txt, encoding="utf-8-sig")
                            except Exception as _e:
                                st.warning(f"TXT 落盘失败（不影响页面预览）：{_e}")

                        st.caption(f"已落盘：{dest_csv.name} / {dest_txt.name}（目录：{ATTN_DIR}）")
                    except Exception as _e:
                        st.warning(f"衍生榜落盘出现异常：{_e}")

            except Exception as e:
                st.error(f"生成失败：{e}")

        # ===== 排名涨速榜 =====
        st.subheader("排名涨速榜")
        s1, s2, s3 = st.columns([2, 2, 1])
        with s1:
            speed_end = st.text_input("观察日（YYYYMMDD）", value=(date_end or ""), key="speed_end_date")
        with s2:
            speed_n = st.number_input("涨速窗口 N（日）", min_value=1, max_value=180, value=3, key="speed_win_n")
        with s3:
            speed_topn = st.number_input("输出 Top-N", min_value=5, max_value=1000, value=100, key="speed_topn")
        btn_speed = st.button("生成涨速榜", key="btn_speed_rank", width='stretch')
        st.caption("计算方式：排名映射为贡献度（对数+二次积分，100 名内封顶0.33），贡献度差值越大表示排名提升越快。")

        def _speed_file_path(start_date: str, end_date: str, n_days: int) -> Path:
            return SPEED_DIR / f"speed_rank_{start_date}_{end_date}_n{int(n_days)}.csv"

        def _ensure_speed_dir():
            try:
                SPEED_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                st.warning(f"创建涨速榜目录失败：{e}")

        def _load_rank(date_str: str) -> pd.DataFrame:
            path = ALL_DIR / f"score_all_{date_str}.csv"
            return _read_df(path, dtype={"ts_code": str})

        def _ensure_rank(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()
            _df = _normalize_rank_date_cols(df)
            if "rank" not in _df.columns:
                if "score" not in _df.columns:
                    return pd.DataFrame()
                _df = _df.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
                _df["rank"] = np.arange(1, len(_df) + 1)
            _df["rank"] = pd.to_numeric(_df["rank"], errors="coerce")
            return _df[["ts_code", "rank"]].dropna()

        SPEED_RANK_MAX_DEFAULT = 6000.0  # 默认榜尾名次，用于计算贡献度的对数刻度
        SPEED_RANK_FLAT = 100.0          # 100 名以内贡献封顶
        SPEED_CONTRIB_CAP = 0.33         # 贡献封顶值（与前期讨论保持一致）

        def _rank_contribution_score(rank_series: pd.Series, *, max_rank: float | None = None) -> pd.Series:
            """将排名映射到贡献度（封顶），越靠前贡献越高，100 名内平坦。"""
            if rank_series is None or len(rank_series) == 0:
                return pd.Series([], dtype=float)
            # 选用数据中的最大排名与默认值的较大者，保持对数尺度稳定
            max_in_series = pd.to_numeric(rank_series, errors="coerce").max()
            _max_rank = max(
                float(max_rank) if max_rank is not None and np.isfinite(max_rank) else 0.0,
                float(max_in_series) if pd.notna(max_in_series) else 0.0,
                SPEED_RANK_MAX_DEFAULT,
                SPEED_RANK_FLAT + 1.0
            )
            denom = math.log(_max_rank / SPEED_RANK_FLAT) if _max_rank > SPEED_RANK_FLAT else 1.0

            def _calc(val):
                if not np.isfinite(val):
                    return np.nan
                r = max(float(val), 1.0)
                x = math.log(_max_rank / r) / denom if denom > 0 else 0.0
                x = min(max(x, 0.0), 1.0)
                contrib = (x ** 2) / 2.0
                return min(contrib, SPEED_CONTRIB_CAP)

            return pd.to_numeric(rank_series, errors="coerce").apply(_calc)

        def _build_speed_board(start_use: str, end_use: str, n_days: int, *, reuse_file: bool = True, save_file: bool = True) -> tuple[pd.DataFrame, Path]:
            """返回涨速榜数据及文件路径，优先复用落盘文件。"""
            if not start_use or not end_use:
                return pd.DataFrame(), None
            file_path = _speed_file_path(start_use, end_use, n_days)
            if reuse_file and file_path.exists():
                try:
                    df_cached = _read_df(file_path, dtype={"ts_code": str})
                    if not df_cached.empty and {"contrib_delta"}.issubset(df_cached.columns):
                        return df_cached, file_path
                except Exception as e:
                    st.warning(f"读取缓存涨速榜失败（将重算）：{e}")

            df_end = _ensure_rank(_load_rank(end_use))
            df_start = _ensure_rank(_load_rank(start_use))
            if df_end.empty or df_start.empty:
                st.warning(f"缺少 {start_use} 或 {end_use} 的排名数据，无法计算涨速。")
                return pd.DataFrame(), file_path

            df_speed = (df_start.rename(columns={"rank": "rank_old"})
                                .merge(df_end.rename(columns={"rank": "rank_new"}), on="ts_code"))
            if df_speed.empty:
                st.warning("两日没有交集代码，无法计算涨速。")
                return pd.DataFrame(), file_path

            n_days_float = float(n_days)
            df_speed["rank_delta"] = df_speed["rank_old"] - df_speed["rank_new"]

            max_rank_for_scale = pd.concat([df_speed["rank_old"], df_speed["rank_new"]], ignore_index=True).max()
            df_speed["contrib_old"] = _rank_contribution_score(df_speed["rank_old"], max_rank=max_rank_for_scale)
            df_speed["contrib_new"] = _rank_contribution_score(df_speed["rank_new"], max_rank=max_rank_for_scale)
            df_speed["contrib_delta"] = (df_speed["contrib_new"] - df_speed["contrib_old"]).round(6)
            df_speed["speed_norm"] = (df_speed["contrib_delta"] / SPEED_CONTRIB_CAP).round(6)
            df_speed["start_date"] = start_use
            df_speed["end_date"] = end_use
            df_speed = df_speed.sort_values(
                ["contrib_delta", "rank_new", "rank_old", "ts_code"],
                ascending=[False, True, True, True]
            ).reset_index(drop=True)

            if save_file:
                _ensure_speed_dir()
                try:
                    df_speed.to_csv(file_path, index=False, encoding="utf-8-sig")
                except Exception as e:
                    st.warning(f"涨速榜落盘失败（不影响预览）：{e}")
            return df_speed, file_path

        def _build_concept_board_from_speed(df_base: pd.DataFrame, head_n: int, ref_date: str) -> pd.DataFrame:
            """基于涨速榜 Top-N 汇总概念强度，加入小样本置信收缩。"""
            if df_base is None or df_base.empty or head_n <= 0:
                return pd.DataFrame()
            top_df = df_base.head(int(head_n)).copy()
            top_df = add_concept_column(top_df, blacklist=CONCEPT_BLACKLIST, out_col="概念")
            if "概念" not in top_df.columns:
                return pd.DataFrame()
            # 采用总贡献度增量作为概念强度基准
            prior_speed = pd.to_numeric(df_base.get("contrib_delta"), errors="coerce").mean()

            rank_meta = _read_rank_all_sorted(ref_date) if ref_date else pd.DataFrame()
            if not rank_meta.empty:
                keep_cols = ["ts_code"]
                for c in ["name", "score", "tiebreak_j", "tiebreak", "rank"]:
                    if c in rank_meta.columns:
                        keep_cols.append(c)
                rank_meta = rank_meta[keep_cols].copy()
                if "rank" in rank_meta.columns:
                    rank_meta = rank_meta.rename(columns={"rank": "rank_meta"})
                top_df = top_df.merge(rank_meta, on="ts_code", how="left")

            tb_col = "tiebreak_j" if "tiebreak_j" in top_df.columns else ("tiebreak" if "tiebreak" in top_df.columns else None)
            concept_bucket: Dict[str, List[Dict[str, Any]]] = {}

            for _, r in top_df.iterrows():
                concepts = [c for c in str(r.get("概念", "")).split(",") if c]
                if not concepts:
                    continue
                tb_val = None
                if tb_col and pd.notna(r.get(tb_col)):
                    tb_val = pd.to_numeric(r.get(tb_col), errors="coerce")
                    concept_item = {
                        "ts_code": r.get("ts_code", ""),
                        "name": r.get("name", ""),
                        "speed": float(r.get("contrib_delta", 0) or 0),
                        "rank_new": pd.to_numeric(r.get("rank_new"), errors="coerce"),
                    "rank_old": pd.to_numeric(r.get("rank_old"), errors="coerce"),
                    "rank_meta": pd.to_numeric(r.get("rank_meta"), errors="coerce"),
                    "rank_delta": float(r.get("rank_delta", 0) or 0),
                    "tiebreak": tb_val,
                }
                for c in concepts:
                    concept_bucket.setdefault(c, []).append(concept_item)

            concept_rows: List[Dict[str, Any]] = []
            for cname, items in concept_bucket.items():
                df_c = pd.DataFrame(items)
                if df_c.empty:
                    continue
                df_c["rank_for_sort"] = df_c["rank_new"].fillna(df_c["rank_meta"])
                df_c["rank_for_sort"] = pd.to_numeric(df_c["rank_for_sort"], errors="coerce")
                df_c["_tb_for_sort"] = pd.to_numeric(df_c["tiebreak"], errors="coerce")
                df_c["rank_for_sort"] = df_c["rank_for_sort"].fillna(np.inf)
                df_c["_tb_for_sort"] = df_c["_tb_for_sort"].fillna(np.inf)

                df_c_sorted = df_c.sort_values(
                    ["rank_for_sort", "_tb_for_sort", "ts_code"],
                    ascending=[True, True, True]
                )
                stocks_str = "; ".join(
                    [
                        f"{row.ts_code}({int(row.rank_for_sort) if np.isfinite(row.rank_for_sort) else '-'})"
                        for row in df_c_sorted.itertuples()
                    ]
                )
                concept_rows.append(
                    {
                        "概念": cname,
                        "综合强度": df_c["speed"].sum(),
                        "平均涨速": _shrink_mean(df_c["speed"].mean(), len(df_c), prior_speed, CONCEPT_SHRINK_ALPHA),
                        "覆盖数": len(df_c),
                        "最佳排名": int(df_c_sorted.iloc[0]["rank_for_sort"]) if np.isfinite(df_c_sorted.iloc[0]["rank_for_sort"]) else "",
                        "最佳tiebreak": float(df_c_sorted.iloc[0]["_tb_for_sort"]) if np.isfinite(df_c_sorted.iloc[0]["_tb_for_sort"]) else np.inf,
                        "股票": stocks_str,
                    }
                )

            concept_df = pd.DataFrame(concept_rows)
            if concept_df.empty:
                return concept_df
            concept_df = concept_df.sort_values(
                ["平均涨速", "综合强度", "最佳排名", "最佳tiebreak", "覆盖数", "概念"],
                ascending=[False, False, True, True, False, True]
            )
            return concept_df.reset_index(drop=True)

        def _render_speed_outputs(df_speed: pd.DataFrame, start_use: str, end_use: str, speed_n: float, rows_eff: int):
            """常驻渲染涨速榜及概念榜。"""
            if df_speed is None or df_speed.empty:
                return
            with st.container(border=True):
                st.caption(f"窗口：{start_use} → {end_use}，N={int(speed_n)}")
                preview_speed = add_concept_column(df_speed.head(rows_eff), blacklist=CONCEPT_BLACKLIST)
                st.dataframe(preview_speed, width='stretch', height=420)
                if "ts_code" in df_speed.columns:
                    codes = df_speed["ts_code"].astype(str).head(rows_eff).tolist()
                    txt = _codes_to_txt(
                        codes,
                        st.session_state["export_pref"]["style"],
                        st.session_state["export_pref"]["with_suffix"]
                    )
                    copy_txt_button(txt, label="复制以上", key=f"copy_speed_{start_use}_{end_use}")

        if btn_speed:
            try:
                days = _cached_trade_dates(DATA_ROOT, API_ADJ) or []
                if not days:
                    st.warning("无法获取交易日历。"); raise Exception("no trade dates")
                # 处理观察日
                if speed_end and speed_end in days:
                    end_idx = days.index(speed_end)
                    end_use = speed_end
                else:
                    end_idx = len(days) - 1
                    end_use = days[end_idx]
                    if speed_end and speed_end not in days:
                        st.info(f"观察日不在交易日历中，已改用最近交易日：{end_use}")
                rows_eff = int(speed_topn)
                results_multi = []
                saved_files = []
                for n_win in range(int(speed_n), 0, -1):
                    start_idx = end_idx - int(n_win)
                    if start_idx < 0:
                        st.warning(f"交易日不足以回溯 N={n_win}，已跳过。")
                        continue
                    start_use = days[start_idx]
                    df_speed, fp = _build_speed_board(start_use, end_use, n_win, reuse_file=True, save_file=True)
                    if df_speed is None or df_speed.empty:
                        continue
                    results_multi.append({
                        "df_speed": df_speed,
                        "start_use": start_use,
                        "end_use": end_use,
                        "speed_n": n_win,
                        "rows_eff": rows_eff,
                        "file_path": fp,
                    })
                    if fp:
                        saved_files.append(fp.name)

                if results_multi:
                    st.session_state["speed_results"] = results_multi
                    for res in results_multi:
                        _render_speed_outputs(
                            res["df_speed"],
                            res["start_use"],
                            res["end_use"],
                            res["speed_n"],
                            res["rows_eff"],
                        )
                    if saved_files:
                        st.caption(f"已落盘：{', '.join(saved_files)}（目录：{SPEED_DIR}）")
                else:
                    st.warning("未能生成任何涨速榜，请检查参考日和排名数据。")
            except Exception:
                pass
        elif "speed_results" in st.session_state:
            res_list = st.session_state.get("speed_results") or []
            for res in res_list:
                _render_speed_outputs(
                    res.get("df_speed"),
                    res.get("start_use", ""),
                    res.get("end_use", ""),
                    res.get("speed_n", speed_n),
                    int(res.get("rows_eff", speed_topn)),
                )

        # ===== 板块榜（挪入衍生榜） =====
        st.subheader("板块榜")
        with st.container(border=True):
            c1, c2 = st.columns([2, 1])
            with c1:
                ref_sector = st.text_input("参考日（YYYYMMDD；留空=自动最新）", value=_get_latest_date_from_files() or "", key="sector_ref_date")
            with c2:
                sector_speed_n = st.number_input("涨速最大 N（日）", min_value=1, max_value=30, value=5, key="sector_speed_n")
            gen_sector_btn = st.button("生成板块榜", width="stretch", key="sector_generate")
        
        if gen_sector_btn:
            ref_real = ref_sector.strip() or _get_latest_date_from_files() or ""
            if not ref_real:
                st.warning("未能确定参考日，无法生成板块榜。")
            else:
                days = _cached_trade_dates(DATA_ROOT, API_ADJ) or []
                if not days:
                    st.warning("无法获取交易日历。")
                else:
                    if ref_real in days:
                        end_idx = days.index(ref_real)
                        end_use = ref_real
                    else:
                        end_idx = len(days) - 1
                        end_use = days[end_idx]
                        st.info(f"参考日不在交易日历中，已改用最近交易日：{end_use}")

                    results_speed = []
                    saved_speed_files = []
                    for n_win in range(int(sector_speed_n), 0, -1):
                        start_idx = end_idx - int(n_win)
                        if start_idx < 0:
                            st.warning(f"交易日不足以回溯 N={n_win}，已跳过。")
                            continue
                        start_use = days[start_idx]
                        df_speed, fp_speed = _build_speed_board(start_use, end_use, n_win, reuse_file=True, save_file=True)
                        if df_speed is None or df_speed.empty:
                            continue
                        concept_board_n = _build_concept_board_from_speed(df_speed, head_n=len(df_speed), ref_date=end_use)
                        if not concept_board_n.empty:
                            concept_board_n.insert(0, "窗口N", int(n_win))
                            results_speed.append(concept_board_n)
                        if fp_speed:
                            saved_speed_files.append(fp_speed.name)

                    if results_speed:
                        with st.expander("各窗口涨速概念榜（按窗口分组，不加权）", expanded=False):
                            for df_cb in results_speed:
                                n_win = int(df_cb.iloc[0]["窗口N"]) if not df_cb.empty else 0
                                st.markdown(f"**窗口 N={n_win}**")
                                st.dataframe(df_cb.head(100), width="stretch", height=360)
                        try:
                            merged_speed = pd.concat(results_speed, ignore_index=True)
                            st.markdown("**合并视图（含窗口N，按窗口分开）**")
                            st.dataframe(merged_speed.head(200), width="stretch", height=420)
                        except Exception:
                            pass
                    else:
                        st.info("未生成涨速概念榜。")

                    if saved_speed_files:
                        st.caption(f"涨速榜已落盘：{', '.join(saved_speed_files)}（目录：{SPEED_DIR}）")

        st.subheader("本地读取")

        c1, c2 = st.columns([1,1])
        with c1:
            ref_inp_attn = st.text_input("参考日（YYYYMMDD；留空=自动取最新）", value="", key="attn_ref_input")
        with c2:
            sort_key = st.selectbox("排序依据", ["score ↓", "rank ↑", "保持原文件顺序"], index=0, key="attn_sort_key")
        topn_attn = st.number_input("Top-K 显示行数", min_value=5, max_value=1000, value=cfg_int("SC_ATTENTION_TOP_K", 50), key="attn_topn")
        # 决定参考日与文件路径
        ref_attn = (ref_inp_attn.strip() or _pick_latest_attn_date())
        if not ref_attn:
            st.info("未在 attention 目录发现任何 CSV，请先产出强度榜或检查输出路径。")
        else:
            attn_path = _find_attn_file_by_date(ref_attn)
            st.caption(f"参考日：{ref_attn}")
            if not attn_path or (not attn_path.exists()):
                st.warning("未找到该日的强度榜文件。")
            else:
                # 读取强度榜
                df_attn = _read_df(attn_path)
                if df_attn is None or df_attn.empty:
                    st.warning("强度榜文件为空或无法读取。")

        # 只有在有有效数据时才进行排序和显示
        if 'df_attn' in locals() and df_attn is not None and not df_attn.empty:
            # 统一/容错排序：默认优先按 score 降序，同分时按J值升序；没有 score 则按 rank 升序；否则保持原顺序
            def _auto_sort(df: pd.DataFrame) -> pd.DataFrame:
                if "score" in df.columns:
                    if "tiebreak_j" in df.columns:
                        return df.sort_values(["score", "tiebreak_j"], ascending=[False, True])
                    else:
                        return df.sort_values(["score"], ascending=[False])
                if "rank" in df.columns:
                    return df.sort_values(["rank"], ascending=[True])
                return df

            if sort_key == "score ↓" and "score" in df_attn.columns:
                if "tiebreak_j" in df_attn.columns:
                    df_attn = df_attn.sort_values(["score", "tiebreak_j"], ascending=[False, True])
                else:
                    df_attn = df_attn.sort_values(["score"], ascending=[False])
            elif sort_key == "rank ↑" and "rank" in df_attn.columns:
                df_attn = df_attn.sort_values(["rank"], ascending=[True])
            # "保持原文件顺序" 就不动

            # 预览 + 导出/复制，行为与"排名"页尽量一致
            st.divider()
            with st.container(border=True):
                rows_eff = int(topn_attn)
                st.markdown("**强度榜 Top-N 预览**")
                preview_attn = add_concept_column(df_attn.head(rows_eff), blacklist=CONCEPT_BLACKLIST)
                st.dataframe(preview_attn, width='stretch', height=420)

                # TXT 复制（按你的导出偏好）
                if "ts_code" in df_attn.columns:
                    codes = df_attn["ts_code"].astype(str).head(rows_eff).tolist()
                    txt = _codes_to_txt(
                        codes,
                        st.session_state["export_pref"]["style"],
                        st.session_state["export_pref"]["with_suffix"]
                    )
                    copy_txt_button(txt, label="复制以上", key=f"copy_attn_{ref_attn}")

        # # --- 轻量：前几日 Top-K 扫描（只看 Top，不算强度） ---
        # with st.expander("前几日 Top-K 扫描（轻量）", expanded=True):
        #     # —— 参数区 ——
        #     c1, c2, c3, c4 = st.columns(4)
        #     with c1:
        #         end_use = st.text_input("观察日（YYYYMMDD）", value=_get_latest_date_from_database() or "", key="lite_end")
        #     with c2:
        #         lookback_days = st.number_input("回看天数 D（不含今天）", min_value=1, max_value=60, value=3, key="lite_D")
        #     with c3:
        #         k_min = st.number_input("K 最小（含）", min_value=1, max_value=10000, value=1, key="lite_kmin")
        #     with c4:
        #         k_max = st.number_input("K 最大（含）", min_value=1, max_value=10000, value=50, key="lite_kmax")

        #     c5, c6, c7 = st.columns(3)
        #     with c5:
        #         hit_mode = st.selectbox(
        #             "命中口径",
        #             ["与今天Top交集", "累计上榜次数≥N", "连续上榜天数≥N"],
        #             index=0, key="lite_mode"
        #         )
        #     with c6:
        #         n_th = st.number_input("N（阈值）", min_value=1, max_value=60, value=2, key="lite_N",
        #                             disabled=(hit_mode == "与今天Top交集"))
        #     with c7:
        #         today_topk = st.number_input("今天对比 Top-K", min_value=1, max_value=5000,
        #                                     value=50, key="lite_todayK",
        #                                     disabled=(hit_mode != "与今天Top交集"))

        #     limit = st.number_input("输出条数上限", min_value=5, max_value=2000, value=200, key="lite_limit")

        #     btn_lite_calc = st.button("计算", width='stretch', key="btn_lite_calc")

        #     if btn_lite_calc:
        #         try:
        #             days = _cached_trade_dates(DATA_ROOT, API_ADJ) or []
        #             if not days:
        #                 st.warning("无法获取交易日历。");
        #             # 观察日处理：若手填不在交易日里，取最近一个交易日
        #             if not end_use or end_use not in days:
        #                 end_idx = len(days) - 1
        #                 end = days[end_idx]
        #                 if end_use and end_use not in days:
        #                     st.caption(f"观察日不在交易日历内，已改用最近交易日：{end}")
        #             else:
        #                 end_idx = days.index(end_use)
        #                 end = end_use
        #             if end_idx <= 0:
        #                 st.info("观察日前没有更早交易日可统计。");

        #             # 回看窗口（不含今天 end）
        #             D = int(lookback_days)
        #             start_idx = max(0, end_idx - D)
        #             win_days = days[start_idx:end_idx]  # t-D ~ t-1

        #             # K 范围校正
        #             k1, k2 = int(min(k_min, k_max)), int(max(k_min, k_max))

        #             # —— 汇总前 D 日 Top-K（K∈[k1,k2]）——
        #             occ = {}             # 累计命中次数
        #             best_rank = {}       # 窗口内最好名次（越小越好）
        #             last_seen = {}       # 最近出现日
        #             day_index = {d:i for i,d in enumerate(win_days)}   # 便于算连续
        #             appear_idx = {}      # ts_code -> 出现的日序号列表

        #             for d in win_days:
        #                 df = _slice_top_from_all(d, k2)
        #                 if df is None or df.empty:
        #                     continue
        #                 df = df[(df["rank"] >= k1) & (df["rank"] <= k2)]
        #                 for ts, rk in zip(df["ts_code"].astype(str), df["rank"].astype(int)):
        #                     occ[ts] = occ.get(ts, 0) + 1
        #                     best_rank[ts] = min(best_rank.get(ts, 10**9), rk)
        #                     last_seen[ts] = d if (ts not in last_seen or d > last_seen[ts]) else last_seen[ts]
        #                     appear_idx.setdefault(ts, []).append(day_index[d])

        #             # 连续天数（窗口内的最大连续段）
        #             max_streak = {}
        #             for ts, idxs in appear_idx.items():
        #                 idxs = sorted(set(idxs))
        #                 if not idxs:
        #                     max_streak[ts] = 0
        #                     continue
        #                 best = cur = 1
        #                 for a, b in zip(idxs, idxs[1:]):
        #                     if b == a + 1:
        #                         cur += 1
        #                         best = max(best, cur)
        #                     else:
        #                         cur = 1
        #                 max_streak[ts] = best

        #             # 汇总 DataFrame（只含在窗口内出现过的）
        #             rows = []
        #             for ts in sorted(occ.keys()):
        #                 rows.append({
        #                     "ts_code": ts,
        #                     "prev_hits": int(occ.get(ts, 0)),
        #                     "max_streak": int(max_streak.get(ts, 0)),
        #                     "best_rank_prev": int(best_rank.get(ts, 10**9)),
        #                     "last": last_seen.get(ts, None),
        #                 })
        #             df_prev = pd.DataFrame(rows).sort_values(
        #                 ["prev_hits","best_rank_prev","ts_code"], ascending=[False, True, True]
        #             ).reset_index(drop=True)

        #             # —— 命中计算 —— 
        #             if hit_mode == "与今天Top交集":
        #                 df_today = _slice_top_from_all(end, int(today_topk))
        #                 if df_today is None or df_today.empty:
        #                     st.info(f"{end} 的全量排名文件不存在或为空。");
        #                     hit = pd.DataFrame()
        #                 else:
        #                     today_set = set(df_today["ts_code"].astype(str))
        #                     hit = df_prev[df_prev["ts_code"].astype(str).isin(today_set)].copy()
        #                     hit = hit.merge(
        #                         df_today[["ts_code","rank"]].rename(columns={"rank":"today_rank"}),
        #                         on="ts_code", how="left"
        #                     ).sort_values(
        #                         ["prev_hits","today_rank","best_rank_prev","ts_code"],
        #                         ascending=[False, True, True, True]
        #                     ).head(int(limit))
        #                 st.markdown(
        #                     f"**窗口：{win_days[0] if win_days else '—'} ~ {win_days[-1] if win_days else '—'}（不含今天 {end}）｜K∈[{k1},{k2}]，今天对比 Top-K={int(today_topk)}**"
        #                 )
        #                 st.markdown("**命中：与今天 Top 交集（延续/再上榜）**")
        #                 if hit.empty:
        #                     st.caption("无命中。")
        #                 else:
        #                     hit = add_concept_column(hit, blacklist=CONCEPT_BLACKLIST)
        #                     st.dataframe(hit, width='stretch', height=360)
        #                     # 复制代码
        #                     codes = hit["ts_code"].astype(str).tolist()
        #                     txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"],
        #                                         st.session_state["export_pref"]["with_suffix"])
        #                     copy_txt_button(txt, label="📋 复制命中代码", key=f"copy_lite_inter_{end}")

        #             elif hit_mode == "累计上榜次数≥N":
        #                 hit = df_prev[df_prev["prev_hits"] >= int(n_th)].copy()
        #                 hit = hit.sort_values(
        #                     ["prev_hits","best_rank_prev","ts_code"],
        #                     ascending=[False, True, True]
        #                 ).head(int(limit))
        #                 st.markdown(
        #                     f"**窗口：{win_days[0] if win_days else '—'} ~ {win_days[-1] if win_days else '—'}（不含今天 {end}）｜K∈[{k1},{k2}]**"
        #                 )
        #                 st.markdown(f"**命中：累计上榜次数 ≥ {int(n_th)}**")
        #                 if hit.empty:
        #                     st.caption("无命中。")
        #                 else:
        #                     st.dataframe(hit, width='stretch', height=360)
        #                     codes = hit["ts_code"].astype(str).tolist()
        #                     txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"],
        #                                         st.session_state["export_pref"]["with_suffix"])
        #                     copy_txt_button(txt, label="📋 复制命中代码", key=f"copy_lite_cnt_{end}")

        #             else:  # 连续上榜天数≥N
        #                 hit = df_prev[df_prev["max_streak"] >= int(n_th)].copy()
        #                 hit = hit.sort_values(
        #                     ["max_streak","best_rank_prev","ts_code"],
        #                     ascending=[False, True, True]
        #                 ).head(int(limit))
        #                 st.markdown(
        #                     f"**窗口：{win_days[0] if win_days else '—'} ~ {win_days[-1] if win_days else '—'}（不含今天 {end}）｜K∈[{k1},{k2}]**"
        #                 )
        #                 st.markdown(f"**命中：连续上榜天数 ≥ {int(n_th)}**")
        #                 if hit.empty:
        #                     st.caption("无命中。")
        #                 else:
        #                     st.dataframe(hit, width='stretch', height=360)
        #                     codes = hit["ts_code"].astype(str).tolist()
        #                     txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"],
        #                                         st.session_state["export_pref"]["with_suffix"])
        #                     copy_txt_button(txt, label="📋 复制命中代码", key=f"copy_lite_streak_{end}")

        #         except Exception as e:
        #             st.error(f"计算失败：{e}")

            # # CSV 下载（Top-N）
            # st.download_button(
            #     "⬇️ 导出 Top-N（CSV）",
            #     data=df_attn.head(rows_eff).to_csv(index=False).encode("utf-8-sig"),
            #     file_name=f"attention_top{rows_eff}_{ref_attn}.csv",
            #     width='stretch',
            #     key=f"dl_attn_{ref_attn}"
            # )

    # ================== 自选榜 ==================
    with tab_custom:
        st.subheader("自选榜（策略组合）")
        
        # 从 strategies_repo.py 加载策略组合
        def load_custom_combos():
            """从 strategies_repo.py 加载策略组合配置，转换为字典格式（name -> combo_data）"""
            try:
                from strategies_repo import CUSTOM_COMBOS
                if isinstance(CUSTOM_COMBOS, list):
                    # 列表格式转换为字典格式
                    return {combo.get('name', ''): combo for combo in CUSTOM_COMBOS if combo.get('name')}
                elif isinstance(CUSTOM_COMBOS, dict):
                    # 兼容旧格式（字典）
                    return dict(CUSTOM_COMBOS)
                return {}
            except Exception as e:
                logger.warning(f"加载策略组合失败: {e}")
                return {}
        
        def save_custom_combos(combos: dict):
            """保存策略组合配置到 strategies_repo.py"""
            try:
                import re
                from pathlib import Path
                
                repo_path = Path("strategies_repo.py")
                if not repo_path.exists():
                    st.error("找不到 strategies_repo.py 文件")
                    return False
                
                # 读取文件内容
                content = repo_path.read_text(encoding="utf-8-sig")
                
                # 构建新的 CUSTOM_COMBOS 内容（列表格式）
                new_combos_str = "# 自选榜策略组合配置\n"
                new_combos_str += "# 格式：列表，每个元素包含 name（组合名称）、rules（策略名列表）、agg_mode（聚合方法：OR/AND）、output_name（落盘名称）、explain（说明）、exclude_rules（排除策略列表）；可选 rule_groups（混合逻辑组）与 group_mode（组间聚合）\n"
                new_combos_str += "CUSTOM_COMBOS = [\n"
                for combo_name, combo_data in combos.items():
                    rules = combo_data.get("rules", [])
                    agg_mode = combo_data.get("agg_mode", "OR")
                    output_name = combo_data.get("output_name", combo_name)
                    explain = combo_data.get("explain", "")
                    exclude_rules = combo_data.get("exclude_rules", [])
                    rule_groups = combo_data.get("rule_groups") or combo_data.get("logic_groups") or []
                    group_mode = combo_data.get("group_mode")
                    new_combos_str += "    {\n"
                    new_combos_str += f"        'name': '{combo_name}',\n"
                    new_combos_str += f"        'rules': {repr(rules)},\n"
                    new_combos_str += f"        'agg_mode': '{agg_mode}',\n"
                    new_combos_str += f"        'output_name': '{output_name}',\n"
                    if explain:
                        new_combos_str += f"        'explain': '{explain}',\n"
                    if exclude_rules:
                        new_combos_str += f"        'exclude_rules': {repr(exclude_rules)},\n"
                    if rule_groups:
                        new_combos_str += f"        'rule_groups': {repr(rule_groups)},\n"
                    if group_mode:
                        new_combos_str += f"        'group_mode': '{group_mode}',\n"
                    new_combos_str += "    },\n"
                new_combos_str += "]\n"
                
                # 使用正则表达式替换 CUSTOM_COMBOS 部分（包括注释）
                # 匹配从注释开始到 CUSTOM_COMBOS 列表/字典结束的整个块
                # 使用平衡括号匹配来正确匹配嵌套的结构
                pattern = r"# 自选榜策略组合配置.*?CUSTOM_COMBOS\s*=\s*[\[{]"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    # 找到列表/字典开始位置
                    start_pos = match.end() - 1  # 回退到 [ 或 { 的位置
                    # 从开始位置找到匹配的结束括号
                    brace_count = 1
                    end_pos = start_pos + 1
                    open_char = content[start_pos]
                    close_char = ']' if open_char == '[' else '}'
                    while brace_count > 0 and end_pos < len(content):
                        if content[end_pos] == open_char:
                            brace_count += 1
                        elif content[end_pos] == close_char:
                            brace_count -= 1
                        end_pos += 1
                    # 替换整个块（包括注释）
                    content = content[:match.start()] + new_combos_str.rstrip() + content[end_pos:]
                else:
                    # 如果不存在，在 POSITION_POLICIES 后添加
                    # 找到 POSITION_POLICIES 的结束位置（包括空行）
                    pos_pattern = r"(POSITION_POLICIES\s*=\s*\[[^\]]*\]\s*\n)"
                    if re.search(pos_pattern, content, re.DOTALL):
                        # 在 POSITION_POLICIES 后添加
                        content = re.sub(
                            pos_pattern,
                            r"\1\n" + new_combos_str,
                            content,
                            flags=re.DOTALL
                        )
                    else:
                        # 在文件末尾添加
                        content += "\n\n" + new_combos_str
                
                # 写回文件
                repo_path.write_text(content, encoding="utf-8-sig")
                
                # 重新加载模块
                import importlib
                import strategies_repo
                importlib.reload(strategies_repo)
                
                return True
            except Exception as e:
                logger.error(f"保存策略组合失败: {e}")
                st.error(f"保存失败: {e}")
                return False
        
        def format_rule_groups(rule_groups: list, group_mode: str = "AND") -> str:
            """将混合逻辑组格式化为简短描述"""
            if not rule_groups:
                return ""
            parts = []
            for g in rule_groups:
                rules = g.get("rules", [])
                mode = g.get("mode", "OR")
                if rules:
                    parts.append(f"({mode}: {', '.join(rules[:3])}{'...' if len(rules) > 3 else ''})")
            if not parts:
                return ""
            connector = " AND " if str(group_mode).upper() == "AND" else " OR "
            return connector.join(parts)
        
        def parse_rule_groups_input(text: str, default_group_mode: str = "AND") -> tuple[list, str]:
            """解析高级逻辑输入，返回规范化的 rule_groups 与 group_mode"""
            import json
            if not text or not text.strip():
                return [], default_group_mode
            try:
                raw = json.loads(text)
            except Exception as e:
                raise ValueError(f"JSON 解析失败: {e}")
            
            group_mode = str(default_group_mode).upper()
            groups_raw = []
            if isinstance(raw, dict):
                group_mode = str(raw.get("group_mode", group_mode)).upper()
                groups_raw = raw.get("groups") or raw.get("rule_groups") or []
            elif isinstance(raw, list):
                groups_raw = raw
            else:
                raise ValueError("格式需为列表或包含 rule_groups 的字典")
            
            normalized = []
            for g in groups_raw:
                if not isinstance(g, dict):
                    continue
                rules = [str(r) for r in g.get("rules", []) if str(r)]
                if not rules:
                    continue
                mode = str(g.get("mode", "OR")).upper()
                if mode not in ("OR", "AND"):
                    mode = "OR"
                normalized.append({"mode": mode, "rules": rules})
            return normalized, group_mode
        
        # 获取所有规则名
        rule_names = _get_rule_names()
        
        # 策略组合管理
        with st.expander("策略组合管理", expanded=False):
            combos = load_custom_combos()
            
            # 显示现有组合
            if combos:
                st.markdown("**现有策略组合：**")
                for combo_name, combo_data in combos.items():
                    with st.container(border=True):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{combo_name}**")
                            rules_list = combo_data.get("rules", [])
                            agg_mode = combo_data.get("agg_mode", "OR")
                            explain = combo_data.get("explain", "")
                            exclude_rules_list = combo_data.get("exclude_rules", [])
                            rule_groups = combo_data.get("rule_groups") or combo_data.get("logic_groups") or []
                            group_mode = combo_data.get("group_mode", "AND")
                            st.caption(f"策略组: {', '.join(rules_list[:3])}{'...' if len(rules_list) > 3 else ''} | 聚合方法: {agg_mode}")
                            logic_desc = format_rule_groups(rule_groups, group_mode)
                            if logic_desc:
                                st.caption(f"混合逻辑: {logic_desc}（组间 {group_mode}）")
                            if exclude_rules_list:
                                st.caption(f"排除策略: {', '.join(exclude_rules_list[:3])}{'...' if len(exclude_rules_list) > 3 else ''}")
                            if explain:
                                st.caption(f"说明: {explain}")
                        with col2:
                            if st.button("删除", key=f"del_{combo_name}"):
                                del combos[combo_name]
                                if save_custom_combos(combos):
                                    st.success(f"已删除策略组合: {combo_name}")
                                    st.rerun()
                        with col3:
                            if st.button("使用", key=f"use_{combo_name}"):
                                st.session_state["selected_combo"] = combo_name
                                st.session_state["selected_rules"] = rules_list
                                st.session_state["selected_agg_mode"] = agg_mode
                                st.rerun()
            
            st.divider()
            
            # 新建/编辑策略组合
            st.markdown("**新建/编辑策略组合：**")
            combo_name_input = st.text_input("组合名称（name）", value="", key="combo_name_input", placeholder="例如：突破组合")
            selected_rules_input = st.multiselect("策略组（rules）", options=rule_names, default=[], key="combo_rules_input")
            agg_mode_input = st.radio("聚合方法（agg_mode）", ["OR（任一命中）", "AND（全部命中）"], index=0, horizontal=True, key="combo_agg_mode")
            exclude_rules_input = st.multiselect("排除策略（exclude_rules）", options=rule_names, default=[], key="combo_exclude_rules_input", help="如果触发这些策略则排除该股票")
            combo_output_name_input = st.text_input("落盘名称（output_name）", value="", key="combo_output_name_input", placeholder="例如：突破组合（用于生成文件名）")
            combo_explain_input = st.text_input("说明（explain）", value="", key="combo_explain_input", placeholder="例如：突破相关策略组合")
            combo_rule_groups_input = st.text_area(
                "高级：混合逻辑（rule_groups，JSON，可选）",
                value="",
                key="combo_rule_groups_input",
                placeholder='示例：[{"mode": "OR", "rules": ["b1买点", "量价共振_上涨放量"]},{"mode": "OR", "rules": ["底部放量", "长期成本线附近"]}]'
            )
            combo_group_mode_input = st.radio("组间逻辑（group_mode，作用于高级逻辑）", ["AND（全部组满足）", "OR（任一组满足）"], index=0, horizontal=True, key="combo_group_mode")
            
            col_save1, col_save2 = st.columns([1, 1])
            with col_save1:
                if st.button("保存策略组合", key="save_combo"):
                    group_mode_val = "AND" if combo_group_mode_input.startswith("AND") else "OR"
                    try:
                        rule_groups_data, parsed_group_mode = parse_rule_groups_input(combo_rule_groups_input, group_mode_val)
                        if parsed_group_mode:
                            group_mode_val = parsed_group_mode
                    except ValueError as e:
                        st.warning(f"高级逻辑解析失败：{e}")
                        st.stop()
                    
                    rules_to_save = list(selected_rules_input)
                    if not rules_to_save and rule_groups_data:
                        rules_to_save = sorted({r for g in rule_groups_data for r in g.get("rules", [])})
                    if not combo_name_input.strip():
                        st.warning("请输入组合名称")
                    elif not rules_to_save:
                        st.warning("请至少选择一个策略或填写高级逻辑")
                    else:
                        # 落盘名称默认为组合名称
                        output_name = combo_output_name_input.strip() if combo_output_name_input.strip() else combo_name_input.strip()
                        combo_data = {
                            "rules": rules_to_save,
                            "agg_mode": "OR" if agg_mode_input.startswith("OR") else "AND",
                            "output_name": output_name,
                            "explain": combo_explain_input.strip() if combo_explain_input.strip() else ""
                        }
                        if exclude_rules_input:
                            combo_data["exclude_rules"] = exclude_rules_input
                        if rule_groups_data:
                            combo_data["rule_groups"] = rule_groups_data
                            combo_data["group_mode"] = group_mode_val
                        combos[combo_name_input.strip()] = combo_data
                        if save_custom_combos(combos):
                            st.success(f"已保存策略组合: {combo_name_input.strip()}")
                            st.rerun()
            with col_save2:
                if st.button("清空输入", key="clear_combo"):
                    st.session_state["combo_name_input"] = ""
                    st.session_state["combo_rules_input"] = []
                    st.session_state["combo_exclude_rules_input"] = []
                    st.session_state["combo_output_name_input"] = ""
                    st.session_state["combo_explain_input"] = ""
                    st.session_state["combo_rule_groups_input"] = ""
                    st.session_state["combo_group_mode"] = "AND（全部组满足）"
                    st.rerun()
        
        st.divider()
        
        # 生成自选榜
        st.markdown("### 生成自选榜")
        
        # 加载repo中的策略组合
        combos_repo = load_custom_combos()
        selected_rule_groups = []
        selected_group_mode = "AND"
        rule_groups_prefill = ""
        
        # 选择策略组合或手动选择
        if combos_repo:
            # 有repo中的策略组合，提供选择
            combo_options = ["手动选择策略"] + list(combos_repo.keys())
            selected_combo_key = st.selectbox("选择策略组合", options=combo_options, index=0, key="select_combo_from_repo")
            
            if selected_combo_key == "手动选择策略":
                # 手动选择策略
                selected_rules = st.multiselect("选择策略（可多选）", options=rule_names, default=[], key="manual_rules")
                selected_agg_mode = st.radio("聚合模式", ["OR（任一命中）", "AND（全部命中）"], index=0, horizontal=True, key="manual_agg_mode")
                selected_agg_mode = "OR" if selected_agg_mode.startswith("OR") else "AND"
                selected_exclude_rules = st.multiselect("排除策略（可多选）", options=rule_names, default=[], key="manual_exclude_rules", help="如果触发这些策略则排除该股票")
                selected_combo_name = None
                selected_output_name = None
                selected_combo_data = None
                selected_rule_groups = []
                selected_group_mode = "AND"
                rule_groups_prefill = ""
            else:
                # 使用repo中的策略组合
                selected_combo_data = combos_repo.get(selected_combo_key)
                if selected_combo_data:
                    selected_rules = selected_combo_data.get("rules", [])
                    selected_agg_mode = selected_combo_data.get("agg_mode", "OR")
                    selected_exclude_rules = selected_combo_data.get("exclude_rules", [])
                    selected_rule_groups = selected_combo_data.get("rule_groups") or selected_combo_data.get("logic_groups") or []
                    selected_group_mode = selected_combo_data.get("group_mode", "AND")
                    if selected_rule_groups:
                        try:
                            import json
                            rule_groups_prefill = json.dumps(selected_rule_groups, ensure_ascii=False)
                        except Exception:
                            rule_groups_prefill = ""
                    else:
                        rule_groups_prefill = ""
                    selected_combo_name = selected_combo_key
                    selected_output_name = selected_combo_data.get("output_name", selected_combo_name)
                    exclude_info = f" | 排除策略: {', '.join(selected_exclude_rules[:3])}{'...' if len(selected_exclude_rules) > 3 else ''}" if selected_exclude_rules else ""
                    st.info(f"**策略组合：{selected_combo_name}** | 策略组: {', '.join(selected_rules[:5])}{'...' if len(selected_rules) > 5 else ''} | 聚合方法: {selected_agg_mode}{exclude_info}")
                    st.caption(f"落盘名称: {selected_output_name}")
                    explain = selected_combo_data.get("explain", "")
                    if explain:
                        st.caption(f"说明: {explain}")
                    logic_desc = format_rule_groups(selected_rule_groups, selected_group_mode)
                    if logic_desc:
                        st.caption(f"混合逻辑: {logic_desc}（组间 {selected_group_mode}）")
                else:
                    selected_rules = []
                    selected_agg_mode = "OR"
                    selected_exclude_rules = []
                    selected_combo_name = None
                    selected_output_name = None
                    selected_combo_data = None
                    selected_rule_groups = []
                    selected_group_mode = "AND"
                    rule_groups_prefill = ""
        else:
            # 没有repo中的策略组合，只能手动选择
            selected_rules = st.multiselect("选择策略（可多选）", options=rule_names, default=[], key="manual_rules")
            selected_agg_mode = st.radio("聚合模式", ["OR（任一命中）", "AND（全部命中）"], index=0, horizontal=True, key="manual_agg_mode")
            selected_agg_mode = "OR" if selected_agg_mode.startswith("OR") else "AND"
            selected_exclude_rules = st.multiselect("排除策略（可多选）", options=rule_names, default=[], key="manual_exclude_rules", help="如果触发这些策略则排除该股票")
            selected_combo_name = None
            selected_output_name = None
            selected_combo_data = None
            selected_rule_groups = []
            selected_group_mode = "AND"
            rule_groups_prefill = ""
        
        # 参数设置
        with st.form("custom_rank_form"):
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                ref_date_custom = st.text_input("参考日（YYYYMMDD；留空=自动最新）", value="", key="custom_ref_date")
            with c2:
                universe_custom = st.selectbox("选股范围", ["全市场", "仅白名单", "仅黑名单", "仅强度榜"], index=0, key="custom_universe")
            with c3:
                tiebreak_custom = st.selectbox("同分排序", ["none", "kdj_j_asc"], index=1, key="custom_tiebreak")
            with c4:
                topN_custom = st.number_input("输出 Top-N（留空=不限制）", min_value=0, max_value=5000, value=200, step=10, key="custom_topN")
            
            # 榜单名称：如果使用了repo中的策略组合，使用其落盘名称；手动选择时才需要填写
            if selected_output_name:
                # 使用repo中的策略组合，显示落盘名称（只读）
                st.text_input("落盘名称（用于文件名）", value=selected_output_name, key="custom_combo_name", disabled=True)
                combo_name_output = selected_output_name
            else:
                # 手动选择策略，需要填写榜单名称
                combo_name_output = st.text_input("榜单名称（用于文件名）", value="custom", key="custom_combo_name", placeholder="例如：突破组合")
            
            rule_groups_override = st.text_area(
                "高级：混合逻辑（留空=沿用组合配置或简单 OR/AND）",
                value=rule_groups_prefill,
                key="custom_rule_groups_override",
                placeholder='示例：[{"mode": "OR", "rules": ["b1买点", "量价共振_上涨放量"]},{"mode": "OR", "rules": ["底部放量", "长期成本线附近"]}]'
            )
            group_mode_override = st.radio(
                "组间逻辑（用于高级逻辑）",
                ["AND（全部组满足）", "OR（任一组满足）"],
                index=0 if str(selected_group_mode).upper() == "AND" else 1,
                horizontal=True,
                key="custom_group_mode_override"
            )
            
            gen_btn = st.form_submit_button("生成并落盘", width='stretch')
        
        if gen_btn:
            if not selected_rules and not rule_groups_override.strip() and not selected_rule_groups:
                st.warning("请至少选择一个策略或填写高级逻辑")
            elif not combo_name_output.strip():
                st.warning("请输入榜单名称")
            else:
                try:
                    # 自动启用数据库读取（和选股页面的按触发规则筛选一样）
                    if not is_details_db_reading_enabled():
                        st.session_state["details_db_reading_enabled"] = True
                    
                    ref_real = ref_date_custom.strip() or _get_latest_date_from_files() or ""
                    if not ref_real:
                        st.error("未能确定参考日")
                    else:
                        universe_map = {"全市场": "all", "仅白名单": "white", "仅黑名单": "black", "仅强度榜": "strength"}
                        universe = universe_map.get(universe_custom, "all")
                        
                        # 获取排除策略
                        exclude_rules = None
                        if selected_combo_data:
                            exclude_rules = selected_combo_data.get("exclude_rules", [])
                        else:
                            # 手动选择策略时，使用已定义的 selected_exclude_rules 变量
                            exclude_rules = selected_exclude_rules if 'selected_exclude_rules' in locals() else []
                        if not exclude_rules:
                            exclude_rules = None
                        
                        # 处理高级混合逻辑
                        group_mode_to_use = "AND" if group_mode_override.startswith("AND") else "OR"
                        rule_groups_to_use = list(selected_rule_groups) if selected_rule_groups else []
                        try:
                            parsed_groups, parsed_group_mode = parse_rule_groups_input(rule_groups_override, group_mode_to_use)
                            if parsed_groups:
                                rule_groups_to_use = parsed_groups
                            if parsed_group_mode:
                                group_mode_to_use = parsed_group_mode
                        except ValueError as e:
                            st.warning(f"高级逻辑解析失败：{e}")
                            st.stop()
                        rule_names_to_use = list(selected_rules)
                        if not rule_names_to_use and rule_groups_to_use:
                            rule_names_to_use = sorted({r for g in rule_groups_to_use for r in g.get("rules", [])})
                        
                        # 调用生成函数
                        from scoring_core import build_custom_rank
                        result = build_custom_rank(
                            combo_name=combo_name_output.strip(),
                            rule_names=rule_names_to_use,
                            agg_mode=selected_agg_mode,
                            rule_groups=rule_groups_to_use,
                            group_mode=group_mode_to_use,
                            ref_date=ref_real,
                            universe=universe,
                            tiebreak=tiebreak_custom,
                            topN=topN_custom if topN_custom > 0 else None,
                            write=True,
                            exclude_rules=exclude_rules
                        )
                        
                        if result:
                            st.success(f"自选榜已生成并落盘: {result}")
                            # 读取并显示结果
                            try:
                                df_result = pd.read_csv(result)
                                df_result = add_concept_column(df_result, blacklist=CONCEPT_BLACKLIST)
                                st.dataframe(df_result, width='stretch', height=480)
                                
                                # 导出 TXT
                                if "ts_code" in df_result.columns:
                                    codes = df_result["ts_code"].astype(str).tolist()
                                    txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"], st.session_state["export_pref"]["with_suffix"])
                                    copy_txt_button(txt, label="📋 复制代码", key=f"copy_custom_{ref_real}")
                            except Exception as e:
                                st.warning(f"读取结果失败: {e}")
                        else:
                            st.info("未筛到命中标的")
                except Exception as e:
                    st.error(f"生成失败: {e}")
        
        # 查看历史榜单
        st.divider()
        st.markdown("### 查看历史榜单")
        custom_dir = SC_OUTPUT_DIR / "custom"
        if custom_dir.exists():
            csv_files = sorted(custom_dir.glob("custom_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
            if csv_files:
                selected_file = st.selectbox("选择文件", options=[f.name for f in csv_files[:20]], key="custom_file_select")
                if selected_file:
                    try:
                        df_view = pd.read_csv(custom_dir / selected_file)
                        df_view = add_concept_column(df_view, blacklist=CONCEPT_BLACKLIST)
                        st.dataframe(df_view, width='stretch', height=420)
                        if "ts_code" in df_view.columns:
                            codes = df_view["ts_code"].astype(str).tolist()
                            txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"], st.session_state["export_pref"]["with_suffix"])
                            copy_txt_button(txt, label="📋 复制代码", key=f"copy_custom_file_{selected_file}")
                    except Exception as e:
                        st.error(f"读取文件失败: {e}")
            else:
                st.info("暂无历史榜单文件")
        else:
            st.info("自定义榜单目录不存在")

    # ================== 选股 ==================
    with tab_screen:
        st.subheader("选股")

        # === 统一参考日 & 范围 ===
        c_top1, c_top2 = st.columns([1,1])
        with c_top1:
            refD_unified = st.text_input("参考日（YYYYMMDD，留空=自动最新）", value=st.session_state.get("screen_refD",""), key="screen_refD")
        with c_top2:
            uni_choice = st.selectbox("选股范围", ["全市场","仅白名单","仅黑名单","仅特别关注榜"], index=0, key="screen_uni_choice")
        _uni_map = {"全市场":"all", "仅白名单":"white", "仅黑名单":"black", "仅特别关注榜":"attention"}

        # ========== 1) 表达式筛选 ==========
        with st.form("expression_screening_form"):
            st.markdown("### 表达式筛选")
            exp = st.text_input("表达式（示例：CLOSE>MA(CLOSE,20) AND VOL>MA(VOL,5)）", value=st.session_state.get("screen_expr",""), key="screen_expr")
            c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])
            with c1:
                level = st.selectbox("时间级别", ["D","W","M"], index=0, key="screen_level")
            with c2:
                window = st.number_input("计分窗口 (score_windows)", min_value=1, max_value=500, value=30, 
                                         help="用于计分判断的历史数据条数。系统会自动根据表达式中的函数（如REF、MA、TS_RANK等）计算所需的历史数据窗口，总数据窗口 = score_windows + 表达式窗口", 
                                         key="screen_window")
            with c3:
                scope_logic = st.selectbox("命中范围(scope)", ["LAST","ANY","ALL","COUNT>=k","CONSEC>=m","ANY_n","ALL_n"], index=0, key="screen_scope_logic")
            with c4:
                n_k_m = st.number_input("k/m/n(特定选择才生效)", min_value=1, max_value=500, value=3, key="screen_nkm")
            with c5:
                tiebreak_expr = st.selectbox("同分排序", ["none", "kdj_j_asc"], index=1, key="screen_tiebreak_expr")
            with c6:
                run_btn = st.form_submit_button("运行筛选", width='stretch')
            cap_sel_expr = st.multiselect(
                "总市值筛选（亿元，留空=不限；多选=并集）",
                options=[b[0] for b in _CAP_BUCKETS],
                default=[],
                key="screen_cap_expr",
                help="基于 stock_list.csv 的 total_mv；查不到则无法命中筛选。"
            )

        if run_btn:
            logger.info(f"用户点击运行筛选: 表达式={exp[:50]}..., 级别={level}, 窗口={window}, 范围={scope_logic}")
            try:
                if not exp.strip():
                    st.warning("请先输入表达式。")
                else:
                    # 组装 scope
                    scope = scope_logic
                    if scope_logic.startswith("COUNT"):
                        scope = f"COUNT>={int(n_k_m)}"
                    elif scope_logic.startswith("CONSEC"):
                        scope = f"CONSEC>={int(n_k_m)}"
                    elif scope_logic == "ANY_n":
                        scope = f"ANY_{int(n_k_m)}"
                    elif scope_logic == "ALL_n":
                        scope = f"ALL_{int(n_k_m)}"

                    df_sel = run_se_screen_in_bg(
                        when_expr=exp.strip(),
                        ref_date=(refD_unified.strip() or None),
                        timeframe=level,
                        window=_safe_int(window, 30),
                        scope=scope,
                        universe=_uni_map.get(uni_choice,"all"),
                        write_white=False,
                        write_black_rest=False,
                        return_df=True,
                    )
                    if df_sel is None or df_sel.empty:
                        st.info("无命中。")
                    else:
                        df_sel = _apply_total_mv_filter(df_sel, cap_sel_expr)
                        if df_sel.empty:
                            st.info("总市值筛选后无命中。")
                        else:
                            ref_for_rank = ""
                            if "ref_date" in df_sel.columns:
                                try:
                                    ref_candidates = df_sel["ref_date"].dropna().astype(str)
                                    if not ref_candidates.empty:
                                        ref_for_rank = ref_candidates.iloc[0]
                                except Exception:
                                    ref_for_rank = ""
                            if not ref_for_rank:
                                ref_for_rank = refD_unified.strip() or _get_latest_date_from_files() or ""
                            df_sel_ranked = _attach_all_rank(df_sel, ref_for_rank)
                            # 调整显示列顺序：rank → ts_code → score → 其余
                            df_show = add_concept_column(df_sel_ranked, blacklist=CONCEPT_BLACKLIST)
                            if "rank" in df_show.columns:
                                base_cols = ["rank", "ts_code", "score", "total_mv_亿"]
                                ordered = [c for c in base_cols if c in df_show.columns]
                                ordered += [c for c in df_show.columns if c not in ordered]
                                df_show = df_show[ordered]
                            # 结果已经按得分排序，直接显示
                            st.caption(f"命中 {len(df_sel)} 只；参考日：{(df_sel['ref_date'].iloc[0] if 'ref_date' in df_sel.columns and len(df_sel)>0 else (refD_unified or '自动'))}")
                            if 'score' in df_sel.columns:
                                st.caption("已按得分排序（降序），同分时按J值升序")
                            st.dataframe(df_show, width='stretch', height=480)
                            # 导出 TXT（代码）
                            if "ts_code" in df_show.columns:
                                txt = _codes_to_txt(df_show["ts_code"].astype(str).tolist(),
                                                    st.session_state["export_pref"]["style"],
                                                    st.session_state["export_pref"]["with_suffix"])
                                copy_txt_button(txt, label="📋 复制以上（按当前预览）", key=f"copy_screen_expr_{refD_unified or 'auto'}")
            except Exception as e:
                st.error(f"筛选失败：{e}")

        st.divider()

        # ========== 2) 按触发规则筛选（当日全市场，多选） ==========
        with st.form("rule_screening_form"):
            st.markdown("### 按触发规则筛选（当日全市场，多选）")
            st.caption("说明：读取当日 details 数据；按所选规则名判断：只要策略触发（ok=True）就视为命中；支持\"任一/全部\"聚合。")
            # 规则名来自 se.SC_RULES（使用缓存）
            rule_names = _get_rule_names()
            picked = st.multiselect("规则名（可多选）", options=rule_names, default=[], key="detail_multi_rules")
            agg_mode = st.radio("命中逻辑", ["任一命中（OR）","全部命中（AND）"], index=0, horizontal=True, key="detail_hit_mode")
            cA, cB, cC = st.columns([1,1,1])
            with cA:
                limit_n = st.number_input("最多显示/导出 N 条（-1=全量）", min_value=-1, value=100, step=10, key="detail_limit_n")
            with cB:
                tiebreak_rule = st.selectbox("同分排序", ["none", "kdj_j_asc"], index=1, key="screen_tiebreak_rule")
            with cC:
                run_detail = st.form_submit_button("筛选当日命中标的", width='stretch')
            cap_sel_rule = st.multiselect(
                "总市值筛选（亿元，留空=不限；多选=并集）",
                options=[b[0] for b in _CAP_BUCKETS],
                default=[],
                key="screen_cap_rule",
                help="基于 stock_list.csv 的 total_mv；查不到则无法命中筛选。"
            )

        if run_detail:
            # 自动启用数据库读取（和个股详情里的解锁逻辑一样）
            if not is_details_db_reading_enabled():
                st.session_state["details_db_reading_enabled"] = True
            
            ref_real = refD_unified.strip() or _get_latest_date_from_files() or ""
            if not ref_real:
                st.error("未能确定参考日。")
            elif not picked:
                st.warning("请先选择至少一个规则名。")
            else:
                rows = []
                try:
                    # 检查是否允许读取数据库（使用统一的函数）
                    db_reading_enabled = is_details_db_reading_enabled()
                    
                    # 优先使用数据库查询（只有当db_reading_enabled为True且数据库可用时才读取数据库）
                    if db_reading_enabled and is_details_db_available():
                        # 使用 database_manager 查询详情
                        logger.info("[数据库连接] 开始获取数据库管理器实例 (查询股票详情用于UI显示)")
                        df_all = pd.DataFrame()
                        try:
                            manager = get_database_manager()
                            if manager:
                                # 使用统一的函数获取details数据库路径（包含回退逻辑）
                                details_db_path = get_details_db_path_with_fallback()
                                if details_db_path:
                                    sql = "SELECT * FROM stock_details WHERE ref_date = ?"
                                    df_all = manager.execute_sync_query(details_db_path, sql, [ref_real], timeout=30.0)
                        except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                            # 数据库文件不存在或管理器获取失败
                            logger.debug(f"数据库访问失败: {ref_real}: {e}")
                            df_all = pd.DataFrame()
                        except Exception as e:
                            # 数据库读取失败（可能是表不存在、连接错误等）
                            error_msg = str(e).lower()
                            if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                                logger.debug(f"Details数据库表不存在: {ref_real}")
                            else:
                                logger.debug(f"数据库读取失败: {ref_real}: {e}")
                            df_all = pd.DataFrame()
                        
                        if not df_all.empty:
                            for _, row in df_all.iterrows():
                                ts2 = str(row.get("ts_code", "")).strip()
                                if not ts2:
                                    continue
                                
                                # 使用统一的 _load_detail_json 函数获取数据
                                data = _load_detail_json(str(ref_real), ts2)
                                if not data:
                                    continue
                                
                                # 从统一格式中提取数据
                                summary = data.get("summary", {})
                                sc = float(summary.get("score", 0.0))
                                rules = data.get("rules", [])
                                
                                names_today = set()
                                hit_dates_map = {}  # 策略名 -> 触发日期列表
                                for rr in rules:
                                    # 只要策略触发（ok=True），就视为命中，无需检查add字段
                                    # 或者add>0也视为命中（兼容原有逻辑）
                                    ok_val = rr.get("ok")
                                    add_val = rr.get("add")
                                    if bool(ok_val) or (add_val is not None and float(add_val) > 0.0):
                                        n = rr.get("name")
                                        if n: 
                                            names_today.add(str(n))
                                            # 收集触发日期列表
                                            hit_date = rr.get("hit_date")
                                            hit_dates = rr.get("hit_dates", [])
                                            # 合并hit_date和hit_dates
                                            all_dates = []
                                            if hit_date:
                                                all_dates.append(str(hit_date))
                                            if hit_dates:
                                                all_dates.extend([str(d) for d in hit_dates if d])
                                            # 去重并排序
                                            all_dates = sorted(set(all_dates))
                                            if all_dates:
                                                hit_dates_map[str(n)] = all_dates
                                
                                if names_today:
                                    if agg_mode.startswith("任一"):
                                        hit = any((n in names_today) for n in picked)
                                    else:
                                        hit = all((n in names_today) for n in picked)
                                    if hit:
                                        # 收集所有选中策略的触发日期列表
                                        trigger_dates_list = []
                                        for rule_name in picked:
                                            if rule_name in hit_dates_map:
                                                trigger_dates_list.extend(hit_dates_map[rule_name])
                                        # 去重并排序
                                        trigger_dates_list = sorted(set(trigger_dates_list))
                                        rows.append({
                                            "ts_code": ts2, 
                                            "score": sc,
                                            "trigger_dates": trigger_dates_list if trigger_dates_list else []
                                        })
                    
                    # 回退到JSON文件查询
                    else:
                        ddir = DET_DIR / str(ref_real)
                        allow_set = None
                        if ddir.exists():
                            for p in ddir.glob("*.json"):
                                try:
                                    j = json.loads(p.read_text(encoding="utf-8-sig"))
                                except Exception:
                                    continue
                                ts2 = str(j.get("ts_code","")).strip()
                                if not ts2:
                                    continue
                                if (allow_set is not None) and (ts2 not in allow_set):
                                    continue
                                sm = j.get("summary") or {}
                                sc = float(sm.get("score", 0.0))
                                names_today = set()
                                hit_dates_map = {}  # 策略名 -> 触发日期列表
                                for rr in (j.get("rules") or []):
                                    # 只要策略触发（ok=True），就视为命中，无需检查add字段
                                    # 或者add>0也视为命中（兼容原有逻辑）
                                    ok_val = rr.get("ok")
                                    add_val = rr.get("add")
                                    if bool(ok_val) or (add_val is not None and float(add_val) > 0.0):
                                        n = rr.get("name")
                                        if n: 
                                            names_today.add(str(n))
                                            # 收集触发日期列表
                                            hit_date = rr.get("hit_date")
                                            hit_dates = rr.get("hit_dates", [])
                                            # 合并hit_date和hit_dates
                                            all_dates = []
                                            if hit_date:
                                                all_dates.append(str(hit_date))
                                            if hit_dates:
                                                all_dates.extend([str(d) for d in hit_dates if d])
                                            # 去重并排序
                                            all_dates = sorted(set(all_dates))
                                            if all_dates:
                                                hit_dates_map[str(n)] = all_dates
                                if names_today:
                                    if agg_mode.startswith("任一"):
                                        hit = any((n in names_today) for n in picked)
                                    else:
                                        hit = all((n in names_today) for n in picked)
                                    if hit:
                                        # 收集所有选中策略的触发日期列表
                                        trigger_dates_list = []
                                        for rule_name in picked:
                                            if rule_name in hit_dates_map:
                                                trigger_dates_list.extend(hit_dates_map[rule_name])
                                        # 去重并排序
                                        trigger_dates_list = sorted(set(trigger_dates_list))
                                        rows.append({
                                            "ts_code": ts2, 
                                            "score": sc,
                                            "trigger_dates": trigger_dates_list if trigger_dates_list else []
                                        })
                    
                    df_hit = pd.DataFrame(rows)
                    if df_hit.empty:
                        st.info("未筛到命中标的。")
                    else:
                        df_hit = _apply_total_mv_filter(df_hit, cap_sel_rule)
                        if df_hit.empty:
                            st.info("总市值筛选后无命中。")
                        else:
                        # 应用Tie-break排序
                            df_hit_sorted = _apply_tiebreak_sorting(df_hit, tiebreak_rule)
                            df_hit_sorted = _attach_all_rank(df_hit_sorted, ref_real)
                            n = int(limit_n)
                            df_show = df_hit_sorted if n < 0 else df_hit_sorted.head(n)
                            # 调整列顺序：rank -> ts_code -> score -> trigger_dates
                            col_order = ["rank", "ts_code", "score", "total_mv_亿", "trigger_dates"]
                            col_order += [c for c in df_show.columns if c not in col_order]
                            df_show = df_show[[c for c in col_order if c in df_show.columns]]
                            st.caption(f"命中 {len(df_hit_sorted)} 只；显示 {len(df_show)} 只；参考日：{ref_real}")
                            st.dataframe(df_show, width='stretch', height=420)
                            # 导出 TXT
                            if "ts_code" in df_show.columns:
                                txt = _codes_to_txt(df_show["ts_code"].astype(str).tolist(),
                                                    st.session_state["export_pref"]["style"],
                                                    st.session_state["export_pref"]["with_suffix"])
                                copy_txt_button(txt, label="📋 复制以上（按当前预览）", key=f"copy_screen_rule_{ref_real}")
                except Exception as e:
                    st.error(f"读取明细失败：{e}")

        st.markdown("---")
        # === 按概念筛选（放在末尾，按需显示） ===
        with st.container(border=True):
            st.markdown("### 按概念筛选")
            ref_for_concept = refD_unified.strip() or _get_latest_date_from_files() or ""
            if not ref_for_concept:
                st.caption("未找到参考日，无法加载概念数据。")
            else:
                df_concept_base = _read_rank_all_sorted(ref_for_concept)
                if df_concept_base is None or df_concept_base.empty:
                    st.caption(f"参考日 {ref_for_concept} 的排名文件为空或缺失。")
                else:
                    df_concept_base = add_concept_column(df_concept_base, blacklist=CONCEPT_BLACKLIST, out_col="概念")
                    concept_set = set()
                    ser_concept = df_concept_base["概念"] if "概念" in df_concept_base.columns else pd.Series([], dtype=str)
                    for raw in ser_concept.astype(str):
                        for c in raw.split(","):
                            c = c.strip()
                            if c:
                                concept_set.add(c)
                    concept_options = sorted(concept_set)

                    with st.form("concept_screen_form"):
                        sel_concepts = st.multiselect("选择概念（可搜索，多选=取并集）", options=concept_options, default=[], key="screen_concept_filter")
                        top_limit = int(st.number_input("显示前 N 条（-1=全量，按排名升序）", min_value=-1, value=100, step=20, key="screen_concept_topn"))
                        cap_sel_concept = st.multiselect(
                            "总市值筛选（亿元，留空=不限；多选=并集）",
                            options=[b[0] for b in _CAP_BUCKETS],
                            default=[],
                            key="screen_cap_concept",
                            help="基于 stock_list.csv 的 total_mv；查不到则无法命中筛选。"
                        )
                        show_btn = st.form_submit_button("显示筛选结果", use_container_width=True, type="primary")

                    if sel_concepts:
                        mask = df_concept_base["概念"].apply(
                            lambda s: any(c in str(s).split(",") for c in sel_concepts)
                        )
                        df_filtered = df_concept_base.loc[mask].copy()
                    else:
                        df_filtered = df_concept_base.copy()
                    df_filtered = _apply_total_mv_filter(df_filtered, cap_sel_concept)

                    if show_btn:
                        if df_filtered.empty:
                            st.info("无匹配股票。")
                        else:
                            j_col = None
                            for cand in ("tiebreak_j", "kdj_j", "J值", "j"):
                                if cand in df_filtered.columns:
                                    j_col = cand
                                    break
                            if j_col:
                                df_filtered["J值"] = pd.to_numeric(df_filtered[j_col], errors="coerce")
                            if "name" not in df_filtered.columns:
                                name_map = _get_stock_name_map()
                                df_filtered["name"] = df_filtered["ts_code"].map(name_map)
                            df_view = df_filtered.copy()
                            if "rank" in df_view.columns:
                                df_view = df_view.sort_values("rank")
                            cols_show = ["ts_code", "name", "rank", "total_mv_亿"]
                            if "J值" in df_view.columns:
                                cols_show.append("J值")
                            cols_show.append("概念")
                            cols_show = [c for c in cols_show if c in df_view.columns]
                            df_to_show = df_view[cols_show] if top_limit < 0 else df_view[cols_show].head(top_limit)
                            st.dataframe(df_to_show, width='stretch', height=420)
                            if "ts_code" in df_to_show.columns:
                                txt = _codes_to_txt(
                                    df_to_show["ts_code"].astype(str).tolist(),
                                    st.session_state["export_pref"]["style"],
                                    st.session_state["export_pref"]["with_suffix"]
                                )
                                copy_txt_button(
                                    txt,
                                    label="📋 复制以上（按当前预览）",
                                    key=f"copy_screen_concept_{ref_for_concept}"
                                )

    # ================== 工具箱 ==================
    with tab_tools:
        st.subheader("工具箱")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**自动补算最近 N 个交易日**")
            n_back = st.number_input("天数 N", min_value=1, max_value=100, value=15)
            inc_today = st.checkbox("包含参考日当天", value=True,
                                    help="勾选后窗口包含参考日（例如 N=5 → [ref-(N-1), ref]；未勾选则 [ref-N, ref-1]）")
            do_force = st.checkbox("强制重建（覆盖已有）", value=False,
                                help="若之前失败留下了 0 字节文件或想重算，勾选此项。")

            go_fill = st.button("执行自动补算", width='stretch')
            if go_fill:
                try:
                    if hasattr(se, "backfill_prev_n_days"):
                        out = se.backfill_prev_n_days(n=int(n_back), include_today=bool(inc_today), force=bool(do_force))
                        st.success(f"已处理：{out}")
                    else:
                        st.warning("未检测到 backfill_prev_n_days。")
                except Exception as e:
                    st.error(f"补算失败：{e}")

        with colB:
            st.markdown("**补齐缺失的 All 排名文件**")
            # start = st.text_input("起始日 YYYYMMDD", value="")
            start = st.text_input("起始日 YYYYMMDD", value="", key="tools_fix_start")
            end = st.text_input("结束日 YYYYMMDD", value="", key="tools_fix_end")
            do_force_fix = st.checkbox("强制重建（覆盖已有）", value=False)
            go_fix = st.button("补齐缺失", width='stretch')
            if go_fix and start and end:
                try:
                    if hasattr(se, "backfill_missing_ranks"):                   
                        out = se.backfill_missing_ranks(start, end, force=bool(do_force_fix))
                        st.success(f"已补齐：{out}")
                    else:
                        st.warning("未检测到 backfill_missing_ranks。")
                except Exception as e:
                    st.error(f"处理失败：{e}")
        st.markdown("---")
        with st.expander("查看已有数据（All / Details / 日历）", expanded=True):
            if "scan_inventory_loaded" not in st.session_state:
                st.session_state["scan_inventory_loaded"] = False
            col0, col1 = st.columns([1,3])
            with col0:
                do_scan = st.button("加载/刷新列表", key="btn_scan_inventory", width='stretch')
                if do_scan:
                    st.session_state["scan_inventory_loaded"] = True
            if not st.session_state["scan_inventory_loaded"]:
                st.info("（首次进入不扫描磁盘，点击上方 **加载/刷新列表** 才读取文件清单。）")
            if st.session_state["scan_inventory_loaded"]:
                try:
                    all_files = sorted(ALL_DIR.glob("score_all_*.csv"))
                    det_dirs  = sorted([p for p in DET_DIR.glob("*") if p.is_dir()])

                    all_dates = [p.stem.replace("score_all_", "") for p in all_files]
                    det_dates = [p.name for p in det_dirs]

                    zero_all = [p.name for p in all_files if p.stat().st_size == 0]

                    cov_min = min(all_dates) if all_dates else ""
                    cov_max = max(all_dates) if all_dates else ""

                    # 交易日日历（若存在则用于对比缺失）
                    missing: list[str] = []
                    try:
                        trade_dates = get_trade_dates() or []
                        if trade_dates and cov_min and cov_max:
                            rng = [d for d in trade_dates if cov_min <= d <= cov_max]
                            aset = set(all_dates)
                            missing = [d for d in rng if d not in aset]
                    except Exception:
                        trade_dates = []

                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("All 文件数", len(all_files))
                    with col2: st.metric("Details 日期目录", len(det_dirs))
                    with col3: st.metric("0 字节 All 文件", len(zero_all))

                    if cov_min:
                        st.caption(f"All 覆盖区间：{cov_min} ~ {cov_max}（缺失 {len(missing)} 天）")
                    else:
                        st.caption("All 目录为空。")

                    if zero_all:
                        names = zero_all[:8]
                        st.warning("检测到 0 字节文件（可用“强制重建”覆盖）：\n" + "，".join(names) + (" ……" if len(zero_all) > len(names) else ""))
                    colL, colR = st.columns([1, 2])
                    with colL:
                        kind = st.radio("数据类型", ["All 排名", "Details"], horizontal=True, key="view_kind")
                        cand = all_dates if kind == "All 排名" else det_dates
                        sel_date = st.selectbox("选择日期（倒序）", cand[::-1] if cand else [], key="view_date") if cand else None
                        show_missing = st.checkbox("显示缺失日期（基于交易日历）", value=False, disabled=not missing)
                    with colR:
                        if sel_date:
                            if kind == "All 排名":
                                p = _path_all(sel_date)
                                if p.exists() and p.stat().st_size > 0:
                                    st.dataframe(_read_df(p).head(200), width='stretch', height=360)
                                else:
                                    st.info("该日 All 文件不存在或为空。")
                            else:
                                pdir = DET_DIR / sel_date
                                if pdir.exists():
                                    st.info(f"{sel_date} 共有 {len(list(pdir.glob('*.json')))} 个详情文件。")
                                else:
                                    st.info("该日没有 Details 目录。")

                    if show_missing and missing:
                        st.markdown("**缺失日期（相对 All 覆盖区间）**")
                        txt = " ".join(missing[:200]) + (" ..." if len(missing) > 200 else "")
                        st.code(txt)
                except Exception as e:
                    st.error(f"扫描失败：{e}")

    # ================== 组合模拟 / 持仓 ==================
    with tab_port:
        st.subheader("组合模拟 / 持仓")
        # PortfolioManager 已在本文件中定义
        pm = PortfolioManager()

        # —— 全局配置（用于新建组合的默认值） ——
        with st.expander("全局配置（默认用于新建组合；来自 config.PF_*）", expanded=True):
            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.text_input("账本名称", value=cfg_str("PF_LEDGER_NAME", "账本1"), key="pf_ledger")
                st.number_input("初始资金（总额）", min_value=0.0, value=float(getattr(cfg, "PF_INIT_CASH", 1_000_000.0)), key="pf_init_cash")
            with colB:
                st.number_input("初始可用资金", min_value=0.0, value=float(getattr(cfg, "PF_INIT_AVAILABLE", getattr(cfg, "PF_INIT_CASH", 1_000_000.0))), key="pf_init_avail")
                st.selectbox("成交价口径", ["next_open","close"], index=(0 if cfg_str("PF_TRADE_PRICE_MODE","next_open")=="next_open" else 1), key="pf_pxmode")
            with colC:
                st.number_input("买入费率（bp）", min_value=0.0, value=float(getattr(cfg, "PF_FEE_BPS_BUY", 15.0)), key="pf_fee_buy")
                st.number_input("卖出费率（bp）", min_value=0.0, value=float(getattr(cfg, "PF_FEE_BPS_SELL", 15.0)), key="pf_fee_sell")
            with colD:
                st.number_input("最低费用（元）", min_value=0.0, value=float(getattr(cfg, "PF_MIN_FEE", 0.0)), key="pf_min_fee")
            st.caption("以上为默认值；新建组合时会带入（每个组合可覆盖）。")

        # —— 新建/选择组合 ——
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown("**新建组合**")
            new_name = st.text_input("名称", value=st.session_state.get("pf_ledger","default"))
            if st.button("创建组合", width='stretch'):
                pid = pm.create_portfolio(
                    name=new_name,
                    init_cash=float(st.session_state["pf_init_cash"]),
                    init_available=float(st.session_state["pf_init_avail"]),
                    trade_price_mode=str(st.session_state["pf_pxmode"]),
                    fee_bps_buy=float(st.session_state["pf_fee_buy"]),
                    fee_bps_sell=float(st.session_state["pf_fee_sell"]),
                    min_fee=float(st.session_state["pf_min_fee"]),
                )
                st.success(f"已创建：{new_name}（id={pid}）")

        with col2:
            st.markdown("**当前组合**")
            ports = pm.list_portfolios()
                #
            # 以 name 排序
            ports_items = sorted(list(ports.items()), key=lambda kv: kv[1].name) if ports else []
            if not ports_items:
                st.info("暂无组合，请先创建。")
                cur_pid, cur_pf = None, None
                st.session_state['cur_pid'] = None
                st.session_state['cur_pf'] = None
                st.session_state['cur_pid'] = cur_pid
                st.session_state['cur_pf'] = cur_pf
            else:
                names = [f"{p.name} ({pid[:6]})" for pid, p in ports_items]
                sel = st.selectbox("选择组合", options=list(range(len(ports_items))), format_func=lambda i: names[i], index=0)
                cur_pid, cur_pf = ports_items[sel]
                st.session_state['cur_pid'] = cur_pid
                st.session_state['cur_pf'] = cur_pf

        st.divider()

        # —— 录入成交（价格参考区间） ——
        st.markdown("**录入成交（带参考价区间）**")
        colx, coly, colz, colw = st.columns([1.2, 1.2, 1.2, 2])
        with colx:
            side = st.selectbox("方向", ["BUY","SELL"], index=0)
        with coly:
            d_exec = st.text_input("成交日（YYYYMMDD）", value=_get_latest_date_from_database() or "")
        with colz:
            ts = st.text_input("代码", value="")
        # 读取当日 O/H/L/C 作为参考
        ref_low = ref_high = px_open = px_close = None
        try:
            ts_norm = normalize_ts(ts) if ts else ""
            if ts_norm and d_exec:
                try:
                    from config import DATA_ROOT, UNIFIED_DB_PATH
                    db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                    df_one = query_stock_data(
                        db_path=db_path,
                        ts_code=ts_norm,
                        start_date=d_exec,
                        end_date=d_exec,
                        adj_type="qfq"
                    )
                except:
                    # 回退到直接查询
                    from config import DATA_ROOT, UNIFIED_DB_PATH
                    db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                    logger.info(f"[数据库连接] 开始获取数据库管理器实例 (回退查询K线数据: {ts_norm}, {d_exec})")
                    manager = get_database_manager()
                    sql = "SELECT open,high,low,close FROM stock_data WHERE ts_code = ? AND trade_date = ?"
                    df_one = manager.execute_sync_query(db_path, sql, [ts_norm, d_exec], timeout=30.0)
                if df_one is not None and not df_one.empty:
                    row = df_one.iloc[-1]
                    px_open = float(row.get("open", float("nan")))
                    px_close = float(row.get("close", float("nan")))
                    ref_low  = float(row.get("low", float("nan")))
                    ref_high = float(row.get("high", float("nan")))
        except Exception:
            pass
        with colw:
            st.write({"open": px_open, "close": px_close, "low": ref_low, "high": ref_high})

        colq, colp = st.columns([1.2, 1.8])
        with colq:
            qty = st.number_input("数量（股）", min_value=0, value=0, step=100)
        with colp:
            price_mode = st.radio("成交价来源", ["按口径自动","自定义价格"], index=0, horizontal=True)
            if price_mode == "自定义价格":
                price = st.number_input("成交价（留空则用口径价）", min_value=0.0, value=float(px_close or px_open or 0.0), step=0.01)
            else:
                price = None

        if cur_pf and st.button("记录成交", width='stretch', key="btn_rec_trade"):
            try:
                pm.record_trade(pid=cur_pid, date=str(d_exec), ts_code=str(ts_norm), side=str(side), qty=int(qty),
                                price_mode=(None if price is not None else cur_pf.trade_price_mode),
                                price=(None if price is None else float(price)), note="manual")
                st.success("已记录")
            except Exception as e:
                st.error(f"记录失败：{e}")

        st.divider()

        # —— 观察日估值 / 净值 ——
        st.markdown("**观察日收益与持仓估值**")
        obs = st.text_input("观察日（YYYYMMDD；默认=最新交易日）", value=_get_latest_date_from_database() or "")
        if obs and cur_pf:
            # 回放估值（从组合创建日至观察日）
            # 我们用 read_nav() 读取结果
            try:
                # 执行估值
                # pm.reprice_and_nav(cur_pid, date_start="19000101", date_end=str(obs), benchmarks=())
                tr = pm.read_trades(cur_pid)
                if tr is not None and not tr.empty:
                    # 组合首笔成交日
                    first_trade = str(pd.to_datetime(tr["date"].astype(str), errors="coerce").dt.strftime("%Y%m%d").min())
                    # 起点 = 首笔成交日前一个“交易日”
                    date_start_use = _prev_trade_date(first_trade, 1)
                else:
                    # 没有成交记录就从观察日开始（避免从远古起算）
                    date_start_use = str(obs)

                pm.reprice_and_nav(
                    cur_pid,
                    date_start=str(date_start_use),
                    date_end=str(obs),
                    benchmarks=(),
                )
                nav_df = pm.read_nav(cur_pid)
                pos_df = pm.read_positions(cur_pid)
            except Exception as e:
                st.error(f"估值失败：{e}")
                nav_df, pos_df = pd.DataFrame(), pd.DataFrame()
            if not nav_df.empty:
                row = nav_df.iloc[-1]
                if not pos_df.empty and "date" in pos_df.columns:
                    cur_pos = pos_df[pos_df["date"] == str(obs)].copy()
                    if not cur_pos.empty:
                        st.markdown("**当前持仓**")
                        show_cols = [c for c in ["ts_code","qty","cost","mkt_price","mkt_value","unreal_pnl"] if c in cur_pos.columns]
                        cur_pos = cur_pos[show_cols].sort_values("mkt_value", ascending=False)
                        st.dataframe(cur_pos, width='stretch', height=300)
                    else:
                        st.caption("观察日无持仓记录。")
                st.metric("组合市值", f"{(row.get('nav',1.0) * float(cur_pf.init_cash)):.0f}")
                st.metric("区间收益率", f"{(row.get('nav',1.0) - 1.0):.2%}")
                cols = [c for c in ["date","cash","position_mv","nav","ret_d","max_dd"] if c in nav_df.columns]
                st.dataframe(nav_df[cols].tail(5), width='stretch')
                st.markdown("**净值曲线（NAV）**")
                try:
                    st.line_chart(nav_df.set_index("date")["nav"])
                except Exception:
                    pass
            else:
                st.caption("暂无净值数据（可能还未有成交或行情数据缺失）")

    # ================== 统计（普通页签） ==================
    with tab_stats:
        st.subheader("统计")
        sub_tabs = st.tabs(["概念分布", "排名跟踪", "涨幅榜", "策略触发统计"])

        # --- 概念分布 ---
        with sub_tabs[0]:
            st.markdown("**名单概念统计**")
            colc1, colc2 = st.columns([2, 1])
            with colc1:
                codes_input = st.text_area(
                    "股票名单（支持多分隔符，ts_code/简写均可）",
                    height=120,
                    placeholder="例：\n000001.SZ 600000.SH 300750\n或 000001,600000;300750",
                    key="concept_list_input",
                )
                st.button(
                    "清空名单输入",
                    key="btn_clear_concept_list",
                    use_container_width=True,
                    on_click=lambda: st.session_state.__setitem__("concept_list_input", "")
                )
            with colc2:
                ref_for_concept = st.text_input(
                    "参考日（用于得分排序，留空=最新 all 文件）",
                    value=st.session_state.get("concept_ref_date", _get_latest_date_from_files() or ""),
                    key="concept_ref_date",
                )
                btn_concept_stat = st.button("统计概念分布", key="btn_concept_stat")

            if btn_concept_stat:
                import re

                def _parse_codes(txt: str):
                    if not txt:
                        return []
                    parts = re.split(r"[\s,\n\r;\|]+", txt)
                    out = []
                    for p in parts:
                        s = p.strip()
                        if not s:
                            continue
                        try:
                            out.append(normalize_ts(s))
                        except Exception:
                            continue
                    return sorted(set(out))

                codes_list = _parse_codes(codes_input)
                if not codes_list:
                    st.warning("请输入有效的股票名单。")
                else:
                    concept_map = get_concept_mapping(blacklist=CONCEPT_BLACKLIST)
                    counts = {}
                    code_concepts = {}
                    no_concept = []
                    for c in codes_list:
                        raw = concept_map.get(c)
                        if not raw:
                            no_concept.append(c)
                            continue
                        concepts = [x for x in raw.split(",") if x]
                        code_concepts[c] = concepts
                        for cn in concepts:
                            counts[cn] = counts.get(cn, 0) + 1

                    if not counts:
                        st.info("名单内未匹配到概念。")
                    else:
                        # 参考日排名/得分，用于排序
                        ref_use = ref_for_concept.strip() or _get_latest_date_from_files() or ""
                        rank_df = _read_rank_all_sorted(ref_use) if ref_use else pd.DataFrame()
                        rank_df["ts_code"] = rank_df.get("ts_code", pd.Series([], dtype=str)).astype(str)
                        score_map = {
                            row["ts_code"]: (
                                row.get("score"),
                                row.get("tiebreak_j", row.get("tiebreak")),
                                row.get("rank"),
                            )
                            for _, row in rank_df.iterrows()
                        } if not rank_df.empty else {}

                        # 概念分布饼图
                        import plotly.graph_objects as go

                        # 只画出现次数最多的前 N 个概念
                        top_n_concepts = int(st.number_input("饼图显示 Top-N 概念", min_value=1, max_value=200, value=10, step=1, key="concept_topn_pie"))
                        top_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n_concepts]
                        pie_labels = [k for k, _ in top_items]
                        pie_values = [counts[k] for k in pie_labels]
                        fig_pie = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, hole=0.3)])
                        # 使用标签引线，将标签放到外侧以避免遮挡
                        fig_pie.update_traces(
                            textinfo="label+percent",
                            textposition="outside",
                            hovertemplate="%{label}: %{value} 支 (%{percent})",
                            showlegend=False,
                            pull=[0.02] * len(pie_labels),
                        )
                        fig_pie.update_layout(
                            margin=dict(l=80, r=120, t=100, b=100),
                            height=600,
                            uniformtext_minsize=10,
                            uniformtext_mode="hide",
                        )

                        col_plot, col_tbl = st.columns([2, 3])
                        with col_plot:
                            st.plotly_chart(fig_pie, width="stretch")

                        # 最频繁概念列表
                        with col_tbl:
                            st.markdown("**概念出现频次（降序）**")
                            top_concepts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                            rows = []
                            for concept_name, cnt in top_concepts:
                                stocks = [
                                    (c, score_map.get(c, (None, None, None)))
                                    for c, cs in code_concepts.items()
                                    if concept_name in cs
                                ]
                                def _sort_key(item):
                                    vals = item[1] if isinstance(item[1], tuple) else (None, None, None)
                                    sc = vals[0] if len(vals) > 0 else None
                                    tb = vals[1] if len(vals) > 1 else None
                                    sc_val = float(sc) if sc not in (None, "") else -1e9
                                    tb_val = float(tb) if tb not in (None, "") else 1e9
                                    return (-sc_val, tb_val, item[0])
                                stocks = sorted(stocks, key=_sort_key)
                                names = []
                                for c, _ in stocks:
                                    nm = _stock_name_of(c) or ""
                                    names.append(f"{c} {nm}".strip())
                                rows.append({
                                    "概念": str(concept_name or ""),
                                    "支数": int(cnt),
                                    "股票": "、".join(names) if names else "（无）",
                                })
                            df_concept = pd.DataFrame(rows, columns=["概念", "支数", "股票"])
                            if df_concept.empty:
                                st.info("无概念数据可展示。")
                            else:
                                st.dataframe(df_concept, width="stretch", hide_index=True)
                        if no_concept:
                            st.info(f"未匹配到概念的股票（共 {len(no_concept)} 只）：")
                            names_nc = []
                            for c in no_concept:
                                nm = _stock_name_of(c) or ""
                                names_nc.append(f"{c} {nm}".strip())
                            st.text("、".join(names_nc))

                    # 加权概念榜（按得分均值权重汇总）
                    st.markdown("**概念加权榜（基于得分均值）**")
                    prior_score = pd.to_numeric(rank_df.get("score"), errors="coerce").mean()
                    weight_rows = []
                    for concept_name, cnt in counts.items():
                        stocks = [
                            (c, score_map.get(c, (None, None, None)))
                            for c, cs in code_concepts.items()
                            if concept_name in cs
                        ]
                        if not stocks:
                            continue
                        total_w = 0.0
                        best_rank = None
                        stock_strs = []
                        for c, (sc, tb, rk) in stocks:
                            sc_val = pd.to_numeric(sc, errors="coerce")
                            rk_val = pd.to_numeric(rk, errors="coerce")
                            # 权重：优先用 score；无 score 时用 1/rank 作为兜底权重
                            w = float(sc_val) if pd.notna(sc_val) else (1.0 / float(rk_val) if pd.notna(rk_val) and rk_val > 0 else 0.0)
                            total_w += w
                            nm = _stock_name_of(c) or ""
                            rk_disp = int(rk_val) if pd.notna(rk_val) else "-"
                            sc_disp = f"{float(sc_val):.2f}" if pd.notna(sc_val) else "-"
                            stock_strs.append(f"{c}({rk_disp} | {sc_disp}) {nm}".strip())
                            if pd.notna(rk_val):
                                if best_rank is None or rk_val < best_rank:
                                    best_rank = rk_val
                        avg_w = total_w / len(stocks) if stocks else 0.0
                        weight_rows.append(
                            {
                                "概念": concept_name,
                                "加权总分": total_w,
                                "平均分": _shrink_mean(avg_w, cnt, prior_score, CONCEPT_SHRINK_ALPHA),
                                "覆盖数": cnt,
                                "最佳排名": int(best_rank) if best_rank is not None and pd.notna(best_rank) else "",
                                "股票": "、".join(stock_strs),
                            }
                        )
                    if weight_rows:
                        df_weight = pd.DataFrame(weight_rows)
                        df_weight = df_weight.sort_values(
                            ["平均分", "覆盖数", "加权总分", "最佳排名", "概念"],
                            ascending=[False, False, False, True, True],
                        )
                        st.dataframe(df_weight, width="stretch", height=420)
                    else:
                        st.info("无可计算的概念加权榜。")

        # --- Tracking ---
        with sub_tabs[1]:
            st.markdown("**使用过去某一天的排名，跟踪至今的涨幅、最大涨幅、最大回撤**")
            st.caption("计算方法：以参考日次日的开盘价为基准，至今/最大涨幅为收盘价相对该基准的收益，最大回撤为同一基准下的最高点回撤。")
            
            # 获取可用的排名文件日期
            available_dates = []
            try:
                all_files = sorted(ALL_DIR.glob("score_all_*.csv"))
                if all_files:
                    available_dates = [p.stem.replace("score_all_", "") for p in all_files]
                    available_dates = sorted(available_dates, reverse=True)  # 最新的在前
            except Exception:
                pass
            
            if not available_dates:
                st.warning("未找到可用的排名文件，请先在\"排名\"页签生成排名数据")
            
            
            # 默认选择最老的日期（列表最后一个）
            default_idx = len(available_dates) - 1
            if "ref_tracking_selected" in st.session_state:
                # 如果之前选择过，尝试保持选择
                if st.session_state["ref_tracking_selected"] in available_dates:
                    default_idx = available_dates.index(st.session_state["ref_tracking_selected"])
            
            refT = st.selectbox(
                "参考日（选择要跟踪的排名日期）",
                options=available_dates,
                index=default_idx,
                key="ref_tracking",
                help=f"共 {len(available_dates)} 个可用日期"
            )
            
            # 保存选择
            st.session_state["ref_tracking_selected"] = refT

            # 添加跟踪前N名的配置项
            top_n = st.number_input(
                "只跟踪前多少名",
                min_value=1,
                max_value=10000,
                value=SC_TRACKING_TOP_N,
                step=1,
                key="tracking_top_n",
                help=f"默认值：{SC_TRACKING_TOP_N}（可在 config.py 中修改 SC_TRACKING_TOP_N）"
            )

            if st.button("生成跟踪表", key="btn_run_tracking", width='stretch'):
                
                try:
                    # 1) 读取参考日的排名数据
                    rank_path = _path_all(refT)
                    if not rank_path.exists():
                        st.error(f"未找到参考日 {refT} 的排名文件：{rank_path}")
                    
                    
                    df_rank = _read_df(rank_path, dtype={"ts_code": str})
                    if df_rank.empty:
                        st.error(f"参考日 {refT} 的排名文件为空")
                    
                    
                    # 确保有 rank 列
                    if "rank" not in df_rank.columns and "score" in df_rank.columns:
                        df_rank = df_rank.sort_values(["score"], ascending=False).reset_index(drop=True)
                        df_rank["rank"] = np.arange(1, len(df_rank) + 1)
                    
                    # 根据配置只跟踪前N名
                    if "rank" in df_rank.columns:
                        original_count = len(df_rank)
                        df_rank = df_rank[df_rank["rank"] <= top_n].copy()
                        if df_rank.empty:
                            st.error(f"过滤后没有数据（前 {top_n} 名），请检查排名数据或调整跟踪数量")
                            st.stop()
                        st.info(f"已过滤为前 {top_n} 名，共 {len(df_rank)} 只股票（原始数据：{original_count} 只）")
                    
                    codes = df_rank["ts_code"].astype(str).unique().tolist()
                    st.info(f"读取到 {len(codes)} 只股票的排名数据（参考日：{refT}）")
                    
                    # 2) 获取最新交易日
                    latest_date = _get_latest_date_from_database()
                    if not latest_date:
                        st.error("无法获取最新交易日，请检查数据库连接")
                    
                    
                    if latest_date <= refT:
                        st.warning(f"最新交易日 {latest_date} 不晚于参考日 {refT}，无法计算跟踪数据")
                    
                    
                    st.info(f"跟踪区间：{refT} → {latest_date}")
                    
                    # 3) 读取价格数据
                    with st.spinner(f"正在读取 {len(codes)} 只股票的价格数据..."):
                        # _read_stock_prices 已在本文件中定义
                        df_prices = _read_stock_prices(codes, start=refT, end=latest_date)
                    
                    if df_prices.empty:
                        st.error("无法读取价格数据，请检查数据库")
                    
                    
                    # 4) 计算涨幅和回撤
                    with st.spinner("正在计算涨幅和回撤..."):
                        # 确保 ts_code 为字符串类型
                        df_prices["ts_code"] = df_prices["ts_code"].astype(str)
                        df_rank["ts_code"] = df_rank["ts_code"].astype(str)
                        
                        # 按 ts_code 和 trade_date 排序
                        df_prices = df_prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
                        
                        results = []
                        
                        # 按股票分组处理
                        for code_str, group in df_prices.groupby("ts_code"):
                            if group.empty:
                                continue
                            group = group.copy()
                            
                            # 参考日次日开盘价
                            ref_data = group[group["trade_date"] == refT]
                            if ref_data.empty:
                                continue
                            
                            next_day_data = group[group["trade_date"] > refT].head(1)
                            if next_day_data.empty:
                                continue
                            ref_price = next_day_data["open"].iloc[0]
                            
                            if pd.isna(ref_price) or ref_price <= 0:
                                continue
                            
                            # 最新收盘价
                            latest_data = group[group["trade_date"] == latest_date]
                            if latest_data.empty:
                                continue
                            latest_close = latest_data["close"].iloc[0]
                            
                            if pd.isna(latest_close):
                                continue
                            
                            # 至今涨幅
                            ret_to_date = (latest_close / ref_price - 1.0) * 100.0
                            
                            # 最大涨幅和最大回撤
                            # 计算每日相对参考日的收益率
                            group["ret_vs_ref"] = (group["close"] / ref_price - 1.0) * 100.0
                            
                            max_ret = group["ret_vs_ref"].max()
                            
                            # 最大回撤：从最高点到最低点的跌幅
                            # 先计算累计最高收益率
                            group["cummax_ret"] = group["ret_vs_ref"].cummax()
                            group["drawdown"] = group["ret_vs_ref"] - group["cummax_ret"]
                            max_drawdown = group["drawdown"].min()
                            
                            # 获取排名信息
                            rank_info = df_rank[df_rank["ts_code"] == code_str]
                            rank_val = rank_info["rank"].iloc[0] if not rank_info.empty else None
                            score_val = rank_info["score"].iloc[0] if not rank_info.empty and "score" in rank_info.columns else None
                            j_val = rank_info["tiebreak_j"].iloc[0] if not rank_info.empty and "tiebreak_j" in rank_info.columns else None
                            
                            results.append({
                                "ts_code": code_str,
                                "rank": rank_val,
                                "score": score_val,
                                "J值": j_val,
                                "至今涨幅": ret_to_date,
                                "最大涨幅": max_ret,
                                "最大回撤": max_drawdown
                            })
                        
                        df_result = pd.DataFrame(results)
                    
                    if df_result.empty:
                        st.warning("未能计算出任何跟踪数据")
                    
                    
                    # 5) 格式化显示
                    df_display = df_result.copy()
                    
                    # 格式化百分比列
                    for col in ["至今涨幅", "最大涨幅", "最大回撤"]:
                        if col in df_display.columns:
                            df_display[col] = df_display[col].map(
                                lambda x: f"{x:.2f}%" if pd.notna(x) else None
                            )
                    
                    # 按排名排序
                    df_display = df_display.sort_values("rank").reset_index(drop=True)
                    
                    # 选择要显示的列
                    display_cols = ["rank", "ts_code", "至今涨幅", "最大涨幅", "最大回撤"]
                    if "score" in df_display.columns:
                        display_cols.insert(2, "score")
                    if "J值" in df_display.columns:
                        # J值放在score之后
                        if "score" in display_cols:
                            score_idx = display_cols.index("score")
                            display_cols.insert(score_idx + 1, "J值")
                        else:
                            display_cols.insert(2, "J值")
                    
                    available_cols = [c for c in display_cols if c in df_display.columns]
                    
                    # 注意：use_container_width 已废弃，使用 width='stretch' 替代
                    st.dataframe(
                        df_display[available_cols],
                        width='stretch',
                        height=600
                    )
                    
                    st.caption(f"跟踪结果：参考日 {refT} 的排名，跟踪至 {latest_date}。涨幅和回撤均为百分比。")
                    
                    # 显示统计信息
                    with st.expander("统计摘要", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_ret = df_result["至今涨幅"].mean()
                            st.metric("平均至今涨幅", f"{avg_ret:.2f}%")
                        with col2:
                            avg_max_ret = df_result["最大涨幅"].mean()
                            st.metric("平均最大涨幅", f"{avg_max_ret:.2f}%")
                        with col3:
                            avg_dd = df_result["最大回撤"].mean()
                            st.metric("平均最大回撤", f"{avg_dd:.2f}%")
                        
                        # 正收益比例
                        positive_ratio = (df_result["至今涨幅"] > 0).sum() / len(df_result) * 100
                        st.metric("正收益比例", f"{positive_ratio:.1f}%")
                    
                except Exception as e:
                    st.error(f"生成失败：{e}")
                    import traceback
                    st.code(traceback.format_exc())

        # --- Surge ---
        with sub_tabs[2]:
            st.markdown("**按最近n日涨幅排序，区分不同市场**")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                refS = st.text_input("参考日（YYYYMMDD，留空=最新交易日）", value="", key="surge_ref",
                                    help="计算涨幅的截止日期")
            with col2:
                n_days = st.number_input("最近N日", min_value=1, max_value=60, value=5, key="surge_n_days",
                                        help="计算最近N日的累计涨幅")
            with col3:
                surge_top_k = st.number_input("排名Top-K", min_value=1, max_value=2000, value=100, key="surge_top_k",
                                              help="生成排名Top涨幅榜时使用的Top-K阈值（基于参考日排名文件）")
            
            # 显示最新交易日
            latest_date_hint = _get_latest_date_from_database()
            if latest_date_hint:
                st.caption(f"最新交易日：{latest_date_hint}")

            if st.button("生成涨幅榜", key="btn_run_surge", width='stretch'):
                try:
                    # 1) 确定参考日
                    if not refS or not refS.strip():
                        refS = _get_latest_date_from_database()
                        if not refS:
                            st.error("无法获取最新交易日，请手动输入参考日")
                        
                    else:
                        refS = refS.strip()
                    
                    st.info(f"参考日：{refS}，计算最近 {n_days} 日涨幅")
                    
                    # 2) 获取股票列表（优先从最新排名文件读取，否则从数据库获取）
                    codes = []
                    try:
                        # 尝试从最新排名文件读取
                        latest_rank_file = _get_latest_date_from_files()
                        if latest_rank_file:
                            rank_path = _path_all(latest_rank_file)
                            if rank_path.exists():
                                df_rank = _read_df(rank_path, dtype={"ts_code": str})
                                if not df_rank.empty and "ts_code" in df_rank.columns:
                                    codes = df_rank["ts_code"].astype(str).unique().tolist()
                                    st.info(f"从排名文件读取到 {len(codes)} 只股票")
                    except Exception:
                        pass
                    
                    # 如果从排名文件读取失败，从数据库获取
                    if not codes:
                        with st.spinner("正在获取全市场股票列表..."):
                            try:
                                codes = get_stock_list(adj_type="qfq")
                                if codes:
                                    st.info(f"从数据库获取到 {len(codes)} 只股票")
                            except Exception as e:
                                st.error(f"获取股票列表失败：{e}")
                            
                    
                    if not codes:
                        st.error("无法获取股票列表")
                    
                    
                    # 3) 计算起始日期
                    # _pick_trade_dates 已在本文件中定义
                    dates = _pick_trade_dates(refS, n_days)
                    if not dates or len(dates) < 2:
                        st.error(f"无法获取足够的交易日（需要至少2个交易日）")
                    
                    
                    start_date = dates[0]  # 最早日期
                    end_date = dates[-1]   # 参考日
                    
                    st.info(f"计算区间：{start_date} → {end_date}（共 {len(dates)} 个交易日）")

                    # 3.1) 读取排名映射（n天前/参考日）
                    start_rank_map: dict[str, int] = {}
                    rank_days_ago_col = f"{int(n_days)}天前排名"
                    df_start_rank = _read_rank_all_sorted(start_date)
                    if not df_start_rank.empty and {"ts_code", "rank"}.issubset(df_start_rank.columns):
                        try:
                            tmp_df = df_start_rank[["ts_code", "rank"]].copy()
                            tmp_df["rank"] = pd.to_numeric(tmp_df["rank"], errors="coerce")
                            tmp_df = tmp_df.dropna(subset=["rank"])
                            start_rank_map = dict(zip(tmp_df["ts_code"].astype(str), tmp_df["rank"].astype(int)))
                        except Exception:
                            start_rank_map = {}
                    ref_rank_map: dict[str, int] = {}
                    df_ref_rank = _read_rank_all_sorted(refS)
                    if not df_ref_rank.empty and {"ts_code", "rank"}.issubset(df_ref_rank.columns):
                        try:
                            tmp_df = df_ref_rank[["ts_code", "rank"]].copy()
                            tmp_df["rank"] = pd.to_numeric(tmp_df["rank"], errors="coerce")
                            tmp_df = tmp_df.dropna(subset=["rank"])
                            ref_rank_map = dict(zip(tmp_df["ts_code"].astype(str), tmp_df["rank"].astype(int)))
                        except Exception:
                            ref_rank_map = {}
                    
                    # 4) 读取价格数据
                    with st.spinner(f"正在读取 {len(codes)} 只股票的价格数据..."):
                        # _read_stock_prices 已在本文件中定义
                        df_prices = _read_stock_prices(codes, start=start_date, end=end_date)
                    
                    if df_prices.empty:
                        st.error("无法读取价格数据，请检查数据库")
                    
                    
                    # 5) 计算涨幅
                    with st.spinner("正在计算涨幅..."):
                        df_prices["ts_code"] = df_prices["ts_code"].astype(str)
                        df_prices = df_prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
                        
                        results = []
                        
                        # 按股票分组计算涨幅
                        for code_str, group in df_prices.groupby("ts_code"):
                            if group.empty:
                                continue
                            
                            # 起始日收盘价
                            start_data = group[group["trade_date"] == start_date]
                            if start_data.empty:
                                continue
                            start_close = start_data["close"].iloc[0]
                            
                            if pd.isna(start_close) or start_close <= 0:
                                continue
                            
                            # 结束日收盘价
                            end_data = group[group["trade_date"] == end_date]
                            if end_data.empty:
                                continue
                            end_close = end_data["close"].iloc[0]
                            
                            if pd.isna(end_close):
                                continue
                            
                            # 计算涨幅
                            ret = (end_close / start_close - 1.0) * 100.0
                            
                            # 获取市场标签
                            market = market_label(code_str)
                            
                            results.append({
                                "ts_code": code_str,
                                "市场": market,
                                f"{n_days}日涨幅": ret
                            })
                        
                        df_result = pd.DataFrame(results)
                    
                    if df_result.empty:
                        st.warning("未能计算出任何涨幅数据")
                    
                    # 5.1) 附加排名信息
                    df_result[rank_days_ago_col] = df_result["ts_code"].map(lambda x: start_rank_map.get(str(x)))
                    df_result[rank_days_ago_col] = df_result[rank_days_ago_col].apply(lambda x: int(x) if pd.notna(x) else None)
                    if ref_rank_map:
                        df_result["参考日排名"] = df_result["ts_code"].map(lambda x: ref_rank_map.get(str(x)))
                        df_result["参考日排名"] = df_result["参考日排名"].apply(lambda x: int(x) if pd.notna(x) else None)
                    else:
                        df_result["参考日排名"] = None
                    
                    # 6) 按市场分组并排序
                    df_result = df_result.sort_values(["市场", f"{n_days}日涨幅"], ascending=[True, False]).reset_index(drop=True)
                    
                    # 添加市场内排名
                    df_result["市场内排名"] = df_result.groupby("市场")[f"{n_days}日涨幅"].rank(ascending=False, method="min").astype(int)
                    
                    # 7) 格式化显示
                    df_display = df_result.copy()
                    
                    # 格式化涨幅列
                    df_display[f"{n_days}日涨幅"] = df_display[f"{n_days}日涨幅"].map(
                        lambda x: f"{x:.2f}%" if pd.notna(x) else None
                    )
                    
                    # 选择要显示的列
                    display_cols = ["市场", "市场内排名", "ts_code", rank_days_ago_col, f"{n_days}日涨幅"]
                    available_cols = [c for c in display_cols if c in df_display.columns]
                    
                    # 按市场分组展示
                    markets = df_display["市场"].unique()
                    # 自定义排序：沪深主板放最前，其他按字母顺序
                    market_order = ["沪A", "深A", "创业板", "科创板", "北交所", "其他"]
                    markets_sorted = sorted(markets, key=lambda x: (
                        market_order.index(x) if x in market_order else len(market_order)
                    ))
                    
                    for market in markets_sorted:
                        market_df = df_display[df_display["市场"] == market].head(100)  # 每个市场最多显示100只
                        if not market_df.empty:
                            st.subheader(f"{market}（共 {len(df_display[df_display['市场'] == market])} 只，显示前100）")
                            # 注意：use_container_width 已废弃，使用 width='stretch' 替代
                            st.dataframe(
                                market_df[available_cols],
                                width='stretch',
                                height=400
                            )
                    
                    caption_parts = [f"涨幅榜：参考日 {refS}，最近 {n_days} 日涨幅（区间：{start_date} → {end_date}）"]
                    if start_rank_map:
                        caption_parts.append(f"附 {rank_days_ago_col}（来源：score_all_{start_date}.csv）")
                    else:
                        caption_parts.append(f"未找到 {start_date} 的排名文件，{rank_days_ago_col} 为空")
                    st.caption("；".join(caption_parts))
                    
                    # 额外：排名Top涨幅榜
                    st.subheader("排名Top涨幅榜")
                    if not ref_rank_map:
                        st.info(f"参考日 {refS} 的排名文件不存在，无法生成排名Top榜。")
                    else:
                        df_top_rank = df_result.copy()
                        df_top_rank = df_top_rank[df_top_rank["参考日排名"].notna()]
                        df_top_rank = df_top_rank[df_top_rank["参考日排名"] <= int(surge_top_k)]
                        if df_top_rank.empty:
                            st.info(f"参考日排名 Top-{int(surge_top_k)} 无数据，或对应股票无涨幅记录。")
                        else:
                            df_top_rank = df_top_rank.sort_values(f"{n_days}日涨幅", ascending=False).reset_index(drop=True)
                            df_top_display = df_top_rank.copy()
                            df_top_display[f"{n_days}日涨幅"] = df_top_display[f"{n_days}日涨幅"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else None)
                            display_cols_top = ["参考日排名", rank_days_ago_col, "ts_code", "市场", "市场内排名", f"{n_days}日涨幅"]
                            available_cols_top = [c for c in display_cols_top if c in df_top_display.columns]
                            st.dataframe(
                                df_top_display[available_cols_top],
                                width='stretch',
                                height=400
                            )
                            st.caption(f"排名Top-{int(surge_top_k)} 涨幅榜：参考日排名文件 {refS}，{rank_days_ago_col} 基于 {start_date}。")
                    
                    # 显示统计信息
                    with st.expander("市场统计摘要", expanded=False):
                        market_stats = []
                        for market in markets_sorted:
                            market_data = df_result[df_result["市场"] == market]
                            if not market_data.empty:
                                market_stats.append({
                                    "市场": market,
                                    "股票数": len(market_data),
                                    "平均涨幅": market_data[f"{n_days}日涨幅"].mean(),
                                    "最大涨幅": market_data[f"{n_days}日涨幅"].max(),
                                    "最小涨幅": market_data[f"{n_days}日涨幅"].min(),
                                    "正收益数": (market_data[f"{n_days}日涨幅"] > 0).sum(),
                                    "正收益比例": (market_data[f"{n_days}日涨幅"] > 0).sum() / len(market_data) * 100
                                })
                        
                        if market_stats:
                            df_stats = pd.DataFrame(market_stats)
                            # 格式化百分比
                            for col in ["平均涨幅", "最大涨幅", "最小涨幅"]:
                                if col in df_stats.columns:
                                    df_stats[col] = df_stats[col].map(lambda x: f"{x:.2f}%" if pd.notna(x) else None)
                            df_stats["正收益比例"] = df_stats["正收益比例"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else None)
                            
                            # 注意：use_container_width 已废弃，使用 width='stretch' 替代
                            st.dataframe(df_stats, width='stretch')
                    
                except Exception as e:
                    st.error(f"生成失败：{e}")
                    import traceback
                    st.code(traceback.format_exc())

        # --- 策略触发统计 ---
        with sub_tabs[3]:
            st.markdown("**统计参考日、区间内的策略触发情况，以及触发的后续跟踪**")
            
            # 获取可用的详情日期
            available_detail_dates = []
            try:
                # 尝试从数据库获取
                from database_manager import query_details_recent_dates
                db_path = get_details_db_path_with_fallback()
                if db_path and is_details_db_available():
                    dates_from_db = query_details_recent_dates(365, db_path)  # 获取最近一年的日期
                    if dates_from_db:
                        available_detail_dates = sorted(dates_from_db, reverse=True)
            except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                # 数据库文件不存在或管理器获取失败，忽略错误
                logger.debug(f"数据库访问失败: {e}")
                pass
            except Exception as e:
                # 数据库读取失败（可能是表不存在、连接错误等），忽略错误
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                    logger.debug(f"Details数据库表不存在")
                else:
                    logger.debug(f"数据库读取失败: {e}")
                pass
            
            # 如果数据库没有，尝试从文件系统获取
            if not available_detail_dates:
                try:
                    det_dirs = sorted([p for p in DET_DIR.glob("*") if p.is_dir()], reverse=True)
                    available_detail_dates = [p.name for p in det_dirs if _is_valid_date(p.name)]
                except Exception:
                    pass
            
            if not available_detail_dates:
                st.warning("未找到可用的详情数据，请先在\"排名\"页签生成评分数据")
            else:
                # 参数设置
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    # 默认选择最新的日期
                    default_idx = 0
                    if "strategy_ref_date_selected" in st.session_state:
                        if st.session_state["strategy_ref_date_selected"] in available_detail_dates:
                            default_idx = available_detail_dates.index(st.session_state["strategy_ref_date_selected"])
                    
                    ref_date_strategy = st.selectbox(
                        "参考日（起始日期）",
                        options=available_detail_dates,
                        index=default_idx,
                        key="strategy_ref_date",
                        help=f"共 {len(available_detail_dates)} 个可用日期"
                    )
                    st.session_state["strategy_ref_date_selected"] = ref_date_strategy
                
                with col2:
                    # 区间长度（交易日数）
                    interval_days = st.number_input(
                        "区间长度（交易日数）",
                        min_value=1,
                        max_value=60,
                        value=5,
                        key="strategy_interval_days",
                        help="统计从参考日开始的N个交易日内的策略触发情况"
                    )
                
                with col3:
                    # 跟踪天数
                    track_days = st.number_input(
                        "后续跟踪天数",
                        min_value=1,
                        max_value=30,
                        value=10,
                        key="strategy_track_days",
                        help="跟踪触发策略的股票在后续N个交易日的表现"
                    )
                
                # 样本选择
                sample_choice = st.selectbox(
                    "统计样本",
                    ["全市场", "仅Top-K", "仅白名单", "仅黑名单"],
                    index=0,
                    key="strategy_sample_choice",
                    help="选择要统计的股票范围"
                )
                
                topk_value = None
                if sample_choice == "仅Top-K":
                    topk_value = st.number_input(
                        "Top-K值",
                        min_value=1,
                        max_value=500,
                        value=50,
                        key="strategy_topk"
                    )
                
                if st.button("生成策略触发统计", key="btn_run_strategy_stats", width='stretch'):
                    try:
                        with st.spinner("正在统计策略触发情况..."):
                            # 1) 获取样本股票列表
                            codes_sample = []
                            if sample_choice == "全市场":
                                codes_sample = get_stock_list(adj_type="qfq")
                            elif sample_choice == "仅Top-K":
                                df_rank = _slice_top_from_all(ref_date_strategy, topk_value)
                                if not df_rank.empty:
                                    codes_sample = df_rank["ts_code"].astype(str).tolist()
                                if not codes_sample:
                                    st.error(f"无法从排名文件读取Top-{topk_value}股票")
                            elif sample_choice == "仅白名单":
                                from config import SC_WHITELIST
                                codes_sample = SC_WHITELIST if SC_WHITELIST else []
                            elif sample_choice == "仅黑名单":
                                from config import SC_BLACKLIST
                                codes_sample = SC_BLACKLIST if SC_BLACKLIST else []
                            
                            if not codes_sample:
                                st.error("无法获取样本股票列表")
                            
                            # 2) 获取区间内的所有交易日（从参考日开始往后推N个交易日）
                            all_trade_dates_list = get_trade_dates() or []
                            if not all_trade_dates_list:
                                st.error("无法获取交易日列表")
                            else:
                                # 找到参考日在交易日列表中的位置
                                try:
                                    ref_idx = all_trade_dates_list.index(ref_date_strategy)
                                    # 从参考日开始，往后取 interval_days 个交易日（包含参考日）
                                    end_idx = min(ref_idx + interval_days, len(all_trade_dates_list))
                                    interval_dates = all_trade_dates_list[ref_idx:end_idx]
                                except ValueError:
                                    st.error(f"参考日 {ref_date_strategy} 不在交易日列表中")
                                    interval_dates = []
                            
                            if not interval_dates or len(interval_dates) < 1:
                                st.error(f"无法获取足够的交易日（参考日：{ref_date_strategy}）")
                            else:
                                st.info(f"统计区间：{interval_dates[0]} 至 {interval_dates[-1]}（共 {len(interval_dates)} 个交易日）")
                                st.info(f"样本股票数：{len(codes_sample)}")

                            # 为“触发后最高排名”准备追踪日期窗口（包含触发当日，向后 track_days 个交易日）
                            obs_forward_dates_map: dict[str, list[str]] = {}
                            rank_dates_needed: set[str] = set()
                            if all_trade_dates_list and interval_dates:
                                idx_map = {d: i for i, d in enumerate(all_trade_dates_list)}
                                for d in interval_dates:
                                    idx = idx_map.get(d)
                                    if idx is None:
                                        continue
                                    end_idx = min(idx + track_days, len(all_trade_dates_list))
                                    forward_dates = all_trade_dates_list[idx:end_idx]
                                    if forward_dates:
                                        obs_forward_dates_map[d] = forward_dates
                                        rank_dates_needed.update(forward_dates)
                            
                            # 3) 统计每个交易日的策略触发情况
                            # _count_strategy_triggers 已在本文件中定义
                            
                            all_trigger_stats = []
                            strategy_stocks_map = {}  # 策略名 -> 触发该策略的股票集合（按日期）
                            
                            # 调试信息：检查详情数据来源
                            debug_info = []
                            
                            for obs_date in interval_dates:
                                # 先尝试从数据库读取
                                df_triggers = None
                                db_path = get_details_db_path_with_fallback()
                                
                                if db_path and is_details_db_available():
                                    try:
                                        from database_manager import get_database_manager
                                        manager = get_database_manager()
                                        if not manager:
                                            raise RuntimeError("无法获取数据库管理器")
                                        sql = "SELECT ts_code, rules FROM stock_details WHERE ref_date = ?"
                                        df_details = manager.execute_sync_query(db_path, sql, [obs_date], timeout=30.0)
                                        
                                        if not df_details.empty:
                                            # 从数据库数据构建触发统计
                                            codes_set = set(str(c) for c in codes_sample)
                                            acc_set = {}  # 策略名 -> 触发股票集合
                                            
                                            for _, detail_row in df_details.iterrows():
                                                ts = str(detail_row["ts_code"])
                                                if codes_set and ts not in codes_set:
                                                    continue
                                                try:
                                                    # 解析 rules 字段：优先 json.loads，失败则 ast.literal_eval，最后保证是 list[dict]
                                                    rules_raw = detail_row.get("rules")
                                                    rules = []
                                                    if rules_raw:
                                                        if isinstance(rules_raw, str):
                                                            try:
                                                                rules = json.loads(rules_raw)
                                                            except Exception:
                                                                try:
                                                                    import ast
                                                                    rules = ast.literal_eval(rules_raw)
                                                                except Exception:
                                                                    rules = []
                                                        elif isinstance(rules_raw, list):
                                                            rules = rules_raw
                                                    
                                                    # 确保 rules 是 list[dict] 格式
                                                    if not isinstance(rules, list):
                                                        rules = []
                                                    
                                                    # 处理规则列表（可能是 per_rules 或 rules）
                                                    for r in rules:
                                                        if not isinstance(r, dict):
                                                            continue
                                                        name = str(r.get("name") or "")
                                                        if not name:
                                                            continue
                                                        ok = bool(r.get("ok"))
                                                        hd = str(r.get("hit_date") or "")
                                                        hds = [str(x) for x in (r.get("hit_dates") or [])]
                                                        trig = ok or (hd == str(obs_date)) or (str(obs_date) in hds)
                                                        if trig:
                                                            acc_set.setdefault(name, set()).add(ts)
                                                except Exception:
                                                    continue
                                            
                                            # 构建DataFrame
                                            if acc_set:
                                                rows = []
                                                n_sample = len(codes_set) if codes_set else 0
                                                for name, stocks_set in acc_set.items():
                                                    cnt = int(len(stocks_set))
                                                    cov = (cnt / n_sample if n_sample else 0.0)
                                                    rows.append({
                                                        "name": name,
                                                        "trigger_count": cnt,
                                                        "n_sample": int(n_sample),
                                                        "coverage": cov,
                                                        "trigger_weighted": float(cnt),
                                                        "sample_weight": float(n_sample),
                                                        "coverage_weighted": cov,
                                                    })
                                                
                                                if rows:
                                                    df_triggers = pd.DataFrame(rows, columns=["name","trigger_count","n_sample","coverage","trigger_weighted","sample_weight","coverage_weighted"])
                                                    df_triggers = df_triggers.sort_values(["trigger_count", "name"], ascending=[False, True]).reset_index(drop=True)
                                                    debug_info.append(f"{obs_date}: 从数据库读取，找到 {len(df_details)} 条记录，{len(rows)} 个策略触发")
                                                else:
                                                    debug_info.append(f"{obs_date}: 从数据库读取，找到 {len(df_details)} 条记录，但无策略触发")
                                            else:
                                                debug_info.append(f"{obs_date}: 从数据库读取，找到 {len(df_details)} 条记录，但无策略触发")
                                        else:
                                            debug_info.append(f"{obs_date}: 数据库中没有该日期的详情数据")
                                    except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                                        # 数据库文件不存在或管理器获取失败
                                        debug_info.append(f"{obs_date}: 数据库访问失败: {e}")
                                    except Exception as e:
                                        # 数据库读取失败（可能是表不存在、连接错误等）
                                        error_msg = str(e).lower()
                                        if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                                            debug_info.append(f"{obs_date}: Details数据库表不存在")
                                        else:
                                            debug_info.append(f"{obs_date}: 数据库读取失败: {e}")
                                
                                # 如果数据库读取失败，尝试从文件系统读取
                                if df_triggers is None or df_triggers.empty:
                                    df_triggers = _count_strategy_triggers(obs_date, codes_sample)
                                    if not df_triggers.empty:
                                        debug_info.append(f"{obs_date}: 从文件系统读取，找到策略触发")
                                    else:
                                        # 检查文件是否存在
                                        ddir = Path("output/score/details") / str(obs_date)
                                        if ddir.exists():
                                            file_count = len(list(ddir.glob("*.json")))
                                            debug_info.append(f"{obs_date}: 文件系统目录存在，有 {file_count} 个JSON文件，但无策略触发")
                                        else:
                                            debug_info.append(f"{obs_date}: 文件系统目录不存在")
                                
                                if not df_triggers.empty:
                                    df_triggers["obs_date"] = obs_date
                                    all_trigger_stats.append(df_triggers)
                                    
                                    # 记录每个策略触发的股票（从详情数据中读取）
                                    for _, row in df_triggers.iterrows():
                                        strategy_name = row["name"]
                                        triggered_stocks = set()
                                        
                                        # 优先从数据库读取，失败则从文件系统读取
                                        try:
                                            # 尝试从数据库读取
                                            db_path = get_details_db_path_with_fallback()
                                            if db_path and is_details_db_available():
                                                from database_manager import get_database_manager
                                                manager = get_database_manager()
                                                if not manager:
                                                    raise RuntimeError("无法获取数据库管理器")
                                                sql = "SELECT ts_code, rules FROM stock_details WHERE ref_date = ?"
                                                df_details = manager.execute_sync_query(db_path, sql, [obs_date], timeout=30.0)
                                                
                                                if not df_details.empty:
                                                    for _, detail_row in df_details.iterrows():
                                                        ts = str(detail_row["ts_code"])
                                                        if ts not in codes_sample:
                                                            continue
                                                        try:
                                                            # 解析 rules 字段
                                                            rules_raw = detail_row.get("rules")
                                                            rules = []
                                                            if rules_raw:
                                                                if isinstance(rules_raw, str):
                                                                    try:
                                                                        rules = json.loads(rules_raw)
                                                                    except Exception:
                                                                        try:
                                                                            import ast
                                                                            rules = ast.literal_eval(rules_raw)
                                                                        except Exception:
                                                                            rules = []
                                                                elif isinstance(rules_raw, list):
                                                                    rules = rules_raw
                                                            
                                                            # 确保 rules 是 list[dict] 格式
                                                            if not isinstance(rules, list):
                                                                rules = []
                                                            
                                                            for r in rules:
                                                                if not isinstance(r, dict):
                                                                    continue
                                                                name = str(r.get("name") or "")
                                                                if name == strategy_name:
                                                                    ok = bool(r.get("ok"))
                                                                    hd = str(r.get("hit_date") or "")
                                                                    hds = [str(x) for x in (r.get("hit_dates") or [])]
                                                                    trig = ok or (hd == str(obs_date)) or (str(obs_date) in hds)
                                                                    if trig:
                                                                        triggered_stocks.add(ts)
                                                                        break
                                                        except Exception:
                                                            continue
                                        except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                                            # 数据库文件不存在或管理器获取失败，忽略错误
                                            logger.debug(f"数据库访问失败: {obs_date}: {e}")
                                            pass
                                        except Exception as e:
                                            # 数据库读取失败（可能是表不存在、连接错误等），忽略错误
                                            error_msg = str(e).lower()
                                            if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                                                logger.debug(f"Details数据库表不存在: {obs_date}")
                                            else:
                                                logger.debug(f"数据库读取失败: {obs_date}: {e}")
                                            pass
                                        
                                        # 如果数据库读取失败或没有数据，从文件系统读取
                                        if not triggered_stocks:
                                            ddir = Path("output/score/details") / str(obs_date)
                                            if ddir.exists():
                                                for fp in ddir.glob("*.json"):
                                                    try:
                                                        obj = json.loads(fp.read_text(encoding="utf-8-sig"))
                                                        ts = str(obj.get("ts_code", ""))
                                                        if ts not in codes_sample:
                                                            continue
                                                        rules = (obj.get("per_rules") or obj.get("rules") or [])
                                                        for r in rules:
                                                            name = str(r.get("name") or "")
                                                            if name == strategy_name:
                                                                ok = bool(r.get("ok"))
                                                                hd = str(r.get("hit_date") or "")
                                                                hds = [str(x) for x in (r.get("hit_dates") or [])]
                                                                trig = ok or (hd == str(obs_date)) or (str(obs_date) in hds)
                                                                if trig:
                                                                    triggered_stocks.add(ts)
                                                                    break
                                                    except Exception:
                                                        continue
                                        
                                        if triggered_stocks:
                                            if strategy_name not in strategy_stocks_map:
                                                strategy_stocks_map[strategy_name] = {}
                                            strategy_stocks_map[strategy_name][obs_date] = triggered_stocks
                            
                            # 显示调试信息
                            if debug_info:
                                with st.expander("调试信息（点击查看数据读取情况）", expanded=False):
                                    for info in debug_info:
                                        st.text(info)
                            
                            if not all_trigger_stats:
                                st.warning("未找到任何策略触发数据")
                                st.info('提示：请确保：\n1. 参考日及区间内的日期都有评分数据\n2. 详情数据已正确存储（数据库或文件系统）\n3. 样本股票列表正确\n4. 展开上方的"调试信息"查看详细情况')
                            else:
                                # 4) 汇总统计
                                df_all = pd.concat(all_trigger_stats, ignore_index=True)

                                # 额外计算：触发后最高排名（触发日及之后 track_days 个交易日内的最好名次，越小越好）
                                best_rank_map: dict[tuple[str, str], int] = {}
                                if strategy_stocks_map and rank_dates_needed:
                                    rank_lookup: dict[str, dict[str, int]] = {}
                                    for d in sorted(rank_dates_needed):
                                        df_rank = _read_rank_all_sorted(d)
                                        if df_rank is None or df_rank.empty:
                                            continue
                                        try:
                                            df_rank = df_rank[["ts_code", "rank"]].dropna(subset=["ts_code", "rank"])
                                            if df_rank.empty:
                                                continue
                                            rank_lookup[d] = dict(zip(df_rank["ts_code"].astype(str), df_rank["rank"].astype(int)))
                                        except Exception as e:
                                            logger.debug(f"构建排名映射失败 {d}: {e}")
                                            continue
                                    
                                    if rank_lookup:
                                        for strategy_name, date_stocks in strategy_stocks_map.items():
                                            for obs_date, stocks in date_stocks.items():
                                                forward_dates = obs_forward_dates_map.get(obs_date, [])
                                                best_rank_val: int | None = None
                                                for d in forward_dates:
                                                    rmap = rank_lookup.get(d)
                                                    if not rmap:
                                                        continue
                                                    for ts in stocks:
                                                        rv = rmap.get(str(ts))
                                                        if rv is None:
                                                            continue
                                                        try:
                                                            rv_int = int(rv)
                                                        except Exception:
                                                            continue
                                                        best_rank_val = rv_int if best_rank_val is None else min(best_rank_val, rv_int)
                                                if best_rank_val is not None:
                                                    best_rank_map[(obs_date, strategy_name)] = best_rank_val
                                
                                if not df_all.empty:
                                    if best_rank_map:
                                        df_all["best_rank_after_trigger"] = [
                                            best_rank_map.get((row["obs_date"], row["name"])) for _, row in df_all.iterrows()
                                        ]
                                    else:
                                        df_all["best_rank_after_trigger"] = [None] * len(df_all)
                                
                                # 按策略名汇总
                                strategy_summary = df_all.groupby("name").agg({
                                    "trigger_count": ["sum", "max"],
                                    "coverage": ["max"],
                                    "obs_date": "nunique"
                                }).reset_index()
                                strategy_summary.columns = ["策略名", "总触发次数", "最大单日触发", 
                                                           "最大覆盖率", "触发天数"]
                                
                                # 格式化百分比
                                strategy_summary["最大覆盖率"] = strategy_summary["最大覆盖率"].map(
                                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else None
                                )
                                
                                # 按总触发次数降序排序
                                strategy_summary = strategy_summary.sort_values("总触发次数", ascending=False).reset_index(drop=True)
                                
                                st.subheader("策略触发汇总统计")
                                st.dataframe(strategy_summary, width='stretch', height=400)
                                
                                # 5) 详细触发情况（按日期）
                                st.subheader("每日策略触发详情")
                                df_detail_display = df_all[["obs_date", "name", "trigger_count", "n_sample", "coverage", "best_rank_after_trigger"]].copy()
                                df_detail_display.columns = ["日期", "策略名", "触发次数", "样本数", "覆盖率", "触发后最高排名"]
                                df_detail_display["覆盖率"] = df_detail_display["覆盖率"].map(
                                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else None
                                )
                                df_detail_display["触发后最高排名"] = df_detail_display["触发后最高排名"].map(
                                    lambda x: int(x) if pd.notna(x) else "-"
                                )
                                df_detail_display = df_detail_display.sort_values(["日期", "触发次数"], ascending=[True, False]).reset_index(drop=True)
                                st.dataframe(df_detail_display, width='stretch', height=400)
                                
                                # 6) 未触发的策略
                                try:
                                    show_when_untriggered = st.checkbox(
                                        "显示规则 when 表达式",
                                        value=False,
                                        key="stats_show_when"
                                    )
                                    
                                    # 参考所有策略列表，找出统计区间内未触发的策略
                                    all_rules = getattr(se, "SC_RULES", []) or []
                                    triggered_rule_names = set(df_all["name"].astype(str).unique())
                                    
                                    # 预先构建 name -> when / explain 映射
                                    name_to_when = {}
                                    name_to_explain = {}
                                    for r in all_rules:
                                        rule_name = str(r.get("name", ""))
                                        if not rule_name:
                                            continue
                                        if "clauses" in r and r["clauses"]:
                                            ws = [c.get("when", "") for c in r["clauses"] if c.get("when")]
                                            name_to_when[rule_name] = " AND ".join(ws)
                                        else:
                                            name_to_when[rule_name] = str(r.get("when", ""))
                                        
                                        explain_val = r.get("explain")
                                        if explain_val:
                                            name_to_explain[rule_name] = str(explain_val)
                                    
                                    untriggered_rules = []
                                    for r in all_rules:
                                        rule_name = str(r.get("name", ""))
                                        if not rule_name or rule_name in triggered_rule_names:
                                            continue
                                        
                                        tf = str(r.get("timeframe", "D")).upper()
                                        scope = str(r.get("scope", "ANY")).upper().strip()
                                        win = None if scope == "LAST" else int(r.get("score_windows", 60))
                                        points = float(r.get("points", 0))
                                        
                                        gate = r.get("gate")
                                        gate_str = ""
                                        if gate:
                                            if isinstance(gate, dict):
                                                gate_when = gate.get("when", "")
                                                if gate_when:
                                                    gate_str = f"gate: {gate_when}"
                                            elif isinstance(gate, str):
                                                gate_str = f"gate: {gate}"
                                        
                                        # 汇总子句信息
                                        clauses_info = ""
                                        if "clauses" in r and r.get("clauses"):
                                            clauses = r.get("clauses", [])
                                            clause_parts = []
                                            for c in clauses:
                                                c_tf = str(c.get("timeframe", "D")).upper()
                                                c_scope = str(c.get("scope", "ANY")).upper().strip()
                                                if c_scope == "LAST":
                                                    clause_parts.append(f"{c_tf}/-/LAST")
                                                else:
                                                    c_win = int(c.get("score_windows", 60))
                                                    clause_parts.append(f"{c_tf}/{c_win}/{c_scope}")
                                            if clause_parts:
                                                clauses_info = f"子句: {len(clauses)}个 ({', '.join(clause_parts)})"
                                        
                                        rule_data = {
                                            "name": rule_name,
                                            "timeframe": tf,
                                            "scope": scope,
                                            "points": points,
                                            "gate": gate_str if gate_str else "",
                                            "clauses": clauses_info if clauses_info else "",
                                            "when": name_to_when.get(rule_name, ""),
                                            "explain": name_to_explain.get(rule_name, str(r.get("explain", "")))
                                        }
                                        if scope != "LAST":
                                            rule_data["score_windows"] = win
                                        
                                        untriggered_rules.append(rule_data)
                                    
                                    if untriggered_rules:
                                        st.subheader("未触发的策略")
                                        untriggered_df = pd.DataFrame(untriggered_rules)
                                        
                                        # 可选隐藏 when 列
                                        if not show_when_untriggered and "when" in untriggered_df.columns:
                                            untriggered_df = untriggered_df.drop(columns=["when"])
                                        
                                        # 确定列顺序
                                        col_order = ["name"]
                                        for col in ["timeframe", "scope", "points"]:
                                            if col in untriggered_df.columns:
                                                col_order.append(col)
                                        if "score_windows" in untriggered_df.columns:
                                            if "scope" in untriggered_df.columns:
                                                has_non_last = (untriggered_df["scope"].astype(str).str.upper() != "LAST").any()
                                                if has_non_last:
                                                    col_order.append("score_windows")
                                            else:
                                                col_order.append("score_windows")
                                        for col in ["gate", "clauses"]:
                                            if col in untriggered_df.columns:
                                                col_order.append(col)
                                        if show_when_untriggered and "when" in untriggered_df.columns:
                                            col_order.append("when")
                                        if "explain" in untriggered_df.columns:
                                            col_order.append("explain")
                                        col_order = [c for c in col_order if c in untriggered_df.columns]
                                        untriggered_display = untriggered_df[col_order].copy()
                                        
                                        # 列配置
                                        untriggered_column_config = {}
                                        try:
                                            if "name" in untriggered_display.columns:
                                                untriggered_column_config["name"] = st.column_config.TextColumn("策略名称", help="策略的简短名称")
                                            if "timeframe" in untriggered_display.columns:
                                                untriggered_column_config["timeframe"] = st.column_config.TextColumn("时间周期", help="D(日线)/W(周线)/M(月线)", width="small")
                                            if "score_windows" in untriggered_display.columns:
                                                untriggered_column_config["score_windows"] = st.column_config.TextColumn("计分窗口", help="计分窗口条数（scope 为 LAST 时为空）", width="small")
                                            if "scope" in untriggered_display.columns:
                                                untriggered_column_config["scope"] = st.column_config.TextColumn("命中口径", help="ANY/EACH/PERBAR等", width="small")
                                            if "points" in untriggered_display.columns:
                                                untriggered_column_config["points"] = st.column_config.NumberColumn("分数", help="命中时加/减分", width="small")
                                            if "gate" in untriggered_display.columns:
                                                untriggered_column_config["gate"] = st.column_config.TextColumn("前置门槛", help="前置条件表达式", width="medium")
                                            if "clauses" in untriggered_display.columns:
                                                untriggered_column_config["clauses"] = st.column_config.TextColumn("子句信息", help="多子句组合信息", width="medium")
                                            if "when" in untriggered_display.columns:
                                                untriggered_column_config["when"] = st.column_config.TextColumn("条件表达式", help="TDX风格表达式", width="large")
                                            if "explain" in untriggered_display.columns:
                                                untriggered_column_config["explain"] = st.column_config.TextColumn("详细说明", help="策略的详细说明", width="medium")
                                        except Exception:
                                            untriggered_column_config = None
                                        
                                        st.dataframe(
                                            untriggered_display,
                                            width='stretch',
                                            height=420,
                                            hide_index=True,
                                            column_config=untriggered_column_config if untriggered_column_config else None
                                        )
                                        st.caption("以上策略在所选统计区间内未出现触发。")
                                    else:
                                        st.info("所选区间内所有策略均有触发。")
                                except Exception as e:
                                    logger.debug(f"显示未触发策略失败: {e}")
                                
                                # 7) 后续跟踪
                                if track_days > 0 and strategy_stocks_map:
                                    st.subheader("触发策略的后续跟踪")
                                    
                                    # 获取跟踪日期范围
                                    last_obs_date = interval_dates[-1]
                                    # 从最后一个观察日的下一个交易日开始跟踪
                                    # 获取所有交易日列表，找到 last_obs_date 的下一个交易日
                                    all_trade_dates = get_trade_dates() or []
                                    if not all_trade_dates:
                                        st.warning("无法获取交易日列表")
                                    else:
                                        # 找到 last_obs_date 在交易日列表中的位置
                                        try:
                                            last_idx = all_trade_dates.index(last_obs_date)
                                            # 从下一个交易日开始，取 track_days 个交易日
                                            next_idx = last_idx + 1
                                            if next_idx >= len(all_trade_dates):
                                                st.warning(f"参考日 {last_obs_date} 是最后一个交易日，无法进行后续跟踪")
                                            else:
                                                track_dates = all_trade_dates[next_idx:next_idx + track_days]
                                                if len(track_dates) < 1:
                                                    st.warning("无法获取足够的跟踪日期")
                                                else:
                                                    track_start = track_dates[0]
                                                    track_end = track_dates[-1]
                                                    
                                                    st.info(f"跟踪区间：{track_start} 至 {track_end}（共 {len(track_dates)} 个交易日）")
                                                    
                                                    # 读取价格数据
                                                    # _read_stock_prices 已在本文件中定义
                                                    
                                                    # 收集所有需要跟踪的股票
                                                    all_track_stocks = set()
                                                    for strategy_name, date_stocks in strategy_stocks_map.items():
                                                        for stocks in date_stocks.values():
                                                            all_track_stocks.update(stocks)
                                                    
                                                    if all_track_stocks:
                                                        with st.spinner(f"正在读取 {len(all_track_stocks)} 只股票的价格数据..."):
                                                            df_prices = _read_stock_prices(list(all_track_stocks), start=track_start, end=track_end)
                                                        
                                                        if df_prices.empty:
                                                            st.warning("无法读取价格数据")
                                                        else:
                                                            # 按策略统计后续表现
                                                            track_results = []
                                                            
                                                            for strategy_name, date_stocks in strategy_stocks_map.items():
                                                                # 收集该策略在所有日期触发的股票（去重）
                                                                strategy_stocks = set()
                                                                for stocks in date_stocks.values():
                                                                    strategy_stocks.update(stocks)
                                                                
                                                                if not strategy_stocks:
                                                                    continue
                                                                
                                                                # 计算这些股票在跟踪区间的表现
                                                                strategy_prices = df_prices[df_prices["ts_code"].isin(strategy_stocks)]
                                                                if strategy_prices.empty:
                                                                    continue
                                                                
                                                                # 按股票分组计算涨幅
                                                                stock_returns = []
                                                                for ts_code, group in strategy_prices.groupby("ts_code"):
                                                                    group = group.sort_values("trade_date")
                                                                    if len(group) < 2:
                                                                        continue
                                                                    
                                                                    start_price = group.iloc[0]["close"]
                                                                    end_price = group.iloc[-1]["close"]
                                                                    if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                                                                        ret = (end_price / start_price - 1.0) * 100.0
                                                                        stock_returns.append(ret)
                                                                
                                                                if stock_returns:
                                                                    track_results.append({
                                                                        "策略名": strategy_name,
                                                                        "触发股票数": len(strategy_stocks),
                                                                        "可跟踪股票数": len(stock_returns),
                                                                        "平均涨幅": np.mean(stock_returns),
                                                                        "最大涨幅": np.max(stock_returns),
                                                                        "最小涨幅": np.min(stock_returns),
                                                                        "正收益数": sum(1 for r in stock_returns if r > 0),
                                                                        "正收益比例": sum(1 for r in stock_returns if r > 0) / len(stock_returns) * 100 if stock_returns else 0
                                                                    })
                                                            
                                                            if track_results:
                                                                df_track = pd.DataFrame(track_results)
                                                                
                                                                # 先保存原始数值用于排序
                                                                df_track["_sort_avg_ret"] = df_track["平均涨幅"]
                                                                
                                                                # 格式化百分比
                                                                for col in ["平均涨幅", "最大涨幅", "最小涨幅"]:
                                                                    if col in df_track.columns:
                                                                        df_track[col] = df_track[col].map(
                                                                            lambda x: f"{x:.2f}%" if pd.notna(x) else None
                                                                        )
                                                                df_track["正收益比例"] = df_track["正收益比例"].map(
                                                                    lambda x: f"{x:.1f}%" if pd.notna(x) else None
                                                                )
                                                                
                                                                # 按平均涨幅降序排序（使用原始数值）
                                                                df_track = df_track.sort_values("_sort_avg_ret", ascending=False).reset_index(drop=True)
                                                                # 删除临时排序列
                                                                df_track = df_track.drop(columns=["_sort_avg_ret"])
                                                                
                                                                st.dataframe(df_track, width='stretch', height=400)
                                                                st.caption(f"跟踪结果：统计在区间内触发各策略的股票，在后续 {track_days} 个交易日的表现")
                                                            else:
                                                                st.warning("无法计算后续跟踪数据")
                                        except ValueError:
                                            st.warning(f"参考日 {last_obs_date} 不在交易日列表中")
                                        
                    except Exception as e:
                        st.error(f"生成失败：{e}")
                        import traceback
                        st.code(traceback.format_exc())


    # ================== 数据管理 ==================
    with tab_data_view:
        st.subheader("数据管理")
        
        # ===== 数据库基础检查和下载按钮 =====
        # 延迟导入 download 模块
        dl = _lazy_import_download()
        if dl is None:
            st.error("无法导入 download 模块")
        else:
            # 获取基础配置（优先使用config，如果没有则使用download模块的默认值）
            base = str(getattr(cfg, "DATA_ROOT", "./data"))
            api_adj = str(getattr(cfg, "API_ADJ", getattr(dl, "API_ADJ", "qfq"))).lower()
            
            # 下载配置项
            st.markdown("#### 数据下载配置")
            with st.expander("下载参数配置", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    # 从config获取默认值，如果没有则使用download模块的值
                    start_default = str(getattr(cfg, "START_DATE", getattr(dl, "START_DATE", "20200101")))
                    start_use = st.text_input("起始日 START_DATE (YYYYMMDD)", value=start_default, key="dl_start_date")
                    
                    end_default_cfg = getattr(cfg, "END_DATE", "today")
                    if str(end_default_cfg).strip().lower() == "today":
                        end_default = "today"
                    else:
                        end_default = str(end_default_cfg)
                    end_input = st.text_input("结束日 END_DATE ('today' 或 YYYYMMDD)", value=end_default, key="dl_end_date")
                    
                    assets_default = list(getattr(cfg, "ASSETS", getattr(dl, "ASSETS", ["stock", "index"]))) or ["stock", "index"]
                    assets = st.multiselect("资产 ASSETS", ["stock", "index"], default=assets_default, key="dl_assets")
                
                with c2:
                    api_adj_options = ["qfq", "hfq", "raw"]
                    api_adj_index = api_adj_options.index(api_adj) if api_adj in api_adj_options else 0
                    api_adj = st.selectbox("复权 API_ADJ", api_adj_options, index=api_adj_index, key="dl_api_adj").lower()
                    
                    fast_threads_default = int(getattr(cfg, "FAST_INIT_THREADS", getattr(dl, "FAST_INIT_THREADS", 16)))
                    fast_threads = st.number_input("FAST_INIT 并发", min_value=1, max_value=64, value=fast_threads_default, key="dl_fast_threads")
                    
                    inc_threads_default = int(getattr(cfg, "STOCK_INC_THREADS", getattr(dl, "STOCK_INC_THREADS", 16)))
                    inc_threads = st.number_input("增量下载线程", min_value=1, max_value=64, value=inc_threads_default, key="dl_inc_threads")
                
                with c3:
                    ind_workers_default = int(getattr(cfg, "INC_RECALC_WORKERS", getattr(dl, "INC_RECALC_WORKERS", 32)))
                    ind_workers = st.number_input("指标重算线程(可选)", min_value=0, max_value=128, value=ind_workers_default, key="dl_ind_workers")
                    
                    token_default_cfg = getattr(cfg, "TOKEN", "")
                    token_default = "" if _is_placeholder_token(token_default_cfg) else str(token_default_cfg)
                    token_input = st.text_input(
                        "Tushare Token",
                        value=token_default,
                        key="dl_token",
                        type="password",
                        help="留空则使用环境变量或 config.TOKEN；不保存，仅本次使用",
                    )
                    
                    st.caption(f"数据根目录: {base}")
            
            # 处理结束日期
            end_use = _today_str() if str(end_input).strip().lower() == "today" else str(end_input).strip()
            start_use = str(start_use).strip()
            
            # 应用参数
            _apply_overrides(
                base,
                assets,
                start_use,
                end_use,
                api_adj,
                int(fast_threads),
                int(inc_threads),
                int(ind_workers) if ind_workers else None,
                token=token_input,
            )
            
            # 显示当前状态
            latest = _latest_trade_date(base, api_adj)
            if latest:
                st.caption(f"当前 {api_adj} 最近交易日：{latest}")
            
            # 下载按钮
            run_download = st.button("🚀 运行下载", width='stretch', type="primary", key="run_download_btn")
            
            if run_download:
                token_use = _resolve_token_from_ui(token_input)
                if not token_use:
                    st.error("请填写有效的 Tushare Token 或设置环境变量 TUSHARE_TOKEN")
                else:
                    logger.info(f"用户点击运行下载: 起始日期={start_use}, 结束日期={end_use}")
                    try:
                        # 直接调用download模块，让它自己判断是否为增量
                        steps = [
                            "准备环境",
                            "数据下载（自动判断首次/增量）",
                            "清理与校验",
                        ]
                        sp = Stepper("数据下载", steps, key_prefix="dl_auto")
                        sp.start()
                        sp.step("准备环境")
                        sp.step("数据下载（自动判断首次/增量）")
                        
                        # 调用download模块的主函数，让它自己判断
                        dl = _lazy_import_download()
                        if dl is not None:
                            results = dl.download_data(
                                start_date=start_use,
                                end_date=end_use,
                                adj_type=api_adj,
                                assets=assets,
                                threads=int(inc_threads),
                                enable_warmup=True,
                                enable_adaptive_rate_limit=True,
                                token=token_use,
                            )
                            # 显示下载结果
                            for asset_type, stats in results.items():
                                st.success(f"{asset_type}: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
                        
                        sp.step("清理与校验")
                        sp.finish(True, "下载完成")
                    except Exception as e:
                        st.error(f"下载失败：{e}")
            
            st.divider()

        # ===== 手动维护概念数据 =====
        st.markdown("#### 手动维护概念数据")
        with st.expander("追加 / 更新概念映射", expanded=False):
            colg1, colg2 = st.columns(2)
            with colg1:
                ts_input = st.text_input("股票代码", placeholder="如 000001.SZ 或 600000", key="manual_concept_ts")
                stock_name_input = st.text_input("股票名称（可选）", value="", key="manual_concept_stock_name")
            with colg2:
                concept_name_input = st.text_input("概念名称（自动匹配代码）", placeholder="如 人工智能", key="manual_concept_name")
                st.caption("概念代码自动根据概念名称匹配；不存在则提示。")
            btn_concept_save = st.button("💾 追加/更新概念", key="btn_manual_concept_save")

        if btn_concept_save:
            try:
                ts_norm = normalize_ts(ts_input)
                if not ts_norm:
                    st.error("请填写有效的股票代码（如 000001.SZ）。")
                    st.stop()
                concept_name_val = str(concept_name_input or "").strip()
                if not concept_name_val:
                    st.error("概念名称不能为空。")
                    st.stop()
                concept_dir = Path(DATA_ROOT) / "concepts"
                concept_dir.mkdir(parents=True, exist_ok=True)
                concepts_path = concept_dir / "concepts.csv"
                members_path = concept_dir / "concept_members.csv"
                stock_map_path = concept_dir / "stock_concepts.csv"

                # 读取现有文件
                concepts_df = pd.read_csv(concepts_path, dtype=str) if concepts_path.exists() else pd.DataFrame(columns=["concept_code", "concept_name"])
                members_df = pd.read_csv(members_path, dtype=str) if members_path.exists() else pd.DataFrame(columns=["concept_code", "concept_name", "ts_code", "name"])

                # 根据概念名称匹配概念代码
                concept_code = None
                if not concepts_df.empty:
                    matches = concepts_df.loc[concepts_df["concept_name"] == concept_name_val]
                    if len(matches) == 1:
                        concept_code = matches["concept_code"].iloc[0]
                    elif len(matches) > 1:
                        st.error(f"概念名称 {concept_name_val} 存在多个代码，请先在概念文件中去重后再操作。")
                        st.stop()
                if not concept_code:
                    st.error(f"未找到概念名称 {concept_name_val} 对应的代码，请先更新概念列表。")
                    st.stop()

                # 更新概念列表
                mask_code = concepts_df["concept_code"] == concept_code if not concepts_df.empty else pd.Series([], dtype=bool)
                if mask_code.any():
                    concepts_df.loc[mask_code, "concept_name"] = concept_name_val
                else:
                    concepts_df = pd.concat([concepts_df, pd.DataFrame([{"concept_code": concept_code, "concept_name": concept_name_val}])], ignore_index=True)

                # 更新成分表（去重同一概念-股票组合）
                if not members_df.empty:
                    members_df = members_df[~((members_df["concept_code"] == concept_code) & (members_df["ts_code"] == ts_norm))]
                members_df = pd.concat(
                    [
                        members_df,
                        pd.DataFrame(
                            [
                                {
                                    "concept_code": concept_code,
                                    "concept_name": concept_name_val,
                                    "ts_code": ts_norm,
                                    "name": stock_name_input.strip() or None,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

                # 重新生成 stock_concepts.csv
                members_df_clean = members_df.copy()
                members_df_clean["concept_code"] = members_df_clean["concept_code"].fillna("")
                members_df_clean["concept_name"] = members_df_clean["concept_name"].fillna("")
                stock_map = (
                    members_df_clean.groupby("ts_code")
                    .agg(
                        concepts_code=("concept_code", lambda s: ",".join(sorted({x for x in s if x}))),
                        concepts_name=("concept_name", lambda s: ",".join(sorted({x for x in s if x}))),
                        stock_name=("name", lambda s: next((x for x in s if isinstance(x, str) and x.strip()), "")),
                    )
                    .reset_index()
                )

                # 落盘
                concepts_df.to_csv(concepts_path, index=False)
                members_df.to_csv(members_path, index=False)
                stock_map.to_csv(stock_map_path, index=False)

                st.success(f"已更新概念：{ts_norm} -> {concept_code} / {concept_name_val}")
                try:
                    import utils as ut
                    if hasattr(ut, "_load_concept_files"):
                        ut._load_concept_files.cache_clear()
                except Exception:
                    pass
            except Exception as e:
                st.error(f"更新概念失败：{e}")
        
        # ===== 数据浏览（保留原有功能） =====
        st.markdown("#### 数据浏览")
        st.info("用于可视化读取数据库原数据")
        
        # 数据源选择
        data_source = st.radio(
            "选择数据源",
            ["Details数据库", "股票原数据"],
            horizontal=True,
            help="Details: 存储评分详情数据 | 股票原数据: 存储股票行情和指标数据"
        )
        
        if data_source == "Details数据库":
            # Details数据库查看
            try:
                from database_manager import (
                    query_details_by_date,
                    query_details_by_stock,
                    query_details_top_stocks,
                    query_details_score_range,
                    query_details_recent_dates,
                    get_details_table_info,
                    get_details_db_path_with_fallback,
                    is_details_db_available
                )
                import os
                
                # 使用统一的函数获取details数据库路径（包含回退逻辑）
                db_path = get_details_db_path_with_fallback()
                
                if not db_path or not is_details_db_available():
                    st.warning(f"Details数据库不可用: {db_path if db_path else '路径获取失败'}")
                else:
                    # —— 数据库查询控制按钮 ——
                    data_view_db_enabled = st.session_state.get("data_view_db_enabled", False)
                    col_db_ctrl1, col_db_ctrl2 = st.columns([3, 1])
                    with col_db_ctrl1:
                        if data_view_db_enabled:
                            st.success("✅ 数据库查询已启用（可以查询数据库）")
                        else:
                            st.info("ℹ️ 数据库查询未启用（点击按钮后才会查询数据库，避免与写入操作冲突）")
                    with col_db_ctrl2:
                        if not data_view_db_enabled:
                            if st.button("🔓 启用数据库查询", key="enable_data_view_db"):
                                st.session_state["data_view_db_enabled"] = True
                                st.rerun()
                        else:
                            # 一旦启用就不再显示按钮，保持启用状态
                            pass
                    
                    # 只有在启用数据库查询后才执行查询操作
                    if not data_view_db_enabled:
                        st.warning("⚠️ 请先点击「启用数据库查询」按钮，然后才能查询数据库数据")
                    
                    
                    # 以下是所有数据库查询操作，只有在启用后才会执行
                    # 查询类型选择
                    query_type = st.selectbox(
                        "查询类型",
                        ["按日期查看", "按股票代码查看", "Top-K股票", "分数范围查询"],
                        key="details_query_type"
                    )
                    
                    # 获取并缓存最新日期（所有查询类型共享，含文件兜底）
                    @st.cache_data
                    def get_latest_details_date():
                        try:
                            dates = query_details_recent_dates(1, db_path)
                            if dates:
                                return dates[0]
                        except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                            # 数据库文件不存在或管理器获取失败
                            logger.debug(f"获取最新日期失败: {e}")
                        except Exception as e:
                            # 数据库读取失败（可能是表不存在、连接错误等）
                            error_msg = str(e).lower()
                            if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                                logger.debug(f"Details数据库表不存在，无法获取最新日期")
                            else:
                                logger.debug(f"获取最新日期失败: {e}")
                        # 数据库失败时，回退到文件系统：output/score/details/<YYYYMMDD> 目录
                        try:
                            ddir = Path(SC_OUTPUT_DIR) / "details"
                            if ddir.exists():
                                dirs = [p.name for p in ddir.iterdir() if p.is_dir() and _is_valid_date(p.name)]
                                if dirs:
                                    return max(dirs)
                        except Exception as e:
                            logger.debug(f"文件系统兜底获取最新日期失败: {e}")
                        return None
                    
                    latest_date = get_latest_details_date()
                    
                    if query_type == "按日期查看":
                        # 如果还没有设置默认日期，使用最新日期
                        if "details_date_value" not in st.session_state:
                            st.session_state["details_date_value"] = latest_date if latest_date else ""
                        
                        limit_param = st.number_input("返回记录数（-1为全部）", min_value=-1, value=200, key="details_limit")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            ref_date = st.text_input(
                                "参考日期（YYYYMMDD，留空=最新）",
                                value=st.session_state["details_date_value"],
                                key="details_date"
                            )
                        with col2:
                            st.write("")  # 占位
                            st.write("")
                            refresh_btn = st.button("刷新", key="details_refresh_btn")
                        
                        # 处理刷新或查询
                        date_to_use = ref_date.strip() if ref_date.strip() else latest_date
                        
                        if refresh_btn or date_to_use:
                            if not date_to_use:
                                st.error("未能确定参考日，无法查询。请手动输入或检查数据库。")
                            else:
                                try:
                                    df = query_details_by_date(date_to_use, limit=limit_param, db_path=db_path)
                                    if not df.empty:
                                        st.dataframe(df, width='stretch')
                                        st.info(f"共找到 {len(df)} 条记录 | 查询日期: {date_to_use}")
                                    else:
                                        st.warning("未找到数据")
                                except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                                    # 数据库文件不存在或管理器获取失败
                                    logger.debug(f"查询Details数据失败: {e}")
                                    st.error(f"查询失败: 数据库不可用（{e}）")
                                except Exception as e:
                                    # 数据库读取失败（可能是表不存在、连接错误等）
                                    error_msg = str(e).lower()
                                    if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                                        logger.debug(f"Details数据库表不存在，查询失败")
                                        st.error("查询失败: Details数据库表不存在，请先运行评分生成数据")
                                    else:
                                        logger.debug(f"查询Details数据失败: {e}")
                                        st.error(f"查询失败: {e}")
                        
                        # 显示最近的日期列表
                        try:
                            recent_dates = query_details_recent_dates(7, db_path)
                            if recent_dates:
                                st.caption(f"最近的交易日: {', '.join(recent_dates)}")
                        except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                            # 数据库文件不存在或管理器获取失败，忽略错误
                            logger.debug(f"获取最近日期列表失败: {e}")
                            pass
                        except Exception as e:
                            # 数据库读取失败（可能是表不存在、连接错误等），忽略错误
                            error_msg = str(e).lower()
                            if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                                logger.debug(f"Details数据库表不存在，无法获取最近日期列表")
                            else:
                                logger.debug(f"获取最近日期列表失败: {e}")
                            pass
                    
                    elif query_type == "按股票代码查看":
                        ts_code = st.text_input("股票代码（如000001）", key="details_ts_code")
                        limit = st.number_input("返回记录数", min_value=1, max_value=100, value=10)
                        
                        if ts_code:
                            try:
                                # 自动补后缀
                                ts_code_normalized = normalize_ts(ts_code.strip(), asset="stock")
                                df = query_details_by_stock(ts_code_normalized, limit, db_path)
                                if not df.empty:
                                    st.dataframe(df, width='stretch')
                                else:
                                    st.warning("未找到该股票的数据")
                            except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                                # 数据库文件不存在或管理器获取失败
                                logger.debug(f"查询Details数据失败: {e}")
                                st.error(f"查询失败: 数据库不可用（{e}）")
                            except Exception as e:
                                # 数据库读取失败（可能是表不存在、连接错误等）
                                error_msg = str(e).lower()
                                if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                                    logger.debug(f"Details数据库表不存在，查询失败")
                                    st.error("查询失败: Details数据库表不存在，请先运行评分生成数据")
                                else:
                                    logger.debug(f"查询Details数据失败: {e}")
                                    st.error(f"查询失败: {e}")
                    
                    elif query_type == "Top-K股票":
                        # 如果还没有设置默认日期，使用最新日期
                        if "details_topk_date_value" not in st.session_state:
                            st.session_state["details_topk_date_value"] = latest_date if latest_date else ""
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            ref_date = st.text_input("参考日期（YYYYMMDD）", value=st.session_state["details_topk_date_value"], key="details_topk_date")
                        with col2:
                            st.write("")  # 占位
                            st.write("")
                            refresh_topk_btn = st.button("刷新", key="details_topk_refresh_btn")
                        
                        top_k = st.number_input("Top-K", min_value=1, max_value=500, value=50)
                        
                        date_to_use = ref_date.strip() if ref_date.strip() else latest_date
                        
                        if refresh_topk_btn or date_to_use:
                            if not date_to_use:
                                st.error("未能确定参考日，无法查询。请手动输入或检查数据库。")
                            else:
                                try:
                                    df = query_details_top_stocks(date_to_use, top_k, db_path)
                                    if not df.empty:
                                        st.dataframe(df, width='stretch')
                                        st.info(f"查询日期: {date_to_use}")
                                    else:
                                        st.warning("未找到数据")
                                except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                                    # 数据库文件不存在或管理器获取失败
                                    logger.debug(f"查询Details数据失败: {e}")
                                    st.error(f"查询失败: 数据库不可用（{e}）")
                                except Exception as e:
                                    # 数据库读取失败（可能是表不存在、连接错误等）
                                    error_msg = str(e).lower()
                                    if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                                        logger.debug(f"Details数据库表不存在，查询失败")
                                        st.error("查询失败: Details数据库表不存在，请先运行评分生成数据")
                                    else:
                                        logger.debug(f"查询Details数据失败: {e}")
                                        st.error(f"查询失败: {e}")
                    
                    elif query_type == "分数范围查询":
                        # 如果还没有设置默认日期，使用最新日期
                        if "details_score_date_value" not in st.session_state:
                            st.session_state["details_score_date_value"] = latest_date if latest_date else ""
                        
                        col1_date, col2_date = st.columns([3, 1])
                        with col1_date:
                            ref_date = st.text_input("参考日期（YYYYMMDD）", value=st.session_state["details_score_date_value"], key="details_score_date")
                        with col2_date:
                            st.write("")  # 占位
                            st.write("")
                            refresh_score_btn = st.button("刷新", key="details_score_refresh_btn")
                        
                        col1_score, col2_score = st.columns(2)
                        with col1_score:
                            min_score = st.number_input("最低分数", value=50.0, key="details_score_min")
                        with col2_score:
                            max_score = st.number_input("最高分数", value=100.0, key="details_score_max")
                        
                        date_to_use = ref_date.strip() if ref_date.strip() else latest_date
                        
                        if refresh_score_btn or date_to_use:
                            if not date_to_use:
                                st.error("未能确定参考日，无法查询。请手动输入或检查数据库。")
                            else:
                                try:
                                    df = query_details_score_range(date_to_use, min_score, max_score, db_path)
                                    if not df.empty:
                                        st.dataframe(df, width='stretch')
                                        st.info(f"共找到 {len(df)} 条记录 | 查询日期: {date_to_use}")
                                    else:
                                        st.warning("未找到数据")
                                except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
                                    # 数据库文件不存在或管理器获取失败
                                    logger.debug(f"查询Details数据失败: {e}")
                                    st.error(f"查询失败: 数据库不可用（{e}）")
                                except Exception as e:
                                    # 数据库读取失败（可能是表不存在、连接错误等）
                                    error_msg = str(e).lower()
                                    if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
                                        logger.debug(f"Details数据库表不存在，查询失败")
                                        st.error("查询失败: Details数据库表不存在，请先运行评分生成数据")
                                    else:
                                        logger.debug(f"查询Details数据失败: {e}")
                                        st.error(f"查询失败: {e}")
                    
                    # 数据库信息（美化显示）
                    with st.expander("数据库信息"):
                        try:
                            info = get_details_table_info(db_path)
                            # 美化显示数据库信息
                            if isinstance(info, dict):
                                with st.container(border=True):
                                    for key, value in info.items():
                                        if isinstance(value, (int, float)):
                                            st.metric(key, value)
                                        elif isinstance(value, list):
                                            st.text(f"{key}: {len(value)} 项")
                                            if value and len(value) <= 10:
                                                for item in value:
                                                    st.text(f"  • {item}")
                                        elif isinstance(value, dict):
                                            st.text(f"{key}:")
                                            for sub_key, sub_value in value.items():
                                                st.text(f"  {sub_key}: {sub_value}")
                                        else:
                                            st.text(f"{key}: {value}")
                            else:
                                st.text(str(info))
                        except Exception as e:
                            st.error(f"获取信息失败: {e}")
            
            except Exception as e:
                st.error(f"初始化Details数据库读取器失败: {e}")
                import traceback
                st.text(traceback.format_exc())
        
        else:
            # 股票原数据查看
            from config import DATA_ROOT, UNIFIED_DB_PATH
            
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            if not os.path.exists(db_path):
                st.warning(f"股票数据数据库不存在: {db_path}")
            else:
                # 查询参数
                col1, col2 = st.columns(2)
                with col1:
                    asset_type = st.selectbox("资产类型", ["stock", "index"], index=0)
                    # 根据资产类型设置默认adj
                    default_adj = "ind" if asset_type == "index" else "qfq"
                with col2:
                    adj_type = st.selectbox(
                        "复权类型",
                        ["raw", "qfq", "hfq", "ind"],
                        index=["raw", "qfq", "hfq", "ind"].index(default_adj)
                    )
                
                view_mode = st.radio(
                    "查看模式",
                    ["单日查看", "区间查询", "单股历史"],
                    horizontal=True
                )
                
                @st.cache_data
                def get_trade_dates_list():
                    """获取交易日期列表"""
                    try:
                        dates = get_trade_dates(db_path)
                        return dates
                    except Exception as e:
                        st.error(f"获取交易日期失败: {e}")
                        return []
                
                # 延迟加载交易日期
                if 'trade_dates' not in st.session_state:
                    with st.spinner("正在加载交易日期..."):
                        st.session_state['trade_dates'] = get_trade_dates_list()
                
                if view_mode == "单日查看":
                    trade_dates = st.session_state['trade_dates']
                    if trade_dates:
                        selected_date = st.selectbox(
                            "选择日期",
                            trade_dates,
                            index=len(trade_dates)-1 if trade_dates else 0
                        )
                        
                        limit = st.number_input("显示行数（-1为全部）", value=100, min_value=-1, max_value=5000, step=50)
                        if limit == -1:
                            limit = None
                        
                        if st.button("查询", key="btn_query_day"):
                            try:
                                df = query_stock_data(
                                    db_path=db_path,
                                    start_date=selected_date,
                                    end_date=selected_date,
                                    adj_type=adj_type if asset_type != "index" else "ind",
                                    limit=limit
                                )
                                if not df.empty:
                                    st.dataframe(df, width='stretch')
                                    # 统计总行数（忽略limit）
                                    try:
                                        from database_manager import count_stock_data as _count_stock_data
                                        total_rows = _count_stock_data(
                                            db_path=db_path,
                                            ts_code=None,
                                            start_date=selected_date,
                                            end_date=selected_date,
                                            adj_type=adj_type if asset_type != "index" else "ind"
                                        )
                                    except Exception:
                                        total_rows = len(df)
                                    st.info(f"总行数: {total_rows} | 本次显示: {len(df)}")
                                else:
                                    st.warning("未找到数据")
                            except Exception as e:
                                st.error(f"查询失败: {e}")
                    else:
                        st.warning("无法获取交易日期列表")
                
                elif view_mode == "区间查询":
                    trade_dates = st.session_state['trade_dates']
                    if trade_dates:
                        col1, col2 = st.columns(2)
                        with col1:
                            start_date = st.selectbox("起始日期", trade_dates, index=len(trade_dates)-10 if len(trade_dates) >= 10 else 0)
                        with col2:
                            end_date = st.selectbox("结束日期", trade_dates, index=len(trade_dates)-1)
                        
                        ts_code = st.text_input("股票代码（留空=全市场，如000001.SZ）")
                        
                        limit = st.number_input("显示行数（-1为全部）", value=200, min_value=-1, max_value=10000, step=100)
                        if limit == -1:
                            limit = None
                        
                        if st.button("查询", key="btn_query_range"):
                            try:
                                # 自动补后缀
                                ts_code_normalized = normalize_ts(ts_code.strip(), asset=asset_type) if ts_code else None
                                df = query_stock_data(
                                    db_path=db_path,
                                    ts_code=ts_code_normalized,
                                    start_date=start_date,
                                    end_date=end_date,
                                    adj_type=adj_type if asset_type != "index" else "ind",
                                    limit=limit
                                )
                                if not df.empty:
                                    st.dataframe(df, width='stretch')
                                    # 统计总行数（忽略limit）
                                    try:
                                        from database_manager import count_stock_data as _count_stock_data
                                        total_rows = _count_stock_data(
                                            db_path=db_path,
                                            ts_code=ts_code_normalized,
                                            start_date=start_date,
                                            end_date=end_date,
                                            adj_type=adj_type if asset_type != "index" else "ind"
                                        )
                                    except Exception:
                                        total_rows = len(df)
                                    st.info(f"总行数: {total_rows} | 本次显示: {len(df)}")
                                else:
                                    st.warning("未找到数据")
                            except Exception as e:
                                st.error(f"查询失败: {e}")
                    
                    if not trade_dates:
                        st.warning("无法获取交易日期列表")
                
                elif view_mode == "单股历史":
                    ts_code = st.text_input("股票代码（如000001.SZ）", key="single_stock_code")
                    
                    limit = st.number_input("显示行数（-1为全部，>0=仅显示最新N行）", value=-1, min_value=-1, max_value=10000, step=100, key="single_limit")
                    if limit == -1:
                        limit = None
                    
                    if st.button("查询", key="btn_query_single"):
                        if not ts_code:
                            st.error("请输入股票代码")
                        else:
                            try:
                                # 自动补后缀
                                ts_code_normalized = normalize_ts(ts_code.strip(), asset=asset_type)

                                # 根据显示行数自动确定起止日期（取最新的 limit 个交易日）
                                start_date = end_date = None
                                effective_limit = int(limit) if limit else None
                                trade_dates = st.session_state.get('trade_dates', [])
                                if trade_dates and effective_limit:
                                    end_date = trade_dates[-1]
                                    start_idx = max(0, len(trade_dates) - effective_limit)
                                    start_date = trade_dates[start_idx]
                                
                                df = query_stock_data(
                                    db_path=db_path,
                                    ts_code=ts_code_normalized,
                                    start_date=start_date,
                                    end_date=end_date,
                                    adj_type=adj_type if asset_type != "index" else "ind",
                                    limit=None  # 手动控制行数，避免默认按最早日期排序
                                )
                                if not df.empty:
                                    # 按日期倒序展示，优先显示最新数据
                                    if "trade_date" in df.columns:
                                        df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)
                                    if effective_limit:
                                        df = df.head(effective_limit)

                                    st.dataframe(df, width='stretch')
                                    
                                    # 如果有收盘价数据，绘制图表
                                    if "trade_date" in df.columns and {"open","high","low","close"}.issubset(df.columns):
                                        try:
                                            df_chart = df.copy().sort_values("trade_date")
                                            df_chart = df_chart.tail(90)  # 仅展示最近90个交易日
                                            df_chart["trade_date"] = pd.to_datetime(df_chart["trade_date"])
                                            
                                            fig = make_subplots(
                                                rows=2, cols=1, shared_xaxes=True,
                                                row_heights=[0.7, 0.3],
                                                vertical_spacing=0.05,
                                                specs=[[{"secondary_y": False}], [{}]]
                                            )
                                            fig.add_trace(
                                                go.Candlestick(
                                                    x=df_chart["trade_date"],
                                                    open=df_chart["open"],
                                                    high=df_chart["high"],
                                                    low=df_chart["low"],
                                                    close=df_chart["close"],
                                                    increasing_line_color="#e74c3c",
                                                    decreasing_line_color="#2ecc71",
                                                    name="K线"
                                                ),
                                                row=1, col=1, secondary_y=False
                                            )
                                            # 多空信号叠加为连续曲线（short=黑，long=红；使用 duokong_short/duokong_long），列缺失则不显示
                                            try:
                                                signal_cols = [c for c in ("duokong_short", "duokong_long") if c in df_chart.columns]
                                                if signal_cols:
                                                    if "duokong_short" in df_chart.columns:
                                                        ser_short = pd.to_numeric(df_chart["duokong_short"], errors="coerce")
                                                        if ser_short.notna().any():
                                                            fig.add_trace(
                                                                go.Scatter(
                                                                    x=df_chart["trade_date"],
                                                                    y=ser_short,
                                                                    mode="lines",
                                                                    line=dict(color="#000000", width=1),
                                                                    name="duokong_short"
                                                                ),
                                                                row=1, col=1, secondary_y=False
                                                            )
                                                    if "duokong_long" in df_chart.columns:
                                                        ser_long = pd.to_numeric(df_chart["duokong_long"], errors="coerce")
                                                        if ser_long.notna().any():
                                                            fig.add_trace(
                                                                go.Scatter(
                                                                x=df_chart["trade_date"],
                                                                y=ser_long,
                                                                mode="lines",
                                                                line=dict(color="#e74c3c", width=1),
                                                                name="duokong_long"
                                                            ),
                                                            row=1, col=1, secondary_y=False
                                                        )
                                            except Exception:
                                                pass
                                            # 成交量柱
                                            try:
                                                if "vol" in df_chart.columns and "close" in df_chart.columns:
                                                    vols = df_chart["vol"]
                                                    closes = df_chart["close"]
                                                    prev_close = closes.shift(1)
                                                    colors = np.where(closes >= prev_close, "#e74c3c", "#2ecc71")
                                                    fig.add_trace(
                                                        go.Bar(
                                                            x=df_chart["trade_date"],
                                                            y=vols,
                                                            marker_color=colors,
                                                            name="成交量"
                                                        ),
                                                        row=2, col=1
                                                    )
                                            except Exception:
                                                pass
                                            fig.update_layout(
                                                margin=dict(l=10, r=10, t=10, b=10),
                                                height=520,
                                                xaxis_rangeslider_visible=False,
                                                xaxis_title="日期",
                                                yaxis_title="价格",
                                            )
                                            # 纵轴刻度：按 10% 迭代（乘以 1.1）
                                            try:
                                                y_min, y_max = df_chart["low"].min(), df_chart["high"].max()
                                                if pd.notna(y_min) and pd.notna(y_max) and y_min > 0 and y_max > 0:
                                                    tickvals = []
                                                    val = y_min / 1.1
                                                    while val <= y_max * 1.05 and len(tickvals) < 120:
                                                        tickvals.append(round(val, 2))
                                                        val *= 1.1
                                                    if len(tickvals) >= 2:
                                                        fig.update_yaxes(tickmode="array", tickvals=tickvals, row=1, col=1, secondary_y=False)
                                            except Exception:
                                                pass
                                            # 参考日标记：在成交量子图下方绘制一个原点
                                            try:
                                                ref_dt = None
                                                if end_date:
                                                    ref_dt = pd.to_datetime(end_date)
                                                if ref_dt is None and not df_chart.empty:
                                                    ref_dt = df_chart["trade_date"].max()

                                                if pd.notna(ref_dt):
                                                    # 若不在区间内，用最近端点；否则用距离最近的交易日
                                                    min_dt, max_dt = df_chart["trade_date"].min(), df_chart["trade_date"].max()
                                                    if ref_dt < min_dt:
                                                        ref_dt = min_dt
                                                    elif ref_dt > max_dt:
                                                        ref_dt = max_dt
                                                    else:
                                                        idx = (df_chart["trade_date"] - ref_dt).abs().idxmin()
                                                        ref_dt = df_chart.loc[idx, "trade_date"]

                                                    row_ref = df_chart.loc[df_chart["trade_date"] == ref_dt].tail(1)
                                                    if not row_ref.empty:
                                                        fig.add_trace(
                                                            go.Scatter(
                                                                x=[ref_dt],
                                                                y=[0],
                                                                mode="markers",
                                                                marker=dict(color="#34495e", size=10, symbol="circle"),
                                                                name="参考日标记",
                                                                hoverinfo="skip",
                                                                showlegend=False
                                                            ),
                                                            row=2, col=1
                                                        )
                                            except Exception:
                                                pass
                                            # 隐藏周末并去掉缺口日期
                                            try:
                                                rangebreaks = [
                                                    dict(bounds=["sat", "mon"]),
                                                    dict(bounds=["sun", "mon"])
                                                ]
                                                trade_days = set(df_chart["trade_date"].dt.normalize())
                                                full_days = pd.date_range(df_chart["trade_date"].min(), df_chart["trade_date"].max(), freq="D")
                                                holidays = [d for d in full_days if d not in trade_days and d.weekday() < 5]
                                                # 避免过长的范围删除列表
                                                if holidays and len(holidays) <= 500:
                                                    rangebreaks.append(dict(values=holidays))
                                                fig.update_xaxes(rangebreaks=rangebreaks, row=1, col=1)
                                                fig.update_xaxes(rangebreaks=rangebreaks, row=2, col=1)
                                            except Exception:
                                                pass

                                            fig.update_yaxes(title_text="价格", row=1, col=1, secondary_y=False)
                                            fig.update_yaxes(title_text="成交量", row=2, col=1)

                                            st.plotly_chart(fig, width="stretch", key=f"data_view_chart_{ts_code_normalized}_{start_date}_{end_date}")
                                        except Exception as e:
                                            st.warning(f"无法绘制图表: {e}")
                                    # 统计总行数（忽略limit）
                                    try:
                                        from database_manager import count_stock_data as _count_stock_data
                                        total_rows = _count_stock_data(
                                            db_path=db_path,
                                            ts_code=ts_code_normalized,
                                            start_date=start_date,
                                            end_date=end_date,
                                            adj_type=adj_type if asset_type != "index" else "ind"
                                        )
                                    except Exception:
                                        total_rows = len(df)
                                    st.info(f"总行数: {total_rows} | 本次显示: {len(df)}")
                                else:
                                    st.warning("未找到该股票的数据")
                            except Exception as e:
                                st.error(f"查询失败: {e}")
                
                # 数据库信息（美化显示）
                with st.expander("数据库信息"):
                    try:
                        info = get_database_info()
                        # 美化显示数据库信息
                        if isinstance(info, dict):
                            with st.container(border=True):
                                for key, value in info.items():
                                    if isinstance(value, (int, float)):
                                        st.metric(key, value)
                                    elif isinstance(value, list):
                                        st.text(f"{key}: {len(value)} 项")
                                        if value and len(value) <= 10:
                                            for item in value:
                                                st.text(f"  • {item}")
                                    elif isinstance(value, dict):
                                        st.text(f"{key}:")
                                        for sub_key, sub_value in value.items():
                                            st.text(f"  {sub_key}: {sub_value}")
                                    else:
                                        st.text(f"{key}: {value}")
                        else:
                            st.text(str(info))
                    except Exception as e:
                        st.error(f"获取信息失败: {e}")

    # ================== 日志 ==================
    with tab_logs:
        st.subheader("日志")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**score.log（尾部 400 行）**")
            st.code(_tail(LOG_DIR / "score.log", 400), language="bash")
        with col2:
            st.markdown("**score_ui.log（尾部 400 行）**")
            st.code(_tail(LOG_DIR / "score_ui.log", 400), language="bash")

    _anchor = st.session_state.pop("scroll_after_rerun", None)
    if _anchor:
        components.html(f"""
        <script>
        (function() {{
        const id = {_anchor!r};
        function go() {{
            const doc = parent.document || document;
            // 1) 激活“个股详情”页签（按钮 role="tab"，文本以“个股详情”开头）
            const tabs = doc.querySelectorAll('button[role="tab"]');
            for (const btn of tabs) {{
            if ((btn.innerText || '').trim().startsWith('个股详情')) {{ btn.click(); break; }}
            }}
            // 2) 滚动到锚点
            const el = doc.getElementById(id);
            if (el) {{
            el.scrollIntoView({{behavior:'instant', block:'start'}});
            }} else {{
            // 兜底：把 hash 设置为锚点
            parent.location.hash = id;
            }}
        }}
        // 多次尝试，等外层 DOM 稳定
        setTimeout(go, 0); setTimeout(go, 200); setTimeout(go, 600);
        }})();
        </script>
        """, height=0)
