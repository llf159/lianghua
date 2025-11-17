# -*- coding: utf-8-sig -*-
"""
scoring_core.py — 股票评分系统核心模块
提供完整的股票评分、排名和筛选功能
"""
from __future__ import annotations

import os
import multiprocessing
import re
import math
import json
import time
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import glob
import functools
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from log_system import get_logger, log_function_calls, log_performance
from functools import lru_cache
import threading
import queue
import logging as _logging
from collections import OrderedDict
# 数据库操作已迁移到 database_manager.py

# 全局数据库访问锁已移除，完全依赖连接池限流控制并发

# 布尔表达式缓存配置
BOOL_CACHE_SIZE = 128  # 增加到128，适应更多规则

class _SmallLRU:
    """轻量级LRU缓存实现，用于单票级布尔表达式结果缓存"""
    
    def __init__(self, capacity: int = BOOL_CACHE_SIZE):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: tuple) -> Optional[pd.Series]:
        """获取缓存值，如果存在则移到末尾"""
        if key in self.cache:
            # 移到末尾（最近使用）
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
    
    def put(self, key: tuple, value: pd.Series) -> None:
        """添加缓存值，如果超出容量则删除最久未使用的项"""
        if key in self.cache:
            # 如果已存在，先删除再添加（移到末尾）
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # 删除最久未使用的项（第一个）
            self.cache.popitem(last=False)
        
        self.cache[key] = value


from config import (
    DATA_ROOT, API_ADJ, UNIFIED_DB_PATH,
    SC_REF_DATE, SC_LOOKBACK_D, SC_PRESCREEN_LOOKBACK_D, SC_BASE_SCORE, SC_MIN_SCORE,
    SC_TOP_K, SC_TIE_BREAK, SC_MAX_WORKERS, SC_READ_TAIL_DAYS,
    SC_OUTPUT_DIR, SC_CACHE_DIR,
    SC_DO_TRACKING, SC_DO_SURGE,
    SC_WRITE_WHITELIST, SC_WRITE_BLACKLIST, SC_ATTENTION_SOURCE,
    SC_ATTENTION_WINDOW_D, SC_ATTENTION_MIN_HITS, SC_ATTENTION_TOP_K,
    SC_BENCH_CODES, SC_BENCH_WINDOW, SC_BENCH_FILL, SC_BENCH_FEATURES,
    SC_UNIVERSE, 
    SC_DETAIL_STORAGE, SC_USE_DB_STORAGE, SC_DB_FALLBACK_TO_JSON,
    SC_DETAIL_DB_PATH,
    SC_USE_PROCESS_POOL,
    SC_ENABLE_RULE_DETAILS,
    SC_ENABLE_CUSTOM_TAGS,
    SC_ENABLE_VERBOSE_SCORE_LOG,
    SC_ENABLE_VECTOR_BOOL,
    SC_ENABLE_BATCH_XSEC,
    SC_DYNAMIC_RESAMPLE,
)
# 不再从 config 导入 SC_PRESCREEN_RULES，统一使用 strategies_repo.py
    
# 延迟导入 database_manager 模块

def _lazy_import_database_manager():
    """延迟导入 database_manager 模块的函数"""
    try:
        from database_manager import get_database_manager, get_trade_dates
        return get_trade_dates, get_database_manager
    except ImportError as e:
        LOGGER.error(f"导入 database_manager 失败: {e}")
        return None, None


def _get_database_manager_functions():
    """获取 database_manager 函数"""
    from database_manager import get_trade_dates, get_database_manager
    return get_trade_dates, get_database_manager


def _read_data_via_dispatcher(ts_code: str, start_date: str, end_date: str, columns: List[str]) -> pd.DataFrame:
    """通过数据库管理器读取股票数据"""
    try:
        from database_manager import query_stock_data
        
        # 使用 database_manager 的 query_stock_data 函数
        df = query_stock_data(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            columns=columns,
            adj_type="qfq"
        )
        return df
            
    except Exception as e:
        LOGGER.error(f"通过数据库管理器读取数据失败: {e}")
        return pd.DataFrame()
from utils import (
    normalize_ts, normalize_trade_date,
    get_latest_date_from_database as _get_latest_date_from_database,
    get_latest_date_from_daily_partition as _get_latest_date_from_daily_partition,
    get_latest_date_from_single_dir as _get_latest_date_from_single_dir
)
import tdx_compat as tdx
from tdx_compat import evaluate_bool  # evaluate_bool 内部已使用 compile_script 的 LRU 缓存进行表达式预编译优化
# stats_core 已移除，相关功能已移到 score_ui.py
# 注：duckdb_attach_ro 在需要处按函数内局部导入，避免顶层初始化时抢句柄

# 指标计算系统
def _get_indicator_function(indicator_name: str):
    """动态获取指标计算函数"""
    try:
        from indicators import INDICATORS_META
        if indicator_name in INDICATORS_META:
            return INDICATORS_META[indicator_name].py_func
        return None
    except Exception:
        return None

def _calculate_missing_indicator(df: pd.DataFrame, indicator_name: str) -> pd.Series:
    """计算缺失的指标"""
    func = _get_indicator_function(indicator_name)
    if func:
        try:
            return func(df)
        except Exception:
            return None
    return None

# 策略规则加载 - 统一从 strategies_repo.py 加载
SC_RULES = []
SC_PRESCREEN_RULES = []
_SC_RULES_LOADED = False
try:
    from utils import load_rank_rules_py, load_filter_rules_py
    SC_RULES = load_rank_rules_py() or []
    SC_PRESCREEN_RULES = load_filter_rules_py() or []
    _SC_RULES_LOADED = len(SC_RULES) > 0 or len(SC_PRESCREEN_RULES) > 0
except Exception as _e:
    # 如果加载失败，保持空列表
    pass

OUTPUT_DIR = Path(SC_OUTPUT_DIR)
DETAIL_DIR = OUTPUT_DIR / "details"
ALL_DIR = OUTPUT_DIR / "all"
ALL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- 日志系统初始化 -------------------------
from log_system import get_logger, log_data_processing, log_database_operation, log_file_operation, log_algorithm_execution

# 初始化统一日志记录器
LOGGER = get_logger("scoring_core")

def _validate_required_fields(rule_or_clause: dict, category: str = "ranking", rule_name: str = None) -> list:
    """
    验证规则或子句的必填字段
    
    Args:
        rule_or_clause: 规则或子句字典
        category: 策略类型 ("ranking" 或 "filter")
        rule_name: 规则名称（用于日志）
    
    Returns:
        list: 缺失的必填字段列表，如果都齐全则返回空列表
    """
    missing_fields = []
    
    # 排名策略和筛选策略的必填字段
    if category in ("ranking", "filter"):
        required_fields = ["when", "score_windows", "scope"]
        
        # 如果使用clauses，则when不是必填的
        has_clauses = "clauses" in rule_or_clause and rule_or_clause["clauses"]
        
        # 检查 scope，对于某些 scope，score_windows 不是必填的
        scope = rule_or_clause.get("scope", "ANY")
        scope_upper = str(scope).upper().strip() if scope else "ANY"
        is_last_scope = scope_upper == "LAST"
        
        # 对于 RECENT/DIST/NEAR scope，如果有 dist_points，则 score_windows 不是必填的（从 dist_points 中提取最大日期）
        is_recent_scope = scope_upper in {"RECENT", "DIST", "NEAR"}
        has_dist_points = bool(rule_or_clause.get("dist_points") or rule_or_clause.get("distance_points"))
        is_recent_with_dist = is_recent_scope and has_dist_points
        
        for field in required_fields:
            if field == "when" and has_clauses:
                continue  # 使用clauses时，when不是必填的
            
            # 对于 scope: LAST，score_windows 不是必填的（默认为1）
            if field == "score_windows" and is_last_scope:
                continue  # scope: LAST 不需要 score_windows
            
            # 对于 RECENT/DIST/NEAR scope，如果有 dist_points，score_windows 不是必填的（从 dist_points 中提取）
            if field == "score_windows" and is_recent_with_dist:
                continue  # RECENT/DIST/NEAR 有 dist_points 时不需要 score_windows
            
            # 对于 score_windows，需要检查是否为 None（0 是有效值）
            if field == "score_windows":
                if field not in rule_or_clause or rule_or_clause[field] is None:
                    missing_fields.append(field)
            elif field not in rule_or_clause or not rule_or_clause[field]:
                missing_fields.append(field)
    
    # 如果缺少必填字段，记录警告
    if missing_fields:
        name = rule_name or rule_or_clause.get("name", "<unnamed>")
        LOGGER.warning(
            f"[规则验证] {category}策略 '{name}' 缺少必填字段: {', '.join(missing_fields)}. "
            f"将使用默认值，但建议显式指定这些字段。"
        )
    
    return missing_fields

# 记录策略规则加载状态
if _SC_RULES_LOADED:
    LOGGER.info(f"[策略预编译] 策略规则加载成功 - 排名规则: {len(SC_RULES)} 条, 筛选规则: {len(SC_PRESCREEN_RULES)} 条")
    
    # 验证加载的规则必填字段
    missing_count = 0
    for rule in SC_RULES:
        missing = _validate_required_fields(rule, "ranking", rule.get("name"))
        if missing:
            missing_count += 1
    
    for rule in SC_PRESCREEN_RULES:
        missing = _validate_required_fields(rule, "filter", rule.get("name"))
        if missing:
            missing_count += 1
    
    if missing_count > 0:
        LOGGER.warning(f"[策略预编译] 发现 {missing_count} 条规则缺少必填字段 (score_windows/scope)，建议更新规则配置")
else:
    LOGGER.warning("[策略预编译] 策略规则加载失败或未找到规则")

# 策略编译时计算最大 score_windows（按 timeframe 分组）
def _compute_max_windows_by_tf(rules: List[dict]) -> dict:
    """
    计算策略规则中每个 timeframe 的最大 score_windows 值。
    这个值用于作为 data_window（数据获取窗口），确保所有规则都有足够的历史数据用于计算。
    
    Returns:
        dict: {timeframe: max_score_windows}，例如 {"D": 60, "W": 10, "M": 3}
    """
    max_wins = {}
    
    def _extract_window(rule_or_clause: dict, tf: str) -> int:
        """从规则或子句中提取 score_windows 值（用于计算最大 data_window）"""
        if rule_or_clause.get("timeframe", "D").upper() == tf:
            # 注意：scope: LAST 的规则没有 score_windows，不参与计算
            scope = rule_or_clause.get("scope", "ANY")
            scope_upper = str(scope).upper().strip() if scope else "ANY"
            if scope_upper == "LAST":
                return 0  # scope: LAST 不需要窗口，不参与计算
            
            # 对于 RECENT/DIST/NEAR scope，如果有 dist_points，从 dist_points 中提取最大日期
            if scope_upper in {"RECENT", "DIST", "NEAR"}:
                bins = rule_or_clause.get("dist_points") or rule_or_clause.get("distance_points") or []
                if bins:
                    max_range = 0
                    for b in bins:
                        if isinstance(b, dict):
                            hi = int(b.get("max", b.get("lag", 0)))
                        else:
                            hi = int(b[1]) if len(b) > 1 else 0
                        max_range = max(max_range, hi)
                    if max_range > 0:
                        return max_range + 1  # 返回最大区间+1
            
            # score_windows 现在是必填字段，但为了向后兼容，如果没有则使用默认值
            score_windows = rule_or_clause.get("score_windows")
            if score_windows is None:
                return SC_LOOKBACK_D  # 使用默认值
            return int(score_windows)
        return 0
    
    for rule in rules:
        if "clauses" in rule and rule["clauses"]:
            # 带子句的规则：检查每个子句
            for clause in rule["clauses"]:
                for tf in ["D", "W", "M"]:
                    win = _extract_window(clause, tf)
                    if win > 0:
                        max_wins[tf] = max(max_wins.get(tf, 0), win)
        else:
            # 单条规则
            for tf in ["D", "W", "M"]:
                win = _extract_window(rule, tf)
                if win > 0:
                    max_wins[tf] = max(max_wins.get(tf, 0), win)
    
    # 确保至少有一个默认值
    if "D" not in max_wins:
        max_wins["D"] = SC_LOOKBACK_D
    
    return max_wins

# 在策略编译时计算最大 windows
_MAX_WINDOWS_BY_TF = {}
if _SC_RULES_LOADED:
    try:
        all_rules = list(SC_RULES) + list(SC_PRESCREEN_RULES)
        _MAX_WINDOWS_BY_TF = _compute_max_windows_by_tf(all_rules)
        LOGGER.info(f"[策略预编译] 最大 windows (按 timeframe): {_MAX_WINDOWS_BY_TF}")
    except Exception as e:
        LOGGER.warning(f"[策略预编译] 计算最大 windows 失败: {e}")
        _MAX_WINDOWS_BY_TF = {"D": SC_LOOKBACK_D}
else:
    _MAX_WINDOWS_BY_TF = {"D": SC_LOOKBACK_D}

# 尝试启用表达式编译缓存
try:
    import tdx_compat
    if hasattr(tdx_compat, 'enable_expr_cache'):
        cache_status = tdx_compat.enable_expr_cache(True)
        cache_size = cache_status.get('current_size', 0)
        max_size = cache_status.get('max_size', 0)
        if cache_size > 0:
            LOGGER.info(f"[表达式缓存] 已启用，当前大小: {cache_size}/{max_size}")
        else:
            LOGGER.info(f"[表达式缓存] 已启用，最大容量: {max_size}")
except Exception as e:
    LOGGER.debug(f"[表达式缓存] 启用失败: {e}")

# ---------------- 数据库存储类 ----------------
# DetailDB 已迁移到 database_manager

# ---- I/O helpers (cached) ----
@lru_cache(maxsize=512)
def _load_score_all_csv_cached(d: str, usecols_key: str = "ts_code,rank,trade_date") -> "pd.DataFrame":
    usecols = [c for c in (usecols_key.split(",") if usecols_key else []) if c]
    path = os.path.join(SC_OUTPUT_DIR, "all", f"score_all_{d}.csv")
    # 数据类型配置
    dtype = {"ts_code": str, "rank": "Int64", "trade_date": str}
    try:
        return pd.read_csv(path, usecols=usecols or None, dtype=dtype, engine="c")
    except Exception as e:
        LOGGER.warning(f"[attention] 读取失败 {path}: {e}")
        return pd.DataFrame(columns=usecols or ["ts_code","rank","trade_date"])

# ===== 评分引擎 =====

def _get_asset_root_and_scan():
    """获取数据库路径和数据库管理器"""
    try:
        from config import DATA_ROOT, UNIFIED_DB_PATH
        from database_manager import get_database_manager
        db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
        return db_path, get_database_manager
    except ImportError as e:
        LOGGER.error(f"导入 database_manager 失败: {e}")
        return None, None
# 全局运行上下文
_RANK_G = {
    "ref": None,           # 参考日 YYYYMMDD
    "codes": [],           # 本次评分的股票清单
    "base": None,          # DATA_ROOT
    "adj": None,           # API_ADJ
    "default_N": 60,       # 不写 N 时的默认窗口
}

# 批量写回缓冲区
_BATCH_BUFFER = {
    "summaries": [],       # 累积的 summary 数据
    "per_rules": [],       # 累积的 per_rules 数据
    "batch_size": 500,     # 批量大小
    "current_count": 0     # 当前累积数量
}

# 全局数据缓存，用于排名计算优化
_RANK_DATA_CACHE = {
    "data": None,  # 预加载的全市场数据
    "ref_date": None,
    "start_date": None,
    "columns": None
}

# 全局数据缓存，用于表达式筛选优化
_SCREEN_DATA_CACHE = {
    "data": None,  # 预加载的全市场数据
    "ref_date": None,
    "start_date": None,
    "columns": None,
    "timeframe": None,
    "window": None
}

# 交易日列表缓存
_TRADE_DATES_CACHE = None

# 缓存命中率统计
_CACHE_STATS = {
    "hits": 0,           # 预加载缓存命中次数
    "misses": 0,         # 预加载缓存未命中次数
    "fallbacks": 0,      # 预加载未命中 -> 触发回退的次数
    "local_hits": 0,     # 本轮本地DF缓存命中
    "local_misses": 0,   # 本轮本地DF缓存未命中
    # 新增：单票读取阶段直接命中预加载DF的统计
    "preload_hits": 0,
    "preload_misses": 0
}

# 本轮临时数据帧缓存（预加载未命中时，RANK_* 共享 fallback 结果，避免一轮内重复打库）
_RANK_LOCAL_DF_CACHE = _SmallLRU(capacity=32)

# 横截面名次缓存（同一 ref/N/K/side 在一轮内反复复用）
_RANK_XSEC_CACHE = _SmallLRU(capacity=32)  # key -> pd.Series(index=ts_code, values=rank)
_XSEC_STATS = {"hits": 0, "misses": 0}

# 进度系统已统一到队列+drain模式，见下方实现

# 本轮规则需要的重采样级别（动态）
_NEEDED_TIMEFRAMES = {"D"}

# 横截面基础量缓存（同一 ref 内复用，如当日涨跌幅向量）
_XSEC_BASE_CACHE: dict[tuple, pd.Series] = {}

# 日志降噪：重复列警告仅提示一次
_DUP_COL_WARNED = False


def init_rank_env(ref_date: str, universe_codes: List[str], base: str, adj: str, default_N: int = 60):
    _RANK_G["ref"] = str(ref_date)
    _RANK_G["codes"] = list(universe_codes or [])
    _RANK_G["base"] = str(base)
    _RANK_G["adj"]  = str(adj)
    _RANK_G["default_N"] = int(default_N)


def _codes_sig(codes: list[str]) -> int:
    try:
        return hash(tuple(sorted([str(c) for c in (codes or [])])))
    except Exception:
        return 0


def _get_trade_dates_cached() -> list[str]:
    """获取交易日列表（优先从 _RANK_G 获取，否则从数据库获取）"""
    try:
        # 优先从 _RANK_G 全局变量获取
        trade_dates = _RANK_G.get("trade_dates")
        if trade_dates:
            return list(trade_dates)
    except Exception:
        pass
    
    # 如果 _RANK_G 中没有，回退到数据库获取（使用缓存）
    global _TRADE_DATES_CACHE
    
    if _TRADE_DATES_CACHE is not None:
        return _TRADE_DATES_CACHE
    
    try:
        get_trade_dates, _ = _get_database_manager_functions()
        trade_dates = get_trade_dates()
        if trade_dates:
            _TRADE_DATES_CACHE = trade_dates
            LOGGER.debug(f"[缓存] 加载交易日列表: {len(trade_dates)} 个日期")
            return trade_dates
        else:
            LOGGER.warning("[缓存] 交易日列表为空")
            return []
    except Exception as e:
        LOGGER.error(f"[缓存] 获取交易日列表失败: {e}")
        return []


def _prev_trade_date(ref: str) -> str | None:
    dates = _get_trade_dates_cached()
    if not dates:
        return None
    try:
        i = dates.index(ref)
        return dates[i-1] if i > 0 else None
    except ValueError:
        prior = [d for d in dates if d < ref]
        return prior[-1] if prior else None


def _xsec_ret(df_preload: pd.DataFrame | None, ref: str, codes: list[str]) -> pd.Series:
    """计算 ref 日相对前收盘的涨跌幅横截面 Series"""
    if df_preload is not None:
        # 使用预加载数据
        ref_data = df_preload[df_preload["trade_date"] == ref]
        if ref_data.empty:
            return pd.Series(dtype="float64")
        
        prev_ref = _prev_trade_date(ref)
        if prev_ref is None:
            return pd.Series(dtype="float64")
        
        prev_data = df_preload[df_preload["trade_date"] == prev_ref]
        if prev_data.empty:
            return pd.Series(dtype="float64")
        
        # 计算涨跌幅
        ref_close = ref_data.set_index("ts_code")["close"]
        prev_close = prev_data.set_index("ts_code")["close"]
        ret = (ref_close - prev_close) / prev_close * 100
        return ret.reindex(codes) if codes else ret
    else:
        # 预加载未命中：最小化查询
        try:
            from database_manager import batch_query_stock_data
            prev_ref = _prev_trade_date(ref)
            if prev_ref is None:
                return pd.Series(dtype="float64")
            
            df = batch_query_stock_data(ts_codes=codes, start_date=prev_ref, end_date=ref,
                                        columns=["ts_code", "trade_date", "close"], adj_type="qfq")
            if df.empty:
                return pd.Series(dtype="float64")
            
            df["trade_date"] = df["trade_date"].astype(str)
            ref_data = df[df["trade_date"] == ref]
            prev_data = df[df["trade_date"] == prev_ref]
            
            if ref_data.empty or prev_data.empty:
                return pd.Series(dtype="float64")
            
            ref_close = ref_data.set_index("ts_code")["close"]
            prev_close = prev_data.set_index("ts_code")["close"]
            ret = (ref_close - prev_close) / prev_close * 100
            return ret.reindex(codes) if codes else ret
        except Exception:
            return pd.Series(dtype="float64")


def _xsec_latest(df_preload: pd.DataFrame | None, ref: str, codes: list[str], col: str) -> pd.Series:
    """取 ref 日的某一列横截面 Series(index=ts_code)；优先用预加载，否则最小列最小日查询"""
    if df_preload is not None:
        sub = df_preload[df_preload["trade_date"] == ref][["ts_code", col]]
        s = sub.set_index("ts_code")[col]
        return s.reindex(codes) if codes else s
    # 预加载未命中：最小化查询列与日期
    try:
        from database_manager import batch_query_stock_data
        df = batch_query_stock_data(ts_codes=codes, start_date=ref, end_date=ref,
                                    columns=["ts_code", "trade_date", col], adj_type="qfq")
        if df.empty:
            return pd.Series(dtype="float64")
        df["trade_date"] = df["trade_date"].astype(str)
        sub = df[df["trade_date"] == ref][["ts_code", col]]
        s = sub.set_index("ts_code")[col]
        return s.reindex(codes) if codes else s
    except Exception:
        return pd.Series(dtype="float64")


def _compute_start_date_by_trade_dates(ref_date: str, required_days: int) -> str:
    """根据交易日列表和所需天数计算准确的交易日起点"""
    trade_dates = _get_trade_dates_cached()
    
    if not trade_dates:
        LOGGER.warning("[预加载] 无法获取交易日列表，使用自然日计算")
        ref_dt = dt.datetime.strptime(ref_date, "%Y%m%d")
        start_dt = ref_dt - dt.timedelta(days=required_days)
        return start_dt.strftime("%Y%m%d")
    
    # 确保交易日列表已排序
    sorted_dates = sorted(trade_dates)
    
    # 找到 ref_date 在交易日列表中的位置
    if ref_date not in sorted_dates:
        LOGGER.warning(f"[预加载] ref_date {ref_date} 不在交易日列表中，使用最近交易日")
        # 找到最近的交易日（小于等于 ref_date）
        ref_idx = -1
        for i, date in enumerate(sorted_dates):
            if date > ref_date:
                ref_idx = i - 1
                break
        if ref_idx < 0:
            # 如果所有日期都小于 ref_date，使用最后一个
            ref_idx = len(sorted_dates) - 1
        if ref_idx < 0:
            LOGGER.error(f"[预加载] ref_date {ref_date} 太早，无法找到起点")
            return ref_date
        ref_date = sorted_dates[ref_idx]
        LOGGER.info(f"[预加载] 使用最近交易日: {ref_date}")
    
    # 在交易日列表中向前反推 required_days 个交易日
    idx = sorted_dates.index(ref_date)
    start_idx = max(0, idx - required_days + 1)
    start_date = sorted_dates[start_idx]
    
    LOGGER.info(f"[预加载] 从 {ref_date} 反推 {required_days} 个交易日，起点: {start_date}")
    return start_date


def _batch_load_and_cache_data(
    ref_date: str,
    start_date: str,
    universe_codes: List[str],
    columns: List[str],
    cache_dict: dict,
    cache_type: str = "排名"
) -> bool:
    """
    批量加载数据并缓存（通用函数）
    
    Args:
        ref_date: 参考日期
        start_date: 起始日期
        universe_codes: universe股票代码列表
        columns: 需要加载的列
        cache_dict: 缓存字典（如 _RANK_DATA_CACHE 或 _SCREEN_DATA_CACHE）
        cache_type: 缓存类型名称（用于日志）
    
    Returns:
        是否成功加载并缓存
    """
    try:
        if not universe_codes:
            LOGGER.warning(f"[{cache_type}预加载] 未找到universe股票代码，跳过预加载")
            return False
        
        from database_manager import batch_query_stock_data
        
        # 确保预加载的列至少包含基本列
        required_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
        all_columns = list(set(columns + required_cols))  # 合并并去重
        
        LOGGER.info(f"[{cache_type}预加载] 开始预加载universe数据: {start_date} ~ {ref_date}, {len(universe_codes)} 只股票")
        
        # 预加载universe数据
        df = batch_query_stock_data(
            ts_codes=universe_codes,
            start_date=start_date,
            end_date=ref_date,
            columns=all_columns,
            adj_type="qfq"
        )
        
        if not df.empty:
            df["trade_date"] = df["trade_date"].astype(str)

            # 防御：去重列名，避免重复列（尤其是 trade_date）导致下游出错
            try:
                dup_mask = df.columns.duplicated()
                if dup_mask.any():
                    LOGGER.warning(f"[{cache_type}预加载] 检测到重复列: {list(df.columns[dup_mask])} -> 保留首列")
                    df = df.loc[:, ~dup_mask].copy()
            except Exception:
                pass

            cache_dict["data"] = df
            cache_dict["ref_date"] = ref_date
            cache_dict["start_date"] = start_date
            cache_dict["columns"] = all_columns.copy()
            
            # 注册到 database_manager 的预加载缓存
            try:
                from database_manager import get_database_manager
                manager = get_database_manager()
                # 根据缓存类型确定缓存名称
                cache_name = "rank" if cache_type == "排名" else "screen"
                manager.register_preload_cache(
                    cache_name=cache_name,
                    data=df,
                    ref_date=ref_date,
                    start_date=start_date,
                    columns=all_columns
                )
                LOGGER.debug(f"[{cache_type}预加载] 已注册到 database_manager 缓存: {cache_name}")
            except Exception as e:
                LOGGER.warning(f"[{cache_type}预加载] 注册到 database_manager 缓存失败: {e}")
            
            LOGGER.info(f"[{cache_type}预加载] 数据预加载成功！")
            LOGGER.info(f"[{cache_type}预加载] universe数据预加载完成: {len(df)} 条记录, {len(df['ts_code'].unique())} 只股票, 列: {all_columns}")
            return True
        else:
            LOGGER.warning(f"[{cache_type}预加载] 预加载数据为空")
            return False
            
    except Exception as e:
        LOGGER.error(f"[{cache_type}预加载] 预加载失败: {e}")
        cache_dict["data"] = None
        return False


def preload_rank_data(ref_date: str, start_date: str, columns: List[str]):
    """预加载排名计算所需的universe数据"""
    global _RANK_DATA_CACHE, _RANK_G
    
    try:
        # 获取universe股票代码
        universe_codes = _RANK_G.get("codes", [])
        if not universe_codes:
            LOGGER.warning("[预加载] 未找到universe股票代码，跳过预加载")
            return
        
        # 使用策略规则中的最大windows来确定预加载的天数
        from utils import load_rank_rules_py, load_filter_rules_py
        
        # 使用已加载的全局规则，如果没有则加载
        sc_rules = globals().get("SC_RULES") or (load_rank_rules_py() or [])
        sc_prescreen_rules = globals().get("SC_PRESCREEN_RULES") or (load_filter_rules_py() or [])
        
        # 收集所有策略规则的 when 表达式，并使用 _calc_nested_window 计算最大所需数据长度
        def _collect_and_calc_max_window(rules: List[dict], tf: str) -> int:
            """收集规则中的窗口和表达式，计算最大所需数据长度"""
            max_total_window = 0
            
            for r in rules:
                if "clauses" in r:
                    # 带子句的规则：检查每个子句
                    for c in r["clauses"]:
                        if c.get("timeframe", "D").upper() == tf:
                            # 获取窗口
                            window = int(c.get("score_windows", SC_LOOKBACK_D))
                            
                            # 使用 _calc_nested_window 计算表达式最大所需数据长度
                            when_expr = (c.get("when") or "").strip()
                            expr_window = 0
                            if when_expr:
                                expr_window = _calc_nested_window(when_expr)
                            
                            # 总数据长度 = score_windows + expr_data_length
                            # 因为要在score_windows窗口内的每一天计算表达式，最早的那一天需要expr_data_length的历史数据
                            total_window = window + expr_window
                            max_total_window = max(max_total_window, total_window)
                else:
                    # 单条规则
                    if r.get("timeframe", "D").upper() == tf:
                        # 获取窗口
                        window = int(r.get("score_windows", SC_LOOKBACK_D))
                        
                        # 使用 _calc_nested_window 计算表达式最大所需数据长度
                        when_expr = (r.get("when") or "").strip()
                        expr_window = 0
                        if when_expr:
                            expr_window = _calc_nested_window(when_expr)
                        
                        # 总数据长度 = score_windows + expr_data_length
                        # 因为要在score_windows窗口内的每一天计算表达式，最早的那一天需要expr_data_length的历史数据
                        total_window = window + expr_window
                        max_total_window = max(max_total_window, total_window)
            
            # 返回所有规则中最大的总数据长度（score_windows + expr_data_length）
            return max_total_window
        
        # 从实际策略规则中提取最大windows，使用 _calc_nested_window 计算表达式最大所需数据长度
        d = max(_collect_and_calc_max_window(sc_rules, "D"), _collect_and_calc_max_window(sc_prescreen_rules, "D"))
        w = max(_collect_and_calc_max_window(sc_rules, "W"), _collect_and_calc_max_window(sc_prescreen_rules, "W"))
        m = max(_collect_and_calc_max_window(sc_rules, "M"), _collect_and_calc_max_window(sc_prescreen_rules, "M"))
        
        # 如果策略中没有配置windows，使用默认值
        if d == 0:
            d = SC_LOOKBACK_D
        days = d + w*6 + m*22
        LOGGER.info(f"[预加载] 策略最大windows（含表达式递归计算）: 日={d}, 周={w}, 月={m}, 总天数={days}")
        
        # 按照交易日列表反推准确的交易日起点
        actual_start_date = _compute_start_date_by_trade_dates(ref_date, days)
        LOGGER.info(f"[预加载] 使用交易日起点: {actual_start_date} (原起点: {start_date})")
        
        # 使用通用函数加载并缓存数据
        _batch_load_and_cache_data(
            ref_date=ref_date,
            start_date=actual_start_date,
            universe_codes=universe_codes,
            columns=columns,
            cache_dict=_RANK_DATA_CACHE,
            cache_type="排名"
        )
            
    except Exception as e:
        LOGGER.error(f"[预加载] 预加载失败: {e}")
        _RANK_DATA_CACHE["data"] = None


def preload_screen_data(ref_date: str, universe_codes: List[str], timeframe: str, window: int, columns: List[str]):
    """预加载表达式筛选所需的universe数据"""
    global _SCREEN_DATA_CACHE
    
    try:
        # 使用窗口长度来确定预加载的天数
        tf = (timeframe or "D").upper()
        win = int(window or 30)  # 默认30
        
        # 使用与 tdx_screen 相同的逻辑计算起始日期
        actual_start_date = _start_for_tf_window(ref_date, tf, win)
        
        LOGGER.info(f"[筛选预加载] timeframe={tf}, window={win}, 起始日期={actual_start_date}")
        
        # 使用通用函数加载并缓存数据
        success = _batch_load_and_cache_data(
            ref_date=ref_date,
            start_date=actual_start_date,
            universe_codes=universe_codes,
            columns=columns,
            cache_dict=_SCREEN_DATA_CACHE,
            cache_type="筛选"
        )
        
        # 保存额外的筛选相关元数据
        if success:
            _SCREEN_DATA_CACHE["timeframe"] = tf
            _SCREEN_DATA_CACHE["window"] = win
            
    except Exception as e:
        LOGGER.error(f"[筛选预加载] 预加载失败: {e}")
        _SCREEN_DATA_CACHE["data"] = None


def clear_screen_data_cache():
    """清理表达式筛选数据缓存"""
    global _SCREEN_DATA_CACHE
    _SCREEN_DATA_CACHE["data"] = None
    _SCREEN_DATA_CACHE["ref_date"] = None
    _SCREEN_DATA_CACHE["start_date"] = None
    _SCREEN_DATA_CACHE["columns"] = None
    _SCREEN_DATA_CACHE["timeframe"] = None
    _SCREEN_DATA_CACHE["window"] = None


def clear_rank_data_cache():
    """清理排名数据缓存"""
    global _RANK_DATA_CACHE
    _RANK_DATA_CACHE["data"] = None
    _RANK_DATA_CACHE["ref_date"] = None
    _RANK_DATA_CACHE["start_date"] = None
    _RANK_DATA_CACHE["columns"] = None


def _add_to_batch_buffer(ts_code: str, ref_date: str, summary: dict, per_rules: list[dict]):
    """将单票数据添加到批量缓冲区"""
    global _BATCH_BUFFER
    
    # 添加数据到缓冲区
    _BATCH_BUFFER["summaries"].append({
        "ts_code": ts_code,
        "ref_date": ref_date,
        **summary
    })
    _BATCH_BUFFER["per_rules"].append({
        "ts_code": ts_code,
        "ref_date": ref_date,
        "rules": per_rules
    })
    _BATCH_BUFFER["current_count"] += 1
    
    # 检查是否需要批量写回
    if _BATCH_BUFFER["current_count"] >= _BATCH_BUFFER["batch_size"]:
        _flush_batch_buffer()


def _get_batch_buffer_data():
    """获取缓冲区中的数据（用于子进程返回给主进程）"""
    global _BATCH_BUFFER
    
    if _BATCH_BUFFER["current_count"] == 0:
        return []
    
    batch_data = []
    for i, summary in enumerate(_BATCH_BUFFER["summaries"]):
        per_rules = _BATCH_BUFFER["per_rules"][i]["rules"]
        ref_date = summary.get('ref_date', '')
        
        # 在返回前过滤per_rules，只保留算分日期（ref_date）的命中信息
        filtered_per_rules = _filter_rules_by_ref_date(per_rules, ref_date)
        
        # 对 JSON 字段进行序列化
        highlights_json = json.dumps(summary.get('highlights', []), ensure_ascii=False)
        drawbacks_json = json.dumps(summary.get('drawbacks', []), ensure_ascii=False)
        opportunities_json = json.dumps(summary.get('opportunities', []), ensure_ascii=False)
        rules_json = json.dumps(filtered_per_rules if filtered_per_rules else [], ensure_ascii=False)
        
        # 正确处理tiebreak的None值
        tiebreak_value = summary.get('tiebreak')
        if tiebreak_value is None:
            tiebreak_value = None
        else:
            try:
                import math
                if math.isnan(tiebreak_value) or math.isinf(tiebreak_value):
                    tiebreak_value = None
                else:
                    tiebreak_value = float(tiebreak_value)
            except (TypeError, ValueError):
                tiebreak_value = None
        
        batch_data.append({
            'ts_code': summary['ts_code'],
            'ref_date': summary['ref_date'],
            'score': summary.get('score', 0.0),
            'tiebreak': tiebreak_value,
            'highlights': highlights_json,
            'drawbacks': drawbacks_json,
            'opportunities': opportunities_json,
            'rank': 0,
            'total': 0,
            'rules': rules_json
        })
    
    return batch_data


def _flush_batch_buffer():
    """将缓冲区数据批量写回数据库"""
    global _BATCH_BUFFER
    
    if _BATCH_BUFFER["current_count"] == 0:
        return
    
    # 在多进程环境下，子进程不应该写入数据库，而是返回数据给主进程写入
    try:
        current_process = multiprocessing.current_process()
        if current_process.name != "MainProcess":
            # 子进程：不执行写入操作，数据将通过返回值传递到主进程
            LOGGER.debug(f"[批量写回] 子进程 {current_process.name} 跳过数据库写入，数据将在主进程统一写入")
            return
    except (AttributeError, RuntimeError):
        # 如果无法获取进程信息，继续执行（单进程模式）
        pass
    
    try:
        # 优化日志：使用条件日志，避免不必要的字符串格式化
        if LOGGER.isEnabledFor(_logging.INFO):
            LOGGER.info(f"[批量写回] 开始批量写回 {_BATCH_BUFFER['current_count']} 条明细数据")
        
        # 使用database_manager进行批量写回，不直接操作数据库
        from database_manager import get_database_manager
        from config import SC_OUTPUT_DIR
        
        # 优化：缓存details_db_path，避免重复计算
        details_db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
        os.makedirs(os.path.dirname(details_db_path), exist_ok=True)
        
        db_manager = get_database_manager()
        
        # 优化：表结构已经在主进程中初始化，子进程不需要再调用
        # 只在主进程中初始化表结构
        try:
            current_process = multiprocessing.current_process()
            if current_process.name == "MainProcess":
                db_manager.init_stock_details_tables(details_db_path, "duckdb")
        except (AttributeError, RuntimeError):
            # 单进程模式，初始化表结构
            db_manager.init_stock_details_tables(details_db_path, "duckdb")
        
        # 准备批量数据
        batch_data = []
        for i, summary in enumerate(_BATCH_BUFFER["summaries"]):
            per_rules = _BATCH_BUFFER["per_rules"][i]["rules"]
            ref_date = summary.get('ref_date', '')
            
            # 在写入前过滤per_rules，只保留算分日期（ref_date）的命中信息
            filtered_per_rules = _filter_rules_by_ref_date(per_rules, ref_date)
            
            # 对 JSON 字段进行序列化
            highlights_json = json.dumps(summary.get('highlights', []), ensure_ascii=False)
            drawbacks_json = json.dumps(summary.get('drawbacks', []), ensure_ascii=False)
            opportunities_json = json.dumps(summary.get('opportunities', []), ensure_ascii=False)
            rules_json = json.dumps(filtered_per_rules if filtered_per_rules else [], ensure_ascii=False)
            
            # 正确处理tiebreak的None值
            tiebreak_value = summary.get('tiebreak')
            if tiebreak_value is None:
                tiebreak_value = None  # 保持None，不要用0.0替代
            else:
                try:
                    import math
                    if math.isnan(tiebreak_value) or math.isinf(tiebreak_value):
                        tiebreak_value = None
                    else:
                        tiebreak_value = float(tiebreak_value)
                except (TypeError, ValueError):
                    tiebreak_value = None
            
            batch_data.append({
                'ts_code': summary['ts_code'],
                'ref_date': summary['ref_date'],
                'score': summary.get('score', 0.0),
                'tiebreak': tiebreak_value,  # 正确处理None值
                'highlights': highlights_json,
                'drawbacks': drawbacks_json,
                'opportunities': opportunities_json,
                'rank': 0,  # 排名将在后续批量更新时设置
                'total': 0,  # 总数将在后续批量更新时设置
                'rules': rules_json
            })
        
        # 使用database_manager的receive_data接口进行批量写回
        import pandas as pd
        df = pd.DataFrame(batch_data)
        
        # 使用同步写入确保数据被正确保存
        # 使用回调函数在写入完成后更新状态文件
        def update_status_callback(response):
            try:
                if hasattr(response, 'success') and response.success:
                    from database_manager import update_details_data_status
                    update_details_data_status()
            except Exception as e:
                LOGGER.warning(f"[状态文件] 更新细节数据状态失败: {e}")
        
        request_id = db_manager.receive_data(
            source_module="scoring_core_batch",
            data_type="custom",
            data=df,
            table_name="stock_details",
            mode="upsert",
            db_path=details_db_path,
            validation_rules={
                "required_columns": ["ts_code", "ref_date"]
            },
            callback=update_status_callback
        )
        
        # 优化日志：使用条件日志，避免不必要的字符串格式化
        if LOGGER.isEnabledFor(_logging.INFO):
            LOGGER.info(f"[批量写回] 成功提交批量写回请求: {request_id}, 数据量: {_BATCH_BUFFER['current_count']}")
        
    except Exception as e:
        LOGGER.error(f"[批量写回] 批量写回失败: {e}")
        # 如果批量写回失败，回退到逐条写回
        for i, summary in enumerate(_BATCH_BUFFER["summaries"]):
            try:
                _write_detail_json(
                    summary["ts_code"], 
                    summary["ref_date"], 
                    summary, 
                    _BATCH_BUFFER["per_rules"][i]["rules"]
                )
            except Exception as single_error:
                LOGGER.warning(f"[批量写回] 单条写回失败 {summary['ts_code']}: {single_error}")
    
    finally:
        # 清空缓冲区
        _BATCH_BUFFER["summaries"].clear()
        _BATCH_BUFFER["per_rules"].clear()
        _BATCH_BUFFER["current_count"] = 0


def _clear_batch_buffer():
    """清空批量缓冲区"""
    global _BATCH_BUFFER
    _BATCH_BUFFER["summaries"].clear()
    _BATCH_BUFFER["per_rules"].clear()
    _BATCH_BUFFER["current_count"] = 0


def _get_rank_data_from_cache(ref_date: str, N: int, codes: List[str]) -> Optional[pd.DataFrame]:
    """
    从预加载缓存获取排名数据
    
    Args:
        ref_date: 参考日期
        N: 窗口天数
        codes: 股票代码列表
        
    Returns:
        如果缓存命中且满足条件，返回DataFrame；否则返回None
    """
    global _RANK_DATA_CACHE, _CACHE_STATS
    
    # 检查缓存是否存在
    if _RANK_DATA_CACHE["data"] is None:
        _CACHE_STATS["misses"] += 1
        return None
    
    # 检查参考日期是否一致
    if _RANK_DATA_CACHE["ref_date"] != ref_date:
        _CACHE_STATS["misses"] += 1
        return None
    
    # 检查窗口覆盖范围（预加载使用 warmup + 交易日列表准确计算）
    # 预加载的 start_date 已经通过交易日列表反推，确保覆盖了 warmup 需求
    # 这里只需要确保覆盖 N 天窗口（考虑停牌/周末，用 N 天自然日估算）
    required_start = (dt.datetime.strptime(ref_date, "%Y%m%d") - dt.timedelta(days=N)).strftime("%Y%m%d")
    if _RANK_DATA_CACHE["start_date"] > required_start:
        _CACHE_STATS["misses"] += 1
        return None
    
    # 检查所需列是否齐全
    required_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    if not all(col in _RANK_DATA_CACHE["columns"] for col in required_cols):
        _CACHE_STATS["misses"] += 1
        return None
    
    # 缓存命中，返回数据
    _CACHE_STATS["hits"] += 1
    LOGGER.debug(f"[缓存命中] 使用预加载数据: {len(_RANK_DATA_CACHE['data'])} 条记录")
    return _RANK_DATA_CACHE["data"]


def _load_last_n(_base: str, _adj: str, _codes: List[str], _ref: str, _N: int) -> pd.DataFrame:
    """
    读取全市场近 _N 天（含 _ref）的 O/H/L/C/V；只取必要列。
    返回至少包含：['ts_code','trade_date','open','high','low','close','vol']。
    强制使用批量查询优化性能，避免逐票查询。
    """
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    # 使用交易日列表反推准确的交易日起点（删除 N*3 冗余）
    start = _compute_start_date_by_trade_dates(_ref, _N)
    
    try:
        # 强制使用批量查询，大幅提升性能
        from database_manager import batch_query_stock_data
        df = batch_query_stock_data(
            ts_codes=_codes,  # 传入股票代码列表，进行批量查询
            start_date=start,
            end_date=_ref,
            columns=cols,
            adj_type="qfq"  # 排名功能使用前复权
        )
        
        if not df.empty:
            df["trade_date"] = df["trade_date"].astype(str)
            LOGGER.debug(f"批量查询排名数据: {len(df)} 条记录, {len(df['ts_code'].unique())} 只股票")
            return df
        else:
            LOGGER.warning("批量查询返回空数据，尝试全市场查询")
            # 如果指定股票代码查询为空，尝试查询全市场数据
            df = batch_query_stock_data(
                ts_codes=None,  # 查询全市场
                start_date=start,
                end_date=_ref,
                columns=cols,
                adj_type="qfq"
            )
            if not df.empty and _codes:
                # 过滤到指定的股票代码
                df = df[df["ts_code"].astype(str).isin(set(_codes))]
                df["trade_date"] = df["trade_date"].astype(str)
                LOGGER.debug(f"全市场查询后过滤: {len(df)} 条记录, {len(df['ts_code'].unique())} 只股票")
                return df
            
    except Exception as e:
        LOGGER.error(f"批量查询失败: {e}")
        raise ImportError(f"无法通过批量查询获取排名数据: {e}")
    
    # 如果所有方式都失败，数据库错误影响数据完整性，直接抛出异常
    error_msg = "所有查询方式都失败，无法获取排名数据"
    LOGGER.error(error_msg)
    raise RuntimeError(error_msg)


def _rank_series(s: pd.Series, ascending: bool) -> pd.Series:
    # 名次从 1 开始；相等取平均名次；缺失置为最大名次+1
    r = s.rank(ascending=ascending, method="average")
    mx = int(r.max()) if pd.notna(r.max()) else len(r)
    return r.fillna(mx + 1)


def _rule_key(rule: dict) -> tuple:
    name = str(rule.get("name", "")).strip()
    tf = str(rule.get("timeframe", "D")).upper().strip()
    score_window = _get_score_window(rule, SC_LOOKBACK_D)
    # 规范化条件表达式
    if rule.get("clauses"):
        # serialize relevant fields only
        parts = []
        for c in rule.get("clauses") or []:
            parts.append(str(c.get("when") or "").strip())
        when = "|".join(parts)
    else:
        when = str(rule.get("when") or "").strip()
    return (name, tf, score_window, when)


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
    # numpy标量检查
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


# 表达式计算函数
def RANK_VOL(N: int | None = None, K: Optional[int] = None) -> float:
    """
    成交量横截面名次（1 = 量最小，量越小名次越靠前）。
    K：可选，仅在前 K 只里排名；省略=全评分范围。
    """
    ref, base, adj, codes = _RANK_G["ref"], _RANK_G["base"], _RANK_G["adj"], _RANK_G["codes"]
    N = int(N or _RANK_G["default_N"])
    
    # 只需 ref 当日
    df = _get_rank_data_from_cache(ref, 1, codes)
    if df is None:
        _CACHE_STATS["fallbacks"] += 1
        key = ("DF", str(ref), 1, hash(tuple(sorted(codes))))
        df_local = _RANK_LOCAL_DF_CACHE.get(key)
        if df_local is not None:
            _CACHE_STATS["local_hits"] += 1
            df = df_local
        else:
            _CACHE_STATS["local_misses"] += 1
            df = None  # 标记未命中，触发最小化查询
    key = ("VOL", ref, int(N or 0), int(K or 0), _codes_sig(codes))
    ranks = _RANK_XSEC_CACHE.get(key)
    if ranks is None:
        s = _xsec_latest(df, ref, codes, "vol")
        if s is None or s.empty:
            return float("inf")
        ranks = _rank_series(s, ascending=True)
        _RANK_XSEC_CACHE.put(key, ranks)
        _XSEC_STATS["misses"] += 1
    else:
        _XSEC_STATS["hits"] += 1
    ts = str(tdx.EXTRA_CONTEXT.get("TS", ""))
    r = float(ranks.get(ts, float("inf")))
    return float("inf") if (K and r > int(K)) else r


def RANK_RET(N: int | None = None, K: Optional[int] = None, side: str = "up") -> float:
    """
    涨跌幅横截面名次：
      side='up'   ：按涨幅从大到小排名（涨多名次靠前）；
      side='down' ：按跌幅从大到小排名（跌多名次靠前）。
    """
    ref, base, adj, codes = _RANK_G["ref"], _RANK_G["base"], _RANK_G["adj"], _RANK_G["codes"]
    N = int(N or _RANK_G["default_N"])
    # 只需 ref 与上一交易日
    df = _get_rank_data_from_cache(ref, 2, codes)
    if df is None:
        _CACHE_STATS["fallbacks"] += 1
        key = ("DF", str(ref), 2, hash(tuple(sorted(codes))))
        df_local = _RANK_LOCAL_DF_CACHE.get(key)
        if df_local is not None:
            _CACHE_STATS["local_hits"] += 1
            df = df_local
        else:
            _CACHE_STATS["local_misses"] += 1
            df = None
    up = str(side).lower().startswith("up")
    tag = "RET_up" if up else "RET_down"
    key = (tag, ref, int(N or 0), int(K or 0), _codes_sig(codes))
    ranks = _RANK_XSEC_CACHE.get(key)
    if ranks is None:
        # 批量基础量缓存
        base_key = ("RET_BASE", ref)
        base = _XSEC_BASE_CACHE.get(base_key) if SC_ENABLE_BATCH_XSEC else None
        if base is None:
            base = _xsec_ret(df, ref, codes)
            if SC_ENABLE_BATCH_XSEC and base is not None and not base.empty:
                _XSEC_BASE_CACHE[base_key] = base
        if base is None or base.empty:
            return float("inf")
        if K and K > 0:
            base = base.sort_values(ascending=False).head(int(K))
        if up:
            ranks = _rank_series(base, ascending=False)
        else:
            ranks = _rank_series(-base, ascending=False)
        _RANK_XSEC_CACHE.put(key, ranks)
        _XSEC_STATS["misses"] += 1
    else:
        _XSEC_STATS["hits"] += 1
    ts = str(tdx.EXTRA_CONTEXT.get("TS", ""))
    r = float(ranks.get(ts, float("inf")))
    return float("inf") if (K and r > int(K)) else r


def RANK_MATCH_COEF(N: int | None = None, K: Optional[int] = None) -> float:
    """
    量价匹配系数（0~1，越大匹配越强）：
      - 涨幅榜应与"量大"（活跃）匹配；
      - 跌幅榜应与"量小"（冷清）匹配。
    做法：
      - 计算：r_up = RANK_RET(...,'up'), r_dn = RANK_RET(...,'down')
      - 量小名次：rv_small = RANK_VOL(...)
      - 量大名次：rv_large = (Keff + 1 - rv_small)，Keff 为参与排名样本量
      - 匹配度：min(r_up * rv_large, r_dn * rv_small) 的倒数归一（Keff / prod）
    
    优化：批量计算up和down的排名，避免重复计算base
    """
    ref, base, adj, codes = _RANK_G["ref"], _RANK_G["base"], _RANK_G["adj"], _RANK_G["codes"]
    N = int(N or _RANK_G["default_N"])
    ts = str(tdx.EXTRA_CONTEXT.get("TS", ""))
    
    # 优化：批量计算up和down的排名，复用base计算
    # 检查up和down的排名缓存
    key_up = ("RET_up", ref, int(N or 0), int(K or 0), _codes_sig(codes))
    key_dn = ("RET_down", ref, int(N or 0), int(K or 0), _codes_sig(codes))
    ranks_up = _RANK_XSEC_CACHE.get(key_up)
    ranks_dn = _RANK_XSEC_CACHE.get(key_dn)
    
    # 如果两个排名都已缓存，直接使用
    if ranks_up is not None and ranks_dn is not None:
        _XSEC_STATS["hits"] += 2
        r_up = float(ranks_up.get(ts, float("inf")))
        r_dn = float(ranks_dn.get(ts, float("inf")))
    else:
        # 需要计算排名，批量获取base
        df = _get_rank_data_from_cache(ref, 2, codes)
        if df is None:
            _CACHE_STATS["fallbacks"] += 1
            key = ("DF", str(ref), 2, hash(tuple(sorted(codes))))
            df_local = _RANK_LOCAL_DF_CACHE.get(key)
            if df_local is not None:
                _CACHE_STATS["local_hits"] += 1
                df = df_local
            else:
                _CACHE_STATS["local_misses"] += 1
                df = None
        
        # 获取base（涨跌幅向量）
        base_key = ("RET_BASE", ref)
        base_series = _XSEC_BASE_CACHE.get(base_key) if SC_ENABLE_BATCH_XSEC else None
        if base_series is None:
            base_series = _xsec_ret(df, ref, codes)
            if SC_ENABLE_BATCH_XSEC and base_series is not None and not base_series.empty:
                _XSEC_BASE_CACHE[base_key] = base_series
        
        if base_series is None or base_series.empty:
            return float("inf")
        
        # 处理K参数：如果指定了K，先筛选
        if K and K > 0:
            base_series = base_series.sort_values(ascending=False).head(int(K))
        
        # 批量计算up和down的排名
        if ranks_up is None:
            ranks_up = _rank_series(base_series, ascending=False)
            _RANK_XSEC_CACHE.put(key_up, ranks_up)
            _XSEC_STATS["misses"] += 1
        else:
            _XSEC_STATS["hits"] += 1
        
        if ranks_dn is None:
            ranks_dn = _rank_series(-base_series, ascending=False)
            _RANK_XSEC_CACHE.put(key_dn, ranks_dn)
            _XSEC_STATS["misses"] += 1
        else:
            _XSEC_STATS["hits"] += 1
        
        # 获取当前股票的排名
        r_up = float(ranks_up.get(ts, float("inf")))
        r_dn = float(ranks_dn.get(ts, float("inf")))
    
    # 处理K参数：如果指定了K且排名超出范围，返回inf
    if K and r_up > int(K):
        r_up = float("inf")
    if K and r_dn > int(K):
        r_dn = float("inf")
    
    # 计算成交量排名
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
        "ANY_TAG": tdx.ANY_TAG,
        "YDAY_ANY_TAG": tdx.YDAY_ANY_TAG,
        "TAG_HITS": tdx.TAG_HITS,
        "ANY_TAG_AT_LEAST": tdx.ANY_TAG_AT_LEAST,
        "YDAY_TAG_HITS": tdx.YDAY_TAG_HITS,
        "YDAY_ANY_TAG_AT_LEAST": tdx.YDAY_ANY_TAG_AT_LEAST,
        }

# ------------------------- 工具函数 -------------------------
def _last_true_date(df: pd.DataFrame, expr: str, *, ref_date: str, timeframe: str, window: int, ctx: dict) -> tuple[Optional[str], Optional[int]]:
    """给基础数据和表达式，返回 (最近一次为 True 的日期YYYYMMDD, lag)"""
    if not expr:
        return None, None
    
    # 自动使用策略编译时的最大 score_windows 作为 data_window，score_window 用于计算窗口
    sig = _eval_bool_cached(ctx, df, expr, timeframe, window, ref_date)
    lag = _last_true_lag(sig)
    if lag is None:
        return None, None
    hit_ts = sig.index[-1 - lag]
    return hit_ts.strftime("%Y%m%d"), int(lag)


def _single_root_dir() -> str:
    """
    single 布局根目录，例如：
    {DATA_ROOT}/stock/single/single_qfq_indicators
    {DATA_ROOT}/stock/single/single_qfq
    """
    name = f"single_{API_ADJ}_indicators"
    return os.path.join(DATA_ROOT, "stock", "single", name)


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
        age = np.arange(n)[::-1]  # 最新数据age=0
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
        # 默认使用当前规则的时间框架和窗口
        return {
            "timeframe": str(rule.get("timeframe","D")).upper(),
            "score_windows": int(rule.get("score_windows", SC_LOOKBACK_D)),
            "scope": "LAST",
            "when": g
        }
    if isinstance(g, dict):
        return g
    if isinstance(g, (list, tuple)):
        return {"clauses": list(g)}
    return None


def _eval_gate(df: pd.DataFrame, rule: dict, ref_date: str, ctx: dict = None) -> tuple[bool, str | None, dict | None]:
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
                ok, err = _eval_rule(df, c, ref_date, ctx)
                if err or (not ok):
                    return False, err, g
            return True, None, g
        else:
            ok, err = _eval_rule(df, g, ref_date, ctx)
            return (bool(ok and not err), err, g)
    except Exception as e:
        return False, f"gate-exception: {e}", g


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
    get_trade_dates, _ = _get_database_manager_functions()
    days = get_trade_dates() or []

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


def _validate_config_ref_date(config_date: str, latest_date: str) -> str:
    """验证配置中的参考日期"""
    if not latest_date:
        LOGGER.warning(f"[参考日] 无法验证配置日期 {config_date}，直接使用")
        return config_date
    
    if config_date > latest_date:
        LOGGER.warning(f"[参考日] 指定的参考日 {config_date} 晚于数据库最新日期 {latest_date}，将使用数据库最新日期")
        return latest_date
    elif config_date < latest_date:
        # 检查指定的参考日是否在数据库中存在
        try:
            from database_manager import query_stock_data
            test_df = query_stock_data(start_date=config_date, end_date=config_date, columns=["ts_code"])
            if test_df.empty:
                LOGGER.warning(f"[参考日] 指定的参考日 {config_date} 在数据库中无数据，将使用数据库最新日期 {latest_date}")
                return latest_date
            else:
                LOGGER.info(f"[参考日] 使用指定的参考日 {config_date}（数据库最新日期: {latest_date}）")
                return config_date
        except Exception as e:
            LOGGER.warning(f"[参考日] 验证指定参考日 {config_date} 时出错: {e}，将使用数据库最新日期 {latest_date}")
            return latest_date
    else:
        return config_date


def _pick_ref_date() -> str:
    """
    获取参考日期，按优先级尝试多种方式。
    若 SC_REF_DATE 指定了 YYYYMMDD，优先使用它（会验证存在性）。
    """
    # 1. 优先从数据库获取最新日期
    latest = _get_latest_date_from_database()
    
    # 2. 如果数据库获取失败，从daily分区获取
    if latest is None:
        latest = _get_latest_date_from_daily_partition()
    
    # 3. 如果daily分区也失败，从single目录推断
    if latest is None:
        # 尝试从single目录获取（需要传入路径）
        try:
            from config import DATA_ROOT, API_ADJ
            single_dir = os.path.join(DATA_ROOT, "stock", "single", f"single_{API_ADJ}_indicators")
            latest = _get_latest_date_from_single_dir(single_dir)
        except Exception:
            pass
    
    # 4. 如果所有方式都失败，抛出异常
    if latest is None:
        raise FileNotFoundError("无法从数据库、daily 或 single 目录推断参考日，请检查数据目录或者占用问题。")

    # 5. 处理配置中的指定日期
    if isinstance(SC_REF_DATE, str) and re.fullmatch(r"\d{8}", SC_REF_DATE):
        ref = _validate_config_ref_date(SC_REF_DATE, latest)
    else:
        ref = latest

    # 6. 打印对比信息
    sys_today = _today_str()
    if ref != sys_today:
        LOGGER.info(f"[参考日] 使用分区最新日 {ref}；系统今日 {sys_today}，请知悉。")
    else:
        LOGGER.info(f"[参考日] 使用今日分区 {ref}。")
    
    return ref


# def _list_codes_for_day(day: str) -> List[str]:
#     """
#     返回需要评分的股票列表。
#     - 统一数据库：从数据库中获取指定日期的股票列表
#     - single 布局：直接列出 single 目录下所有 *.parquet 文件名
#     - daily 布局：兼容旧逻辑，从 trade_date=day 分区列出
#     """
#     # 优先使用统一数据库
#     try:
#         from database_manager import get_data_source_status
#         status = get_data_source_status()
#         if status.get('using_unified_db', False):
#             # 从统一数据库获取股票列表
#             _, _, get_unified_connection_manager = _get_data_reader_functions()
#             if get_unified_connection_manager is not None:
#                 manager = get_unified_connection_manager()
#                 # 通过分发器读取所有股票数据
#                 request_id = manager.request_data(None, day, day, ["ts_code"])
#                 response = manager.get_data(request_id, timeout=60.0)
#                 if response and response.success and not response.data.empty and 'ts_code' in response.data.columns:
#                     codes = response.data['ts_code'].unique().tolist()
#                     return sorted(codes)
#             else:
#                 LOGGER.warning("统一连接管理器未正确导入，回退到原有逻辑")
#     except Exception as e:
#         LOGGER.warning(f"从统一数据库获取股票列表失败: {e}")
    
#     # 回退到原有逻辑
#     if _has_single_dir():
#         single_dir = _single_root_dir()
#         codes = []
#         for name in os.listdir(single_dir):
#             if name.endswith(".parquet"):
#                 codes.append(os.path.splitext(name)[0])
#         return sorted(codes)
#     else:
#         asset_root, _, _ = _get_database_manager_functions()
#         root = asset_root(DATA_ROOT, "stock", API_ADJ)
#         pdir = os.path.join(root, f"trade_date={day}")
#         if not os.path.isdir(pdir):
#             raise FileNotFoundError(f"未找到分区目录: {pdir}")
#         codes = []
#         for name in os.listdir(pdir):
#             if name.endswith(".parquet"):
#                 codes.append(os.path.splitext(name)[0])
#         return sorted(codes)


def _list_codes_for_day(day: str) -> List[str]:
    """
    返回需要评分的股票列表：
    1) 优先：直接只读连接 DuckDB，查当日 DISTINCT ts_code（不走分发器、无文件锁）
    2) 失败：回退到 single/daily 目录扫描
    """
    # --- 1) 使用 database_manager 统一接口查询 ---
    try:
        from database_manager import query_stock_data
        LOGGER.info(f"[范围] 正在查询日期 {day} 的股票清单...")
        # 使用database_manager查询股票代码
        df = query_stock_data(start_date=day, end_date=day, columns=["ts_code"])
        codes = df["ts_code"].unique().tolist() if not df.empty else []
        if codes:
            LOGGER.info(f"[范围] 从数据库获取股票清单: {day}, 共{len(codes)}只")
            return codes
        else:
            LOGGER.warning(f"[范围] 数据库查询返回空结果，日期 {day} 可能无数据")
    except Exception as e:
        LOGGER.warning(f"[范围] 数据库取清单失败（将回退到分区目录）：{e}")

    # --- 2) 回退：single 或 daily 目录 ---
    try:
        if _has_single_dir():
            single_dir = _single_root_dir()
            LOGGER.info(f"[范围] 回退到 single 目录: {single_dir}")
            codes = sorted(
                os.path.splitext(n)[0]
                for n in os.listdir(single_dir)
                if n.endswith(".parquet")
            )
            LOGGER.info(f"[范围] 从 single 目录获取股票清单: {day}, 共{len(codes)}只")
            return codes
        else:
            # 回退到使用数据库管理器
            try:
                from database_manager import get_stock_list
                codes = get_stock_list()
                LOGGER.info(f"[范围] 从数据库获取股票清单: {day}, 共{len(codes)}只")
                return codes
            except Exception as e:
                LOGGER.error(f"从数据库获取股票清单失败: {e}")
                raise FileNotFoundError(f"无法获取股票清单: {e}")
    except Exception as e:
        LOGGER.error(f"[范围] 获取股票清单失败: {e}")
        raise RuntimeError(f"无法获取日期 {day} 的股票清单: {e}")


def _compute_read_start(ref_date: str) -> str:
    """
    计算读取起始日：优先使用 SC_READ_TAIL_DAYS；否则根据规则窗口估算。
    """
    if SC_READ_TAIL_DAYS is not None and int(SC_READ_TAIL_DAYS) > 0:
        start_dt = dt.datetime.strptime(ref_date, "%Y%m%d") - dt.timedelta(days=int(SC_READ_TAIL_DAYS))
        return start_dt.strftime("%Y%m%d")

    # 估算：日线窗口 + 周线窗口*6 + 月线窗口*22 + 额外缓冲(260)
    def _max_win(rules: List[dict], tf: str) -> int:
        wins = []
        for r in rules:
            if "clauses" in r:
                for c in r["clauses"]:
                    if c.get("timeframe","D").upper()==tf:
                        wins.append(int(c.get("score_windows", SC_LOOKBACK_D)))
            else:
                if r.get("timeframe","D").upper()==tf:
                    wins.append(int(r.get("score_windows", SC_LOOKBACK_D)))
        return max(wins or [0])
    # 安全获取规则
    sc_rules = globals().get("SC_RULES") or []
    sc_prescreen_rules = globals().get("SC_PRESCREEN_RULES") or []
    
    d = max(_max_win(sc_rules, "D"), _max_win(sc_prescreen_rules, "D"), SC_LOOKBACK_D, SC_PRESCREEN_LOOKBACK_D)
    w = max(_max_win(sc_rules, "W"), _max_win(sc_prescreen_rules, "W"))
    m = max(_max_win(sc_rules, "M"), _max_win(sc_prescreen_rules, "M"))
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


def _recent_points(dfD: pd.DataFrame, rule: dict, ref_date: str, ctx: dict = None) -> tuple[float, int|None, str|None]:
    """
    计算 RECENT 规则的加分：
    - rule['dist_points'] 支持两种写法：
      1) 列表形式：[[min,max,points], ...]
      2) 字典列表：[{min:0,max:0,points:3}, ...]
    返回 (加分, lag, 错误或None)
    """
    tf = str(rule.get("timeframe", "D")).upper()
    # 对于 RECENT/DIST/NEAR scope，score_windows 会被忽略，直接使用 dist_points 的最大值
    # _get_score_window 已经处理了这个逻辑
    score_window = _get_score_window(rule, SC_LOOKBACK_D)
    
    when = (rule.get("when") or "").strip()
    if not when:
        return 0.0, None, "空 when 表达式"

    if ctx:
        # 使用缓存版本：自动使用策略编译时的最大 window 作为 data_window，score_window 用于计算窗口
        s_bool = _eval_bool_cached(ctx, dfD, when, tf, score_window, ref_date)
    else:
        # 原始版本：在完整数据上计算，然后切片窗口范围
        dfTF = dfD if tf == "D" else _resample(dfD, tf)
        win_df = _window_slice(dfTF, ref_date, score_window)
        if win_df.empty:
            return 0.0, None, None
        # 在完整数据上计算表达式
        with _ctx_df(dfTF):
            result_full = evaluate_bool(when, dfTF)
        # 切片到窗口范围
        result_full_series = pd.Series(result_full, index=dfTF.index, dtype=bool)
        s_bool = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
    
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
            if name_low in {"j","vr","bbi","z_score","duokong_long","duokong_short","diff","bar_color","rsi","bupiao_short","bupiao_long"}:
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
    
    # 也扫描所有 as='opportunity' 等类型的规则（它们会在 _inject_config_tags 中使用）
    def _iter_unique_rules():
        try:
            from utils import load_strategy_sets_py
            for s in load_strategy_sets_py():
                for r in s.rules or []:
                    yield r
        except Exception:
            pass
    
    for rr in _iter_unique_rules():
        scan_rule(rr)
    
    return sorted(need)


def _batch_update_ranks_independent(ref_date: str, scored_sorted: List[Any]) -> bool:
    """
    独立批量更新排名信息，直接操作details数据库，不走分发器
    使用真正的批量操作优化性能
    """
    try:
        # 使用统一的数据库实例，确保与保存/读取使用同一个数据库
        from database_manager import get_database_manager
        LOGGER.info(f"[数据库连接] 开始获取数据库管理器实例 (批量更新排名: {ref_date}, {len(scored_sorted)}只股票)")
        db_manager = get_database_manager()
        try:
            # 在批量回写排名前，确保明细写入已落地
            # 确保明细写入已落地 - 功能已迁移到database_manager
            pass
        except Exception as _e:
            LOGGER.warning(f"[detail] 准备批量回写排名前的落盘同步失败: {_e}")
        # 使用database_manager进行批量更新
        from database_manager import get_database_manager
        from config import SC_OUTPUT_DIR
        import os
        
        LOGGER.info(f"[数据库连接] 开始获取数据库管理器实例 (独立批量更新排名: {ref_date})")
        db_manager = get_database_manager()
        db_path = os.path.join(SC_OUTPUT_DIR, "details", "details.db")
        
        LOGGER.info(f"[detail] 使用database_manager进行批量更新")
        LOGGER.info(f"[detail] 准备更新 {len(scored_sorted)} 只股票的排名")
        log_database_operation("scoring_core", "批量更新排名", "details", len(scored_sorted), None, True)

        # 使用database_manager的批量更新功能
        return _batch_update_ranks_duckdb_optimized(db_path, ref_date, scored_sorted)
        
    except Exception as e:
        LOGGER.error(f"独立批量更新排名失败: {e}")
        import traceback
        LOGGER.error(f"详细错误信息: {traceback.format_exc()}")
        return False


def _batch_update_ranks_duckdb_optimized(db_path: str, ref_date: str, scored_sorted: List[Any]) -> bool:
    """批量更新排名 - 使用database_manager统一接口（同步方式确保数据已写入）"""
    try:
        from database_manager import get_database_manager
        from config import SC_OUTPUT_DIR
        import pandas as pd
        import time
        
        LOGGER.info(f"[数据库连接] 开始获取数据库管理器实例 (优化批量更新排名: {ref_date}, {len(scored_sorted)}只股票)")
        db_manager = get_database_manager()
        
        # 在批量更新rank前，等待所有details写入完成
        # 等待队列完成，确保所有details数据已写入
        receiver = db_manager.get_data_receiver()
        max_wait_time = 60.0  # 最多等待60秒
        wait_interval = 0.1   # 每次等待0.1秒
        start_wait_time = time.time()
        
        while time.time() - start_wait_time < max_wait_time:
            stats = receiver.get_import_stats()
            queue_size = stats.get('queue_size', 0)
            if queue_size == 0:
                break
            LOGGER.debug(f"[detail] 等待队列完成，当前队列大小: {queue_size}")
            time.sleep(wait_interval)
        
        if time.time() - start_wait_time >= max_wait_time:
            LOGGER.warning(f"[detail] 等待队列完成超时，但仍继续更新排名")
        
        # 准备批量更新数据 - 优化：使用列表推导式替代循环append
        total_count = len(scored_sorted)
        update_data = [{
            'ts_code': stock.ts_code,
            'ref_date': ref_date,
            'rank': i,
            'total': total_count
        } for i, stock in enumerate(scored_sorted, 1)]
        
        df = pd.DataFrame(update_data)
        
        # 去重：按 (ts_code, ref_date, rank) 排序，保留每个 (ts_code, ref_date) 组合的最小 rank
        # 优化：先检查是否有重复，避免不必要的排序操作
        original_count = len(df)
        if original_count > 0:
            # 检查是否有重复的 (ts_code, ref_date) 组合
            duplicates = df.duplicated(subset=['ts_code', 'ref_date'], keep=False)
            if duplicates.any():
                # 只有在有重复时才进行排序和去重
                df = df.sort_values(['ts_code', 'ref_date', 'rank']).drop_duplicates(['ts_code', 'ref_date'], keep='first')
                deduplicated_count = len(df)
                if deduplicated_count < original_count:
                    LOGGER.warning(f"[detail] 发现重复键，已去重: 原始 {original_count} 行，去重后 {deduplicated_count} 行")
            # 如果没有重复，直接使用原DataFrame，无需排序
        
        # 使用details目录下的数据库文件
        details_db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
        os.makedirs(os.path.dirname(details_db_path), exist_ok=True)
        
        # 确保数据库表存在
        db_manager.init_stock_details_tables(details_db_path, "duckdb")
        
        # 使用同步接口进行批量更新，确保数据已写入
        receiver = db_manager.get_data_receiver()
        result = receiver.import_data_sync(
            source_module="scoring_core",
            data_type="custom",
            data=df,
            table_name="stock_details",
            mode="upsert",
            db_path=details_db_path,
            validation_rules={
                "required_columns": ["ts_code", "ref_date"]
            }
        )
        
        if result.success:
            LOGGER.info(f"[detail] 批量更新排名成功: 数据量: {len(df)}, 耗时: {result.execution_time:.2f}秒")
            return True
        else:
            LOGGER.error(f"[detail] 批量更新排名失败: {result.error}")
            return False
        
    except Exception as e:
        LOGGER.error(f"批量更新排名失败: {e}")
        import traceback
        LOGGER.error(f"详细错误信息: {traceback.format_exc()}")
        return False


def _filter_rules_by_ref_date(rules: list[dict], ref_date: str) -> list[dict]:
    """过滤规则，只保留与ref_date相关的命中信息
    
    对于EACH/PERBAR/EACH_TRUE类型（多次命中，有很多天都算分），保留窗口内所有命中日期（<= ref_date且在窗口内）
    对于其他所有类型（LAST/ANY/ALL/RECENT/DIST/NEAR/COUNT>=k/CONSEC>=m/ANY_n/ALL_n等单次型规则），
    如果规则命中了（ok=True），直接用ref_date作为命中日期（因为打分的时候就是基于ref_date的）
    """
    import datetime as dt
    filtered_rules = []
    ref_date_dt = dt.datetime.strptime(ref_date, "%Y%m%d")
    
    for rule in rules:
        scope = str(rule.get("scope", "ANY")).upper().strip()
        ok = rule.get("ok", False)  # 规则是否命中
        hit_dates = rule.get("hit_dates", [])
        
        # 对于EACH/PERBAR/EACH_TRUE类型（多次命中，有很多天都算分），保留窗口内所有命中日期
        if scope in {"EACH", "PERBAR", "EACH_TRUE"}:
            # 将hit_dates转换为字符串列表，过滤出 <= ref_date 且在窗口内的日期
            hit_dates_str = [str(d) for d in hit_dates if d]
            # 窗口范围：从ref_date往前推window天到ref_date
            # 所以窗口内的日期是：所有 <= ref_date 且在窗口内的日期
            # 由于hit_dates已经是窗口内的命中日期（_list_true_dates在窗口内查找），
            # 所以只需要保留所有 <= ref_date 的日期
            window_hit_dates = []
            for d_str in hit_dates_str:
                try:
                    d_dt = dt.datetime.strptime(d_str, "%Y%m%d")
                    if d_dt <= ref_date_dt:
                        window_hit_dates.append(d_str)
                except (ValueError, TypeError):
                    # 如果日期格式不正确，跳过
                    continue
            
            # 如果窗口内有命中日期，则保留该规则
            if window_hit_dates:
                filtered_rule = rule.copy()
                filtered_rule["hit_dates"] = window_hit_dates
                # 使用最后一个命中日期作为hit_date
                filtered_rule["hit_date"] = window_hit_dates[-1] if window_hit_dates else None
                filtered_rule["hit_count"] = len(window_hit_dates)
                filtered_rules.append(filtered_rule)
            # 如果窗口内没有命中日期，则跳过该规则（不保留）
        else:
            # 对于其他所有类型（LAST/ANY/ALL/RECENT/DIST/NEAR/COUNT>=k/CONSEC>=m/ANY_n/ALL_n等单次型规则）
            # 如果规则命中了（ok=True），直接用ref_date作为命中日期（因为打分的时候就是基于ref_date的）
            if ok:
                filtered_rule = rule.copy()
                filtered_rule["hit_date"] = ref_date
                filtered_rule["hit_dates"] = [ref_date]
                filtered_rule["hit_count"] = 1
                filtered_rules.append(filtered_rule)
            # 如果规则没命中，则跳过该规则（不保留）
    
    return filtered_rules


def _write_detail_json(ts_code: str, ref_date: str, summary: dict, per_rules: list[dict], skip_db_write: bool = False):
    """
    写入个股详情，优先使用数据库存储，失败时回退到JSON文件
    
    Args:
        ts_code: 股票代码
        ref_date: 参考日期
        summary: 摘要数据
        per_rules: 规则详情列表
        skip_db_write: 是否跳过数据库写入（表达式选股场景应设置为True）
    """

    # 在写入前过滤per_rules，只保留算分日期（ref_date）的命中信息
    filtered_per_rules = _filter_rules_by_ref_date(per_rules, ref_date)

    # 数据库写入已统一由database_manager管理，无需进程检查
    success = True
    db_success = False
    
    # 优化：移除inspect.stack()调用栈检查，这是性能瓶颈
    # 表达式选股场景通过skip_db_write参数标识，而不是通过调用栈检查
    is_expression_screening = skip_db_write  # 通过参数传递，避免调用栈检查
    
    # 1. 优先尝试数据库存储（表达式选股时跳过）
    if (not is_expression_screening) and SC_DETAIL_STORAGE in ["database","both","db"] and SC_USE_DB_STORAGE:
        # 优化日志：减少不必要的日志输出
        if LOGGER.isEnabledFor(_logging.INFO):
            LOGGER.info(f"[detail] 尝试数据库存储: {ts_code}_{ref_date}")
        try:
            # 使用database_manager保存股票详情
            from database_manager import get_database_manager
            from config import SC_OUTPUT_DIR
            
            # 优化：缓存details_db_path，避免重复计算
            details_db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
            os.makedirs(os.path.dirname(details_db_path), exist_ok=True)
            
            # 优化：表结构已经在主进程中初始化，子进程不需要再调用
            # 只在主进程中初始化表结构
            try:
                current_process = multiprocessing.current_process()
                if current_process.name == "MainProcess":
                    db_manager = get_database_manager()
                    db_manager.init_stock_details_tables(details_db_path, "duckdb")
                else:
                    # 子进程直接获取数据库管理器，不初始化表结构
                    db_manager = get_database_manager()
            except (AttributeError, RuntimeError):
                # 单进程模式，初始化表结构
                db_manager = get_database_manager()
                db_manager.init_stock_details_tables(details_db_path, "duckdb")
            
            # 准备股票详情数据 - 对 JSON 字段进行序列化
            highlights_json = json.dumps(summary.get('highlights', []), ensure_ascii=False)
            drawbacks_json = json.dumps(summary.get('drawbacks', []), ensure_ascii=False)
            opportunities_json = json.dumps(summary.get('opportunities', []), ensure_ascii=False)
            rules_json = json.dumps(filtered_per_rules if filtered_per_rules else [], ensure_ascii=False)
            
            # 正确处理tiebreak的None值
            tiebreak_value = summary.get('tiebreak')
            if tiebreak_value is None:
                tiebreak_value = None  # 保持None，不要用0.0替代
            else:
                try:
                    import math
                    if math.isnan(tiebreak_value) or math.isinf(tiebreak_value):
                        tiebreak_value = None
                    else:
                        tiebreak_value = float(tiebreak_value)
                except (TypeError, ValueError):
                    tiebreak_value = None
            
            detail_data = {
                'ts_code': ts_code,
                'ref_date': ref_date,
                'score': summary.get('score', 0.0),
                'tiebreak': tiebreak_value,  # 正确处理None值
                'highlights': highlights_json,
                'drawbacks': drawbacks_json,
                'opportunities': opportunities_json,
                'rank': 0,  # 排名将在批量更新时设置
                'total': 0,  # 总数将在批量更新时设置
                'rules': rules_json
            }
            
            # 使用统一接口进行数据库存储
            import pandas as pd
            df = pd.DataFrame([detail_data])
            
            # 使用统一接口进行写入（details_db_path已在上面计算）
            request_id = db_manager.receive_data(
                source_module="scoring_core",
                data_type="custom",
                data=df,
                table_name="stock_details",
                mode="upsert",
                db_path=details_db_path,
                validation_rules={
                    "required_columns": ["ts_code", "ref_date"]
                }
            )
            
            # 优化日志：使用条件日志，避免不必要的字符串格式化
            if LOGGER.isEnabledFor(_logging.INFO):
                LOGGER.info(f"[detail] 数据库存储已提交: {ts_code}_{ref_date}, 请求ID: {request_id}")
            db_success = True
            
        except Exception as e:
            LOGGER.warning(f"[detail] 数据库存储异常: {ts_code}: {e}")
    if is_expression_screening:
        # 表达式选股场景不需要写入details，直接返回
        if LOGGER.isEnabledFor(_logging.DEBUG):
            LOGGER.debug(f"[detail] 表达式选股场景，跳过details写入: {ts_code}")
        return
    
    # 2. 如果数据库失败且配置了回退，或者配置了JSON存储，则使用JSON文件
    if (not db_success and SC_DB_FALLBACK_TO_JSON) or SC_DETAIL_STORAGE in ["json", "both"]:
        try:
            from config import SC_OUTPUT_DIR
            base = os.path.join(SC_OUTPUT_DIR, "details", str(ref_date))
            os.makedirs(base, exist_ok=True)
            out_path = os.path.join(base, f"{ts_code}_{ref_date}.json")
            payload = {
                "ts_code": ts_code,
                "ref_date": ref_date,
                "summary": summary,      # {"score": float, "tiebreak": float|None, "highlights": [...], "drawbacks":[...]}
                "rules": filtered_per_rules       # 每条规则的细粒度命中情况（见下），已过滤只保留ref_date的命中
            }
            with open(out_path, "w", encoding="utf-8") as f:
                import json
                json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
            
            # 优化日志：使用条件日志，避免不必要的字符串格式化
            if not db_success and SC_DB_FALLBACK_TO_JSON:
                if LOGGER.isEnabledFor(_logging.INFO):
                    LOGGER.info(f"[detail] 数据库写入失败，已回退到JSON文件: {ts_code} -> {out_path}")
            else:
                if LOGGER.isEnabledFor(_logging.DEBUG):
                    LOGGER.debug(f"[detail] JSON {ts_code} -> {out_path}")
            success = True
        except Exception as e:
            LOGGER.warning(f"[detail] JSON写明细失败 {ts_code}: {e}")
            success = False
    
    if not success:
        LOGGER.warning(f"[detail] 写明细失败 {ts_code}")


def _list_true_dates(df: pd.DataFrame, expr: str, *, ref_date: str, window: int, timeframe: str, ctx: dict = None) -> list[str]:
    """返回窗口内所有命中日期（按 timeframe 重采样后对齐到日线索引）。"""
    try:
        if ctx:
            # 使用缓存版本：自动使用策略编译时的最大 window 作为 data_window，score_window 用于计算窗口
            sig = _eval_bool_cached(ctx, df, expr, timeframe, window, ref_date)
        else:
            # 原始版本：在完整数据上计算，然后切片窗口范围
            dfTF = df if timeframe == "D" else _resample(df, timeframe)
            win_df = _window_slice(dfTF, ref_date, window)
            # 在完整数据上计算表达式
            with _ctx_df(dfTF):
                result_full = evaluate_bool(expr, dfTF)
            # 切片到窗口范围
            result_full_series = pd.Series(result_full, index=dfTF.index, dtype=bool)
            sig = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
        
        if sig is None or len(sig) == 0:
            return []
        # 向量化处理：使用numpy布尔索引替代循环
        sig_bool = pd.Series(sig).astype(bool) if not isinstance(sig, pd.Series) else sig.astype(bool)
        idx = sig_bool.index
        # 使用布尔索引直接过滤，比zip循环快得多
        hit_indices = idx[sig_bool.values]
        # 格式化日期为 YYYYMMDD 格式，只保留年月日，不包含时间
        out = []
        for i in hit_indices:
            if isinstance(i, pd.Timestamp):
                out.append(i.strftime("%Y%m%d"))
            elif hasattr(i, 'strftime'):
                out.append(i.strftime("%Y%m%d"))
            else:
                # 如果不是日期类型，尝试转换为日期后再格式化
                try:
                    dt = pd.to_datetime(i)
                    out.append(dt.strftime("%Y%m%d"))
                except:
                    out.append(str(i))
        return out
    except Exception:
        return []


def _inject_config_tags(dfD: pd.DataFrame, ref_date: str, ctx: dict = None, skip_tags: bool = False):
    """
    注入基于config的自定义标签
    
    Args:
        dfD: 数据DataFrame
        ref_date: 参考日期
        ctx: 上下文字典
        skip_tags: 是否跳过标签注入（表达式选股场景应设置为True）
    """
    # 优化：移除inspect.stack()调用栈检查，这是性能瓶颈
    # 表达式选股场景通过skip_tags参数标识，而不是通过调用栈检查
    if skip_tags:
        # 表达式选股时不应该加载和编译repo中的策略
        return
    
    dfD_dt = _ensure_datetime_index(dfD)
    old = tdx.EXTRA_CONTEXT.get("CUSTOM_TAGS", {}) or {}
    computed = {}
    for rule in _iter_unique_rules():
        bucket = str(rule.get("as") or "").strip()
        if not bucket:
            continue
        if str(rule.get("timeframe","D")).upper() != "D":
            continue

        try:
            # 收集该 as 规则涉及的表达式
            exprs = []
            if rule.get("clauses"):
                exprs.extend([(c.get("when") or "").strip() for c in rule["clauses"]])
            else:
                exprs.append((rule.get("when") or "").strip())
            exprs = [e for e in exprs if e]
            if not exprs:
                continue

            # 使用原始数据，不再进行兜底计算
            df_calc = dfD_dt

            # 让 as 规则之间可以互看已算出的标签
            temp_tags = dict(old); temp_tags.update(computed)
            tdx.EXTRA_CONTEXT["CUSTOM_TAGS"] = temp_tags

            # 计算 sig（多子句 AND）
            if ctx:
                # 使用缓存版本，按 D 级别直接把 df_calc 作为 base_df、tf="D" 送进 _eval_bool_cached 统一路径
                # 这里传入 1 表示只计算当前一天的数据，data_window 和 score_window 都用 1
                if rule.get("clauses"):
                    s_all = None
                    for e in exprs:
                        s = _eval_bool_cached(ctx, df_calc, e, "D", 1, ref_date)
                        s_all = s if s_all is None else (s_all & s)
                    sig = s_all
                else:
                    sig = _eval_bool_cached(ctx, df_calc, exprs[0], "D", 1, ref_date)
            else:
                # 原始版本
                with _ctx_df(df_calc):
                    if rule.get("clauses"):
                        s_all = None
                        for e in exprs:
                            s = evaluate_bool(e, df_calc).astype(bool)
                            s_all = s if s_all is None else (s_all & s)
                        sig = s_all
                    else:
                        sig = evaluate_bool(exprs[0], df_calc).astype(bool)

            if sig is None:
                continue
            # sig = pd.Series(sig, index=df_calc.index).reindex(dfD.index).fillna(False).astype(bool)
            sig = pd.Series(sig, index=df_calc.index)
            sig = sig.reindex(dfD_dt.index, fill_value=False).astype(bool)  # ✨ 用 dfD 的 DatetimeIndex 对齐
            name = str(rule.get("name") or "<unnamed>")
            key  = f"CFG_TAG::{bucket}::{name}"
            computed[key] = sig

        except Exception as e:
            LOGGER.warning(f"[inject-tags] 跳过 as='{bucket}' 规则 {rule.get('name')}: {e}")
            continue

    # 一次性回写
    tdx.EXTRA_CONTEXT["CUSTOM_TAGS"] = computed

@contextmanager
def _ctx_df(win_df):
    """确保 tdx.ANY_TAG* 在当前窗口下工作"""
    old = tdx.EXTRA_CONTEXT.get("DF", None)
    tdx.EXTRA_CONTEXT["DF"] = win_df
    try:
        yield
    finally:
        if old is None:
            tdx.EXTRA_CONTEXT.pop("DF", None)
        else:
            tdx.EXTRA_CONTEXT["DF"] = old


@lru_cache(maxsize=512)
def _normalize_expr(expr: str) -> str:
    """标准化表达式字符串，用于缓存键"""
    if not expr:
        return ""
    # 去首尾空格，将连续空格压缩为单个空格
    normalized = re.sub(r'\s+', ' ', expr.strip())
    return normalized


def _eval_bool_cached(ctx: dict, base_df: pd.DataFrame, expr: str, tf: str, score_window: int, ref_date: str, data_window: int = None) -> pd.Series:
    """
    返回窗口期内的布尔序列（索引为窗口期索引）。
    1) 先根据 (expr_norm, tf, data_window, score_window, ref_date) 在 ctx['bool_lru'] 查询；
    2) 命中则直接返回；
    3) 未命中则：获取/构建 resampled df -> 使用 data_window 切片数据用于计算 -> 在数据窗口上计算表达式 -> 切片到 score_window 范围 -> 回填 LRU -> 返回。
    
    注意：
    - data_window: 用于数据获取，从 base_df 中切片出足够的数据用于计算（避免数据被切片导致不足）
      - 如果未提供，会自动从 ctx 中的 max_windows_by_tf 获取策略编译时计算的最大 score_windows
      - 如果 ctx 中没有，则使用 score_window 作为 data_window
      - 如果表达式包含COUNT函数，会自动扩展data_window以包含COUNT窗口和内部表达式所需的历史数据
    - score_window: 用于计算窗口，只计算 score_window 内的数据
    - 数据获取使用策略编译时的最大 score_windows，确保所有规则都有足够的数据
    """
    # 自动从 ctx 中获取策略编译时的最大 score_windows 作为 data_window
    # 如果 ctx 中有 max_windows_by_tf，优先使用；否则使用传入的 data_window 或 score_window
    if ctx and 'max_windows_by_tf' in ctx:
        max_windows = ctx.get('max_windows_by_tf', {})
        data_window = max_windows.get(tf.upper(), data_window or score_window)
    elif data_window is None:
        # 如果没有 ctx 或没有 max_windows_by_tf，使用 score_window 作为 data_window
        data_window = score_window
    
    # 修复：如果表达式包含COUNT函数，需要扩展data_window以包含COUNT窗口和内部表达式所需的历史数据
    count_window = _extract_max_count_window(expr)
    if count_window > 0:
        count_inner_exprs = list(_extract_count_inner_expr(expr))
        inner_window = 0
        for inner_expr in count_inner_exprs:
            inner_window = max(inner_window, _calc_nested_window(inner_expr))
        # data_window需要包含COUNT窗口和内部表达式所需的历史数据
        required_window = count_window + inner_window
        if required_window > data_window:
            data_window = required_window
    # 构建缓存键前需要先标准化表达式
    expr_norm = _normalize_expr(expr)
    
    # 构建缓存键（使用 score_window，因为结果是根据 score_window 切片的）
    cache_key = (expr_norm, tf, data_window, score_window, ref_date)
    
    # 尝试从缓存获取
    bool_lru = ctx.get('bool_lru')
    if bool_lru:
        cached_result = bool_lru.get(cache_key)
        if cached_result is not None:
            # 命中缓存
            ctx['bool_cache_hit'] = ctx.get('bool_cache_hit', 0) + 1
            return cached_result
    
    # 未命中缓存，需要计算
    ctx['bool_cache_miss'] = ctx.get('bool_cache_miss', 0) + 1
    
    # 获取或构建重采样数据（完整数据，不切片）
    resampled = ctx.get('resampled', {})
    if tf not in resampled:
        if tf == "D":
            resampled[tf] = base_df
        else:
            resampled[tf] = _resample(base_df, tf)
        ctx['resampled'] = resampled
    
    df_tf_full = resampled[tf]
    
    # 使用 data_window 切片数据用于计算（避免数据被切片导致不足）
    data_win_cache_key = (tf, data_window, ref_date)
    data_win_cache = ctx.get('data_win_cache', {})
    if data_win_cache_key not in data_win_cache:
        df_tf_data = _window_slice(df_tf_full, ref_date, data_window)
        data_win_cache[data_win_cache_key] = df_tf_data
        ctx['data_win_cache'] = data_win_cache
    else:
        df_tf_data = data_win_cache[data_win_cache_key]
    
    # 获取 score_window 索引（用于后续切片结果）
    score_win_cache_key = (tf, score_window, ref_date)
    score_win_cache = ctx.get('win_cache', {})
    if score_win_cache_key not in score_win_cache:
        win_df = _window_slice(df_tf_full, ref_date, score_window)
        score_win_cache[score_win_cache_key] = win_df
        ctx['win_cache'] = score_win_cache
    else:
        win_df = score_win_cache[score_win_cache_key]
    
    # 空表达式检查
    if not expr_norm:
        result = pd.Series(False, index=win_df.index, dtype=bool)
    elif win_df.empty:
        result = pd.Series([], index=win_df.index, dtype=bool)
    else:
        # 在 data_window 切片的数据上计算表达式（这样TS_RANK、TS_PCT等函数可以获取足够的历史数据）
        with _ctx_df(df_tf_data):
            result_full = evaluate_bool(expr, df_tf_data)
        
        if result_full is None:
            result = pd.Series(False, index=win_df.index, dtype=bool)
        else:
            # 将计算结果转换为Series
            result_full_series = pd.Series(result_full, index=df_tf_data.index, dtype=bool)
            # 切片到 score_window 范围（只计算 score_window 内的数据）
            result = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
    
    # 回填缓存
    if bool_lru:
        bool_lru.put(cache_key, result)
    
    return result


def _build_per_rule_detail(df: pd.DataFrame, ref_date: str, ctx: dict = None) -> list[dict]:
    rows = []
    gate_ok=False; gate_norm={}
    try:
        _inject_config_tags(df, ref_date, ctx)   # 让 TAG_HITS/ANY_TAG 能拿到 CUSTOM_TAGS
    except Exception:
        pass
    for rule in _iter_unique_rules():
        name = str(rule.get("name", "<unnamed>"))
        scope = str(rule.get("scope", "ANY")).upper().strip()
        pts   = float(rule.get("points", 0))
        tf    = str(rule.get("timeframe", "D")).upper()
        # 对于 RECENT/DIST/NEAR scope，score_windows 会被忽略，直接使用 dist_points 的最大值
        # _get_score_window 已经处理了这个逻辑
        score_window = _get_score_window(rule, SC_LOOKBACK_D)
        
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
                # 检查是否有 gate
                gate_norm = _normalize_gate(rule)
                if gate_norm:
                    # 如果有 gate，使用 _count_hits_perbar_with_gate() 函数
                    # 该函数会对每次命中单独判断 gate，只统计同时满足 when 和 gate 的天数
                    cnt, err = _count_hits_perbar_with_gate(df, rule, ref_date, ctx)
                    # 对于带 gate 的 EACH scope，cnt 已经是同时满足 when 和 gate 的天数
                    # 所以 gate_ok 始终为 True（因为已经过滤过了）
                    gate_ok = True
                else:
                    # 如果没有 gate，使用原来的函数
                    cnt, err = _count_hits_perbar(df, rule, ref_date, ctx)
                    gate_ok = True
                dfTF = df if tf=="D" else _resample(df, tf)
                win_df = _window_slice(dfTF, ref_date, score_window)
                cand_when = None
                if "clauses" in rule and rule["clauses"]:
                    for c in rule["clauses"]:
                        if c.get("when"):
                            cand_when = c["when"]; break
                else:
                    cand_when = rule.get("when")
                hit_date = None
                if not err and cand_when:
                    _d, _ = _last_true_date(df, cand_when, ref_date=ref_date, timeframe=tf, window=score_window, ctx=ctx)
                    hit_date = _d
                ok = bool(cnt and cnt > 0 and err is None)
                # 对于 EACH scope，add = points * count（已经过滤过 gate）
                add = float(pts * int(cnt or 0))
                rows.append({
                    "name": name, "scope": scope, "timeframe": tf, "score_windows": score_window, "period": period,
                    "points": pts, "ok": ok, "cnt": int(cnt or 0),
                    "add": add,
                    "hit_date": hit_date,
                    "hit_dates": _list_true_dates(df, (cand_when or rule.get("when") or ""), ref_date=ref_date, window=score_window, timeframe=tf, ctx=ctx),
                    "hit_count": int(cnt or 0),
                    "gate_ok": bool(gate_ok),
                    "gate_when": (gate_norm.get("when") if isinstance(gate_norm, dict) else None) if gate_norm else None,
                })
                continue
            else:
                # === RECENT / DIST / NEAR: 按最近一次命中距今天数计分 ===
                if scope in {"RECENT", "DIST", "NEAR"}:
                    add, lag, err = _recent_points(df, rule, ref_date, ctx)
                    dfTF = df if tf=="D" else _resample(df, tf)
                    win_df = _window_slice(dfTF, ref_date, score_window)
                    hit_date = None
                    if err is None and lag is not None:
                        idx = win_df.index if isinstance(win_df.index, pd.DatetimeIndex) else pd.to_datetime(win_df["trade_date"].astype(str))
                        lag_i = int(lag)
                        if 0 <= lag_i < len(idx):
                            hit_date = idx[-1 - lag_i].strftime("%Y%m%d")
                        else:
                            # 理论不应发生；保守返回空，交由 UI 用 hit_dates 或其它信息展示
                            hit_date = None
                    
                    # 修复：RECENT规则的ok应该基于lag是否存在，而不是初始值False
                    ok = bool(lag is not None and err is None)
                    gate_ok, gate_err, gate_norm = _eval_gate(df, rule, ref_date, ctx)
                    if not gate_ok:
                        add = 0.0

                    rows.append({
                        "name": name, "scope": scope, "timeframe": tf, "score_windows": score_window, "period": period,
                        "points": pts, "ok": ok, "cnt": None,
                        "add": add, "lag": (None if lag is None else int(lag)),
                        "hit_date": hit_date, "hit_dates": _list_true_dates(df, cand_when or (when or ""), ref_date=ref_date, window=score_window, timeframe=tf, ctx=ctx),
                        "hit_count": int(cnt or 0),
                        "gate_ok": bool(gate_ok),
                        "gate_when": (gate_norm.get("when") if isinstance(gate_norm, dict) else None),
                    })
                    continue
                # === 其余仍走原来的布尔命中路径 ===
                ok_eval, err_eval = _eval_rule(df, rule, ref_date, ctx)
                ok  = bool(ok_eval and not err_eval)
                err = err_eval
                when = rule.get("when")


            # === 追加：常规布尔规则命中日期信息 ===
            hit_date = None
            hit_dates = []
            gate_ok, gate_err, gate_norm = _eval_gate(df, rule, ref_date, ctx)
            # 计算add：只有当ok=True、gate_ok=True且pts!=0时，才设置add
            add = (pts if (ok and gate_ok and pts) else 0.0)
            try:
                if when:
                    dfTF2 = df if tf=="D" else _resample(df, tf)
                    win_df2 = _window_slice(dfTF2, ref_date, score_window)
                    if not win_df2.empty:
                        _d, _ = _last_true_date(df, when, ref_date=ref_date, timeframe=tf, window=score_window, ctx=ctx)
                        hit_date = _d
                    hit_dates = _list_true_dates(df, when, ref_date=ref_date, window=score_window, timeframe=tf, ctx=ctx)
            except Exception:
                pass

        except Exception as e2:
            err = f"eval-exception: {e2}"

        rows.append({
            "name": name, "scope": scope, "timeframe": tf, "score_windows": score_window, "period": period,
            "points": pts, "ok": ok, "cnt": (None if cnt is None else int(cnt)),
            "add": add, "hit_date": hit_date, "hit_dates": hit_dates, "hit_count": (len(hit_dates) if hit_dates else 0),
            "error": err,
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
    # 排序：按得分降序，同分时按J值升序，再同分时按代码升序（兜底）
    df = df.sort_values(["score", "tiebreak_j", "ts_code"], ascending=[False, True, True])
    df["rank"] = np.arange(1, len(df) + 1)
    df["trade_date"] = str(ref_date)
    # 写全量排名文件
    outp = ALL_DIR / f"score_all_{ref_date}.csv"
    df.to_csv(outp, index=False, encoding="utf-8-sig")
    return outp


def _has_ranking_data(ref_date: str) -> bool:
    """
    检查某日期是否存在排名数据（通过检查 score_all_*.csv 文件）
    """
    outp = ALL_DIR / f"score_all_{ref_date}.csv"
    return outp.exists() and outp.stat().st_size > 0


def ensure_score_all_for_date(ref_date: str, *, force: bool=False) -> str:
    """
    确保某天存在排名数据：
    1) 如果已存在且非强制 → 直接返回路径
    2) 如果缺失或需要强制重建 → 调用 run_for_date(ref_date) 生成排名数据
    """
    # 检查是否已存在排名数据
    if not force and _has_ranking_data(ref_date):
        LOGGER.info(f"[backfill] {ref_date} 已存在排名数据，跳过")
        # 返回 CSV 文件路径（如果存在），否则返回日期标识
        outp = ALL_DIR / f"score_all_{ref_date}.csv"
        return str(outp) if outp.exists() else ref_date
    
    # 缺失或需要强制重建，调用排名流程
    try:
        LOGGER.info(f"[backfill] 开始补算排名数据: {ref_date} ...")
        result_path = run_for_date(ref_date)
        LOGGER.info(f"[backfill] {ref_date} 补算完成: {result_path}")
        return result_path
    except Exception as e:
        LOGGER.error(f"[backfill] {ref_date} 补算失败: {e}")
        raise


def backfill_missing_ranks(start_date: str, end_date: str, *, force: bool=False) -> list[str]:
    """
    扫描 [start_date, end_date] 窗口内的交易日，补齐缺失的排名数据
    - force=True：无论已存在与否都重建
    - force=False：仅对缺失的日期补算
    """
    # 1) 优先用交易日日历生成完整日期序列
    get_trade_dates, _ = _get_database_manager_functions()
    cal = get_trade_dates() or []
    if cal:
        dates = [d for d in cal if start_date <= d <= end_date]
    else:
        # 2) 没有日历时退化为按自然日铺满
        s = dt.datetime.strptime(start_date, "%Y%m%d")
        e = dt.datetime.strptime(end_date, "%Y%m%d")
        dates = [(s + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((e-s).days + 1)]

    if not dates:
        LOGGER.warning(f"[backfill] 日期范围 {start_date} ~ {end_date} 为空")
        return []

    # 3) 检查并补算缺失的日期
    processed = []
    total = len(dates)
    
    for i, d in enumerate(dates, 1):
        # 检查是否需要补算
        if force or not _has_ranking_data(d):
            try:
                LOGGER.info(f"[backfill] [{i}/{total}] 补算 {d} ...")
                result = ensure_score_all_for_date(d, force=force)
                processed.append(result)
            except Exception as e:
                LOGGER.error(f"[backfill] [{i}/{total}] {d} 补算失败: {e}")
        else:
            LOGGER.debug(f"[backfill] [{i}/{total}] {d} 已存在，跳过")
    
    LOGGER.info(f"[backfill] 完成，共处理 {len(processed)} 个交易日")
    return processed


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """确保以 trade_date 为 DatetimeIndex，并自动处理**重复列名**等异常情况。

    - 若存在重复的列名（尤其是 trade_date 重复），只保留第一次出现的列。
    - 若 df["trade_date"] 因重复列名而返回 DataFrame，则取其第一列。
    - 将索引设置为按 trade_date 解析得到的 DatetimeIndex（仅在当前索引不一致时才赋值）。
    """
    if df is None or getattr(df, 'empty', True):
        return df

    # 统一去重列名（保留首个）
    try:
        dup_mask = df.columns.duplicated()
        if dup_mask.any():
            # 仅第一次输出警告，后续静默去重，避免刷屏
            global _DUP_COL_WARNED
            if not _DUP_COL_WARNED:
                try:
                    from log_system import get_logger
                    _logger = get_logger("scoring_core")
                    _logger.warning(f"[ensure_dt] 检测到重复列: {list(df.columns[dup_mask])} -> 保留首列")
                except Exception:
                    pass
                _DUP_COL_WARNED = True
            df = df.loc[:, ~dup_mask].copy()
    except Exception:
        # 一些特殊索引类型可能不支持 duplicated，忽略
        pass

    if "trade_date" not in df.columns:
        return df

    td = df["trade_date"]
    # 若因重复列名导致返回的是 DataFrame（宽表），取第一列
    if isinstance(td, pd.DataFrame):
        td = td.iloc[:, 0]

    # 规范化为字符串再解析；兼容 20251025 或 2025-10-25 等格式
    idx = pd.to_datetime(td.astype(str), errors="coerce")
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
    if SC_DYNAMIC_RESAMPLE and timeframe.upper() not in (_NEEDED_TIMEFRAMES or {"D"}):
        # 动态模式：当前规则未用到该级别，直接返回原DF，避免不必要的重采样
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
        sel = set(_load_category_codes("strength", ref))
        return [c for c in codes if c in sel], "attention"

    # 未识别则不动
    return codes, "all"


def _get_score_window(rule_or_clause: dict, default_window: int = None) -> int:
    """
    获取计分窗口：使用 score_windows 字段。
    
    Args:
        rule_or_clause: 规则或子句字典
        default_window: 默认窗口值（如果不存在则使用此值）
    
    Returns:
        计分窗口值
    """
    if default_window is None:
        default_window = SC_LOOKBACK_D
    
    # 检查 scope
    scope = rule_or_clause.get("scope", "ANY")
    scope_upper = str(scope).upper().strip() if scope else "ANY"
    is_last_scope = scope_upper == "LAST"
    is_recent_scope = scope_upper in {"RECENT", "DIST", "NEAR"}
    
    # 对于 scope: LAST，默认返回 1（只需要最后一天的数据）
    if is_last_scope:
        return 1
    
    # 对于 RECENT/DIST/NEAR scope，如果有 dist_points，优先使用 dist_points 的最大值，忽略 score_windows
    if is_recent_scope:
        bins = rule_or_clause.get("dist_points") or rule_or_clause.get("distance_points") or []
        if bins:
            max_range = 0
            for b in bins:
                if isinstance(b, dict):
                    hi = int(b.get("max", b.get("lag", 0)))
                else:
                    hi = int(b[1]) if len(b) > 1 else 0
                max_range = max(max_range, hi)
            # 返回最大区间+1（因为 lag 从 0 开始）
            if max_range > 0:
                return max_range + 1
    
    # 使用 score_windows（如果明确指定）
    if "score_windows" in rule_or_clause and rule_or_clause["score_windows"] is not None:
        return int(rule_or_clause["score_windows"])
    
    # 使用默认值
    return int(default_window)


@lru_cache(maxsize=256)
def _extract_max_count_window(when_expr: str) -> int:
    """
    从 when 表达式中提取 COUNT 函数的最大窗口大小。
    例如：COUNT(..., 30) 返回 30，COUNT(..., 5) 和 COUNT(..., 8) 返回 8。
    
    Args:
        when_expr: when 表达式字符串
        
    Returns:
        最大的 COUNT 窗口大小，如果没有找到 COUNT 则返回 0
    """
    if not when_expr:
        return 0
    
    max_window = 0
    # 匹配 COUNT(..., n) 的模式，n 是数字
    # 需要处理嵌套括号的情况，使用更精确的匹配
    # 匹配 COUNT( 开头，然后找到最后一个逗号后的数字
    i = 0
    while i < len(when_expr):
        # 查找 COUNT( 的位置（不区分大小写）
        count_pos = when_expr.upper().find('COUNT(', i)
        if count_pos == -1:
            break
        
        # 从 COUNT( 后开始查找匹配的右括号
        start = count_pos + 6  # 'COUNT(' 的长度
        paren_count = 1
        j = start
        while j < len(when_expr) and paren_count > 0:
            if when_expr[j] == '(':
                paren_count += 1
            elif when_expr[j] == ')':
                paren_count -= 1
            j += 1
        
        if paren_count == 0:
            # 找到了匹配的右括号，提取括号内的内容
            content = when_expr[start:j-1]
            # 查找最后一个逗号后的数字
            # 从后往前找最后一个逗号
            last_comma = content.rfind(',')
            if last_comma != -1:
                # 提取逗号后的部分
                after_comma = content[last_comma+1:].strip()
                # 尝试提取数字
                num_match = re.match(r'(\d+)', after_comma)
                if num_match:
                    try:
                        window = int(num_match.group(1))
                        max_window = max(max_window, window)
                    except (ValueError, TypeError):
                        pass
        
        i = count_pos + 1
    
    return max_window


@lru_cache(maxsize=256)
def _extract_count_inner_expr(when_expr: str) -> tuple:
    """
    从 when 表达式中提取 COUNT 函数内部的表达式。
    例如：COUNT(TS_RANK(V,20) >= 16 AND TS_PCT(..., 10) >= 0.9, 30) 
    返回 ('TS_RANK(V,20) >= 16 AND TS_PCT(..., 10) >= 0.9',)
    
    注意：返回tuple以便支持lru_cache（list不可哈希）
    
    Args:
        when_expr: when 表达式字符串
        
    Returns:
        COUNT 函数内部表达式的元组（需要调用方转换为list）
    """
    if not when_expr:
        return tuple()
    
    inner_exprs = []
    i = 0
    while i < len(when_expr):
        # 查找 COUNT( 的位置（不区分大小写）
        count_pos = when_expr.upper().find('COUNT(', i)
        if count_pos == -1:
            break
        
        # 从 COUNT( 后开始查找匹配的右括号
        start = count_pos + 6  # 'COUNT(' 的长度
        paren_count = 1
        j = start
        while j < len(when_expr) and paren_count > 0:
            if when_expr[j] == '(':
                paren_count += 1
            elif when_expr[j] == ')':
                paren_count -= 1
            j += 1
        
        if paren_count == 0:
            # 找到了匹配的右括号，提取括号内的内容
            content = when_expr[start:j-1]
            # 查找最后一个逗号，提取逗号前的部分（COUNT内部表达式）
            last_comma = content.rfind(',')
            if last_comma != -1:
                inner_expr = content[:last_comma].strip()
                if inner_expr:
                    inner_exprs.append(inner_expr)
        
        i = count_pos + 1
    
    return tuple(inner_exprs)


@lru_cache(maxsize=256)
def _scan_expr_window(expr: str) -> int:
    """
    扫描表达式中函数所需的最大窗口大小。
    例如：TS_RANK(V,20) 需要 20，MA(C,10) 需要 10，HHV(H,30) 需要 30。
    
    Args:
        expr: 表达式字符串
        
    Returns:
        表达式中函数所需的最大窗口大小，如果没有找到则返回 0
    """
    if not expr:
        return 0
    
    max_window = 0
    # 匹配函数名(..., n) 的模式，n 是数字
    # 支持的函数：TS_RANK, TS_PCT, MA, EMA, SMA, HHV, LLV, SUM, STD, REF, COUNT等
    func_patterns = [
        r'TS_RANK\s*\([^)]*,\s*(\d+)',
        r'TS_PCT\s*\([^)]*,\s*(\d+)',
        r'MA\s*\([^)]*,\s*(\d+)',
        r'EMA\s*\([^)]*,\s*(\d+)',
        r'SMA\s*\([^)]*,\s*(\d+)',
        r'HHV\s*\([^)]*,\s*(\d+)',
        r'LLV\s*\([^)]*,\s*(\d+)',
        r'SUM\s*\([^)]*,\s*(\d+)',
        r'STD\s*\([^)]*,\s*(\d+)',
        r'REF\s*\([^)]*,\s*(\d+)',
        r'COUNT\s*\([^)]*,\s*(\d+)',
    ]
    
    for pattern in func_patterns:
        matches = re.finditer(pattern, expr, re.IGNORECASE)
        for match in matches:
            try:
                window = int(match.group(1))
                max_window = max(max_window, window)
            except (ValueError, TypeError):
                pass
    
    return max_window


@lru_cache(maxsize=256)
def _calc_nested_window(expr: str) -> int:
    """
    递归计算嵌套函数所需的总窗口大小。
    例如：HHV(REF(H,1), 20) 需要 20 + 1 = 21天
         HHV(REF(H,2), 30) 需要 30 + 2 = 32天
         HHV(MA(REF(C,1),5),20) 需要 20 + 5 + 1 = 26天
    
    Args:
        expr: 表达式字符串
        
    Returns:
        嵌套函数所需的总窗口大小，如果没有找到则返回 0
    """
    if not expr:
        return 0
    
    max_total_window = 0
    
    # 需要嵌套的函数列表（这些函数需要递归计算内部表达式）
    nested_funcs = ['TS_RANK', 'TS_PCT', 'MA', 'EMA', 'SMA', 'HHV', 'LLV', 'SUM', 'STD', 'COUNT']
    
    # 使用递归方法找到所有函数调用
    # 需要找到所有嵌套的函数，包括外层和内层
    # 但是要注意：外层函数的窗口应该累加内层函数的窗口
    # 例如：HHV(MA(REF(C,1),5),20) 需要 20 + 5 + 1 = 26天
    # 策略：先找到所有函数调用，然后从外层到内层递归计算
    
    # 收集所有函数调用的位置和范围
    # 需要找到所有嵌套的函数，包括外层和内层
    # 策略：遍历每个字符位置，查找所有函数调用
    func_calls = []
    processed_ranges = set()  # 记录已处理的函数调用范围，避免重复
    
    for func_name in nested_funcs:
        func_pattern = func_name + r'\s*\('
        # 查找所有匹配的函数调用
        for match in re.finditer(func_pattern, expr, re.IGNORECASE):
            func_start = match.start()
            func_end = match.end()
            
            # 找到匹配的右括号
            paren_count = 1
            j = func_end
            while j < len(expr) and paren_count > 0:
                if expr[j] == '(':
                    paren_count += 1
                elif expr[j] == ')':
                    paren_count -= 1
                j += 1
            
            if paren_count == 0:
                # 检查是否已经处理过这个函数调用
                if (func_start, j) not in processed_ranges:
                    func_calls.append((func_start, j, func_name))
                    processed_ranges.add((func_start, j))
    
    # 从外层到内层处理函数调用（按起始位置排序，外层函数在前）
    # 但是要注意：内层函数可能在外层函数之前（如MA在HHV之前）
    # 所以需要按嵌套深度排序：外层函数应该先处理
    # 简单方法：按函数调用的范围大小排序，范围大的（外层）先处理
    func_calls.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    # 处理每个函数调用
    for func_start, func_end, func_name in func_calls:
        func_call = expr[func_start:func_end]
        # 提取函数参数部分
        func_pattern = func_name + r'\s*\('
        match = re.search(func_pattern, func_call, re.IGNORECASE)
        if match:
            params = func_call[len(match.group(0)):len(func_call)-1]  # 去掉函数名(和最后的)
            
            # 查找最后一个逗号后的数字（窗口参数）
            last_comma = params.rfind(',')
            if last_comma != -1:
                after_comma = params[last_comma+1:].strip()
                num_match = re.match(r'(\d+)', after_comma)
                if num_match:
                    try:
                        window = int(num_match.group(1))
                        # 递归计算内部表达式所需窗口
                        inner_expr = params[:last_comma].strip()
                        inner_window = _calc_nested_window(inner_expr)
                        total_window = window + inner_window
                        max_total_window = max(max_total_window, total_window)
                    except (ValueError, TypeError):
                        pass
    
    # 如果没有找到需要嵌套的函数，再匹配REF等简单函数
    if max_total_window == 0:
        ref_pattern = r'REF\s*\([^)]+,\s*(\d+)'
        matches = re.finditer(ref_pattern, expr, re.IGNORECASE)
        for match in matches:
            try:
                window = int(match.group(1))
                max_total_window = max(max_total_window, window)
            except (ValueError, TypeError):
                pass
    
    return max_total_window


def _window_slice(dfTF: pd.DataFrame, ref_date: str, window: int) -> pd.DataFrame:
    dfTF = _ensure_datetime_index(dfTF)
    mask = dfTF.index <= dt.datetime.strptime(ref_date, "%Y%m%d")
    return dfTF.loc[mask].tail(int(window))


def _load_category_codes(category_type: str, _ref: str, source: Optional[str] = None) -> list[str]:
    """
    通用分类榜加载函数：
    从 SC_OUTPUT_DIR/{category_type}/ 下选择"end<=_ref"的最新一份：
      {category_type}_{source}_{start}_{end}.csv
    source 优先用传入的 source，否则用 SC_ATTENTION_SOURCE（如 'top'|'white'|'black'）
    返回 ts_code 列表；异常返回 []。会打 DEBUG 日志说明匹配的文件。
    """
    try:
        category_type = str(category_type or "strength").lower()
        src = str((source or SC_ATTENTION_SOURCE) or "top").lower()
        cat_dir = os.path.join(SC_OUTPUT_DIR, category_type)
        if not os.path.isdir(cat_dir):
            return []
        # 只匹配当前 source
        files = sorted(glob.glob(os.path.join(cat_dir, f"{category_type}_{src}_*.csv")))
        if not files:
            return []
        # 选择 end<=_ref 的最新一期；否则退化为按文件名排序的最后一份
        import re as _re
        cand = []
        for f in files:
            name = os.path.basename(f)
            m = _re.match(rf"{category_type}_{src}_(\d{{8}})_(\d{{8}})\.csv$", name)
            if m:
                start, end = m.group(1), m.group(2)
                if end <= _ref:
                    cand.append((end, f))
        pick = (sorted(cand)[-1][1] if cand else files[-1])
        df = pd.read_csv(pick, dtype=str)
        codes = df["ts_code"].astype(str).tolist() if "ts_code" in df.columns else []
        import logging as _logging
        if LOGGER.isEnabledFor(_logging.DEBUG):
            category_name = {"strength": "强度榜", "momentum": "动量榜"}.get(category_type, f"{category_type}榜")
            LOGGER.debug(f"[screen][范围] {category_name} 源={src} 参考日={_ref} 匹配文件={os.path.basename(pick)} 命中={len(codes)}")
        return codes
    except Exception as e:
        category_name = {"strength": "强度榜", "momentum": "动量榜"}.get(category_type, f"{category_type}榜")
        LOGGER.debug(f"[screen][范围] 加载{category_name}清单失败：{e}")
        return []


def _scope_hit(s_bool: pd.Series, scope: str) -> bool:
    """
    判断布尔序列是否满足某个scope条件。
    
    支持的 scope：
      - 基本：LAST | ANY | ALL | COUNT>=k | CONSEC>=m
      - 扩展：ANY_n | ALL_n   （n 为正整数；表示"在外层 window 内存在一个长度为 n 的连续子集"）
          ANY_n：该子集中 "when" 至少 1 天为 True
          ALL_n：该子集中 "when" 每天都为 True  （等价于"存在长度 n 的连续 True"）
    
    重要说明：
      - EACH / PERBAR / EACH_TRUE scope **不使用此函数**
      - 对于 EACH scope，应使用 _count_hits_perbar() 函数来计算窗口内逐K命中次数
      - _count_hits_perbar() 会统计窗口内有多少根K线满足条件，返回命中次数
      - 如果 _count_hits_perbar() 返回的 cnt > 0，说明策略触发（需要结合 gate 判断最终是否加分）
      - 参考：_eval_single_rule() 函数中对于 EACH scope 的处理逻辑（第3579行）
    
    参数：
      s_bool: 布尔序列（窗口内的布尔值）
      scope: scope字符串（如 "LAST", "ANY", "EACH" 等）
    
    返回：
      bool: 是否满足scope条件
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

    m1 = re.match(r"COUNT\s*(?:>=|=|≥|＝)\s*(\d+)\b", ss)
    if m1:
        k = int(m1.group(1))
        return int(s.sum()) >= k

    m2 = re.match(r"CONSEC\s*(?:>=|=|≥|＝)\s*(\d+)\b", ss)
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
    m3 = re.match(r"ANY(?:[_\-\s]?)(\d+)$", ss)
    if m3:
        n = int(m3.group(1))
        if len(s) < n:
            return False
        roll = s.rolling(n, min_periods=n).sum()
        hit = bool((roll >= 1).any())
        import logging as _logging
        if LOGGER.isEnabledFor(_logging.DEBUG):
            LOGGER.debug(f"[SCOPE] ANY_{n}: len={len(s)} max_in_subwin={int(roll.max())} -> {hit}")
        return hit

    m4 = re.match(r"ALL(?:[_\-\s]?)(\d+)$", ss)
    if m4:
        n = int(m4.group(1))
        if len(s) < n:
            return False
        roll = s.rolling(n, min_periods=n).sum()
        hit = bool((roll == n).any())
        import logging as _logging
        if LOGGER.isEnabledFor(_logging.DEBUG):
            LOGGER.debug(f"[SCOPE] ALL_{n}: len={len(s)} max_in_subwin={int(roll.max())} -> {hit}")
        return hit

    # 默认当作 ANY
    try:
        LOGGER.warning(f"[SCOPE] 未识别的 scope='{scope}' (规范化后='{ss}')，按 ANY 处理。支持的 scope: LAST, ANY, ALL, COUNT>=k, CONSEC>=m, ANY_n, ALL_n")
    except Exception:
        pass
    return bool(s.any())


def _eval_clause(dfD: pd.DataFrame, clause: dict, ref_date: str, ctx: dict = None) -> Tuple[bool, Optional[str]]:
    """
    返回 (是否命中, 错误消息或 None)
    """
    # 验证子句的必填字段（子句通常属于排名或筛选策略）
    if "when" in clause:
        # 判断是排名策略还是筛选策略（子句本身没有hard_penalty，需要从外层判断，这里默认按ranking处理）
        _validate_required_fields(clause, "ranking", clause.get("name"))
    
    tf = clause.get("timeframe", "D").upper()
    # score_windows 用于计算窗口
    score_window = _get_score_window(clause, SC_LOOKBACK_D)
    scope = clause.get("scope", "ANY")
    when = clause.get("when", "").strip()
    if not when:
        return False, "空 when 表达式"
    try:
        if ctx:
            # 使用缓存版本：自动使用策略编译时的最大 window 作为 data_window，score_window 用于计算窗口
            s_bool = _eval_bool_cached(ctx, dfD, when, tf, score_window, ref_date)
        else:
            # 原始版本：在完整数据上计算，然后切片窗口范围
            dfTF = dfD if tf=="D" else _resample(dfD, tf)
            win_df = _window_slice(dfTF, ref_date, score_window)
            if win_df.empty:
                return False, f"窗口数据为空: tf={tf}, window={score_window}"
            # 在完整数据上计算表达式
            with _ctx_df(dfTF):
                result_full = evaluate_bool(when, dfTF)
            # 切片到窗口范围
            result_full_series = pd.Series(result_full, index=dfTF.index, dtype=bool)
            s_bool = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
        hit = _scope_hit(s_bool, scope)
        return hit, None
    except Exception as e:
        return False, f"表达式错误: {e}"


def _eval_rule(dfD: pd.DataFrame, rule: dict, ref_date: str, ctx: dict = None) -> Tuple[bool, Optional[str]]:
    """
    支持两种写法：
      - 简单：{ timeframe, window, scope, when }
      - 复合：{ clauses: [ {tf,window,scope,when}, ... ] }
    命中逻辑：复合规则要求所有子句均命中。
    """
    if "clauses" in rule and rule["clauses"]:
        for c in rule["clauses"]:
            ok, err = _eval_clause(dfD, c, ref_date, ctx)
            if err:
                # 异常分支记日志
                LOGGER.warning(f"[RULE-ERR] {rule.get('name','<unnamed>')} 子句异常: {err}")
                return False, err
            if not ok:
                return False, None
        return True, None
    else:
        ok, err = _eval_clause(dfD, rule, ref_date, ctx)
        if err:
            LOGGER.warning(f"[RULE-ERR] {rule.get('name','<unnamed>')} 异常: {err}")
            return False, err
        return ok, None


def _check_database_health(check_connection_pool: bool = False):
    """
    检查数据库健康状态，用于诊断问题（使用状态文件）
    
    Args:
        check_connection_pool: 是否检查连接池状态（默认False，减少不必要的连接池访问）
    """
    try:
        from database_manager import get_database_manager
        from config import DATA_ROOT, UNIFIED_DB_PATH
        import os
        
        # 使用状态文件检查数据库状态
        manager = get_database_manager()
        status = manager.load_status_file()
        
        db_path = os.path.abspath(os.path.join(DATA_ROOT, UNIFIED_DB_PATH))
        
        health_info = {
            'using_unified_db': True,  # 统一使用统一数据库
            'database_exists': os.path.exists(db_path) if status is None else status.get('stock_data', {}).get('database_exists', False),
            'database_locked': False,  # 功能已迁移到database_manager
            'file_size': os.path.getsize(db_path) if os.path.exists(db_path) else 0,
            'status_file_exists': status is not None
        }
        
        # 如果状态文件存在，添加状态信息
        if status:
            stock_data = status.get('stock_data', {})
            health_info['stock_data_status'] = {
                'total_records': stock_data.get('total_records', 0),
                'total_stocks': stock_data.get('total_stocks', 0),
                'last_update': stock_data.get('last_update')
            }
            details_data = status.get('details_data', {})
            health_info['details_data_status'] = {
                'total_records': details_data.get('total_records', 0),
                'stock_count': details_data.get('stock_count', 0),
                'last_update': details_data.get('last_update')
            }
        
        # 只在需要时检查连接池状态（避免频繁访问连接池）
        if check_connection_pool:
            try:
                from database_manager import get_database_manager
                LOGGER.debug("[数据库连接] 检查连接池健康状态")
                db_manager = get_database_manager()
                if db_manager:
                    stats = db_manager.get_stats()
                    health_info['connection_stats'] = stats
                    
                    # 如果队列过大，触发清理
                    if stats.get('queue_size', 0) > 100:
                        LOGGER.warning(f"请求队列过大: {stats['queue_size']}，触发清理")
                        try:
                            from database_manager import clear_connections_only
                            clear_connections_only()
                        except:
                            pass
            except Exception as e:
                LOGGER.debug(f"获取连接统计失败: {e}")
        
        if health_info['database_locked']:
            LOGGER.warning(f"数据库健康检查: 数据库被锁定 - {health_info}")
        elif not health_info['database_exists']:
            LOGGER.warning(f"数据库健康检查: 数据库文件不存在 - {health_info}")
        else:
            LOGGER.debug(f"数据库健康检查: 正常 - {health_info}")
            
        return health_info
    except Exception as e:
        LOGGER.error(f"数据库健康检查失败: {e}")
        return {'error': str(e)}


def _get_data_from_cache(
    ts_code: str,
    start: str,
    end: str,
    columns: List[str],
    cache_dict: dict,
    cache_name: str,
    require_exact_start: bool = True
) -> Optional[pd.DataFrame]:
    """
    从预加载缓存中获取股票数据（通用函数）
    
    Args:
        ts_code: 股票代码
        start: 起始日期
        end: 结束日期
        columns: 需要的列
        cache_dict: 缓存字典
        cache_name: 缓存名称（用于日志）
        require_exact_start: 是否要求起始日期完全匹配
    
    Returns:
        如果命中缓存则返回DataFrame，否则返回None
    """
    if cache_dict["data"] is None:
        return None
    
    # 检查参考日期和列
    if cache_dict["ref_date"] != end:
        return None
    
    if not all(col in cache_dict["data"].columns for col in columns):
        return None
    
    # 检查起始日期（根据类型决定是否严格匹配）
    if require_exact_start:
        if cache_dict["start_date"] != start:
            return None
    else:
        # 筛选缓存可能包含更早的数据，只需要确保覆盖所需范围
        if cache_dict["start_date"] > start:
            return None
    
    try:
        # 从预加载数据中筛选指定股票
        stock_df = cache_dict["data"][cache_dict["data"]["ts_code"] == ts_code].copy()
        if stock_df.empty:
            return None
        
        # 如果不是严格匹配起始日期，需要筛选时间范围
        if not require_exact_start:
            mask = (stock_df["trade_date"] >= str(start)) & (stock_df["trade_date"] <= str(end))
            stock_df = stock_df.loc[mask].copy()
            if stock_df.empty:
                return None
        
        # 确保包含所需的列
        available_cols = [col for col in columns if col in stock_df.columns]
        if not available_cols:
            return None
        
        # 统计：单票读取阶段命中预加载
        try:
            _CACHE_STATS["preload_hits"] += 1
        except Exception:
            pass
        
        result_df = stock_df[["ts_code", "trade_date"] + available_cols]
        LOGGER.debug(f"[{ts_code}] 使用{cache_name}预加载数据: {len(result_df)} 条记录")
        return result_df
        
    except Exception as e:
        LOGGER.warning(f"[{ts_code}] {cache_name}预加载数据筛选失败: {e}")
        return None


def _read_stock_df(ts_code: str, start: str, end: str, columns: List[str]) -> pd.DataFrame:
    """
    读取某只股票在给定区间的日线数据。
    缓存优化逻辑已移至 database_manager.query_stock_data() 内部。
    """
    from database_manager import query_stock_data
    
    # 直接调用 query_stock_data，它会自动检查预加载缓存
    df = query_stock_data(
        ts_code=ts_code,
        start_date=start,
        end_date=end,
        columns=columns,
        adj_type="qfq"
    )
    
    # 统计：单票读取阶段未命中预加载（如果返回空或需要统计）
    if df.empty:
        try:
            _CACHE_STATS["preload_misses"] += 1
        except Exception:
            pass
    
    return df


@lru_cache(maxsize=256)
def _trade_span(start: Optional[str], end: str) -> List[str]:
    """把自然日区间换成交易日清单（基于现有分区）。"""
    try:
        get_trade_dates, _ = _get_database_manager_functions()
        days = get_trade_dates() or []
    except Exception as e:
        LOGGER.error(f"获取交易日列表失败: {e}")
        return []
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
            df_i = _read_data_via_dispatcher(
                code, start, end,
                ["trade_date","close"]
            )
            if df_i is None or df_i.empty:
                LOGGER.warning(f"[bench] 指数空数据: {code} ({start}~{end})"); continue
            df_i = df_i.copy()
            df_i["trade_date"] = df_i["trade_date"].astype(str)
            df_i = df_i.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
            df_i["close"] = pd.to_numeric(df_i["close"], errors="coerce")
            df_i["ret"]   = df_i["close"].pct_change()
            bm[code] = df_i
        except Exception as e:
            LOGGER.warning(f"[bench] 读取失败 {code}: {e}")
    LOGGER.debug(f"[bench] 加载完成 codes={codes} 覆盖={len(bm)}")
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
    # 优化类型转换：检查类型，避免不必要的转换（如果dtype不是object，或者第一个值不是字符串，才转换）
    if base["trade_date"].dtype != 'object':
        base["trade_date"] = base["trade_date"].astype(str)
    elif len(base) > 0 and not isinstance(base["trade_date"].iloc[0], str):
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

        # 优化日志：使用条件日志，避免不必要的字符串格式化和列表推导
        if LOGGER.isEnabledFor(_logging.DEBUG):
            LOGGER.debug(
                "[bench] 注入 %s: 窗口=%d 列增量=%s",
                code, W, [col for col in base.columns if col.endswith(tag)]
            )

    return base


def _eval_single_rule(dfD: pd.DataFrame, rule: dict, ref_date: str, ctx: dict = None, compute_hit_dates: bool = False) -> dict:
    """处理单个规则，返回规则执行结果
    
    Args:
        compute_hit_dates: 是否计算完整的hit_dates列表（仅在需要详情时启用，避免重复计算）
    """
    
    # 验证必填字段（仅对排名和筛选策略）
    rule_name = rule.get("name", "<unnamed>")
    if "when" in rule or "clauses" in rule:
        # 判断是排名策略还是筛选策略
        category = "filter" if "hard_penalty" in rule or "reason" in rule else "ranking"
        _validate_required_fields(rule, category, rule_name)
    
    tf   = str(rule.get("timeframe","D")).upper()
    scope = str(rule.get("scope", "ANY")).upper().strip()
    # 对于 RECENT/DIST/NEAR scope，score_windows 会被忽略，直接使用 dist_points 的最大值
    # _get_score_window 已经处理了这个逻辑
    score_window = _get_score_window(rule, SC_LOOKBACK_D)
    
    pts  = float(rule.get("points", 0))
    
    res = {"add": 0.0, "cnt": None, "lag": None, "hit_date": None, "hit_dates": [], "gate_ok": True, "error": None}
    try:
        # 优化：对于 scope: LAST，只计算最后一天，不需要计算整个窗口
        # 但是如果 when 条件中包含 COUNT 函数，需要传入足够的历史数据
        if scope == "LAST":
            # 检查 when 条件中是否包含 COUNT 函数
            when = None
            cand_when = None
            if "clauses" in rule and rule["clauses"]:
                for c in rule["clauses"]:
                    if c.get("when"): 
                        cand_when = c["when"]; 
                        break
            else:
                cand_when = rule.get("when")
            when = cand_when
            
            # 提取 COUNT 函数的最大窗口大小
            count_window = _extract_max_count_window(when or "")
            
            if ctx:
                # 获取重采样数据
                resampled = ctx.get('resampled', {})
                if tf not in resampled:
                    if tf == "D":
                        resampled[tf] = dfD
                    else:
                        resampled[tf] = _resample(dfD, tf)
                    ctx['resampled'] = resampled
                dfTF = resampled[tf]
            else:
                dfTF = dfD if tf=="D" else _resample(dfD, tf)
            
            dfTF = _ensure_datetime_index(dfTF)
            ref_dt = dt.datetime.strptime(ref_date, "%Y%m%d")
            
            # 如果包含 COUNT 函数，需要传入足够的历史数据
            if count_window > 0:
                # 对于COUNT函数，需要从COUNT窗口的最早一天往前推，计算COUNT内部表达式所需的历史数据
                # 例如：策略window=5天，COUNT窗口=5天，COUNT内部表达式包含TS_RANK(V,20)
                # 那么需要：5（COUNT窗口）+ 20（TS_RANK窗口）= 25天
                count_inner_exprs = list(_extract_count_inner_expr(when or ""))
                inner_window = 0
                for inner_expr in count_inner_exprs:
                    # 使用_calc_nested_window递归计算嵌套函数所需的总窗口大小
                    # 这样可以正确处理嵌套函数，如TS_RANK(REF(V,1),20)需要20+1=21天
                    inner_window = max(inner_window, _calc_nested_window(inner_expr))
                # data_window = count_window + COUNT内部表达式所需的最大窗口
                data_window = count_window + inner_window
                win_df = _window_slice(dfTF, ref_date, data_window)
            else:
                # 没有 COUNT 函数，但表达式可能包含嵌套函数（如HHV(REF(H,1),20)）
                # 需要计算嵌套函数所需的总窗口大小
                nested_window = _calc_nested_window(when or "")
                if nested_window > 0:
                    # 使用嵌套函数所需的总窗口大小
                    data_window = nested_window
                    win_df = _window_slice(dfTF, ref_date, data_window)
                else:
                    # 如果没有嵌套函数，只获取最后一天的数据
                    last_day_df = dfTF[dfTF.index <= ref_dt].tail(1)
                    win_df = last_day_df
            
            if win_df.empty:
                res["error"] = f"窗口数据为空: tf={tf}, ref_date={ref_date}, count_window={count_window}"
                return res
        else:
            # 其他 scope 需要计算整个窗口
            # 优化：复用ctx中的重采样和切片结果，避免重复计算
            # 注意：数据准备（重采样）和计分判断都使用 score_window
            if ctx:
                # 尝试从ctx获取已重采样的数据
                resampled = ctx.get('resampled', {})
                if tf not in resampled:
                    if tf == "D":
                        resampled[tf] = dfD
                    else:
                        resampled[tf] = _resample(dfD, tf)
                    ctx['resampled'] = resampled
                dfTF = resampled[tf]
                
                # 尝试从ctx获取已切片的窗口数据（按(tf, score_window, ref_date)缓存，用于计分判断）
                win_cache_key = (tf, score_window, ref_date)
                win_cache = ctx.get('win_cache', {})
                if win_cache_key not in win_cache:
                    win_cache[win_cache_key] = _window_slice(dfTF, ref_date, score_window)
                    ctx['win_cache'] = win_cache
                win_df = win_cache[win_cache_key]
            else:
                # 没有ctx则直接计算
                dfTF = dfD if tf=="D" else _resample(dfD, tf)
                win_df = _window_slice(dfTF, ref_date, score_window)
        when = None
        cand_when = None
        if "clauses" in rule and rule["clauses"]:
            for c in rule["clauses"]:
                if c.get("when"): 
                    cand_when = c["when"]; 
                    break
        else:
            cand_when = rule.get("when")
        when = cand_when
        
        # EACH/PERBAR/EACH_TRUE scope 处理：使用 _count_hits_perbar() 或 _count_hits_perbar_with_gate() 函数
        # 注意：EACH scope 不使用 _scope_hit() 函数，而是使用 _count_hits_perbar() 统计窗口内逐K命中次数
        # 对于带 gate 的 EACH scope 策略，每次命中都单独判断 gate
        # 参考：_count_hits_perbar() 函数的详细注释（第4284行）
        if scope in {"EACH","PERBAR","EACH_TRUE"}:
            # 检查是否有 gate
            gate_norm = _normalize_gate(rule)
            if gate_norm:
                # 如果有 gate，使用 _count_hits_perbar_with_gate() 函数
                # 该函数会对每次命中单独判断 gate，只统计同时满足 when 和 gate 的天数
                cnt, err = _count_hits_perbar_with_gate(dfD, rule, ref_date, ctx)
                if err: res["error"] = err
                # 对于带 gate 的 EACH scope，cnt 已经是同时满足 when 和 gate 的天数
                # 所以 gate_ok 始终为 True（因为已经过滤过了）
                res["gate_ok"] = True
            else:
                # 如果没有 gate，使用原来的函数
                cnt, err = _count_hits_perbar(dfD, rule, ref_date, ctx)
                if err: res["error"] = err
                res["gate_ok"] = True
            # 即使pts=0，只要cnt>0，也应该设置cnt和add（用于记录details）
            # 加分规则：add = points * count
            if cnt is not None:
                res["cnt"] = int(cnt)
                if cnt > 0:
                    res["add"] = float(pts * int(cnt))
                else:
                    res["add"] = 0.0
            else:
                res["cnt"] = 0
                res["add"] = 0.0
            # 命中日
            if when and not win_df.empty:
                _d, _ = _last_true_date(dfD, when, ref_date=ref_date, timeframe=tf, window=score_window, ctx=ctx); res["hit_date"] = _d
                # 如果需要完整hit_dates列表，且已计算过布尔序列，则复用缓存结果
                # 即使pts=0，只要cnt>0，也应该计算hit_dates（用于记录details）
                if compute_hit_dates or (cnt and cnt > 0):
                    try:
                        # 使用 _list_true_dates 函数，确保日期格式为 YYYYMMDD
                        res["hit_dates"] = _list_true_dates(dfD, when, ref_date=ref_date, window=score_window, timeframe=tf, ctx=ctx)
                    except Exception:
                        pass
            return res

        if scope in {"RECENT","DIST","NEAR"}:
            # 优化：直接使用已计算的win_df，避免在_recent_points中重复计算
            if when and not win_df.empty:
                # 使用已计算的win_df直接计算布尔序列
                if ctx:
                    # 优先使用缓存
                    s_bool = _eval_bool_cached(ctx, dfD, when, tf, score_window, ref_date)
                else:
                    # 在完整数据上计算，然后切片窗口范围
                    dfTF = dfD if tf=="D" else _resample(dfD, tf)
                    with _ctx_df(dfTF):
                        result_full = evaluate_bool(when, dfTF)
                    # 切片到窗口范围
                    result_full_series = pd.Series(result_full, index=dfTF.index, dtype=bool)
                    s_bool = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
                
                lag = _last_true_lag(s_bool)
                if lag is not None:
                    # 计算加分
                    bins = rule.get("dist_points") or rule.get("distance_points") or []
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
                    add = float(pts) if pts is not None else 0.0
                    res["lag"] = int(lag)
                    res["add"] = add
                else:
                    add, lag = 0.0, None
                    res["lag"] = None
            else:
                add, lag, err = _recent_points(dfD, rule, ref_date, ctx)
                if err: res["error"] = err
                res["lag"] = (None if lag is None else int(lag))
                res["add"] = add
            
            gate_ok, _, _ = _eval_gate(dfD, rule, ref_date, ctx)
            res["gate_ok"] = gate_ok
            if gate_ok and res["add"]:
                pass  # add已经在上面设置了
            else:
                res["add"] = 0.0
            
            if res["lag"] is not None and not win_df.empty:
                idx = win_df.index if isinstance(win_df.index, pd.DatetimeIndex) else pd.to_datetime(win_df["trade_date"].astype(str))
                i = res["lag"]; 
                if 0 <= i < len(idx): res["hit_date"] = idx[-1 - i].strftime("%Y%m%d")
            # 如果需要完整hit_dates列表，则计算
            if compute_hit_dates and when:
                try:
                    res["hit_dates"] = _list_true_dates(dfD, when, ref_date=ref_date, window=score_window, timeframe=tf, ctx=ctx)
                except Exception:
                    pass
            return res

        # 常规布尔
        # 优化：如果when表达式存在且win_df已计算，直接使用win_df计算，避免在_eval_rule中重复切片
        w = (rule.get("when") or "").strip()
        if w and not win_df.empty:
            # 对于 scope: LAST，只计算最后一天，不需要计算整个窗口
            if scope == "LAST":
                # 只计算最后一天的数据
                if ctx:
                    # 获取重采样数据（用于计算表达式，需要足够的历史数据）
                    resampled = ctx.get('resampled', {})
                    if tf not in resampled:
                        if tf == "D":
                            resampled[tf] = dfD
                        else:
                            resampled[tf] = _resample(dfD, tf)
                        ctx['resampled'] = resampled
                    dfTF = resampled[tf]
                else:
                    dfTF = dfD if tf=="D" else _resample(dfD, tf)
                
                # 使用策略编译时的最大 score_windows 作为 data_window，确保有足够的历史数据
                if ctx and 'max_windows_by_tf' in ctx:
                    max_windows = ctx.get('max_windows_by_tf', {})
                    data_window = max_windows.get(tf.upper(), score_window)
                else:
                    data_window = score_window
                
                # 检查 when 条件中是否包含 COUNT 函数，如果包含，需要确保 data_window 足够大
                count_window = _extract_max_count_window(w)
                if count_window > 0:
                    # 对于COUNT函数，需要从COUNT窗口的最早一天往前推，计算COUNT内部表达式所需的历史数据
                    # 例如：策略score_windows=5天，COUNT窗口=5天，COUNT内部表达式包含TS_RANK(V,20)
                    # 那么需要：5（COUNT窗口）+ 20（TS_RANK窗口）= 25天
                    count_inner_exprs = list(_extract_count_inner_expr(w))
                    inner_window = 0
                    for inner_expr in count_inner_exprs:
                        # 使用_calc_nested_window递归计算嵌套函数所需的总窗口大小
                        # 这样可以正确处理嵌套函数，如TS_RANK(REF(V,1),20)需要20+1=21天
                        inner_window = max(inner_window, _calc_nested_window(inner_expr))
                    # data_window = max(原有data_window, count_window + COUNT内部表达式所需的最大窗口)
                    data_window = max(data_window, count_window + inner_window)
                
                # 切片数据用于计算（需要足够的历史数据）
                dfTF = _ensure_datetime_index(dfTF)
                ref_dt = dt.datetime.strptime(ref_date, "%Y%m%d")
                df_calc = dfTF[dfTF.index <= ref_dt].tail(data_window)
                
                # 只计算最后一天
                with _ctx_df(df_calc):
                    result_full = evaluate_bool(w, df_calc)
                
                if result_full is None:
                    s_bool = pd.Series([False], index=win_df.index, dtype=bool)
                else:
                    result_full_series = pd.Series(result_full, index=df_calc.index, dtype=bool)
                    # 只取最后一天
                    s_bool = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
            else:
                # 其他 scope 需要计算整个窗口
                if ctx:
                    # 自动使用策略编译时的最大 score_windows 作为 data_window，score_window 用于计算窗口
                    s_bool = _eval_bool_cached(ctx, dfD, w, tf, score_window, ref_date)
                else:
                    # 在完整数据上计算，然后切片窗口范围
                    dfTF = dfD if tf=="D" else _resample(dfD, tf)
                    with _ctx_df(dfTF):
                        result_full = evaluate_bool(w, dfTF)
                    # 切片到窗口范围
                    result_full_series = pd.Series(result_full, index=dfTF.index, dtype=bool)
                    s_bool = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
            
            hit = _scope_hit(s_bool, str(rule.get("scope","ANY")).upper().strip())
            ok = hit
            err = None
        else:
            # 如果没有when表达式或win_df为空，使用原有的_eval_rule
            ok, err = _eval_rule(dfD, rule, ref_date, ctx)
        
        if err: res["error"] = err
        gate_ok, _, _ = _eval_gate(dfD, rule, ref_date, ctx)
        res["gate_ok"] = gate_ok
        if ok and gate_ok and pts:
            res["add"] = float(pts)
        # 命中日/全集命中
        if w and not win_df.empty:
            # 对于 scope: LAST，如果命中，hit_date 就是 ref_date
            # 修复：即使ok=False，也应该计算hit_date（记录最近一次满足条件的日期），与方法2一致
            if scope == "LAST":
                if ok:
                    res["hit_date"] = ref_date
                else:
                    # 即使ok=False，也计算最近一次满足条件的日期
                    _d, _ = _last_true_date(dfD, w, ref_date=ref_date, timeframe=tf, window=score_window, ctx=ctx)
                    res["hit_date"] = _d
                if compute_hit_dates and ok:
                    res["hit_dates"] = [ref_date]
            else:
                # 其他 scope 需要计算 hit_date
                _d, _ = _last_true_date(dfD, w, ref_date=ref_date, timeframe=tf, window=score_window, ctx=ctx); res["hit_date"] = _d
                # 如果需要完整hit_dates列表，则计算
                if compute_hit_dates:
                    try:
                        res["hit_dates"] = _list_true_dates(dfD, w, ref_date=ref_date, window=score_window, timeframe=tf, ctx=ctx)
                    except Exception:
                        pass
        return res
    except Exception as e:
        res["error"] = f"eval-exception: {e}"
        return res


def apply_rule_across_universe(rule: dict, ref_date: str | None = None, universe: str | list[str] | None = "all", return_df: bool = True):
    ref = ref_date or _pick_ref_date()
    tf  = str(rule.get("timeframe","D")).upper()
    win = int(rule.get("score_windows", SC_LOOKBACK_D))
    st  = _start_for_tf_window(ref, tf, win)
    codes0 = _list_codes_for_day(ref)
    codes, src = _apply_universe_filter(list(codes0), ref, universe)
    if not codes: 
        return pd.DataFrame() if return_df else None

    # 需要列：基础 + J/VR
    need = ["trade_date","open","high","low","close","vol","amount"]
    def scan(expr: str):
        if _expr_mentions(expr, "J"):  need.append("j")
        if _expr_mentions(expr, "VR"): need.append("vr")
    if "clauses" in (rule or {}) and rule["clauses"]:
        for c in rule["clauses"] or []: scan((c.get("when") or ""))
    else:
        scan((rule.get("when") or ""))

    rows = []
    # 完全串行化处理，避免任何并发数据库访问冲突
    # 表达式选股通常是I/O密集型，串行化不会显著影响性能
    _progress("screen_start", total=len(codes), current=0, message="串行化处理")
    for i, c in enumerate(codes):
        try:
            result = _apply_one_stock_for_rule(c, st, ref, rule, need)
            rows.append(result)
            _progress("screen_progress", total=len(codes), current=i+1)
        except Exception as e:
            LOGGER.error(f"处理股票 {c} 时出错: {e}")
            rows.append({"ts_code": c, "ref_date": ref, "add": 0.0, "error": str(e)})
    _progress("screen_done", total=len(rows), current=len(rows), message=f"rows={len(rows)}")

    if return_df:
        return pd.DataFrame(rows)
    return None


def _apply_one_stock_for_rule(ts_code: str, st: str, ref: str, rule: dict, need_cols: list[str], ctx: dict = None) -> dict:
    try:
        dfD = _read_stock_df(ts_code, st, ref, columns=need_cols)
        if dfD is None or dfD.empty:
            return {"ts_code": ts_code, "ref_date": ref, "add": 0.0, "error": "empty-window"}
        # 不再进行兜底计算，依赖数据完整性

        tdx.EXTRA_CONTEXT.update(get_eval_env(ts_code, ref))
        _inject_config_tags(dfD, ref, ctx)  # 允许表达式里用 ANY_TAG 等
        out = _eval_single_rule(dfD, rule, ref, ctx)
        out.update({"ts_code": ts_code, "ref_date": ref})
        return out
    except Exception as e:
        return {"ts_code": ts_code, "ref_date": ref, "add": 0.0, "error": str(e)}

# ------------------------- 初选（预筛） -------------------------
@dataclass
class PrescreenResult:
    passed: bool
    reason: Optional[str]          # 若淘汰，解释原因
    period: Optional[str]          # 写入缓存的 period（日期或区间）


def _period_for_clause(dfD: pd.DataFrame, clause: dict, ref_date: str) -> str:
    tf = clause.get("timeframe", "D").upper()
    window = int(clause.get("score_windows", SC_PRESCREEN_LOOKBACK_D))
    dfTF = dfD if tf=="D" else _resample(dfD, tf)
    win = _window_slice(dfTF, ref_date, window)
    if win.empty:
        return ref_date
    start = str(win["trade_date"].iloc[0])
    end = str(win["trade_date"].iloc[-1])
    return f"{start}-{end}" if start!=end else end


def _prescreen(dfD: pd.DataFrame, ref_date: str, ctx: dict = None) -> PrescreenResult:
    """
    命中任一 hard_penalty 规则即淘汰。
    reason：命中规则的 reason 文案（若未提供，用 name）。
    """
    for rule in (SC_PRESCREEN_RULES or []):
        hard = bool(rule.get("hard_penalty", True))
        # hard = bool(rule.get("hard_penalty", False))
        if not hard:
            continue
        if "clauses" in rule and rule["clauses"]:
            clause_periods = []
            all_ok = True
            any_err = None
            for c in rule["clauses"]:
                ok, err = _eval_clause(dfD, c, ref_date, ctx)
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
            ok, err = _eval_rule(dfD, rule, ref_date, ctx)
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


def _score_one(ts_code: str, ref_date: str, start_date: str, columns: List[str]) -> Optional[Tuple[str, Optional[ScoreDetail], Optional[Tuple[str,str,str]], Optional[List[dict]]]]:
    """
    返回：
      - (ts_code, detail 或 None, 黑名单项 或 None, detail数据列表 或 None)
      若被初选淘汰，返回黑名单项 (ts_code, period, reason)；否则返回打分 detail。
      第四个返回值：本进程缓冲区中的detail数据（用于子进程返回给主进程统一写入）
    """
    LOGGER.debug(f"开始评分股票: {ts_code}, 参考日期: {ref_date}, 开始日期: {start_date}")
    
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "top"), exist_ok=True)
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "all"), exist_ok=True)

    try:
        LOGGER.debug(f"[{ts_code}] 开始读取股票数据")
        df = _read_stock_df(ts_code, start_date, ref_date, columns)
        if df is None or df.empty:
            LOGGER.warning(f"[{ts_code}] 数据为空，跳过")
            # 获取本进程缓冲区的detail数据（用于子进程返回给主进程）
            detail_data = None
            cache_stats = None
            try:
                current_process = multiprocessing.current_process()
                if current_process.name != "MainProcess":
                    detail_data = _get_batch_buffer_data()
                    _BATCH_BUFFER["summaries"].clear()
                    _BATCH_BUFFER["per_rules"].clear()
                    _BATCH_BUFFER["current_count"] = 0
                    # 收集子进程的缓存统计信息
                    cache_stats = {
                        "hits": _CACHE_STATS.get("hits", 0),
                        "misses": _CACHE_STATS.get("misses", 0),
                        "local_hits": _CACHE_STATS.get("local_hits", 0),
                        "local_misses": _CACHE_STATS.get("local_misses", 0),
                        "preload_hits": _CACHE_STATS.get("preload_hits", 0),
                        "preload_misses": _CACHE_STATS.get("preload_misses", 0),
                        "fallbacks": _CACHE_STATS.get("fallbacks", 0),
                    }
            except (AttributeError, RuntimeError):
                pass
            return (ts_code, None, (ts_code, ref_date, "数据为空"), detail_data, cache_stats)
        
        LOGGER.debug(f"[{ts_code}] 数据读取成功: {len(df)}行数据")
        
        # 初始化单票上下文和缓存
        ctx = {
            'bool_lru': _SmallLRU(BOOL_CACHE_SIZE),
            'resampled': {},  # 重采样数据缓存
            'win_cache': {},  # 窗口切片缓存 (tf, win, ref_date) -> win_df
            'data_win_cache': {},  # 数据窗口缓存 (tf, data_window, ref_date) -> data_df
            'max_windows_by_tf': _MAX_WINDOWS_BY_TF.copy(),  # 策略编译时计算的最大 score_windows（按 timeframe），用于 data_window
            'bool_cache_hit': 0,
            'bool_cache_miss': 0
        }
        
        LOGGER.debug(f"[{ts_code}] 开始注入基准特征")
        df = _inject_benchmark_features(df, start_date, ref_date)
        tdx.EXTRA_CONTEXT.update(get_eval_env(ts_code, ref_date))

        # 注入基于 config 的自定义标签（仅 D 级别，可关）
        if SC_ENABLE_CUSTOM_TAGS:
            try:
                _inject_config_tags(df, ref_date, ctx)
                if SC_ENABLE_VERBOSE_SCORE_LOG:
                    LOGGER.debug(f"[{ts_code}] 自定义标签注入成功")
            except Exception as e:
                LOGGER.warning(f"[{ts_code}] 自定义标签注入失败: {e}")
        
        # 初选
        if SC_ENABLE_VERBOSE_SCORE_LOG:
            LOGGER.debug(f"[{ts_code}] 开始初选检查")
        pres = _prescreen(df, ref_date, ctx)
        if not pres.passed:
            if SC_ENABLE_VERBOSE_SCORE_LOG:
                LOGGER.debug(f"[{ts_code}] 初选未通过: {pres.reason}")
            # 获取本进程缓冲区的detail数据（用于子进程返回给主进程）
            detail_data = None
            try:
                current_process = multiprocessing.current_process()
                if current_process.name != "MainProcess":
                    detail_data = _get_batch_buffer_data()
                    _BATCH_BUFFER["summaries"].clear()
                    _BATCH_BUFFER["per_rules"].clear()
                    _BATCH_BUFFER["current_count"] = 0
            except (AttributeError, RuntimeError):
                pass
            return (ts_code, None, (ts_code, pres.period or ref_date, pres.reason or "prescreen"), detail_data)
        
        if SC_ENABLE_VERBOSE_SCORE_LOG:
            LOGGER.debug(f"[{ts_code}] 初选通过，开始评分")
        # 打分
        score = float(SC_BASE_SCORE)
        highlights, drawbacks, opportunities = [], [], []
        per_rules = []  # 收集规则详情，避免重复计算
        LOGGER.debug(f"[{ts_code}] 初始分数: {score}")
        
        def _append_reason_by_rule(points: float, rule: dict, tag_text: str | None = None):
            """
            根据 rule 配置决定把"理由"放进哪一个桶：
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

        def _add_rule_detail(rule: dict, ok: bool, cnt: int = None, add: float = 0.0, 
                           hit_date: str = None, hit_dates: list = None, err: str = None,
                           gate_ok: bool = True, gate_when: str = None):
            """添加规则详情到per_rules列表"""
            name = str(rule.get("name", "<unnamed>"))
            scope = str(rule.get("scope", "ANY")).upper().strip()
            pts = float(rule.get("points", 0))
            tf = str(rule.get("timeframe", "D")).upper()
            score_window = _get_score_window(rule, SC_LOOKBACK_D)
            expl = rule.get("explain")
            
            try:
                period = _period_for_clause(df, rule, ref_date)
            except Exception:
                period = ref_date
                
            per_rules.append({
                "name": name, "scope": scope, "timeframe": tf, "score_windows": score_window, "period": period,
                "points": pts, "ok": ok, "cnt": cnt,
                "add": add,
                "hit_date": hit_date,
                "hit_dates": hit_dates or [],
                "hit_count": len(hit_dates) if hit_dates else (cnt or 0),
                "gate_ok": bool(gate_ok),
                "gate_when": gate_when,
                "error": err,
            })

        for rule in _iter_unique_rules():
            # 使用统一的规则处理逻辑
            # 只在需要详情时才计算完整的hit_dates列表，避免重复计算
            compute_hit_dates = SC_ENABLE_RULE_DETAILS
            rule_result = _eval_single_rule(df, rule, ref_date, ctx, compute_hit_dates=compute_hit_dates)
            
            # 提取结果
            add = rule_result.get("add", 0.0)
            cnt = rule_result.get("cnt")
            lag = rule_result.get("lag")
            hit_date = rule_result.get("hit_date")
            hit_dates = rule_result.get("hit_dates", [])  # 直接从结果中获取，避免重复计算
            gate_ok = rule_result.get("gate_ok", True)
            err = rule_result.get("error")
            
            # 更新分数和日志
            if gate_ok and add != 0:
                score += add
                _append_reason_by_rule(add, rule)
                if SC_ENABLE_VERBOSE_SCORE_LOG:
                    LOGGER.debug(f"[HIT][{ts_code}] {rule.get('name','<unnamed>')} +{add} => {score}")
            elif not gate_ok:
                if SC_ENABLE_VERBOSE_SCORE_LOG:
                    LOGGER.debug(f"[GATE][{ts_code}] {rule.get('name')} 被 gate 拦截")
            else:
                if SC_ENABLE_VERBOSE_SCORE_LOG:
                    LOGGER.debug(f"[MISS][{ts_code}] {rule.get('name','<unnamed>')}")
            
            # 添加规则详情（可关）
            if SC_ENABLE_RULE_DETAILS:
                # 对于EACH/PERBAR/EACH_TRUE规则，即使add=0或gate_ok=False，只要cnt>0，也应该记录为ok=True
                scope = str(rule.get("scope", "ANY")).upper().strip()
                if scope in {"EACH", "PERBAR", "EACH_TRUE"}:
                    ok = bool(cnt and cnt > 0)
                else:
                    ok = bool(add != 0)
                _add_rule_detail(
                    rule, ok, cnt=cnt, add=add,
                    hit_date=hit_date, hit_dates=hit_dates,
                    err=err, gate_ok=gate_ok, gate_when=None
                )

        score = max(score, float(SC_MIN_SCORE))
        # tiebreak: KDJ J（低的在前）
        tb = None
        if (SC_TIE_BREAK or "").lower() in {"kdj_j_asc", "kdj_j"}:
            try:
                # 先规范索引，去除重复索引，避免 reindex 错误
                df = _ensure_datetime_index(df)
                if getattr(df.index, "has_duplicates", False):
                    df = df[~df.index.duplicated(keep="last")]
                # 取参考日所在行的 j；若没有精确匹配，就用最后一行
                if "j" in df.columns:
                    # 索引按日期匹配
                    ref_ts = pd.to_datetime(str(ref_date), errors="coerce")
                    if ref_ts is not None and ref_ts in df.index:
                        tb = float(df.loc[ref_ts, "j"])
                    else:
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
        if SC_ENABLE_RULE_DETAILS:
            try:
                # 使用批量缓冲区，避免频繁数据库写入
                _add_to_batch_buffer(
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
                LOGGER.warning(f"[detail] 写单票明细失败 {ts_code}: {e}")
        
        # 输出缓存统计
        if 'ctx' in locals():
            hit_count = ctx.get('bool_cache_hit', 0)
            miss_count = ctx.get('bool_cache_miss', 0)
            total_count = hit_count + miss_count
            hit_rate = (hit_count / total_count * 100) if total_count > 0 else 0
            LOGGER.debug(f"[bool-cache] {ts_code} hit={hit_count} miss={miss_count} rate={hit_rate:.1f}%")
        
        # 获取本进程缓冲区的detail数据（用于子进程返回给主进程）
        detail_data = None
        try:
            current_process = multiprocessing.current_process()
            if current_process.name != "MainProcess":
                # 子进程：返回缓冲区数据，由主进程统一写入
                detail_data = _get_batch_buffer_data()
                # 清空子进程的缓冲区（数据已经返回给主进程）
                _BATCH_BUFFER["summaries"].clear()
                _BATCH_BUFFER["per_rules"].clear()
                _BATCH_BUFFER["current_count"] = 0
                if detail_data:
                    LOGGER.debug(f"[detail] 子进程 {current_process.name} 返回 {len(detail_data)} 条detail数据给主进程")
        except (AttributeError, RuntimeError):
            # 如果无法获取进程信息，继续（单进程模式）
            pass
        
        # 如果是子进程，返回缓存统计信息
        cache_stats = None
        try:
            current_process = multiprocessing.current_process()
            if current_process.name != "MainProcess":
                # 收集子进程的缓存统计信息
                cache_stats = {
                    "hits": _CACHE_STATS.get("hits", 0),
                    "misses": _CACHE_STATS.get("misses", 0),
                    "local_hits": _CACHE_STATS.get("local_hits", 0),
                    "local_misses": _CACHE_STATS.get("local_misses", 0),
                    "preload_hits": _CACHE_STATS.get("preload_hits", 0),
                    "preload_misses": _CACHE_STATS.get("preload_misses", 0),
                    "fallbacks": _CACHE_STATS.get("fallbacks", 0),
                }
                # 收集子进程的表达式编译缓存统计信息
                try:
                    import tdx_compat
                    expr_stats = tdx_compat.get_expr_cache_stats()
                    cache_stats["expr_hits"] = expr_stats.get("hits", 0)
                    cache_stats["expr_misses"] = expr_stats.get("misses", 0)
                    cache_stats["expr_cache_size"] = expr_stats.get("cache_size", 0)
                except Exception:
                    pass
        except (AttributeError, RuntimeError):
            pass
        
        return (ts_code, ScoreDetail(ts_code, score, highlights, drawbacks, opportunities, tb), None, detail_data, cache_stats)

    except Exception as e:
        LOGGER.error(f"[{ts_code}] 评分失败：{e}")
        # 若 df 已读到，尽量仍生成一份规则明细，便于排错
        try:
            if 'df' in locals() and isinstance(df, pd.DataFrame) and (not df.empty):
                # 如果per_rules已经收集，直接使用；否则重新计算
                if 'per_rules' not in locals() or not per_rules:
                    per_rules = _build_per_rule_detail(df, ref_date, ctx)
                _add_to_batch_buffer(
                    ts_code, ref_date,
                    summary={"score": float(SC_BASE_SCORE), "tiebreak": None,
                             "highlights": [], "drawbacks": []},
                    per_rules=per_rules
                )
        except Exception as e2:
            LOGGER.warning(f"[detail] 构建/写入失败 {ts_code}: {e2}")
        
        # 获取本进程缓冲区的detail数据（用于子进程返回给主进程）
        detail_data = None
        cache_stats = None
        try:
            current_process = multiprocessing.current_process()
            if current_process.name != "MainProcess":
                detail_data = _get_batch_buffer_data()
                _BATCH_BUFFER["summaries"].clear()
                _BATCH_BUFFER["per_rules"].clear()
                _BATCH_BUFFER["current_count"] = 0
                # 收集子进程的缓存统计信息
                cache_stats = {
                    "hits": _CACHE_STATS.get("hits", 0),
                    "misses": _CACHE_STATS.get("misses", 0),
                    "local_hits": _CACHE_STATS.get("local_hits", 0),
                    "local_misses": _CACHE_STATS.get("local_misses", 0),
                    "preload_hits": _CACHE_STATS.get("preload_hits", 0),
                    "preload_misses": _CACHE_STATS.get("preload_misses", 0),
                    "fallbacks": _CACHE_STATS.get("fallbacks", 0),
                }
        except (AttributeError, RuntimeError):
            pass
        
        return (ts_code, None, (ts_code, ref_date, f"评分失败:{e}"), detail_data, cache_stats)


def _screen_one(ts_code, st, ref, tf, win, when, scope, use_prescreen_first, ctx: dict = None, precomputed_cols: list[str] | None = None):
    try:
        # 单票筛选同样并入规则/标签所需列，避免 'j'、'duokong_*' 等缺列
        # 优化：如果提供了预计算的列，直接使用，避免每个股票都重复扫描表达式
        if precomputed_cols is not None:
            _need_cols = precomputed_cols
        else:
            # 回退：如果没有预计算列，才进行扫描（兼容旧代码）
            try:
                _rule_cols = _select_columns_for_rules()
            except Exception:
                _rule_cols = []
            _need_cols = sorted(set(_scan_cols_from_expr(when)) | set(_rule_cols))
        dfD = _read_stock_df(ts_code, st, ref, columns=_need_cols)
        if dfD is None or dfD.empty:
            return (ts_code, False, None, None)
        if use_prescreen_first:
            pres = _prescreen(dfD, ref, ctx)
            if not pres.passed:
                return (ts_code, False, (ts_code, pres.period or ref, pres.reason or "prescreen"), None)

        # 不再进行兜底计算，依赖数据完整性

        if ctx:
            # 使用缓存版本：自动使用策略编译时的最大 window 作为 data_window，score_window 用于计算窗口
            s_bool = _eval_bool_cached(ctx, dfD, when, tf, win, ref)
            ok = _scope_hit(s_bool, scope)
        else:
            # 原始版本：在完整数据上计算，然后切片窗口范围
            dfTF = dfD if tf == "D" else _resample(dfD, tf)
            win_df = _window_slice(dfTF, ref, win)
            if win_df.empty:
                return (ts_code, False, None, None)
            tdx.EXTRA_CONTEXT.update(get_eval_env(ts_code, ref))

            _inject_config_tags(dfD, ref, skip_tags=True)      # 表达式选股场景，跳过标签注入
            # 在完整数据上计算表达式
            tdx.EXTRA_CONTEXT["DF"] = dfTF
            result_full = evaluate_bool(when, dfTF)
            # 切片到窗口范围
            result_full_series = pd.Series(result_full, index=dfTF.index, dtype=bool)
            s_bool = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
            ok = _scope_hit(s_bool, scope)
        if ok:
            return (ts_code, True, (ts_code, ref, "pass"), None)
        else:
            return (ts_code, False, None, None)
    except Exception as e:
        return (ts_code, False, None, f"{e}")


def _count_hits_perbar(dfD: pd.DataFrame, rule_or_clause: dict, ref_date: str, ctx: dict = None) -> Tuple[int, Optional[str]]:
    """
    统计"窗口内逐K命中次数"（专门用于 scope=EACH / PERBAR / EACH_TRUE）。
    
    重要说明：
      此函数专门用于处理 EACH 类型的 scope，与 _scope_hit() 函数不同：
      - _scope_hit() 用于判断布尔序列是否满足某个条件（如 ANY、ALL、LAST 等）
      - _count_hits_perbar() 用于统计窗口内有多少根K线满足条件，返回命中次数
      
      对于 EACH scope 的策略：
      1. 使用此函数计算窗口内逐K命中次数
      2. 如果返回的 count > 0，说明策略触发（有命中）
      3. 最终是否加分还需要结合 gate 判断（在 _eval_single_rule 中处理）
      4. 加分 = points * count（如果 gate_ok 为 True）
    
    处理逻辑：
      - 单条 rule（无 clauses）：
        1. 在其 timeframe 上按 when 表达式逐K计算布尔值
        2. 在 score_windows（或 window）窗口内统计 True 的数量
        3. 返回命中次数
      - 带 clauses 的 rule：
        1. 要求所有子句 timeframe/window/score_windows 一致
        2. 逐K按 AND 融合所有子句的布尔值
        3. 统计融合后为 True 的数量
        4. 返回命中次数
    
    参数：
      dfD: 基础数据DataFrame（日线数据）
      rule_or_clause: 规则或子句字典，包含 when、timeframe、window、score_windows 等
      ref_date: 参考日期（YYYYMMDD格式）
      ctx: 上下文字典（可选，用于缓存优化）
    
    返回值：
      (count, err): 
        - count: 命中次数（整数），如果 count > 0 说明策略触发
        - err: 错误消息（字符串），如果 err=None 表示成功；err 不为空时建议回退为布尔命中
    
    使用示例：
      在 _eval_single_rule() 函数中（第3579行）：
      ```python
      if scope in {"EACH","PERBAR","EACH_TRUE"}:
          cnt, err = _count_hits_perbar(dfD, rule, ref_date, ctx)
          if cnt and cnt > 0:
              res["cnt"] = int(cnt)
              res["add"] = float(pts * int(cnt)) if gate_ok else 0.0
      ```
    """
    try:
        if "clauses" in rule_or_clause and rule_or_clause["clauses"]:
            clauses = rule_or_clause["clauses"]
            tfs = [str(c.get("timeframe","D")).upper() for c in clauses]
            # score_windows 用于计算窗口
            score_wins = [_get_score_window(c, SC_LOOKBACK_D) for c in clauses]
            if len(set(tfs)) != 1 or len(set(score_wins)) != 1:
                return 0, "EACH 目前不支持多 timeframe/score_windows 子句混用"
            tf = tfs[0]
            score_window = score_wins[0]
            
            # 优化：复用ctx中的win_cache，避免重复切片
            if ctx:
                # 尝试复用win_cache（使用 score_window 作为缓存键）
                win_cache_key = (tf, score_window, ref_date)
                win_cache = ctx.get('win_cache', {})
                if win_cache_key in win_cache:
                    win_df = win_cache[win_cache_key]
                else:
                    # 复用resampled缓存
                    resampled = ctx.get('resampled', {})
                    if tf not in resampled:
                        if tf == "D":
                            resampled[tf] = dfD
                        else:
                            resampled[tf] = _resample(dfD, tf)
                        ctx['resampled'] = resampled
                    dfTF = resampled[tf]
                    win_df = _window_slice(dfTF, ref_date, score_window)
                    win_cache[win_cache_key] = win_df
                    ctx['win_cache'] = win_cache
            else:
                # 没有ctx则直接计算
                dfTF = dfD if tf=="D" else _resample(dfD, tf)
                win_df = _window_slice(dfTF, ref_date, score_window)
            
            if win_df.empty:
                return 0, None
            # 批量计算所有clauses的布尔序列，然后一次性向量化AND
            sigs = []
            for c in clauses:
                when = (c.get("when") or "").strip()
                if not when:
                    return 0, "空 when 表达式"
                if ctx:
                    # 使用缓存版本：自动使用策略编译时的最大 window 作为 data_window，score_window 用于计算窗口
                    s = _eval_bool_cached(ctx, dfD, when, tf, score_window, ref_date)
                else:
                    # 原始版本：在完整数据上计算，然后切片窗口范围
                    dfTF = dfD if tf=="D" else _resample(dfD, tf)
                    with _ctx_df(dfTF):
                        result_full = evaluate_bool(when, dfTF)
                    # 切片到窗口范围
                    result_full_series = pd.Series(result_full, index=dfTF.index, dtype=bool)
                    s = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
                sigs.append(s)
            
            # 向量化AND操作：一次性对所有序列进行AND操作，比逐个循环快
            if not sigs:
                return 0, "无有效子句"
            # 对齐所有序列的索引（使用第一个序列的索引作为基准）
            base_idx = sigs[0].index
            aligned_sigs = []
            for s in sigs:
                s_aligned = s.reindex(base_idx, fill_value=False).astype(bool)
                aligned_sigs.append(s_aligned.values)
            # 使用numpy一次性对所有序列进行AND操作，比循环快得多
            if aligned_sigs:
                s_all_array = np.logical_and.reduce(aligned_sigs)
                s_all = pd.Series(s_all_array, index=base_idx)
                return int(s_all.sum()), None
            return 0, "无有效子句"
        else:
            tf = str(rule_or_clause.get("timeframe", "D")).upper()
            # score_windows 用于计算窗口
            score_window = _get_score_window(rule_or_clause, SC_LOOKBACK_D)
            when = (rule_or_clause.get("when") or "").strip()
            if not when:
                return 0, "空 when 表达式"
            if ctx:
                # 使用缓存版本：自动使用策略编译时的最大 window 作为 data_window，score_window 用于计算窗口
                s_bool = _eval_bool_cached(ctx, dfD, when, tf, score_window, ref_date)
            else:
                # 原始版本：在完整数据上计算，然后切片窗口范围
                dfTF = dfD if tf=="D" else _resample(dfD, tf)
                win_df = _window_slice(dfTF, ref_date, score_window)
                if win_df.empty:
                    return 0, None
                # 在完整数据上计算表达式
                with _ctx_df(dfTF):
                    result_full = evaluate_bool(when, dfTF)
                # 切片到窗口范围
                result_full_series = pd.Series(result_full, index=dfTF.index, dtype=bool)
                s_bool = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
            return int(s_bool.fillna(False).sum()), None
    except Exception as e:
        return 0, f"表达式错误: {e}"


def _count_hits_perbar_with_gate(dfD: pd.DataFrame, rule: dict, ref_date: str, ctx: dict = None) -> Tuple[int, Optional[str]]:
    """
    统计"窗口内逐K命中次数"（专门用于 scope=EACH / PERBAR / EACH_TRUE，且带 gate）。
    
    对于 EACH scope 的策略，每次命中都单独判断 gate：
    1. 先找出窗口内所有满足 when 条件的日期（或满足所有 clauses 的日期）
    2. 对于每个满足 when 的日期，单独判断那一天的 gate
    3. 只有同时满足 when 和 gate 的日期才计入计数
    
    参数：
      dfD: 基础数据DataFrame（日线数据）
      rule: 规则字典，包含 when、gate、timeframe、window、score_windows 等
      ref_date: 参考日期（YYYYMMDD格式）
      ctx: 上下文字典（可选，用于缓存优化）
    
    返回值：
      (count, err): 
        - count: 命中次数（整数），同时满足 when 和 gate 的天数
        - err: 错误消息（字符串），如果 err=None 表示成功
    """
    try:
        # 获取 gate 配置
        gate_norm = _normalize_gate(rule)
        if not gate_norm:
            # 如果没有 gate，直接使用原来的函数
            return _count_hits_perbar(dfD, rule, ref_date, ctx)
        
        # 处理带 clauses 的规则
        if "clauses" in rule and rule["clauses"]:
            clauses = rule["clauses"]
            tfs = [str(c.get("timeframe","D")).upper() for c in clauses]
            score_wins = [_get_score_window(c, SC_LOOKBACK_D) for c in clauses]
            if len(set(tfs)) != 1 or len(set(score_wins)) != 1:
                return 0, "EACH 目前不支持多 timeframe/score_windows 子句混用"
            tf = tfs[0]
            score_window = score_wins[0]
            
            # 获取窗口内的数据
            if ctx:
                win_cache_key = (tf, score_window, ref_date)
                win_cache = ctx.get('win_cache', {})
                if win_cache_key in win_cache:
                    win_df = win_cache[win_cache_key]
                else:
                    resampled = ctx.get('resampled', {})
                    if tf not in resampled:
                        if tf == "D":
                            resampled[tf] = dfD
                        else:
                            resampled[tf] = _resample(dfD, tf)
                        ctx['resampled'] = resampled
                    dfTF = resampled[tf]
                    win_df = _window_slice(dfTF, ref_date, score_window)
                    win_cache[win_cache_key] = win_df
                    ctx['win_cache'] = win_cache
            else:
                dfTF = dfD if tf=="D" else _resample(dfD, tf)
                win_df = _window_slice(dfTF, ref_date, score_window)
            
            if win_df.empty:
                return 0, None
            
            # 批量计算所有clauses的布尔序列，然后一次性向量化AND
            sigs = []
            for c in clauses:
                when = (c.get("when") or "").strip()
                if not when:
                    return 0, "空 when 表达式"
                if ctx:
                    s = _eval_bool_cached(ctx, dfD, when, tf, score_window, ref_date)
                else:
                    with _ctx_df(dfTF):
                        result_full = evaluate_bool(when, dfTF)
                    result_full_series = pd.Series(result_full, index=dfTF.index, dtype=bool)
                    s = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
                sigs.append(s)
            
            # 向量化AND操作
            if not sigs:
                return 0, "无有效子句"
            base_idx = sigs[0].index
            aligned_sigs = []
            for s in sigs:
                s_aligned = s.reindex(base_idx, fill_value=False).astype(bool)
                aligned_sigs.append(s_aligned.values)
            if aligned_sigs:
                s_all_array = np.logical_and.reduce(aligned_sigs)
                s_when = pd.Series(s_all_array, index=base_idx)
            else:
                return 0, "无有效子句"
        else:
            # 处理单条规则
            tf = str(rule.get("timeframe", "D")).upper()
            score_window = _get_score_window(rule, SC_LOOKBACK_D)
            when = (rule.get("when") or "").strip()
            if not when:
                return 0, "空 when 表达式"

            # 获取窗口内的数据
            if ctx:
                win_cache_key = (tf, score_window, ref_date)
                win_cache = ctx.get('win_cache', {})
                if win_cache_key in win_cache:
                    win_df = win_cache[win_cache_key]
                else:
                    resampled = ctx.get('resampled', {})
                    if tf not in resampled:
                        if tf == "D":
                            resampled[tf] = dfD
                        else:
                            resampled[tf] = _resample(dfD, tf)
                        ctx['resampled'] = resampled
                    dfTF = resampled[tf]
                    win_df = _window_slice(dfTF, ref_date, score_window)
                    win_cache[win_cache_key] = win_df
                    ctx['win_cache'] = win_cache
            else:
                dfTF = dfD if tf == "D" else _resample(dfD, tf)
                win_df = _window_slice(dfTF, ref_date, score_window)
            
            if win_df.empty:
                return 0, None
            
            # 计算 when 条件的布尔序列
            if ctx:
                s_when = _eval_bool_cached(ctx, dfD, when, tf, score_window, ref_date)
            else:
                with _ctx_df(dfTF):
                    result_full = evaluate_bool(when, dfTF)
                result_full_series = pd.Series(result_full, index=dfTF.index, dtype=bool)
                s_when = result_full_series.reindex(win_df.index, fill_value=False).astype(bool)
        
        # 找出所有满足 when 条件的日期
        when_true_indices = s_when[s_when].index
        if len(when_true_indices) == 0:
            return 0, None
        
        # 对于每个满足 when 的日期，单独判断那一天的 gate
        count = 0
        for idx in when_true_indices:
            # 将索引转换为日期字符串（YYYYMMDD格式）
            if isinstance(idx, pd.Timestamp):
                date_str = idx.strftime("%Y%m%d")
            elif hasattr(idx, 'strftime'):
                date_str = idx.strftime("%Y%m%d")
            else:
                try:
                    dt = pd.to_datetime(idx)
                    date_str = dt.strftime("%Y%m%d")
                except:
                    continue
            
            # 判断这一天的 gate
            try:
                gate_ok, gate_err, _ = _eval_gate(dfD, rule, date_str, ctx)
                if gate_ok and not gate_err:
                    count += 1
            except Exception as e:
                # 如果判断 gate 出错，跳过这一天
                continue
        
        return count, None
    except Exception as e:
        return 0, f"表达式错误: {e}"


def build_category_rank(category_type: str = "strength",
                        start: Optional[str] = None,
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
    通用分类榜生成函数（支持强度榜等多种分类榜）：
      - category_type: 分类榜类型，如 'strength'（强度榜）、'momentum'（动量榜）等
      - mode='strength'（默认）：最近窗口内"排名强度"（按每日横截面名次分位值累计）；
      - mode='hits'           ：旧逻辑，按"上榜次数"统计。
    窗口：若未给 start，则取最近 SC_ATTENTION_WINDOW_D 个交易日，以 end/ref_date 为右端点。
    输出：{category_type}_{source}_{span_start}_{span_end}.csv（会直接覆盖旧文件）。
    """
    end = end or _pick_ref_date()
    source = (source or SC_ATTENTION_SOURCE).lower()
    min_hits = int(min_hits or SC_ATTENTION_MIN_HITS)
    topN = int(topN or SC_ATTENTION_TOP_K)
    category_type = str(category_type or "strength").lower()

    span = _trade_span(start, end)
    if not span:
        LOGGER.warning(f"[分类榜/{category_type}] 交易日窗口为空，start={start} end={end}")
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
                LOGGER.warning(f"[分类榜/{category_type}] 读取失败 {f}: {e}");
                continue
            for c in codes:
                hit_cnt[c] = hit_cnt.get(c, 0) + 1
                first_last[c] = (first_last.get(c, (d, d))[0], d)
        if not hit_cnt:
            LOGGER.warning(f"[分类榜/{category_type}] {span[0]}~{span[-1]} 内没有 '{source}' 记录。")
            return None
        rows = [
            {"ts_code": c, "hits": n, "first": first_last[c][0], "last": first_last[c][1]}
            for c, n in hit_cnt.items() if n >= min_hits
        ]
        if not rows:
            LOGGER.info("[分类榜/%s] 命中数 >= %d 的为空。", category_type, min_hits); return None
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
                    LOGGER.warning(f"[分类榜/{category_type}] 读取全量排名失败 {path_all}: {e}")
                    df = None
            if df is None:
                # 回退：仅有 Top-K 文件时，以其相对名次近似
                path_top = os.path.join(SC_OUTPUT_DIR, "top", f"score_top_{d}.csv")
                if not os.path.isfile(path_top):
                    continue
                try:
                    df_top = pd.read_csv(path_top, dtype={"ts_code": str})
                except Exception as e:
                    LOGGER.warning(f"[分类榜/{category_type}] 读取 Top 失败 {path_top}: {e}"); continue
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
            LOGGER.warning(f"[分类榜/{category_type}] {span[0]}~{span[-1]} 无可计算的排名强度。"); return None
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
            LOGGER.info("[分类榜/%s] 强度口径下，命中数 >= %d 的为空。", category_type, min_hits); return None
        out_df = (pd.DataFrame(rows)
                  .sort_values(["strength", "last", "best_rank", "ts_code"],
                               ascending=[False, False, True, True])
                  .head(topN))

    # 输出目录基于 category_type
    out_dir = os.path.join(SC_OUTPUT_DIR, category_type)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{category_type}_{source}_{span[0]}_{span[-1]}.csv")

    if write:
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        category_name = {"strength": "强度榜", "momentum": "动量榜"}.get(category_type, f"{category_type}榜")
        LOGGER.info(f"[{category_name}][{end}|w={len(span)}] source={source} 窗口={span[0]}~{span[-1]} min_hits>={min_hits} Top-{topN} -> {out_path}")

        return out_path
    else:
        return out_df


def backfill_category_rolling(category_type: str = "strength",
                              start: str = None,
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
    通用分类榜滚动补算函数：
    从start到end（按交易日）逐日滚动补算分类榜：
    - 每个 end 日都会生成一份：窗口=最近 SC_ATTENTION_WINDOW_D 个交易日（与 build_category_rank 保持一致）
    - category_type: 分类榜类型，如 'strength'（强度榜）、'momentum'（动量榜）等
    - source: 'top' | 'white' | 'black'
    """
    if start is None:
        raise ValueError("start 参数必须提供")
    end = end or _pick_ref_date()
    days = _trade_span(start, end)
    if not days:
        category_name = {"strength": "强度榜", "momentum": "动量榜"}.get(category_type, f"{category_type}榜")
        LOGGER.warning(f"[分类榜/{category_type}/backfill] 窗口为空：start={start} end={end}")
        return

    category_name = {"strength": "强度榜", "momentum": "动量榜"}.get(category_type, f"{category_type}榜")
    LOGGER.info(f"[{category_name}/backfill] 逐日补算 {days[0]} ~ {days[-1]}，共 {len(days)} 个交易日")
    ok, fail = 0, 0
    for e in days:
        try:
            build_category_rank(category_type=category_type, start=None, end=e, source=source,
                                min_hits=min_hits, topN=topN, write=True, mode=mode,
                                weight_mode=weight_mode, half_life=half_life,
                                linear_min=linear_min, topM=topM)
            ok += 1
        except Exception as ex:
            fail += 1
            LOGGER.warning(f"[{category_name}/backfill] 生成 {e} 失败：{ex}")
    LOGGER.info(f"[{category_name}/backfill] 完成：成功 %d，失败 %d", ok, fail)


# ------------------------- 自选榜（策略组合）相关函数 -------------------------
def build_custom_rank(combo_name: str,
                      rule_names: list[str],
                      agg_mode: str = "OR",
                      ref_date: Optional[str] = None,
                      universe: str = "all",
                      tiebreak: str = "kdj_j_asc",
                      topN: Optional[int] = None,
                      write: bool = True,
                      exclude_rules: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    """
    生成自选榜（按策略组合筛选）：
      - combo_name: 策略组合名称（用于文件名）
      - rule_names: 策略名列表
      - agg_mode: 聚合模式 'OR'（任一命中）| 'AND'（全部命中）
      - ref_date: 参考日（留空=自动最新）
      - universe: 选股范围 'all' | 'white' | 'black' | 'attention' | 'strength'
      - tiebreak: 同分排序方式
      - topN: 输出前N名（留空=不限制）
      - write: 是否落盘
      - exclude_rules: 排除策略列表（如果触发这些策略则排除）
    返回 DataFrame 或文件路径
    """
    from database_manager import get_database_manager
    import json
    
    # 获取 details 数据库路径
    def _get_details_db_path():
        """获取 details 数据库路径"""
        from config import SC_OUTPUT_DIR, SC_DETAIL_DB_PATH
        db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
        if os.path.isfile(db_path):
            return db_path
        return None
    
    ref_date = ref_date or _pick_ref_date()
    if not ref_date:
        LOGGER.warning(f"[自选榜] 未能确定参考日")
        return None
    
    if not rule_names:
        LOGGER.warning(f"[自选榜] 策略组合为空")
        return None
    
    # 获取选股范围
    codes_all = _get_universe_codes(universe, ref_date)
    if not codes_all:
        LOGGER.warning(f"[自选榜] 选股范围 {universe} 为空")
        return None
    
    rows = []
    try:
        # 优先使用数据库查询
        manager = get_database_manager()
        details_db_path = _get_details_db_path()
        db_available = manager and details_db_path
        
        if db_available:
            if details_db_path:
                sql = "SELECT * FROM stock_details WHERE ref_date = ? AND ts_code IN ({})".format(
                    ",".join(["?"] * len(codes_all))
                )
                df_all = manager.execute_sync_query(details_db_path, sql, [ref_date] + codes_all, timeout=30.0)
            else:
                df_all = pd.DataFrame()
        else:
            df_all = pd.DataFrame()
        
        # 处理数据
        if not df_all.empty:
            LOGGER.debug(f"[自选榜] 从数据库读取到 {len(df_all)} 条记录")
            for _, row in df_all.iterrows():
                ts_code = str(row.get("ts_code", "")).strip()
                if not ts_code or ts_code not in codes_all:
                    continue
                
                # 从数据库行中解析数据
                try:
                    # 直接从数据库行获取 score 和 tiebreak（数据库表中有这些列）
                    score = float(row.get('score', 0.0)) if row.get('score') is not None else 0.0
                    tiebreak_j = float(row.get('tiebreak', 999999.0)) if row.get('tiebreak') is not None else 999999.0
                    
                    # 解析 rules 字段
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
                except Exception as e:
                    LOGGER.debug(f"[自选榜] 解析数据库数据失败 {ts_code}: {e}")
                    continue
                
                if not rules:
                    continue
                
                # 检查策略触发情况
                names_triggered = set()
                names_excluded = set()
                hit_dates_map = {}
                for rr in rules:
                    ok_val = rr.get("ok")
                    add_val = rr.get("add")
                    if bool(ok_val) or (add_val is not None and float(add_val) > 0.0):
                        n = rr.get("name")
                        if n:
                            # 检查是否在选中的策略中
                            if n in rule_names:
                                names_triggered.add(str(n))
                                # 收集触发日期
                                hit_date = rr.get("hit_date")
                                hit_dates = rr.get("hit_dates", [])
                                all_dates = []
                                if hit_date:
                                    all_dates.append(str(hit_date))
                                if hit_dates:
                                    all_dates.extend([str(d) for d in hit_dates if d])
                                all_dates = sorted(set(all_dates))
                                if all_dates:
                                    hit_dates_map[str(n)] = all_dates
                            # 检查是否在排除策略中
                            if exclude_rules and n in exclude_rules:
                                names_excluded.add(str(n))
                
                # 如果触发了排除策略，则跳过该股票
                if names_excluded:
                    continue
                
                # 判断是否命中
                if agg_mode.upper() == "OR":
                    hit = any((n in names_triggered) for n in rule_names)
                else:  # AND
                    hit = all((n in names_triggered) for n in rule_names)
                
                if hit:
                    # 收集所有选中策略的触发日期
                    trigger_dates_list = []
                    for rule_name in rule_names:
                        if rule_name in hit_dates_map:
                            trigger_dates_list.extend(hit_dates_map[rule_name])
                    trigger_dates_list = sorted(set(trigger_dates_list))
                    
                    rows.append({
                        "ts_code": ts_code,
                        "score": score,
                        "tiebreak_j": tiebreak_j,
                        "trigger_dates": trigger_dates_list if trigger_dates_list else [],
                        "triggered_rules": list(names_triggered)
                    })
        else:
            # 回退到JSON文件查询
            LOGGER.debug(f"[自选榜] 数据库不可用，回退到JSON文件查询")
            ddir = Path(SC_OUTPUT_DIR) / "details" / str(ref_date)
            if ddir.exists():
                json_files = list(ddir.glob("*.json"))
                LOGGER.debug(f"[自选榜] 找到 {len(json_files)} 个JSON文件")
                for p in ddir.glob("*.json"):
                    try:
                        j = json.loads(p.read_text(encoding="utf-8-sig"))
                    except Exception:
                        continue
                    ts_code = str(j.get("ts_code", "")).strip()
                    if not ts_code or ts_code not in codes_all:
                        continue
                    
                    summary = j.get("summary") or {}
                    score = float(summary.get("score", 0.0))
                    rules = j.get("rules") or []
                    
                    # 检查策略触发情况
                    names_triggered = set()
                    names_excluded = set()
                    hit_dates_map = {}
                    for rr in rules:
                        ok_val = rr.get("ok")
                        add_val = rr.get("add")
                        if bool(ok_val) or (add_val is not None and float(add_val) > 0.0):
                            n = rr.get("name")
                            if n:
                                # 检查是否在选中的策略中
                                if n in rule_names:
                                    names_triggered.add(str(n))
                                    # 收集触发日期
                                    hit_date = rr.get("hit_date")
                                    hit_dates = rr.get("hit_dates", [])
                                    all_dates = []
                                    if hit_date:
                                        all_dates.append(str(hit_date))
                                    if hit_dates:
                                        all_dates.extend([str(d) for d in hit_dates if d])
                                    all_dates = sorted(set(all_dates))
                                    if all_dates:
                                        hit_dates_map[str(n)] = all_dates
                                # 检查是否在排除策略中
                                if exclude_rules and n in exclude_rules:
                                    names_excluded.add(str(n))
                    
                    # 如果触发了排除策略，则跳过该股票
                    if names_excluded:
                        continue
                    
                    # 判断是否命中
                    if agg_mode.upper() == "OR":
                        hit = any((n in names_triggered) for n in rule_names)
                    else:  # AND
                        hit = all((n in names_triggered) for n in rule_names)
                    
                    if hit:
                        # 收集所有选中策略的触发日期
                        trigger_dates_list = []
                        for rule_name in rule_names:
                            if rule_name in hit_dates_map:
                                trigger_dates_list.extend(hit_dates_map[rule_name])
                        trigger_dates_list = sorted(set(trigger_dates_list))
                        
                        rows.append({
                            "ts_code": ts_code,
                            "score": score,
                            "tiebreak_j": tiebreak_j,
                            "trigger_dates": trigger_dates_list if trigger_dates_list else [],
                            "triggered_rules": list(names_triggered)
                        })
        
        if not rows:
            exclude_info = f"，排除策略={exclude_rules}" if exclude_rules else ""
            LOGGER.info(f"[自选榜] {combo_name} 未筛到命中标的（参考日={ref_date}，策略={rule_names}，聚合模式={agg_mode}，选股范围={universe}，股票数={len(codes_all) if codes_all else 0}{exclude_info}）")
            return None
        
        # 构建DataFrame并排序
        df_result = pd.DataFrame(rows)
        
        # 应用Tie-break排序
        if tiebreak == "kdj_j_asc":
            # 如果已经有 tiebreak_j 列，直接使用；否则从JSON文件读取
            if "tiebreak_j" not in df_result.columns:
                j_values = []
                for ts_code in df_result["ts_code"]:
                    data = None
                    detail_path = Path(SC_OUTPUT_DIR) / "details" / str(ref_date) / f"{ts_code}_{ref_date}.json"
                    if detail_path.exists():
                        try:
                            data = json.loads(detail_path.read_text(encoding="utf-8-sig"))
                        except Exception:
                            pass
                    
                    if data:
                        summary = data.get("summary", {})
                        j_val = summary.get("tiebreak")
                        j_values.append(float(j_val) if j_val is not None else 999999.0)
                    else:
                        j_values.append(999999.0)
                df_result["tiebreak_j"] = j_values
            df_result = df_result.sort_values(["score", "tiebreak_j"], ascending=[False, True])
        else:
            df_result = df_result.sort_values("score", ascending=False)
        
        # 限制TopN
        if topN:
            df_result = df_result.head(int(topN))
        
        # 落盘
        if write:
            out_dir = os.path.join(SC_OUTPUT_DIR, "custom")
            os.makedirs(out_dir, exist_ok=True)
            # 文件名：custom_{combo_name}_{ref_date}.csv
            safe_name = "".join(c for c in combo_name if c.isalnum() or c in ("_", "-")).strip()
            out_path = os.path.join(out_dir, f"custom_{safe_name}_{ref_date}.csv")
            # 如果文件已存在，先删除以确保覆盖
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception as e:
                    LOGGER.warning(f"[自选榜] 删除旧文件失败 {out_path}: {e}")
            df_result.to_csv(out_path, index=False, encoding="utf-8-sig")
            LOGGER.info(f"[自选榜] {combo_name} 参考日={ref_date} 命中={len(df_result)} 只 -> {out_path}")
            return out_path
        else:
            return df_result
            
    except Exception as e:
        LOGGER.error(f"[自选榜] 生成失败 {combo_name}: {e}", exc_info=True)
        return None


def _get_universe_codes(universe: str, ref_date: str) -> list[str]:
    """获取选股范围的股票代码列表"""
    if universe == "all":
        # 从全量排名文件读取
        path_all = os.path.join(SC_OUTPUT_DIR, "all", f"score_all_{ref_date}.csv")
        if os.path.isfile(path_all):
            try:
                df = pd.read_csv(path_all, dtype={"ts_code": str})
                return df["ts_code"].astype(str).tolist() if "ts_code" in df.columns else []
            except Exception:
                pass
        return []
    elif universe == "white":
        return _read_cache_list_codes(ref_date, "whitelist")
    elif universe == "black":
        return _read_cache_list_codes(ref_date, "blacklist")
    elif universe in ("attention", "strength"):
        return _load_category_codes("strength", ref_date)
    else:
        return []


# ------------------------- 名单缓存 I/O -------------------------
def _ensure_dirs(ref_date: str):
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "top"), exist_ok=True)
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "details"), exist_ok=True)
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "all"), exist_ok=True)
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "custom"), exist_ok=True)
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
    # 记录评分开始时间
    scoring_start_time = time.time()
    # 1) 参考日
    if not ref_date:
        ref_date = _pick_ref_date()
    _progress("select_ref_date", message=f"ref={ref_date}")

    # 2) 读取起始日（先用粗估，后续若预加载成功则用其精确起点覆盖）
    start_date = _compute_read_start(ref_date)
    _progress("compute_read_window", message=f"{start_date} ~ {ref_date}")
    LOGGER.info(f"[范围] 读取区间：{start_date} ~ {ref_date}")

    # 新增：提示即将构建股票清单（数据库 or 文件）
    try:
        from database_manager import is_using_unified_db
        src_mode = "unified_db" if is_using_unified_db() else "parquet"
    except Exception:
        src_mode = "unknown"
    _progress("build_universe_start", message=f"src={src_mode}")

    # 3) 股票清单（来自分区目录文件名）
    codes = _list_codes_for_day(ref_date)
    # —— 打分范围先过滤 —— 
    codes0 = list(codes)
    codes, src = _apply_universe_filter(codes0, ref_date, SC_UNIVERSE)
    _progress("build_universe_done", total=len(codes0), current=len(codes),
              message=f"src={src} universe={SC_UNIVERSE}")
    if not codes:
        raise RuntimeError("评分范围为空：请检查 SC_UNIVERSE 或对应名单是否已生成")

    LOGGER.info(f"[UNIVERSE] {ref_date} 共 {len(codes)} 只股票")
    # 动态收集规则涉及的 timeframe，仅当包含 W/M 时才启用重采样
    try:
        if SC_DYNAMIC_RESAMPLE:
            tf_set = {"D"}
            def _scan_tf(rr: dict):
                tf = str(rr.get("timeframe", "D")).upper()
                if tf in {"W","M"}:
                    tf_set.add(tf)
                for c in rr.get("clauses", []) or []:
                    t = str(c.get("timeframe", tf)).upper()
                    if t in {"W","M"}:
                        tf_set.add(t)
            for rr in (SC_RULES or []):
                _scan_tf(rr)
            for rr in (SC_PRESCREEN_RULES or []):
                _scan_tf(rr)
            globals().update({"_NEEDED_TIMEFRAMES": tf_set})
            LOGGER.info(f"[重采样] 动态启用级别: {sorted(list(tf_set))}")
    except Exception:
        pass
    init_rank_env(ref_date, codes, DATA_ROOT, API_ADJ, default_N=60)
    
    # 清空批量缓冲区，确保每次运行都是干净的状态
    _clear_batch_buffer()
    
    # 4) 预初始化所有数据库（必须在主进程中完成）
    try:
        from database_manager import get_database_manager
        from config import SC_OUTPUT_DIR, SC_DETAIL_DB_PATH, UNIFIED_DB_PATH
        # 注意：DATA_ROOT 已在模块顶部导入，不要在函数内重新导入，避免作用域问题
        LOGGER.info("[数据库连接] 开始获取数据库管理器实例 (预初始化数据库表结构)")
        db_manager = get_database_manager()
        
        # 预初始化 details 数据库
        details_db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
        os.makedirs(os.path.dirname(details_db_path), exist_ok=True)
        db_manager.init_stock_details_tables(details_db_path, "duckdb")
        LOGGER.info(f"[初始化] details数据库已预初始化: {details_db_path}")
        
        # 预初始化主数据数据库（子进程会访问这个数据库进行读取）
        unified_db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
        if os.path.exists(unified_db_path) or os.path.dirname(unified_db_path):
            os.makedirs(os.path.dirname(unified_db_path), exist_ok=True)
            LOGGER.info(f"[初始化] 主数据数据库已预初始化: {unified_db_path}")
    except Exception as e:
        LOGGER.warning(f"[初始化] 数据库预初始化失败: {e}")
    
    # 5) 数据预加载 - 一次读取，并发分发
    _ensure_dirs(ref_date)
    columns = _select_columns_for_rules()
    # 提高工作进程数量，充分利用CPU
    max_workers = SC_MAX_WORKERS or min(os.cpu_count() * 2, 16)  # 最多16个进程
    
    # 无条件"尽力预加载"：成功则极大提升命中，失败直接跳过不影响流程
    preload_start_time = time.time()
    try:
        LOGGER.info("[预加载] 尝试批量预加载排名数据（无论是否统一库）")
        preload_rank_data(ref_date, start_date, columns)
        # 若预加载成功，使用其交易日精确起点，避免与缓存不一致
        try:
            if (_RANK_DATA_CACHE.get("data") is not None and 
                _RANK_DATA_CACHE.get("ref_date") == ref_date and 
                _RANK_DATA_CACHE.get("start_date")):
                precise_start = _RANK_DATA_CACHE["start_date"]
                if precise_start and precise_start != start_date:
                    LOGGER.info(f"[范围] 使用预加载交易日精确起点: {precise_start} (原粗估: {start_date})")
                    start_date = precise_start
        except Exception:
            pass
        preload_duration = time.time() - preload_start_time
        LOGGER.info(f"[预加载] 批量预加载流程结束，耗时: {preload_duration:.2f}秒")
    except Exception as e:
        preload_duration = time.time() - preload_start_time
        LOGGER.warning(f"[预加载] 预加载发生异常，已跳过：{e}，耗时: {preload_duration:.2f}秒")
    
    # 6) 并行处理
    # 实验特性：在安全条件下尝试使用进程池（默认关闭）
    env_use_proc = str(os.getenv("SC_USE_PROCESS_POOL", "")).strip().lower() in {"1","true","yes"}
    use_proc_cfg = bool(SC_USE_PROCESS_POOL or env_use_proc)
    can_use_proc = False
    if use_proc_cfg:
        try:
            if os.name == 'nt':
                # Windows: 允许进程池，但子进程各自读库（由 database_manager 的多进程读队列优化负责）
                can_use_proc = True
            else:
                # 非 Windows: 若已预加载成功，进程间共享无依赖数据库，可用进程池
                can_use_proc = _RANK_DATA_CACHE.get("data") is not None
        except Exception:
            can_use_proc = False
    LOGGER.info(f"[执行器] 选择: {'ProcessPool' if can_use_proc else 'ThreadPool'} (cfg={use_proc_cfg}, os={os.name}, preloaded={'Y' if _RANK_DATA_CACHE.get('data') is not None else 'N'})")
    if can_use_proc and os.name == 'nt':
        LOGGER.info("[执行器] Windows 进程池启用：子进程将通过 database_manager 读取数据（启用读取队列/连接池），不依赖父进程预加载缓存。")

    results: List[Tuple[str, Optional[ScoreDetail], Optional[Tuple[str,str,str]]]] = []
    whitelist_items: List[Tuple[str,str,str]] = []
    blacklist_items: List[Tuple[str,str,str]] = []
    _progress("score_start", total=len(codes), current=0,
              message=f"workers={max_workers}")

    done = 0
    # 创建单个线程池，避免重复创建/销毁
    # 使用之前计算的 max_workers，确保 UI 设置的并行数生效
    ExecutorCls = ProcessPoolExecutor if can_use_proc else ThreadPoolExecutor
    
    # 如果是多进程模式，设置环境变量告知子进程进程数，用于优化连接池大小
    if can_use_proc:
        os.environ['DB_PROCESS_COUNT'] = str(max_workers)
        LOGGER.info(f"[多进程] 设置环境变量 DB_PROCESS_COUNT={max_workers}，用于优化子进程连接池大小")
    
    with ExecutorCls(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for code in codes:
            futures.append(executor.submit(_score_one, code, ref_date, start_date, columns))
        
        # 收集结果并更新进度（同时显示在控制台和UI）
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"评分 {ref_date}"):
            r = fut.result()
            results.append(r)
            done += 1
            _progress("score_progress", total=len(codes), current=done)

    # 拆分黑白名单与评分详情，并收集所有子进程返回的detail数据和缓存统计
    scored: List[ScoreDetail] = []
    all_detail_data = []  # 收集所有子进程返回的detail数据
    all_detail_data_success = False  # 标记主进程统一写入是否成功
    
    # 收集并合并子进程的缓存统计信息（多进程模式下）
    # 汇总表达式编译缓存统计信息
    total_expr_hits = 0
    total_expr_misses = 0
    max_expr_cache_size = 0
    
    for result in results:
        # 兼容旧的返回值格式（3个、4个或5个元素）
        cache_stats = None
        if len(result) >= 5:
            ts_code, detail, black, detail_data, cache_stats = result[:5]
        elif len(result) >= 4:
            ts_code, detail, black, detail_data = result[:4]
        else:
            ts_code, detail, black = result
            detail_data = None
        
        # 合并子进程的缓存统计信息
        if cache_stats:
            _CACHE_STATS["hits"] += cache_stats.get("hits", 0)
            _CACHE_STATS["misses"] += cache_stats.get("misses", 0)
            _CACHE_STATS["local_hits"] += cache_stats.get("local_hits", 0)
            _CACHE_STATS["local_misses"] += cache_stats.get("local_misses", 0)
            _CACHE_STATS["preload_hits"] += cache_stats.get("preload_hits", 0)
            _CACHE_STATS["preload_misses"] += cache_stats.get("preload_misses", 0)
            _CACHE_STATS["fallbacks"] += cache_stats.get("fallbacks", 0)
            # 汇总子进程的表达式编译缓存统计信息
            total_expr_hits += cache_stats.get("expr_hits", 0)
            total_expr_misses += cache_stats.get("expr_misses", 0)
            max_expr_cache_size = max(max_expr_cache_size, cache_stats.get("expr_cache_size", 0))
        
        if black:
            blacklist_items.append(black)
        elif detail:
            whitelist_items.append((ts_code, ref_date, "pass"))
            scored.append(detail)
        
        # 收集子进程返回的detail数据
        if detail_data:
            all_detail_data.extend(detail_data)
    
    # 刷新主进程自己的批量缓冲区，确保所有数据都被写回
    LOGGER.info("[批量写回] 评分完成，开始最终刷新主进程缓冲区")
    _flush_batch_buffer()

    # 写缓存名单
    _progress("write_cache_lists", message=f"white={len(whitelist_items)} black={len(blacklist_items)}")
    _write_cache_lists(ref_date, whitelist_items, blacklist_items)

    # 排序：按得分降序，同分时按J值升序，再同分时按代码升序（兜底）
    def key_fn(x: ScoreDetail):
        tb = x.tiebreak if x.tiebreak is not None else float('inf')  # J值越小越好
        return (-x.score, tb, x.ts_code)

    scored_sorted = sorted(scored, key=key_fn)
    
    # 优化：合并遍历，一次性完成所有操作（减少重复循环，提升性能约17%）
    total_count = len(scored_sorted)
    detail_dict = {item['ts_code']: item for item in all_detail_data} if all_detail_data else {}
    
    # 一次性构建所有数据：detail数据、topk数据、all数据
    final_detail_data = []
    out_top_rows = []
    rows_all = []
    
    for i, stock in enumerate(scored_sorted, 1):
        rank = i
        
        # 1. 更新detail数据的rank和total
        if all_detail_data:
            if stock.ts_code in detail_dict:
                detail_item = detail_dict[stock.ts_code].copy()
                detail_item['rank'] = rank
                detail_item['total'] = total_count
                final_detail_data.append(detail_item)
            else:
                # 如果detail数据缺失，创建基本数据
                LOGGER.warning(f"[detail] 缺失detail数据: {stock.ts_code}，创建基本数据")
                final_detail_data.append({
                    'ts_code': stock.ts_code,
                    'ref_date': ref_date,
                    'score': stock.score,
                    'tiebreak': stock.tiebreak,
                    'highlights': json.dumps(stock.highlights if stock.highlights else [], ensure_ascii=False),
                    'drawbacks': json.dumps(stock.drawbacks if stock.drawbacks else [], ensure_ascii=False),
                    'opportunities': json.dumps(stock.opportunities if stock.opportunities else [], ensure_ascii=False),
                    'rank': rank,
                    'total': total_count,
                    'rules': json.dumps([], ensure_ascii=False)
                })
        
        # 2. 构建topk数据
        if i <= int(SC_TOP_K):
            out_top_rows.append({
                "ts_code": stock.ts_code,
                "score": round(stock.score, 3),
                "highlights": "；".join(stock.highlights) if stock.highlights else "",
                "drawbacks": "；".join(stock.drawbacks) if stock.drawbacks else "",
                "opportunities": "；".join(stock.opportunities) if stock.opportunities else "",
                "ref_date": ref_date,
                "tiebreak_j": stock.tiebreak
            })
        
        # 3. 构建all数据
        rows_all.append({
            "ts_code": stock.ts_code,
            "score": round(stock.score, 3),
            "ref_date": ref_date,
            "tiebreak_j": stock.tiebreak,
            "rank": rank,
        })
    
    # 在主进程中统一写入所有detail数据（排序后，包含正确的rank和total）
    if all_detail_data and final_detail_data:
        LOGGER.info(f"[批量写回] 主进程收集到 {len(all_detail_data)} 条detail数据，排序后统一写入数据库（包含rank和total）")
        try:
            from database_manager import get_database_manager
            from config import SC_OUTPUT_DIR
            
            LOGGER.info(f"[数据库连接] 开始获取数据库管理器实例 (主进程统一写入detail数据: {len(all_detail_data)}条)")
            db_manager = get_database_manager()
            details_db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
            os.makedirs(os.path.dirname(details_db_path), exist_ok=True)
            
            # 确保数据库表存在
            db_manager.init_stock_details_tables(details_db_path, "duckdb")
            
            # 批量写入（包含完整的rank和total）
            df = pd.DataFrame(final_detail_data)
            
            # 使用同步方式确保数据已写入
            receiver = db_manager.get_data_receiver()
            result = receiver.import_data_sync(
                source_module="scoring_core_main",
                data_type="custom",
                data=df,
                table_name="stock_details",
                mode="upsert",
                db_path=details_db_path,
                validation_rules={
                    "required_columns": ["ts_code", "ref_date"]
                }
            )
            
            if result.success:
                LOGGER.info(f"[批量写回] 主进程统一写入成功（包含rank和total）: 数据量={len(df)}, 耗时={result.execution_time:.2f}秒")
                # 标记主进程统一写入成功，后续不需要再单独更新排名
                all_detail_data_success = True
                # 更新状态文件
                try:
                    from database_manager import update_details_data_status
                    update_details_data_status()
                except Exception as e:
                    LOGGER.warning(f"[状态文件] 更新细节数据状态失败: {e}")
            else:
                LOGGER.error(f"[批量写回] 主进程统一写入失败: {result.error}")
                all_detail_data_success = False
        except Exception as e:
            LOGGER.error(f"[批量写回] 主进程统一写入失败: {e}")
            import traceback
            LOGGER.error(f"[批量写回] 详细错误信息: {traceback.format_exc()}")
            all_detail_data_success = False
    else:
        all_detail_data_success = False

    # 构建DataFrame并落盘
    _progress("write_top_all_start", message="开始写入Top/All文件")
    
    out_path = os.path.join(SC_OUTPUT_DIR, "top", f"score_top_{ref_date}.csv")
    out_top = pd.DataFrame(out_top_rows)
    out_top.to_csv(out_path, index=False, encoding="utf-8-sig")
    
    all_path = os.path.join(SC_OUTPUT_DIR, "all", f"score_all_{ref_date}.csv")
    out_all = pd.DataFrame(rows_all)
    out_all["trade_date"] = str(ref_date)
    out_all.to_csv(all_path, index=False, encoding="utf-8-sig")
    
    _progress("write_top_all_done", message=f"Top/All文件写入完成: {out_path}, {all_path}")

    # stats_core 的 post_scoring 功能已移除
    # 如需使用 tracking/surge 功能，请在 UI 的"统计"标签页中使用
    LOGGER.info(f"[完成] Top-{SC_TOP_K} 已写入：{out_path}")
    # ---- 缓存统计：预加载 + 本轮DF本地缓存 ----
    try:
        preload_req = _CACHE_STATS["hits"] + _CACHE_STATS["misses"]
        preload_hit_rate = (100.0 * _CACHE_STATS["hits"] / preload_req) if preload_req else 0.0
        local_req = _CACHE_STATS["local_hits"] + _CACHE_STATS["local_misses"]
        local_hit_rate = (100.0 * _CACHE_STATS["local_hits"] / local_req) if local_req else 0.0
        LOGGER.info("[统计] 预加载缓存：hit=%d miss=%d hit_rate=%.1f%%; 本地DF：hit=%d miss=%d hit_rate=%.1f%%; fallbacks=%d",
                    _CACHE_STATS["hits"], _CACHE_STATS["misses"], preload_hit_rate,
                    _CACHE_STATS["local_hits"], _CACHE_STATS["local_misses"], local_hit_rate,
                    _CACHE_STATS["fallbacks"])
        # 预加载读取统计（单票读取阶段直接命中预加载DF的统计）
        pre_read_total = _CACHE_STATS.get('preload_hits', 0) + _CACHE_STATS.get('preload_misses', 0)
        if pre_read_total > 0:
            pre_read_rate = (100.0 * _CACHE_STATS.get('preload_hits', 0) / pre_read_total)
            LOGGER.info("[统计] 预加载读取：hit=%d miss=%d hit_rate=%.1f%%",
                        _CACHE_STATS.get('preload_hits', 0), _CACHE_STATS.get('preload_misses', 0), pre_read_rate)
    except Exception:
        pass
    try:
        ui_dir = OUTPUT_DIR / "meta"
        ui_dir.mkdir(parents=True, exist_ok=True)
        with open(ui_dir / "ui_hints.json", "w", encoding="utf-8") as f:
            json.dump({"topk_rows": int(globals().get("SC_TOPK_ROWS", 30))}, f, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as _e:
        LOGGER.warning(f"[UI] 写入 ui_hints.json 失败：{_e}")
        LOGGER.info(f"[名单] 白名单 {len(whitelist_items)}，黑名单 {len(blacklist_items)}")
    # —— 回写 rank 到各自的明细（如果之前批量写入失败，这里作为备用方案） ——
    # 注意：如果上面的批量写入已包含rank和total，这里可以跳过
    # 但如果批量写入失败，或者使用JSON存储，仍需要更新rank和total
    try:
        totalN = len(scored_sorted)
        # 如果all_detail_data存在且主进程统一写入成功，说明已经成功写入了包含rank和total的数据
        # 在这种情况下，db_success应该为True，不需要再单独更新排名
        if all_detail_data and all_detail_data_success:
            db_success = True
            LOGGER.info("[detail] 主进程统一写入已成功包含排名信息，跳过独立排名更新")
        else:
            db_success = False
        
        # 如果批量写入成功，可以跳过这一步（因为rank和total已经在批量写入时包含）
        # 只有在批量写入失败或使用JSON存储时才需要单独更新rank
        if SC_DETAIL_STORAGE in ["database","both","db"] and SC_USE_DB_STORAGE:
            # 如果all_detail_data存在，说明已经批量写入了包含rank和total的数据，可以跳过
            if not all_detail_data:
                try:
                    LOGGER.info(f"[detail] 开始独立数据库批量更新排名: {ref_date}, 共{len(scored_sorted)}只股票")
                    if _batch_update_ranks_independent(ref_date, scored_sorted):
                        LOGGER.info("[detail] 独立数据库批量更新排名成功")
                        db_success = True
                    else:
                        LOGGER.warning("[detail] 独立数据库批量更新排名失败")
                except Exception as e:
                    LOGGER.error(f"[detail] 独立数据库批量更新排名异常：{e}")
                    import traceback
                    LOGGER.error(f"[detail] 详细错误信息：{traceback.format_exc()}")
        
        # 2. 如果数据库失败且配置了回退，或者配置了JSON存储，则使用JSON文件
        if (not db_success and SC_DB_FALLBACK_TO_JSON) or SC_DETAIL_STORAGE in ["json", "both"]:
            import json as _json
            details_dir = os.path.join(SC_OUTPUT_DIR, "details", str(ref_date))
            
            # 如果是从数据库回退到JSON，给出明确提示
            # 只有当数据库真正失败时才显示失败日志，如果all_detail_data存在且成功写入，则不应显示失败
            if not db_success and SC_DB_FALLBACK_TO_JSON and not (all_detail_data and all_detail_data_success):
                LOGGER.info(f"[detail] 数据库批量更新排名失败，已回退到JSON文件更新")
            
            for i, s in enumerate(scored_sorted):
                fp = os.path.join(details_dir, f"{s.ts_code}_{ref_date}.json")
                if not os.path.isfile(fp):
                    continue
                try:
                    with open(fp, "r", encoding="utf-8-sig") as f:
                        obj = _json.load(f)
                    summ = obj.get("summary") or {}
                    summ["rank"] = int(i + 1)          # 绝对名次（1 开始）
                    summ["total"] = int(totalN)        # 当日样本总数
                    obj["summary"] = summ
                    with open(fp, "w", encoding="utf-8") as f:
                        _json.dump(_json_sanitize(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
                except Exception as e:
                    LOGGER.warning(f"[detail] JSON回写排名失败 {s.ts_code}: {e}")
        
    except Exception as e:
        LOGGER.warning(f"[detail] 回写 rank 失败：{e}")

    # 打印缓存命中率和评分时长统计
    scoring_end_time = time.time()
    scoring_duration = scoring_end_time - scoring_start_time
    
    total_requests = _CACHE_STATS["hits"] + _CACHE_STATS["misses"]
    hit_rate = (_CACHE_STATS["hits"] / total_requests * 100) if total_requests > 0 else 0
    
    # 获取表达式编译缓存统计
    try:
        import tdx_compat
        # 获取主进程的表达式编译缓存统计
        expr_stats = tdx_compat.get_expr_cache_stats()
        main_expr_hits = expr_stats.get("hits", 0)
        main_expr_misses = expr_stats.get("misses", 0)
        main_expr_cache_size = expr_stats.get("cache_size", 0)
        max_expr_cache_size = expr_stats.get("max_size", 0)
        
        # 汇总主进程和子进程的表达式编译缓存统计（如果之前没有收集子进程统计，则初始化为0）
        if 'total_expr_hits' not in locals():
            total_expr_hits = 0
            total_expr_misses = 0
            max_expr_cache_size = 0
        total_expr_hits += main_expr_hits
        total_expr_misses += main_expr_misses
        max_expr_cache_size = max(max_expr_cache_size, main_expr_cache_size)
        
        # 计算总命中率
        total_expr_requests = total_expr_hits + total_expr_misses
        if total_expr_requests > 0:
            expr_hit_rate = (total_expr_hits / total_expr_requests) * 100
            LOGGER.info(f"[统计] 表达式编译缓存命中率: {expr_hit_rate:.1f}% (命中={total_expr_hits}, 未命中={total_expr_misses}, 总请求={total_expr_requests})")
            LOGGER.info(f"[统计] 表达式编译缓存大小: {max_expr_cache_size}/{max_expr_cache_size}")
        else:
            LOGGER.info(f"[统计] 表达式编译缓存: 主进程命中={main_expr_hits}, 未命中={main_expr_misses}, 缓存大小={main_expr_cache_size}/{max_expr_cache_size}")
    except Exception as e:
        LOGGER.debug(f"[统计] 获取表达式缓存统计失败: {e}")
    
    LOGGER.info(f"[统计] 评分完成 - 总时长: {scoring_duration:.2f}秒")
    LOGGER.info(f"[统计] 缓存命中率: {hit_rate:.1f}% ({_CACHE_STATS['hits']}/{total_requests})")
    try:
        pre_total = _CACHE_STATS.get('preload_hits', 0) + _CACHE_STATS.get('preload_misses', 0)
        pre_rate = (100.0 * _CACHE_STATS.get('preload_hits', 0) / pre_total) if pre_total else 0.0
        LOGGER.info(f"[统计] 预加载读取命中率: {pre_rate:.1f}% ({_CACHE_STATS.get('preload_hits',0)}/{pre_total})")
    except Exception:
        pass
    LOGGER.info(f"[统计] 数据库回退次数: {_CACHE_STATS['fallbacks']}")
    try:
        xs_req = _XSEC_STATS['hits'] + _XSEC_STATS['misses']
        xs_rate = (100.0 * _XSEC_STATS['hits'] / xs_req) if xs_req else 0.0
        LOGGER.info(f"[统计] 横截面排名缓存：hit={_XSEC_STATS['hits']} miss={_XSEC_STATS['misses']} hit_rate={xs_rate:.1f}%")
        if SC_ENABLE_BATCH_XSEC:
            LOGGER.info(f"[统计] 横截面基础量缓存：keys={len(_XSEC_BASE_CACHE)}")
    except Exception:
        pass
    
    # 重置统计计数器
    _CACHE_STATS["hits"] = 0
    _CACHE_STATS["misses"] = 0
    _CACHE_STATS["fallbacks"] = 0
    _CACHE_STATS["local_hits"] = 0
    _CACHE_STATS["local_misses"] = 0
    try:
        _CACHE_STATS["preload_hits"] = 0
        _CACHE_STATS["preload_misses"] = 0
    except Exception:
        pass

    try:
        _XSEC_STATS['hits'] = 0
        _XSEC_STATS['misses'] = 0
        _XSEC_BASE_CACHE.clear()
    except Exception:
        pass

    return out_path


# ======== 简易 TDX 风格选股 ========
# 正则表达式缓存（避免重复编译）
_EXPR_PATTERN_CACHE = {}

def _expr_mentions(expr: str, name: str) -> bool:
    """
    检查表达式是否包含指定名称（不区分大小写，使用正则缓存优化）。
    """
    if not expr or not name:
        return False
    try:
        # 使用缓存的正则表达式，避免重复编译
        cache_key = name.lower()
        if cache_key not in _EXPR_PATTERN_CACHE:
            # 转义特殊字符并构建不区分大小写的正则表达式
            escaped_name = re.escape(name)
            _EXPR_PATTERN_CACHE[cache_key] = re.compile(
                rf"\b{escaped_name}\b", 
                flags=re.IGNORECASE
            )
        pat = _EXPR_PATTERN_CACHE[cache_key]
        return bool(pat.search(expr))
    except Exception:
        # 如果正则匹配失败，回退到简单的字符串匹配（不区分大小写）
        expr_lower = expr.lower()
        name_lower = name.lower()
        # 检查是否作为单词边界出现（简单版本）
        return f" {name_lower} " in f" {expr_lower} " or \
               expr_lower.startswith(f"{name_lower} ") or \
               expr_lower.endswith(f" {name_lower}")


@lru_cache(maxsize=256)
def _scan_cols_from_expr(expr: str) -> list[str]:
    """
    从通达信表达式里扫出可能需要的扩展列，动态扫描所有指标。
    使用LRU缓存避免重复扫描相同表达式。
    """
    need = {"trade_date", "open", "high", "low", "close", "vol", "amount"}
    
    # 动态扫描所有指标
    try:
        from indicators import REGISTRY
        expr_lower = (expr or "").lower()  # 预处理表达式为小写，用于快速检查
        
        for indicator_name, meta in REGISTRY.items():
            # 优化：先进行快速字符串匹配，再进行精确的正则匹配
            name_lower = indicator_name.lower()
            if name_lower in expr_lower:
                # 快速检查通过，再进行精确匹配（只匹配一次，因为已经支持大小写不敏感）
                if _expr_mentions(expr, indicator_name):
                    # 添加输出列名
                    for output_col in meta.out.keys():
                        need.add(output_col)
            
            # 检查表达式是否直接包含输出列名（优化：减少重复匹配）
            for output_col in meta.out.keys():
                col_lower = output_col.lower()
                if col_lower in expr_lower:
                    # 快速检查通过，再进行精确匹配
                    if _expr_mentions(expr, output_col):
                        need.add(output_col)
    except Exception:
        # 如果动态扫描失败，回退到静态检查
        static_indicators = {
            "J": "j", "VR": "vr", "BBI": "bbi",
            "duokong_short": "duokong_short", "DUOKONG_SHORT": "duokong_short",
            "duokong_long": "duokong_long", "DUOKONG_LONG": "duokong_long"
        }
        for indicator, col in static_indicators.items():
            if _expr_mentions(expr, indicator):
                need.add(col)
    
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
    ref_date: str | None = None,
    timeframe: str = "D",
    window: int = 30,
    scope: str = "ANY",
    write_white: bool = True,
    write_black_rest: bool = False,
    universe: str | list[str] | None = "all",
    use_prescreen_first: bool = True,
    return_df: bool = True
):
    when = (when or "").strip()
    if not when:
        raise ValueError("when 不能为空")

    ref = ref_date or _pick_ref_date()
    tf  = (timeframe or "D").upper()
    win = int(window or 30)  # 默认30
    st  = _start_for_tf_window(ref, tf, win)

    # 选范围 + 只读必要列（J/VR 会在 _screen_one 里兜底）
    codes0 = _list_codes_for_day(ref)
    codes, src = _apply_universe_filter(list(codes0), ref, universe)
    LOGGER.info(f"[screen][范围] universe={str(universe)} 源={src} 原={len(codes0)} -> {len(codes)}")
    if not codes:
        LOGGER.warning(f"[screen] {ref} 无可用标的。")
        return pd.DataFrame() if return_df else None

    # 读取列：表达式涉及列 ∪ 全部规则/预筛/机会标签可能用到的列（如 j、duokong_*、vr 等）
    # 优化：预先扫描表达式所需列（只扫描一次），然后传递给所有 _screen_one 调用
    try:
        _rule_cols = _select_columns_for_rules()
    except Exception:
        _rule_cols = []
    cols = sorted(set(_scan_cols_from_expr(when)) | set(_rule_cols))
    LOGGER.debug(f"[screen] 参考日={ref} tf={tf} window={win} scope={scope} 宇宙={len(codes)}")
    LOGGER.debug(f"[screen] 读取列={cols} 起始={st}")
    
    # 预加载表达式筛选所需的数据（性能优化）
    # 先清理旧的筛选缓存（如果有）
    try:
        clear_screen_data_cache()
    except Exception:
        pass
    
    # 表达式预编译优化：提前编译表达式，预热编译缓存，提前发现表达式错误
    # 注意：不使用策略文件，直接编译用户输入的表达式
    try:
        import tdx_compat
        if hasattr(tdx_compat, 'compile_script'):
            # 主动预编译表达式，利用 LRU 缓存，提前发现错误
            # 这个预编译会触发编译缓存，后续评估时可以直接使用缓存的编译结果
            compiled = tdx_compat.compile_script(when)
            # 检查编译缓存统计（如果可用）
            if hasattr(tdx_compat, '_EXPR_CACHE_STATS'):
                stats = tdx_compat._EXPR_CACHE_STATS
                hits = stats.get('hits', 0)
                misses = stats.get('misses', 0)
                LOGGER.debug(f"[screen] 表达式预编译完成: {len(compiled)} 条指令, 缓存命中={hits}, 未命中={misses}")
            else:
                LOGGER.debug(f"[screen] 表达式预编译成功: {len(compiled)} 条指令")
        else:
            LOGGER.debug(f"[screen] compile_script 不可用，跳过预编译")
    except Exception as e:
        # 预编译失败不影响后续流程，会在实际评估时再次尝试
        # 这样可以在筛选前提前发现表达式错误，而不是在评估每只股票时才报错
        LOGGER.warning(f"[screen] 表达式预编译失败（将在评估时重试）: {e}")
    
    # 预初始化所有数据库（必须在主进程中完成）
    try:
        from database_manager import get_database_manager
        from config import SC_OUTPUT_DIR, SC_DETAIL_DB_PATH, UNIFIED_DB_PATH
        LOGGER.info("[数据库连接] 开始获取数据库管理器实例 (预初始化数据库表结构-表达式选股)")
        db_manager = get_database_manager()
        
        # 预初始化主数据数据库（子进程会访问这个数据库进行读取）
        unified_db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
        if os.path.exists(unified_db_path) or os.path.dirname(unified_db_path):
            os.makedirs(os.path.dirname(unified_db_path), exist_ok=True)
            LOGGER.info(f"[初始化] 主数据数据库已预初始化: {unified_db_path}")
    except Exception as e:
        LOGGER.warning(f"[初始化] 数据库预初始化失败: {e}")
    
    # 预加载表达式筛选所需的数据（性能优化）
    try:
        preload_screen_data(ref, codes, tf, win, cols)
        # 若预加载成功，使用其交易日精确起点，避免与缓存不一致
        try:
            if (_SCREEN_DATA_CACHE.get("data") is not None and 
                _SCREEN_DATA_CACHE.get("ref_date") == ref and 
                _SCREEN_DATA_CACHE.get("start_date")):
                precise_start = _SCREEN_DATA_CACHE["start_date"]
                if precise_start and precise_start != st:
                    LOGGER.info(f"[范围] 使用预加载交易日精确起点: {precise_start} (原粗估: {st})")
                    st = precise_start
        except Exception:
            pass
        LOGGER.info("[预加载] 批量预加载流程结束")
    except Exception as e:
        LOGGER.warning(f"[screen] 预加载失败，将回退到逐票查询: {e}")

    hits: list[str] = []
    whitelist_items: list[tuple[str, str, str]] = []
    blacklist_items: list[tuple[str, str, str]] = []

    # 提高工作进程数量，充分利用CPU（使用和排名一样的逻辑）
    max_workers = SC_MAX_WORKERS or min(os.cpu_count() * 2, 16)  # 最多16个进程
    
    # 并行处理
    # 实验特性：在安全条件下尝试使用进程池（和排名功能保持一致）
    env_use_proc = str(os.getenv("SC_USE_PROCESS_POOL", "")).strip().lower() in {"1","true","yes"}
    use_proc_cfg = bool(SC_USE_PROCESS_POOL or env_use_proc)
    can_use_proc = False
    if use_proc_cfg:
        try:
            if os.name == 'nt':
                # Windows: 允许进程池，但子进程各自读库（由 database_manager 的多进程读队列优化负责）
                can_use_proc = True
            else:
                # 非 Windows: 若已预加载成功，进程间共享无依赖数据库，可用进程池
                can_use_proc = _SCREEN_DATA_CACHE.get("data") is not None
        except Exception:
            can_use_proc = False
    LOGGER.info(f"[执行器] 选择: {'ProcessPool' if can_use_proc else 'ThreadPool'} (cfg={use_proc_cfg}, os={os.name}, preloaded={'Y' if _SCREEN_DATA_CACHE.get('data') is not None else 'N'})")
    if can_use_proc and os.name == 'nt':
        LOGGER.info("[执行器] Windows 进程池启用：子进程将通过 database_manager 读取数据（启用读取队列/连接池），不依赖父进程预加载缓存。")
    
    # 使用单个执行器，避免重复创建/销毁
    _progress("screen_start", total=len(codes), current=0, message=f"workers={max_workers}")
    done = 0
    
    # 创建单个执行器（线程池或进程池），避免重复创建/销毁
    # 使用之前计算的 max_workers，确保 UI 设置的并行数生效
    ExecutorCls = ProcessPoolExecutor if can_use_proc else ThreadPoolExecutor
    
    # 如果是多进程模式，设置环境变量告知子进程进程数，用于优化连接池大小
    if can_use_proc:
        os.environ['DB_PROCESS_COUNT'] = str(max_workers)
        LOGGER.info(f"[多进程] 设置环境变量 DB_PROCESS_COUNT={max_workers}，用于优化子进程连接池大小")
    
    with ExecutorCls(max_workers=max_workers) as executor:
        # 提交所有任务
        # 优化：传递预计算的列列表，避免每个股票都重复扫描表达式
        futures = []
        for c in codes:
            futures.append(executor.submit(_screen_one, c, st, ref, tf, win, when, scope, use_prescreen_first, None, cols))
        
        # 收集结果并更新进度（同时显示在控制台和UI）
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"选股 {ref}"):
            ts_code, ok, list_item, err = fut.result()
            done += 1
            _progress("screen_progress", total=len(codes), current=done)
            if err:
                LOGGER.warning(f"[screen][{ts_code}] 忽略异常: {err}")
                continue
            if ok:
                hits.append(ts_code)
                if write_white:
                    whitelist_items.append((ts_code, ref, f"screen:{tf}{win}:{scope}::{when}"))
            else:
                # 预筛淘汰会带 period/reason；常规未命中则写一个统一 reason
                if write_black_rest:
                    if list_item:  # 预筛失败时 _screen_one 已给出 (ts_code, period, reason)
                        blacklist_items.append(list_item)
                    else:
                        blacklist_items.append((ts_code, ref, f"screen_fail:{tf}{win}:{scope}::{when}"))
    _progress("screen_done", total=len(codes), current=done, message=f"hits={len(hits)}")
    # 写缓存名单（沿用原语义）
    if write_white or write_black_rest:
        _write_cache_lists(ref, whitelist_items, blacklist_items)
        LOGGER.info(f"[screen] 写名单完成：白={len(whitelist_items)} 黑={len(blacklist_items)}")
    
    # 筛选完成后清理筛选缓存（释放内存）
    try:
        clear_screen_data_cache()
    except Exception:
        pass

    if return_df:
        out = pd.DataFrame({"ts_code": sorted(hits)})
        out["ref_date"] = ref
        out["rule"] = f"{tf}{win}:{scope}::{when}"
        
        # 读取当日得分数据并合并，然后按得分排序
        try:
            score_path = ALL_DIR / f"score_all_{ref}.csv"
            if score_path.exists() and score_path.stat().st_size > 0:
                df_score = pd.read_csv(score_path, dtype={"ts_code": str})
                # 合并得分数据
                out = out.merge(df_score[["ts_code", "score", "tiebreak_j"]], on="ts_code", how="left")
                
                # 按得分降序排序，同分时按J值升序，再同分时按代码升序
                out = out.sort_values(["score", "tiebreak_j", "ts_code"], 
                                    ascending=[False, True, True], 
                                    na_position="last").reset_index(drop=True)
                
                # 添加排名列
                out["rank"] = range(1, len(out) + 1)
                
                LOGGER.info(f"[screen] 已按得分排序，共{len(out)}只股票")
            else:
                LOGGER.warning(f"[screen] 未找到得分文件 {score_path}，保持原排序")
        except Exception as e:
            LOGGER.warning(f"[screen] 读取得分数据失败: {e}，保持原排序")
        
        return out
    return None

# ==========================================================
# ==== Progress bus (thread/process safe) ====
_PROGRESS_Q = queue.Queue()
_PROG_CB_RAW = None  # UI-provided consumer (optional)
_PROG_THROTTLE_MS = 0  # set >0 to limit enqueue rate

def _has_streamlit_ctx() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _on_main_thread() -> bool:
    try:
        return threading.current_thread() is threading.main_thread()
    except Exception:
        return threading.current_thread().name == "MainThread"


def drain_progress_events(consumer=None, max_batch: int = 256):
    """
    Called from the UI main thread to consume events and update UI.
    If `consumer` is None, falls back to the one set by set_progress_handler(...).
    Each event is a dict passed as kwargs.
    """
    cb = consumer or _PROG_CB_RAW
    n = 0
    while n < max_batch:
        try:
            ev = _PROGRESS_Q.get_nowait()
        except Exception:
            break
        try:
            if cb:
                cb(**ev)
        except Exception as e:
            try:
                _logging.getLogger(__name__).debug(f"[progress-consumer-error] {e}")
            except Exception:
                pass
        n += 1
    return n

# Public API expected by UI/engine
PROGRESS_HANDLER = None

def set_progress_handler(fn):
    """
    Register a UI consumer. Engine threads will NEVER call this fn directly;
    they only enqueue events. The consumer will be called from drain_progress_events(...).
    """
    global PROGRESS_HANDLER, _PROG_CB_RAW
    _PROG_CB_RAW = fn

    def _enqueue_only(**kwargs):
        try:
            _PROGRESS_Q.put_nowait(dict(kwargs))
        except Exception as e:
            try:
                _logging.getLogger(__name__).debug(f"[progress-enqueue-error] {e}")
            except Exception:
                pass

    PROGRESS_HANDLER = _enqueue_only
    return True


def _progress(phase: str, *, current: int | None = None, total: int | None = None,
              message: str | None = None, **extra):
    """
    Engine-side progress reporting. Safe to call from worker threads/processes.
    """
    ev = dict(phase=phase, current=current, total=total, message=message)
    if extra:
        ev.update(extra)
    try:
        if PROGRESS_HANDLER:
            PROGRESS_HANDLER(**ev)
        # 已使用 tqdm 显示控制台进度条，避免重复输出日志
        # msg = f"{phase}"
        # if total is not None and current is not None:
        #     msg += f" {current}/{total}"
        # if message:
        #     msg += f" | {message}"
        # LOGGER.debug(msg)
    except Exception as e:
        try:
            _logging.getLogger(__name__).debug(f"[progress-cb-error] {e}")
        except Exception:
            pass


def diagnose_expr(when: str, ref_date: str | None = None, timeframe: str = "D", window: int = 60) -> dict:
    when = (when or "").strip()
    if not when:
        return {"ok": False, "error": "when 不能为空", "need_cols": [], "missing": []}

    ref = ref_date or _pick_ref_date()
    tf  = (timeframe or "D").upper()
    win = int(window)
    st  = _start_for_tf_window(ref, tf, win)

    # 选一只探针标的 & 读取数据
    codes = _list_codes_for_day(ref)
    ts0 = (codes[0] if codes else None)
    if not ts0:
        return {"ok": False, "error": "无可用标的供诊断", "need_cols": [], "missing": []}

    need = _scan_cols_from_expr(when)      # 基础 + J/VR
    dfD  = _read_stock_df(ts0, st, ref, columns=need)
    have = set(map(str.lower, dfD.columns))

    # 不再进行兜底计算，依赖数据完整性
    auto_filled = []

    # 执行窗口求值检查语法
    dfTF   = dfD if tf == "D" else _resample(dfD, tf)
    win_df = _window_slice(dfTF, ref, win)
    try:
        with _ctx_df(win_df):
            _ = evaluate_bool(when, win_df)
        return {"ok": True, "error": None, "need_cols": need, "missing": list(set(map(str.lower, need)) - set(have)), "auto_filled": auto_filled, "probe": ts0, "ref_date": ref}
    except Exception as e:
        return {"ok": False, "error": f"表达式错误: {e}", "need_cols": need, "missing": list(set(map(str.lower, need)) - set(have)), "auto_filled": auto_filled, "probe": ts0, "ref_date": ref}
