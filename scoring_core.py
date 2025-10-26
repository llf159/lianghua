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
BOOL_CACHE_SIZE = 32

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
)
try:
    from config import SC_PRESCREEN_RULES
except Exception:
    SC_PRESCREEN_RULES = []
    
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
from utils import normalize_ts, normalize_trade_date
import tdx_compat as tdx
from tdx_compat import evaluate_bool
from stats_core import post_scoring
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

# 策略规则加载
try:
    from utils import load_rank_rules_py, load_filter_rules_py
    _loaded_rank = load_rank_rules_py()
    _loaded_filter = load_filter_rules_py()
    if _loaded_rank:
        SC_RULES = _loaded_rank
    if _loaded_filter:
        SC_PRESCREEN_RULES = _loaded_filter
except Exception as _e:
    # 回退到默认配置
    pass

OUTPUT_DIR = Path(SC_OUTPUT_DIR)
DETAIL_DIR = OUTPUT_DIR / "details"
ALL_DIR = OUTPUT_DIR / "all"
ALL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- 日志系统初始化 -------------------------
from log_system import get_logger, log_data_processing, log_database_operation, log_file_operation, log_algorithm_execution

# 初始化统一日志记录器
LOGGER = get_logger("scoring_core")

# 尝试启用表达式编译缓存
try:
    import tdx_compat
    if hasattr(tdx_compat, 'enable_expr_cache'):
        cache_status = tdx_compat.enable_expr_cache(True)
        LOGGER.info(f"[表达式缓存] 已启用，容量: {cache_status.get('max_size', 0)}, 当前大小: {cache_status.get('current_size', 0)}")
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

# 交易日列表缓存
_TRADE_DATES_CACHE = None

# 缓存命中率统计
_CACHE_STATS = {
    "hits": 0,      # 缓存命中次数
    "misses": 0,    # 缓存未命中次数
    "fallbacks": 0  # 回退到数据库查询次数
}

# 进度系统已统一到队列+drain模式，见下方实现


def init_rank_env(ref_date: str, universe_codes: List[str], base: str, adj: str, default_N: int = 60):
    _RANK_G["ref"] = str(ref_date)
    _RANK_G["codes"] = list(universe_codes or [])
    _RANK_G["base"] = str(base)
    _RANK_G["adj"]  = str(adj)
    _RANK_G["default_N"] = int(default_N)


def _get_trade_dates_cached() -> List[str]:
    """获取交易日列表（带缓存）"""
    global _TRADE_DATES_CACHE
    
    if _TRADE_DATES_CACHE is not None:
        return _TRADE_DATES_CACHE
    
    try:
        list_trade_dates, _ = _get_database_manager_functions()
        trade_dates = list_trade_dates()
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


def preload_rank_data(ref_date: str, start_date: str, columns: List[str]):
    """预加载排名计算所需的universe数据"""
    global _RANK_DATA_CACHE, _RANK_G
    
    try:
        # 获取universe股票代码
        universe_codes = _RANK_G.get("codes", [])
        if not universe_codes:
            LOGGER.warning("[预加载] 未找到universe股票代码，跳过预加载")
            return
        
        from database_manager import batch_query_stock_data
        
        # 确保预加载的列至少包含排名计算所需的基本列
        required_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
        all_columns = list(set(columns + required_cols))  # 合并并去重
        
        # 使用 warmup_for 来计算最长指标所需的日期
        from indicators import warmup_for
        from indicators import REGISTRY
        
        # 提取需要计算的指标名称
        indicator_names = set()
        for col in all_columns:
            for ind_name, meta in REGISTRY.items():
                if col in meta.out:
                    indicator_names.add(ind_name)
        
        # 计算所需的 warmup 天数
        if indicator_names:
            warmup_days = warmup_for(list(indicator_names))
            LOGGER.info(f"[预加载] 最长指标所需的 warmup 天数: {warmup_days}")
        else:
            # 如果没有找到指标，使用默认值
            warmup_days = 120
            LOGGER.warning(f"[预加载] 未找到指标，使用默认 warmup 天数: {warmup_days}")
        
        # 还需要考虑排名计算所需的 N*3 天冗余
        default_N = _RANK_G.get("default_N", 60)
        min_required_days = max(warmup_days, default_N * 3)
        LOGGER.info(f"[预加载] 综合所需天数: {min_required_days} (warmup={warmup_days}, N*3={default_N*3})")
        
        # 按照交易日列表反推准确的交易日起点
        actual_start_date = _compute_start_date_by_trade_dates(ref_date, min_required_days)
        LOGGER.info(f"[预加载] 使用交易日起点: {actual_start_date} (原起点: {start_date})")
        
        LOGGER.info(f"[预加载] 开始预加载universe排名数据: {actual_start_date} ~ {ref_date}, {len(universe_codes)} 只股票")
        
        # 预加载universe数据
        df = batch_query_stock_data(
            ts_codes=universe_codes,  # 只查询universe中的股票
            start_date=actual_start_date,
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
                    LOGGER.warning(f"[预加载] 检测到重复列: {list(df.columns[dup_mask])} -> 保留首列")
                    df = df.loc[:, ~dup_mask].copy()
            except Exception:
                pass

            _RANK_DATA_CACHE["data"] = df
            _RANK_DATA_CACHE["ref_date"] = ref_date
            _RANK_DATA_CACHE["start_date"] = actual_start_date
            _RANK_DATA_CACHE["columns"] = all_columns.copy()
            
            LOGGER.info(f"[预加载] universe排名数据预加载完成: {len(df)} 条记录, {len(df['ts_code'].unique())} 只股票, 列: {all_columns}")
        else:
            LOGGER.warning("[预加载] 预加载数据为空")
            
    except Exception as e:
        LOGGER.error(f"[预加载] 预加载失败: {e}")
        _RANK_DATA_CACHE["data"] = None

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

def _flush_batch_buffer():
    """将缓冲区数据批量写回数据库"""
    global _BATCH_BUFFER
    
    if _BATCH_BUFFER["current_count"] == 0:
        return
    
    try:
        LOGGER.info(f"[批量写回] 开始批量写回 {_BATCH_BUFFER['current_count']} 条明细数据")
        
        # 使用database_manager进行批量写回，不直接操作数据库
        from database_manager import get_database_manager
        from config import SC_OUTPUT_DIR
        
        db_manager = get_database_manager()
        details_db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
        os.makedirs(os.path.dirname(details_db_path), exist_ok=True)
        
        # 确保数据库表存在
        db_manager.init_stock_details_tables(details_db_path, "duckdb")
        
        # 准备批量数据
        batch_data = []
        for i, summary in enumerate(_BATCH_BUFFER["summaries"]):
            per_rules = _BATCH_BUFFER["per_rules"][i]["rules"]
            # 对 JSON 字段进行序列化
            highlights_json = json.dumps(summary.get('highlights', []), ensure_ascii=False)
            drawbacks_json = json.dumps(summary.get('drawbacks', []), ensure_ascii=False)
            opportunities_json = json.dumps(summary.get('opportunities', []), ensure_ascii=False)
            rules_json = json.dumps(per_rules if per_rules else [], ensure_ascii=False)
            
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
        request_id = db_manager.receive_data(
            source_module="scoring_core_batch",
            data_type="custom",
            data=df,
            table_name="stock_details",
            mode="upsert",
            db_path=details_db_path,
            validation_rules={
                "required_columns": ["ts_code", "ref_date"]
            }
        )
        
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
    
    # 检查窗口覆盖范围（至少需要N*3天的冗余）
    required_start = (dt.datetime.strptime(ref_date, "%Y%m%d") - dt.timedelta(days=N*3)).strftime("%Y%m%d")
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
    # 给足冗余，按自然日回溯（停牌/周末）
    start = (dt.datetime.strptime(_ref, "%Y%m%d") - dt.timedelta(days=_N*3)).strftime("%Y%m%d")
    
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
    
    # 如果所有方式都失败，返回空DataFrame
    LOGGER.error("所有查询方式都失败，返回空数据")
    return pd.DataFrame(columns=cols)


def _rank_series(s: pd.Series, ascending: bool) -> pd.Series:
    # 名次从 1 开始；相等取平均名次；缺失置为最大名次+1
    r = s.rank(ascending=ascending, method="average")
    mx = int(r.max()) if pd.notna(r.max()) else len(r)
    return r.fillna(mx + 1)


def _rule_key(rule: dict) -> tuple:
    name = str(rule.get("name", "")).strip()
    tf = str(rule.get("timeframe", "D")).upper().strip()
    win = int(rule.get("window", SC_LOOKBACK_D))
    # 规范化条件表达式
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
    
    # 先尝试从预加载缓存获取数据
    df = _get_rank_data_from_cache(ref, N, codes)
    if df is None:
        # 缓存未命中，回退到数据库查询
        _CACHE_STATS["fallbacks"] += 1
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
    
    # 先尝试从预加载缓存获取数据
    df = _get_rank_data_from_cache(ref, N, codes)
    if df is None:
        # 缓存未命中，回退到数据库查询
        _CACHE_STATS["fallbacks"] += 1
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
            "window": int(rule.get("window", SC_LOOKBACK_D)),
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


def _after_scoring(ref_date: str, df_all_scores: pd.DataFrame | None = None):
    windows = [1,2,3,5,10,20]              # 时间窗口配置
    benchmarks = ["000300.SH", "399006.SZ"]  # 只用"已下载"的指数
    try:
        from stats_core import run_tracking
        run_tracking(ref_date, windows, benchmarks, score_df=df_all_scores, group_by_board=True, save=True)
    except ImportError:
        LOGGER.warning("stats_core模块未找到，跳过跟踪功能")


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
    list_trade_dates, _ = _get_database_manager_functions()
    days = list_trade_dates() or []

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


def _get_latest_date_from_database() -> Optional[str]:
    """从数据库获取最新交易日期"""
    try:
        from database_manager import get_latest_trade_date
        latest = get_latest_trade_date()
        if latest:
            LOGGER.info(f"[参考日] 从数据库获取最新日期: {latest}")
            log_database_operation("scoring_core", "查询最新日期", "latest_trade_date", 1, None, True)
            return latest
    except Exception as e:
        LOGGER.warning(f"从数据库获取最新日期失败: {e}")
    return None


def _get_latest_date_from_daily_partition() -> Optional[str]:
    """从daily分区获取最新交易日期"""
    try:
        list_trade_dates, _ = _get_database_manager_functions()
        dates = list_trade_dates()
        if dates:
            latest = dates[-1]
            LOGGER.info(f"[参考日] 从daily分区获取最新日期: {latest}")
            return latest
    except Exception:
        pass
    return None


def _get_latest_date_from_single_dir() -> Optional[str]:
    """从single目录推断最新交易日期"""
    if not _has_single_dir():
        return None
    
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
        LOGGER.info(f"[参考日] 从single目录获取最新日期: {mx}")
        return mx
    return None


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
        latest = _get_latest_date_from_single_dir()
    
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


def _recent_points(dfD: pd.DataFrame, rule: dict, ref_date: str, ctx: dict = None) -> tuple[float, int|None, str|None]:
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

    if ctx:
        # 使用缓存版本
        s_bool = _eval_bool_cached(ctx, dfD, when, tf, window, ref_date)
    else:
        # 原始版本
        dfTF = dfD if tf == "D" else _resample(dfD, tf)
        win_df = _window_slice(dfTF, ref_date, window)
        if win_df.empty:
            return 0.0, None, None
        with _ctx_df(win_df):
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


# 所有数据库操作已统一使用database_manager，不再需要分散的优化函数


def _batch_update_ranks_duckdb_optimized(db_path: str, ref_date: str, scored_sorted: List[Any]) -> bool:
    """批量更新排名 - 使用database_manager统一管理"""
    try:
        from database_manager import get_database_manager
        db_manager = get_database_manager()
        
        # 准备批量更新数据
        update_data = []
        for i, stock in enumerate(scored_sorted, 1):
            update_data.append({
                'ts_code': stock.ts_code,
                'ref_date': ref_date,
                'rank': i,
                'total': len(scored_sorted)
            })
        
        # 使用database_manager的receive_data接口进行批量更新
        import pandas as pd
        df = pd.DataFrame(update_data)
        
        # 去重：按 (ts_code, ref_date, rank) 排序，保留每个 (ts_code, ref_date) 组合的最小 rank
        original_count = len(df)
        df = df.sort_values(['ts_code', 'ref_date', 'rank']).drop_duplicates(['ts_code', 'ref_date'], keep='first')
        deduplicated_count = len(df)
        
        # 如果发生去重，打印告警
        if deduplicated_count < original_count:
            LOGGER.warning(f"[detail] 发现重复键，已去重: 原始 {original_count} 行，去重后 {deduplicated_count} 行")
        
        # 使用details目录下的数据库文件
        from config import SC_OUTPUT_DIR
        details_db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
        
        # 确保数据库表存在
        os.makedirs(os.path.dirname(details_db_path), exist_ok=True)
        db_manager.init_stock_details_tables(details_db_path, "duckdb")
        
        # 使用同步写入确保排名更新完成，单事务批量操作
        try:
            # 直接使用数据库连接进行同步批量更新
            with db_manager.get_connection(details_db_path, read_only=False) as conn:
                # 开始事务
                conn.execute("BEGIN TRANSACTION")
                
                try:
                    import time
                    start_time = time.time()
                    
                    # 使用 MERGE 语句进行一次性批量更新
                    # 创建临时表
                    conn.execute("""
                        CREATE TEMP TABLE _ranks(
                            ts_code VARCHAR, 
                            ref_date VARCHAR, 
                            rank INTEGER, 
                            total INTEGER
                        )
                    """)
                    
                    # 一次性批量导入：将DataFrame注册为DuckDB临时视图
                    conn.register('df_view', df)
                    
                    # 直接从DataFrame视图一次性插入到临时表
                    conn.execute("""
                        INSERT INTO _ranks (ts_code, ref_date, rank, total) 
                        SELECT ts_code, ref_date, rank, total FROM df_view
                    """)
                    
                    batch_insert_time = time.time() - start_time
                    
                    # 使用 MERGE 语句一次性更新
                    conn.execute("""
                        MERGE INTO stock_details d
                        USING _ranks r
                        ON d.ts_code = r.ts_code AND d.ref_date = r.ref_date
                        WHEN MATCHED THEN UPDATE SET rank = r.rank, total = r.total
                        WHEN NOT MATCHED THEN INSERT (ts_code, ref_date, rank, total) 
                        VALUES (r.ts_code, r.ref_date, r.rank, r.total)
                    """)
                    
                    # 清理临时表
                    conn.execute("DROP TABLE _ranks")
                    
                    # 提交事务
                    conn.execute("COMMIT")
                    
                except Exception as e:
                    # 回滚事务
                    conn.execute("ROLLBACK")
                    raise e
                
            LOGGER.info(f"[detail] MERGE批量更新排名成功, 影响行数: {len(scored_sorted)}, 批量导入耗时: {batch_insert_time:.3f}秒")
            return True
            
        except Exception as sync_error:
            LOGGER.warning(f"[detail] 同步批量更新失败，尝试异步写入: {sync_error}")
            # 回退到异步写入
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
            
            LOGGER.info(f"[detail] 异步批量更新排名请求已提交: {request_id}")
            return True
        
    except Exception as e:
        LOGGER.error(f"批量更新排名失败: {e}")
        return False


# 所有数据库操作已统一使用database_manager，不再需要分散的优化函数


def _write_detail_json(ts_code: str, ref_date: str, summary: dict, per_rules: list[dict]):
    """
    写入个股详情，优先使用数据库存储，失败时回退到JSON文件
    在表达式选股场景下，禁用数据库写入以避免冲突
    """

    # 数据库写入已统一由database_manager管理，无需进程检查
    success = True
    db_success = False
    
    # 检查是否在表达式选股场景下（通过调用栈判断）
    import inspect
    call_stack = [frame.function for frame in inspect.stack()]
    is_expression_screening = any(func in call_stack for func in ['apply_rule_across_universe', '_apply_one_stock_for_rule', 'tdx_screen', '_screen_one'])
    
    # 1. 优先尝试数据库存储（表达式选股时跳过）
    if (not is_expression_screening) and SC_DETAIL_STORAGE in ["database","both","db"] and SC_USE_DB_STORAGE:
        LOGGER.info(f"[detail] 尝试数据库存储: {ts_code}_{ref_date}")
        try:
            # 使用database_manager保存股票详情
            from database_manager import get_database_manager
            db_manager = get_database_manager()
            
            # 确保数据库表存在
            from config import SC_OUTPUT_DIR
            details_db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
            os.makedirs(os.path.dirname(details_db_path), exist_ok=True)
            db_manager.init_stock_details_tables(details_db_path, "duckdb")
            
            # 准备股票详情数据 - 对 JSON 字段进行序列化
            highlights_json = json.dumps(summary.get('highlights', []), ensure_ascii=False)
            drawbacks_json = json.dumps(summary.get('drawbacks', []), ensure_ascii=False)
            opportunities_json = json.dumps(summary.get('opportunities', []), ensure_ascii=False)
            rules_json = json.dumps(per_rules if per_rules else [], ensure_ascii=False)
            
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
            
            # 使用同步写入确保数据被正确保存
            import pandas as pd
            df = pd.DataFrame([detail_data])
            
            # 使用details目录下的数据库文件
            from config import SC_OUTPUT_DIR
            details_db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
            
            # 使用同步写入，确保数据被正确保存
            try:
                # 直接使用数据库连接进行同步写入
                with db_manager.get_connection(details_db_path, read_only=False) as conn:
                    # 准备插入数据
                    columns = list(detail_data.keys())
                    values = list(detail_data.values())
                    
                    # 构建INSERT OR REPLACE语句
                    columns_str = ', '.join(columns)
                    placeholders = ', '.join(['?' for _ in columns])
                    
                    sql = f"INSERT OR REPLACE INTO stock_details ({columns_str}) VALUES ({placeholders})"
                    conn.execute(sql, values)
                    
                LOGGER.info(f"[detail] 数据库存储成功: {ts_code}_{ref_date}")
                db_success = True
                
            except Exception as sync_error:
                LOGGER.warning(f"[detail] 同步写入失败，尝试异步写入: {sync_error}")
                # 回退到异步写入
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
                
                LOGGER.info(f"[detail] 异步数据库存储已提交: {ts_code}_{ref_date}, 请求ID: {request_id}")
                db_success = True
            
        except Exception as e:
            LOGGER.warning(f"[detail] 数据库存储异常: {ts_code}: {e}")
    elif is_expression_screening:
        LOGGER.debug(f"[detail] 表达式选股场景，跳过数据库存储: {ts_code}")
    
    # 2. 如果数据库失败且配置了回退，或者配置了JSON存储，则使用JSON文件
    if (not db_success and SC_DB_FALLBACK_TO_JSON) or SC_DETAIL_STORAGE in ["json", "both"]:
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
            
            # 如果是从数据库回退到JSON，给出明确提示
            if not db_success and SC_DB_FALLBACK_TO_JSON:
                LOGGER.info(f"[detail] 数据库写入失败，已回退到JSON文件: {ts_code} -> {out_path}")
            else:
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
            # 使用缓存版本
            sig = _eval_bool_cached(ctx, df, expr, timeframe, window, ref_date)
        else:
            # 原始版本
            dfTF = df if timeframe == "D" else _resample(df, timeframe)
            win_df = _window_slice(dfTF, ref_date, window)
            with _ctx_df(win_df):
                sig = evaluate_bool(expr, win_df)
        
        if sig is None:
            return []
        idx = sig.index
        out = [str(i) for i, v in zip(idx, list(sig)) if bool(v)]
        return out
    except Exception:
        return []


def _inject_config_tags(dfD: pd.DataFrame, ref_date: str, ctx: dict = None):
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


def _normalize_expr(expr: str) -> str:
    """标准化表达式字符串，用于缓存键"""
    if not expr:
        return ""
    # 去首尾空格，将连续空格压缩为单个空格
    normalized = re.sub(r'\s+', ' ', expr.strip())
    return normalized


def _eval_bool_cached(ctx: dict, base_df: pd.DataFrame, expr: str, tf: str, win: int, ref_date: str) -> pd.Series:
    """
    返回窗口期内的布尔序列（索引为窗口期索引）。
    1) 先根据 (expr_norm, tf, win, ref_date) 在 ctx['bool_lru'] 查询；
    2) 命中则直接返回；
    3) 未命中则：获取/构建 resampled df -> 按 ref_date 向左切 win 根 -> 注入上下文后调用原有 evaluate 逻辑 -> 回填 LRU -> 返回。
    """
    # 构建缓存键前需要先标准化表达式
    expr_norm = _normalize_expr(expr)
    
    # 构建缓存键
    cache_key = (expr_norm, tf, win, ref_date)
    
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
    
    # 获取或构建重采样数据
    resampled = ctx.get('resampled', {})
    if tf not in resampled:
        if tf == "D":
            resampled[tf] = base_df
        else:
            resampled[tf] = _resample(base_df, tf)
        ctx['resampled'] = resampled
    
    df_tf = resampled[tf]
    
    # 切窗
    win_df = _window_slice(df_tf, ref_date, win)
    
    # 空表达式检查移到切窗之后，这样可以用 win_df.index 构造返回值
    if not expr_norm:
        result = pd.Series(False, index=win_df.index, dtype=bool)
    elif win_df.empty:
        result = pd.Series([], index=win_df.index, dtype=bool)
    else:
        # 计算布尔序列
        with _ctx_df(win_df):
            result = evaluate_bool(expr, win_df)
        
        if result is None:
            result = pd.Series(False, index=win_df.index, dtype=bool)
        else:
            # 确保返回的是布尔类型的Series，索引与窗口对齐
            result = pd.Series(result, index=win_df.index, dtype=bool)
    
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
                cnt, err = _count_hits_perbar(df, rule, ref_date, ctx)
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
                    _d, _ = _last_true_date(df, cand_when, ref_date=ref_date, timeframe=tf, window=win, ctx=ctx)
                    hit_date = _d
                ok = bool(cnt and cnt > 0 and err is None)
                add = float(pts * int(cnt or 0))
                gate_ok, gate_err, gate_norm = _eval_gate(df, rule, ref_date, ctx)
                if not gate_ok:
                    add = 0.0
                rows.append({
                    "name": name, "scope": scope, "timeframe": tf, "window": win, "period": period,
                    "points": pts, "ok": ok, "cnt": int(cnt or 0),
                    "add": add,
                    "hit_date": hit_date,
                    "hit_dates": _list_true_dates(df, (cand_when or rule.get("when") or ""), ref_date=ref_date, window=win, timeframe=tf, ctx=ctx),
                    "hit_count": int(cnt or 0),
                    "gate_ok": bool(gate_ok),
                    "gate_when": (gate_norm.get("when") if isinstance(gate_norm, dict) else None),
                    "explain": expl,
                })
                continue
            else:
                # === RECENT / DIST / NEAR: 按最近一次命中距今天数计分 ===
                if scope in {"RECENT", "DIST", "NEAR"}:
                    add, lag, err = _recent_points(df, rule, ref_date, ctx)
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
                        "hit_date": hit_date, "hit_dates": _list_true_dates(df, cand_when or (when or ""), ref_date=ref_date, window=win, timeframe=tf, ctx=ctx),
                        "hit_count": int(cnt or 0),
                        "explain": expl,
                        "gate_ok": bool(gate_ok),
                        "gate_when": (gate_norm.get("when") if isinstance(gate_norm, dict) else None),
                    })
                    continue
                # === 其余仍走原来的布尔命中路径 ===
                ok_eval, err_eval = _eval_rule(df, rule, ref_date, ctx)
                ok  = bool(ok_eval and not err_eval)
                err = err_eval
                add = (pts if ok else 0.0)
                when = rule.get("when")


            # === 追加：常规布尔规则命中日期信息 ===
            hit_date = None
            hit_dates = []
            gate_ok, gate_err, gate_norm = _eval_gate(df, rule, ref_date, ctx)
            try:
                if when:
                    dfTF2 = df if tf=="D" else _resample(df, tf)
                    win_df2 = _window_slice(dfTF2, ref_date, win)
                    if not win_df2.empty:
                        _d, _ = _last_true_date(df, when, ref_date=ref_date, timeframe=tf, window=win, ctx=ctx)
                        hit_date = _d
                    hit_dates = _list_true_dates(df, when, ref_date=ref_date, window=win, timeframe=tf, ctx=ctx)
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
    # 排序：按得分降序，同分时按J值升序，再同分时按代码升序（兜底）
    df = df.sort_values(["score", "tiebreak_j", "ts_code"], ascending=[False, True, True])
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
    try:
        LOGGER.info(f"[backfill] run_for_date({ref_date}) ...")
        run_for_date(ref_date)  # 如果你的函数签名有其它参数，这里照你的实际传入
    except Exception as e:
        LOGGER.error(f"[backfill] run_for_date 失败：{e}")

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
    # 1) 优先用交易日日历生成完整日期序列
    list_trade_dates, _ = _get_database_manager_functions()
    cal = list_trade_dates() or []
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
    LOGGER.info(f"[backfill] done, wrote {len(written)} days.")
    return written


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
            try:
                from log_system import get_logger
                _logger = get_logger("scoring_core")
                _logger.warning(f"[ensure_dt] 检测到重复列: {list(df.columns[dup_mask])} -> 保留首列")
            except Exception:
                pass
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
        import logging as _logging
        if LOGGER.isEnabledFor(_logging.DEBUG):
            LOGGER.debug(f"[screen][范围] attention 源={src} 参考日={_ref} 匹配文件={os.path.basename(pick)} 命中={len(codes)}")
        return codes
    except Exception as e:
        LOGGER.debug(f"[screen][范围] 加载特别关注清单失败：{e}")
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
        LOGGER.warning(f"[SCOPE] 未识别的 scope='{scope}'，按 ANY 处理")
    except Exception:
        pass
    return bool(s.any())


def _eval_clause(dfD: pd.DataFrame, clause: dict, ref_date: str, ctx: dict = None) -> Tuple[bool, Optional[str]]:
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
        if ctx:
            # 使用缓存版本
            s_bool = _eval_bool_cached(ctx, dfD, when, tf, window, ref_date)
        else:
            # 原始版本
            dfTF = dfD if tf=="D" else _resample(dfD, tf)
            win_df = _window_slice(dfTF, ref_date, window)
            if win_df.empty:
                return False, f"窗口数据为空: tf={tf}, window={window}"
            with _ctx_df(win_df):
                s_bool = evaluate_bool(when, win_df)
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


def _check_database_health():
    """检查数据库健康状态，用于诊断问题"""
    try:
        from database_manager import get_data_source_status
        from config import DATA_ROOT, UNIFIED_DB_PATH
        import os
        
        status = get_data_source_status()
        db_path = os.path.abspath(os.path.join(DATA_ROOT, UNIFIED_DB_PATH))
        
        health_info = {
            'using_unified_db': status.get('use_unified_db', False),
            'database_exists': os.path.exists(db_path),
            'database_locked': False,  # 功能已迁移到database_manager
            'file_size': os.path.getsize(db_path) if os.path.exists(db_path) else 0
        }
        
        # 检查连接池状态
        try:
            from database_manager import get_database_manager
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

def _read_stock_df(ts_code: str, start: str, end: str, columns: List[str]) -> pd.DataFrame:
    """
    读取某只股票在给定区间的日线数据。
    - 优先使用预加载的全市场数据缓存
    - 回退到 database_manager 统一接口
    """
    global _RANK_DATA_CACHE
    
    # 优先使用预加载的数据缓存
    if (_RANK_DATA_CACHE["data"] is not None and 
        _RANK_DATA_CACHE["ref_date"] == end and 
        _RANK_DATA_CACHE["start_date"] == start and
        all(col in _RANK_DATA_CACHE["data"].columns for col in columns)):
        
        try:
            # 从预加载数据中筛选指定股票
            df = _RANK_DATA_CACHE["data"][_RANK_DATA_CACHE["data"]["ts_code"] == ts_code].copy()
            if not df.empty:
                # 确保包含所需的列
                available_cols = [col for col in columns if col in df.columns]
                if available_cols:
                    df = df[["ts_code", "trade_date"] + available_cols]
                    LOGGER.debug(f"[{ts_code}] 使用预加载数据: {len(df)} 条记录")
                    return df
        except Exception as e:
            LOGGER.warning(f"[{ts_code}] 预加载数据筛选失败: {e}")
    
    # 回退到 database_manager 统一接口
    from database_manager import query_stock_data
    
    # 带重试机制的数据查询
    max_retries = 3  # 减少重试次数，避免长时间等待
    retry_delay = 0.5  # 减少初始延迟时间
    
    for attempt in range(max_retries):
        try:
            # 在重试前进行数据库健康检查
            if attempt == 0:
                health_info = _check_database_health()
                if health_info.get('database_locked', False):
                    LOGGER.warning(f"[{ts_code}] 数据库被锁定，等待释放...")
                    import time
                    time.sleep(2.0)  # 等待2秒让其他进程释放锁
            elif attempt > 0:
                LOGGER.info(f"[{ts_code}] 第{attempt + 1}次重试，检查数据库状态...")
                _check_database_health()
            
            # 直接使用 database_manager 统一接口查询
            from database_manager import query_stock_data
            df = query_stock_data(
                ts_code=ts_code,
                start_date=start,
                end_date=end,
                columns=columns
            )
            
            if not df.empty and "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str)
            LOGGER.debug(f"[{ts_code}] 数据库查询完成: n_cols={len(df.columns)} 前12列={list(df.columns)[:12]}")
            return df
                    
        except Exception as e:
            error_msg = str(e).lower()
            LOGGER.debug(f"[{ts_code}] 数据库访问异常 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
            
            # 根据错误类型采用不同的处理策略
            if any(keyword in error_msg for keyword in [
                "database is locked", "file is locked", "database is busy",
                "file is being used", "access denied", "另一个程序正在使用此文件"
            ]):
                if attempt < max_retries - 1:
                    # 数据库锁定错误：使用固定延迟 + 随机抖动，避免指数退避
                    import random
                    random_delay = retry_delay + random.uniform(0, 1.0)  # 减少随机延迟范围
                    LOGGER.warning(f"[{ts_code}] 数据库被占用，{random_delay:.1f}秒后重试 (尝试 {attempt + 1}/{max_retries}): {e}")
                    import time
                    time.sleep(random_delay)
                    # 移除指数退避，使用固定延迟
                    continue
                else:
                    LOGGER.warning(f"[{ts_code}] 数据库重试失败，回退到Parquet文件: {e}")
                    # 强制清理可能残留的连接
                    try:
                        from database_manager import clear_connections_only
                        clear_connections_only()
                    except:
                        pass
                    break
            elif "timeout" in error_msg or "timed out" in error_msg:
                # 超时错误：快速重试，不增加延迟
                if attempt < max_retries - 1:
                    LOGGER.warning(f"[{ts_code}] 数据库超时，立即重试 (尝试 {attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    LOGGER.warning(f"[{ts_code}] 数据库超时重试失败: {e}")
                    break
            elif "connection" in error_msg or "connect" in error_msg:
                # 连接错误：清理连接后重试
                LOGGER.warning(f"[{ts_code}] 数据库连接错误: {e}")
                try:
                    from database_manager import clear_connections_only
                    clear_connections_only()
                except:
                    pass
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1.0)  # 短暂等待后重试
                    continue
                else:
                    break
            elif any(keyword in error_msg for keyword in [
                "no such table", "table", "sql", "query", "execute", "cursor",
                "database", "db", "sqlite", "duckdb", "parquet"
            ]):
                # 数据库查询错误：可能是表不存在或查询问题，尝试重试
                LOGGER.warning(f"[{ts_code}] 数据库查询错误: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)  # 短暂等待后重试
                    continue
                else:
                    break
            else:
                # 其他错误：记录详细信息，但尝试重试一次
                LOGGER.warning(f"[{ts_code}] 未知数据库错误: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    # 对于未知错误，也尝试重试一次
                    import time
                    time.sleep(0.5)
                    continue
                else:
                    # 强制清理可能残留的连接
                    try:
                        from database_manager import clear_connections_only
                        clear_connections_only()
                    except:
                        pass
                    break
    
    # 回退到原有逻辑
    LOGGER.info(f"[{ts_code}] 数据库访问失败，回退到备用数据源")
    
    if _has_single_dir():
        f = os.path.join(_single_root_dir(), f"{ts_code}.parquet")
        if not os.path.isfile(f):
            LOGGER.error(f"[{ts_code}] single 文件不存在: {f}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(f)
            # 统一为字符串比较 + 区间切片
            if "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str)
                mask = (df["trade_date"] >= str(start)) & (df["trade_date"] <= str(end))
                df = df.loc[mask].copy()
            # ✅ 不再按 columns 裁列：保留 single_* 中的全量指标/特征
            LOGGER.debug(f"[{ts_code}] single列保留: n_cols={len(df.columns)} 前12列={list(df.columns)[:12]}")
        except Exception as e:
            LOGGER.error(f"[{ts_code}] 读取single文件失败: {e}")
            return pd.DataFrame()
    else:
        # 按日分区读取时，仍然只取所需列以提高效率
        try:
            df = _read_data_via_dispatcher(ts_code, start, end, columns)
            if df.empty:
                LOGGER.error(f"[{ts_code}] 通过分发器无法读取数据")
                return pd.DataFrame()
        except Exception as e:
            LOGGER.error(f"[{ts_code}] 分发器读取失败: {e}")
            return pd.DataFrame()
    
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str)

    # 不再进行兜底计算，依赖数据完整性
    return df


def _trade_span(start: Optional[str], end: str) -> List[str]:
    """把自然日区间换成交易日清单（基于现有分区）。"""
    try:
        list_trade_dates, _ = _get_database_manager_functions()
        days = list_trade_dates() or []
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


def _eval_single_rule(dfD: pd.DataFrame, rule: dict, ref_date: str, ctx: dict = None) -> dict:
    """处理单个规则，返回规则执行结果"""
    
    tf   = str(rule.get("timeframe","D")).upper()
    win  = int(rule.get("window", SC_LOOKBACK_D))
    pts  = float(rule.get("points", 0))
    dfTF = dfD if tf=="D" else _resample(dfD, tf)
    win_df = _window_slice(dfTF, ref_date, win)
    res = {"add": 0.0, "cnt": None, "lag": None, "hit_date": None, "gate_ok": True, "error": None}
    try:
        scope = str(rule.get("scope","ANY")).upper().strip()
        if scope in {"EACH","PERBAR","EACH_TRUE"}:
            cnt, err = _count_hits_perbar(dfD, rule, ref_date, ctx)
            if err: res["error"] = err
            gate_ok, _, _ = _eval_gate(dfD, rule, ref_date, ctx)
            res["gate_ok"] = gate_ok
            if cnt and pts and gate_ok:
                res["cnt"] = int(cnt); res["add"] = float(pts * int(cnt))
            # 命中日
            cand_when = None
            if "clauses" in rule and rule["clauses"]:
                for c in rule["clauses"]:
                    if c.get("when"): cand_when = c["when"]; break
            else:
                cand_when = rule.get("when")
            if cand_when and not win_df.empty:
                _d, _ = _last_true_date(dfD, cand_when, ref_date=ref_date, timeframe=tf, window=win, ctx=ctx); res["hit_date"] = _d
            return res

        if scope in {"RECENT","DIST","NEAR"}:
            add, lag, err = _recent_points(dfD, rule, ref_date, ctx)
            if err: res["error"] = err
            gate_ok, _, _ = _eval_gate(dfD, rule, ref_date, ctx)
            res["gate_ok"] = gate_ok
            res["lag"] = (None if lag is None else int(lag))
            if gate_ok and add:
                res["add"] = float(add)
            if res["lag"] is not None and not win_df.empty:
                idx = win_df.index if isinstance(win_df.index, pd.DatetimeIndex) else pd.to_datetime(win_df["trade_date"].astype(str))
                i = res["lag"]; 
                if 0 <= i < len(idx): res["hit_date"] = idx[-1 - i].strftime("%Y%m%d")
            return res

        # 常规布尔
        ok, err = _eval_rule(dfD, rule, ref_date, ctx)
        if err: res["error"] = err
        gate_ok, _, _ = _eval_gate(dfD, rule, ref_date, ctx)
        res["gate_ok"] = gate_ok
        if ok and gate_ok and pts:
            res["add"] = float(pts)
        # 命中日/全集命中
        w = (rule.get("when") or "").strip()
        if w and not win_df.empty:
            _d, _ = _last_true_date(dfD, w, ref_date=ref_date, timeframe=tf, window=win, ctx=ctx); res["hit_date"] = _d
        return res
    except Exception as e:
        res["error"] = f"eval-exception: {e}"
        return res


def apply_rule_across_universe(rule: dict, ref_date: str | None = None, universe: str | list[str] | None = "all", return_df: bool = True):
    ref = ref_date or _pick_ref_date()
    tf  = str(rule.get("timeframe","D")).upper()
    win = int(rule.get("window", SC_LOOKBACK_D))
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
    window = int(clause.get("window", SC_PRESCREEN_LOOKBACK_D))
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


def _score_one(ts_code: str, ref_date: str, start_date: str, columns: List[str]) -> Optional[Tuple[str, Optional[ScoreDetail], Optional[Tuple[str,str,str]]]]:
    """
    返回：
      - (ts_code, detail 或 None, 黑名单项 或 None)
      若被初选淘汰，返回黑名单项 (ts_code, period, reason)；否则返回打分 detail。
    """
    LOGGER.debug(f"开始评分股票: {ts_code}, 参考日期: {ref_date}, 开始日期: {start_date}")
    
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "top"), exist_ok=True)
    os.makedirs(os.path.join(SC_OUTPUT_DIR, "all"), exist_ok=True)

    try:
        LOGGER.debug(f"[{ts_code}] 开始读取股票数据")
        df = _read_stock_df(ts_code, start_date, ref_date, columns)
        if df is None or df.empty:
            LOGGER.warning(f"[{ts_code}] 数据为空，跳过")
            # return (ts_code, None, ("", "", "数据为空"))
            return (ts_code, None, (ts_code, ref_date, "数据为空"))
        
        LOGGER.debug(f"[{ts_code}] 数据读取成功: {len(df)}行数据")
        
        # 初始化单票上下文和缓存
        ctx = {
            'bool_lru': _SmallLRU(BOOL_CACHE_SIZE),
            'resampled': {},
            'bool_cache_hit': 0,
            'bool_cache_miss': 0
        }
        
        LOGGER.debug(f"[{ts_code}] 开始注入基准特征")
        df = _inject_benchmark_features(df, start_date, ref_date)
        tdx.EXTRA_CONTEXT.update(get_eval_env(ts_code, ref_date))

        # 注入基于 config 的自定义标签（仅 D 级别）
        try:
            _inject_config_tags(df, ref_date, ctx)
            LOGGER.debug(f"[{ts_code}] 自定义标签注入成功")
        except Exception as e:
            LOGGER.warning(f"[{ts_code}] 自定义标签注入失败: {e}")
        
        # 初选
        LOGGER.debug(f"[{ts_code}] 开始初选检查")
        pres = _prescreen(df, ref_date, ctx)
        if not pres.passed:
            LOGGER.info(f"[{ts_code}] 初选未通过: {pres.reason}")
            return (ts_code, None, (ts_code, pres.period or ref_date, pres.reason or "prescreen"))
        
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
            win = int(rule.get("window", SC_LOOKBACK_D))
            expl = rule.get("explain")
            
            try:
                period = _period_for_clause(df, rule, ref_date)
            except Exception:
                period = ref_date
                
            per_rules.append({
                "name": name, "scope": scope, "timeframe": tf, "window": win, "period": period,
                "points": pts, "ok": ok, "cnt": cnt,
                "add": add,
                "hit_date": hit_date,
                "hit_dates": hit_dates or [],
                "hit_count": len(hit_dates) if hit_dates else (cnt or 0),
                "gate_ok": bool(gate_ok),
                "gate_when": gate_when,
                "explain": expl,
                "error": err,
            })

        for rule in _iter_unique_rules():
            # 使用统一的规则处理逻辑
            rule_result = _eval_single_rule(df, rule, ref_date, ctx)
            
            # 提取结果
            add = rule_result.get("add", 0.0)
            cnt = rule_result.get("cnt")
            lag = rule_result.get("lag")
            hit_date = rule_result.get("hit_date")
            gate_ok = rule_result.get("gate_ok", True)
            err = rule_result.get("error")
            
            # 计算hit_dates
            tf = str(rule.get("timeframe", "D")).upper()
            win = int(rule.get("window", SC_LOOKBACK_D))
            when = rule.get("when")
            hit_dates = []
            if when:
                try:
                    hit_dates = _list_true_dates(df, when, ref_date=ref_date, window=win, timeframe=tf, ctx=ctx)
                except Exception:
                    pass
            
            # 更新分数和日志
            if gate_ok and add != 0:
                score += add
                _append_reason_by_rule(add, rule)
                LOGGER.debug(f"[HIT][{ts_code}] {rule.get('name','<unnamed>')} +{add} => {score}")
            elif not gate_ok:
                LOGGER.debug(f"[GATE][{ts_code}] {rule.get('name')} 被 gate 拦截")
            else:
                LOGGER.debug(f"[MISS][{ts_code}] {rule.get('name','<unnamed>')}")
            
            # 添加规则详情
            _add_rule_detail(rule, bool(add != 0), cnt=cnt, add=add, hit_date=hit_date, hit_dates=hit_dates, 
                           err=err, gate_ok=gate_ok, gate_when=None)

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
        
        return (ts_code, ScoreDetail(ts_code, score, highlights, drawbacks, opportunities, tb), None)

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
        return (ts_code, None, (ts_code, ref_date, f"评分失败:{e}"))


def _screen_one(ts_code, st, ref, tf, win, when, scope, use_prescreen_first, ctx: dict = None):
    try:
        # 单票筛选同样并入规则/标签所需列，避免 'j'、'duokong_*' 等缺列
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
            # 使用缓存版本
            s_bool = _eval_bool_cached(ctx, dfD, when, tf, win, ref)
            ok = _scope_hit(s_bool, scope)
        else:
            # 原始版本
            dfTF = dfD if tf == "D" else _resample(dfD, tf)
            win_df = _window_slice(dfTF, ref, win)
            if win_df.empty:
                return (ts_code, False, None, None)
            tdx.EXTRA_CONTEXT.update(get_eval_env(ts_code, ref))

            _inject_config_tags(dfD, ref)      # 忽略异常同原逻辑
            tdx.EXTRA_CONTEXT["DF"] = win_df
            ok = _scope_hit(evaluate_bool(when, win_df), scope)
        if ok:
            return (ts_code, True, (ts_code, ref, "pass"), None)
        else:
            return (ts_code, False, None, None)
    except Exception as e:
        return (ts_code, False, None, f"{e}")


def _count_hits_perbar(dfD: pd.DataFrame, rule_or_clause: dict, ref_date: str, ctx: dict = None) -> Tuple[int, Optional[str]]:
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
                if ctx:
                    # 使用缓存版本
                    s = _eval_bool_cached(ctx, dfD, when, tf, window, ref_date)
                else:
                    # 原始版本
                    with _ctx_df(win_df):
                        s = evaluate_bool(when, win_df).astype(bool)
                s_all = s if s_all is None else (s_all & s)
            return int(s_all.fillna(False).sum()), None
        else:
            tf = str(rule_or_clause.get("timeframe", "D")).upper()
            window = int(rule_or_clause.get("window", SC_LOOKBACK_D))
            when = (rule_or_clause.get("when") or "").strip()
            if not when:
                return 0, "空 when 表达式"
            if ctx:
                # 使用缓存版本
                s_bool = _eval_bool_cached(ctx, dfD, when, tf, window, ref_date)
            else:
                # 原始版本
                dfTF = dfD if tf=="D" else _resample(dfD, tf)
                win_df = _window_slice(dfTF, ref_date, window)
                if win_df.empty:
                    return 0, None
                with _ctx_df(win_df):
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
        LOGGER.warning(f"[attention] 交易日窗口为空，start={start} end={end}")
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
                LOGGER.warning(f"[attention] 读取失败 {f}: {e}");
                continue
            for c in codes:
                hit_cnt[c] = hit_cnt.get(c, 0) + 1
                first_last[c] = (first_last.get(c, (d, d))[0], d)
        if not hit_cnt:
            LOGGER.warning(f"[attention] {span[0]}~{span[-1]} 内没有 '{source}' 记录。")
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
                    LOGGER.warning(f"[attention] 读取全量排名失败 {path_all}: {e}")
                    df = None
            if df is None:
                # 回退：仅有 Top-K 文件时，以其相对名次近似
                path_top = os.path.join(SC_OUTPUT_DIR, "top", f"score_top_{d}.csv")
                if not os.path.isfile(path_top):
                    continue
                try:
                    df_top = pd.read_csv(path_top, dtype={"ts_code": str})
                except Exception as e:
                    LOGGER.warning(f"[attention] 读取 Top 失败 {path_top}: {e}"); continue
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
            LOGGER.warning(f"[attention] {span[0]}~{span[-1]} 无可计算的排名强度。"); return None
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
        LOGGER.info(f"[特别关注][{end}|w={len(span)}] source={source} 窗口={span[0]}~{span[-1]} min_hits>={min_hits} Top-{topN} -> {out_path}")

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
        LOGGER.warning(f"[attention/backfill] 窗口为空：start={start} end={end}")
        return

    LOGGER.info(f"[attention/backfill] 逐日补算 {days[0]} ~ {days[-1]}，共 {len(days)} 个交易日")
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
            LOGGER.warning(f"[attention/backfill] 生成 {e} 失败：{ex}")
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
    # 记录评分开始时间
    scoring_start_time = time.time()
    # 1) 参考日
    if not ref_date:
        ref_date = _pick_ref_date()
    _progress("select_ref_date", message=f"ref={ref_date}")

    # 2) 估算读取起始日
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
    init_rank_env(ref_date, codes, DATA_ROOT, API_ADJ, default_N=60)
    
    # 清空批量缓冲区，确保每次运行都是干净的状态
    _clear_batch_buffer()
    
    # 4) 数据预加载 - 一次读取，并发分发
    _ensure_dirs(ref_date)
    columns = _select_columns_for_rules()
    # 提高工作进程数量，充分利用CPU
    max_workers = SC_MAX_WORKERS or min(os.cpu_count() * 2, 16)  # 最多16个进程
    
    # 预加载所有股票数据到缓存
    try:
        from database_manager import get_data_source_status
        status = get_data_source_status()
        if status.get('use_unified_db', False):
            LOGGER.info(f"[预加载] 启用预加载功能，开始批量加载排名数据")
            preload_rank_data(ref_date, start_date, columns)
            LOGGER.info(f"[预加载] 批量预加载完成，开始并发评分")
        else:
            LOGGER.info(f"[预加载] 未使用统一数据库，跳过预加载")
    except Exception as e:
        LOGGER.warning(f"[预加载] 数据预加载失败，继续使用原有方式: {e}")
    
    # 5) 并行处理
    
    results: List[Tuple[str, Optional[ScoreDetail], Optional[Tuple[str,str,str]]]] = []
    whitelist_items: List[Tuple[str,str,str]] = []
    blacklist_items: List[Tuple[str,str,str]] = []
    _progress("score_start", total=len(codes), current=0,
              message=f"workers={max_workers}")

    done = 0
    # 创建单个线程池，避免重复创建/销毁
    # 使用之前计算的 max_workers，确保 UI 设置的并行数生效
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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

    # 拆分黑白名单与评分详情
    scored: List[ScoreDetail] = []
    for ts_code, detail, black in results:
        if black:
            blacklist_items.append(black)
        elif detail:
            whitelist_items.append((ts_code, ref_date, "pass"))
            scored.append(detail)
    
    # 刷新批量缓冲区，确保所有数据都被写回
    LOGGER.info("[批量写回] 评分完成，开始最终刷新缓冲区")
    _flush_batch_buffer()

    # 写缓存名单
    _progress("write_cache_lists", message=f"white={len(whitelist_items)} black={len(blacklist_items)}")
    _write_cache_lists(ref_date, whitelist_items, blacklist_items)

    # 排序：按得分降序，同分时按J值升序，再同分时按代码升序（兜底）
    def key_fn(x: ScoreDetail):
        tb = x.tiebreak if x.tiebreak is not None else float('inf')  # J值越小越好
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
    # 写入Top/All文件
    _progress("write_top_all_start", message="开始写入Top/All文件")
    
    out_path = os.path.join(SC_OUTPUT_DIR, "top", f"score_top_{ref_date}.csv")
    out_top.to_csv(out_path, index=False, encoding="utf-8-sig")
    # —— 全量排名（用于"单票详情"显示名次） ——
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
    
    _progress("write_top_all_done", message=f"Top/All文件写入完成: {out_path}, {all_path}")

    try:
        hooks_res = post_scoring(ref_date, df_all_scores=out_all)
        LOGGER.info(f"[HOOKS] tracking/surge 已完成：{hooks_res}")
    except Exception as _e:
        LOGGER.exception(f"[HOOKS] 执行失败：{_e}")
    LOGGER.info(f"[完成] Top-{SC_TOP_K} 已写入：{out_path}")
    try:
        ui_dir = OUTPUT_DIR / "meta"
        ui_dir.mkdir(parents=True, exist_ok=True)
        with open(ui_dir / "ui_hints.json", "w", encoding="utf-8") as f:
            json.dump({"topk_rows": int(globals().get("SC_TOPK_ROWS", 30))}, f, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as _e:
        LOGGER.warning(f"[UI] 写入 ui_hints.json 失败：{_e}")
        LOGGER.info(f"[名单] 白名单 {len(whitelist_items)}，黑名单 {len(blacklist_items)}")
    # —— 回写 rank 到各自的明细（独立写入details数据库） ——
    try:
        totalN = len(scored_sorted)
        db_success = False
        
        # 1. 优先尝试独立数据库批量更新排名
        if SC_DETAIL_STORAGE in ["database","both","db"] and SC_USE_DB_STORAGE:
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
            if not db_success and SC_DB_FALLBACK_TO_JSON:
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
        expr_stats = tdx_compat.get_expr_cache_stats()
        expr_hit_rate = expr_stats["hit_rate"] * 100
        LOGGER.info(f"[统计] 表达式编译缓存命中率: {expr_hit_rate:.1f}% ({expr_stats['hits']}/{expr_stats['hits'] + expr_stats['misses']})")
        LOGGER.info(f"[统计] 表达式编译缓存大小: {expr_stats['cache_size']}/{expr_stats['max_size']}")
    except Exception as e:
        LOGGER.debug(f"[统计] 获取表达式缓存统计失败: {e}")
    
    LOGGER.info(f"[统计] 评分完成 - 总时长: {scoring_duration:.2f}秒")
    LOGGER.info(f"[统计] 缓存命中率: {hit_rate:.1f}% ({_CACHE_STATS['hits']}/{total_requests})")
    LOGGER.info(f"[统计] 数据库回退次数: {_CACHE_STATS['fallbacks']}")
    
    # 重置统计计数器
    _CACHE_STATS["hits"] = 0
    _CACHE_STATS["misses"] = 0
    _CACHE_STATS["fallbacks"] = 0

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
    从通达信表达式里扫出可能需要的扩展列，动态扫描所有指标。
    """
    need = {"trade_date", "open", "high", "low", "close", "vol", "amount"}
    
    # 动态扫描所有指标
    try:
        from indicators import REGISTRY
        for indicator_name, meta in REGISTRY.items():
            # 检查表达式是否包含该指标（支持大小写）
            if (_expr_mentions(expr, indicator_name) or 
                _expr_mentions(expr, indicator_name.upper()) or
                _expr_mentions(expr, indicator_name.lower())):
                # 添加输出列名
                for output_col in meta.out.keys():
                    need.add(output_col)
            
            # 检查表达式是否直接包含输出列名（支持大小写）
            for output_col in meta.out.keys():
                if (_expr_mentions(expr, output_col) or 
                    _expr_mentions(expr, output_col.upper()) or
                    _expr_mentions(expr, output_col.lower())):
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
    window: int = 60,
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
    win = int(window)
    st  = _start_for_tf_window(ref, tf, win)

    # 选范围 + 只读必要列（J/VR 会在 _screen_one 里兜底）
    codes0 = _list_codes_for_day(ref)
    codes, src = _apply_universe_filter(list(codes0), ref, universe)
    LOGGER.info(f"[screen][范围] universe={str(universe)} 源={src} 原={len(codes0)} -> {len(codes)}")
    if not codes:
        LOGGER.warning(f"[screen] {ref} 无可用标的。")
        return pd.DataFrame() if return_df else None

    # 读取列：表达式涉及列 ∪ 全部规则/预筛/机会标签可能用到的列（如 j、duokong_*、vr 等）
    try:
        _rule_cols = _select_columns_for_rules()
    except Exception:
        _rule_cols = []
    cols = sorted(set(_scan_cols_from_expr(when)) | set(_rule_cols))
    LOGGER.debug(f"[screen] 参考日={ref} tf={tf} window={win} scope={scope} 宇宙={len(codes)}")
    LOGGER.debug(f"[screen] 读取列={cols} 起始={st}")

    hits: list[str] = []
    whitelist_items: list[tuple[str, str, str]] = []
    blacklist_items: list[tuple[str, str, str]] = []

    max_workers = (SC_MAX_WORKERS or max(os.cpu_count() - 1, 1))
    
    # 现在 _read_stock_df 已使用文件锁机制，避免数据库访问冲突

    
    # 使用单个线程池，避免重复创建/销毁
    _progress("screen_start", total=len(codes), current=0, message=f"workers={max_workers}")
    done = 0
    
    # 创建单个线程池，避免重复创建/销毁
    # 使用之前计算的 max_workers，确保 UI 设置的并行数生效
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for c in codes:
            futures.append(executor.submit(_screen_one, c, st, ref, tf, win, when, scope, use_prescreen_first))
        
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
