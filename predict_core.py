# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterable, Literal, Tuple, Callable, Union
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import datetime as dt
import re
import warnings
from scipy.optimize import minimize, minimize_scalar
from log_system import get_logger, log_function_calls, log_data_processing, log_algorithm_execution

# 初始化日志记录器
logger = get_logger("predict_core")

# ---------------- 依赖模块导入 ----------------
try:
    import config as cfg
    import tdx_compat as tdx
    import indicators as ind
    from indicators import estimate_warmup
    from database_manager import (
        get_database_manager, query_stock_data, get_trade_dates, 
        get_stock_list, get_latest_trade_date, get_smart_end_date,
        pv_asset_root, scan_with_duckdb, read_range, list_trade_dates
    )
    
    # 直接使用 database_manager 函数，不再需要包装器
    
    logger.info("成功导入所有依赖模块")
except Exception as e:
    logger.warning(f"部分依赖模块导入失败，使用兜底实现: {e}")
    
    class _Dummy: 
        pass
    
    # 配置兜底
    cfg = type("Cfg", (), {"DATA_ROOT": ".", "API_ADJ": "qfq"})()
    
    # 数据读取兜底
    pv_asset_root = lambda base, asset, adj: base
    def list_trade_dates(root): 
        return []
    read_range = None
    scan_with_duckdb = None
    
    # 兜底实现：直接返回空数据
    def query_stock_data(*args, **kwargs):
        return pd.DataFrame()
    
    def get_trade_dates(*args, **kwargs):
        return []
    
    def get_database_manager(*args, **kwargs):
        return None
    
    # 指标计算兜底
    ind = _Dummy()
    ind.REGISTRY = {}
    def estimate_warmup(exprs: Optional[Iterable[str]], recompute):
        return 60
    
    # TDX兼容兜底
    tdx = _Dummy()
    def _eval_stub(*a, **k):
        data = k.get("data", None)
        if k.get("as_bool"):
            if data is None: 
                return [False]
            return [False] * len(data)
        return {"sig": [False] * (len(data) if data is not None else 1)}
    tdx.evaluate = lambda code, data=None: _eval_stub(code, data=data)
    tdx.evaluate_bool = lambda code, data=None: _eval_stub(code, data=data, as_bool=True)
    tdx.EXTRA_CONTEXT: Dict[str, Any] = {}

# ---------------- 缓存系统 ----------------
class CacheBackend:
    """缓存后端接口：独立目录存储，按key存取模拟结果"""
    def get(self, key: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError
    def set(self, key: str, df: pd.DataFrame) -> None:
        raise NotImplementedError


class FileCache(CacheBackend):
    """文件缓存实现：将结果存储到指定目录的parquet/csv文件中"""
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
    """生成Scenario的哈希值，用作缓存键的一部分"""
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
    """生成股票代码列表的哈希值"""
    arr = sorted(str(c) for c in (codes or []))
    raw = "|".join(arr)
    import hashlib as _hashlib
    return _hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _make_cache_key(ref_date: str, sim_date: str, scene_hash: str, uni_hash: str, ind_flag: str) -> str:
    """生成完整的缓存键"""
    return f"{ref_date}_{sim_date}_{scene_hash}_{uni_hash}_{ind_flag}"

# ---------------- 数据模型定义 ----------------
DType = Literal["stock", "index"]

@dataclass
class PriceBounds:
    """价格约束范围定义"""
    open_min: float = 0.01
    open_max: float = 1000.0
    high_min: float = 0.01
    high_max: float = 1000.0
    low_min: float = 0.01
    low_max: float = 1000.0
    close_min: float = 0.01
    close_max: float = 1000.0
    
    def __post_init__(self):
        """初始化后验证价格范围合理性"""
        self.open_min = max(0.01, self.open_min)
        self.high_min = max(0.01, self.high_min)
        self.low_min = max(0.01, self.low_min)
        self.close_min = max(0.01, self.close_min)

@dataclass
class SolveResult:
    """价格求解结果"""
    success: bool
    prices: Dict[str, float]  # {"open": x, "high": x, "low": x, "close": x}
    target_value: float
    actual_value: float
    error: float
    iterations: int
    method: str
    message: str = ""

class PriceSolver:
    """价格求解器：根据技术指标条件反推股票价格"""
    
    def __init__(self, 
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
    @log_function_calls(logger)
    def solve_price(self,
                   condition: str,
                   target_value: float,
                   historical_data: pd.DataFrame,
                   price_bounds: Optional[PriceBounds] = None,
                   method: str = "optimize",
                   initial_guess: Optional[Dict[str, float]] = None) -> SolveResult:
        """
        根据技术指标条件反推股票价格
        
        Args:
            condition: 指标条件表达式，如 "j <= 13" 或 "rsi >= 70"
            target_value: 目标指标值
            historical_data: 历史数据（包含O,H,L,C,V等）
            price_bounds: 价格约束范围
            method: 求解方法 ("optimize", "binary_search", "grid_search")
            initial_guess: 初始价格猜测
            
        Returns:
            SolveResult: 求解结果
        """
        logger.debug(f"开始价格求解: 条件={condition}, 目标值={target_value}, 方法={method}")
        log_algorithm_execution("predict_core", "价格求解", {
            "condition": condition, 
            "target_value": target_value, 
            "method": method
        }, None, None, True)
        if price_bounds is None:
            price_bounds = PriceBounds()
            
        if initial_guess is None:
            initial_guess = self._get_initial_guess(historical_data)
            
        try:
            if method == "optimize":
                return self._solve_optimize(condition, target_value, historical_data, 
                                          price_bounds, initial_guess)
            elif method == "binary_search":
                return self._solve_binary_search(condition, target_value, historical_data, 
                                               price_bounds, initial_guess)
            elif method == "grid_search":
                return self._solve_grid_search(condition, target_value, historical_data, 
                                             price_bounds, initial_guess)
            else:
                raise ValueError(f"不支持的求解方法: {method}")
        except Exception as e:
            logger.error(f"价格求解失败: {e}", exc_info=True)
            return SolveResult(
                success=False,
                prices=initial_guess,
                target_value=target_value,
                actual_value=0.0,
                error=float('inf'),
                iterations=0,
                method=method,
                message=f"求解异常: {str(e)}"
            )
    
    def _get_initial_guess(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """获取初始价格猜测值"""
        if historical_data.empty:
            return {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0}
            
        last_row = historical_data.iloc[-1]
        return {
            "open": float(last_row.get("open", 10.0)),
            "high": float(last_row.get("high", 11.0)),
            "low": float(last_row.get("low", 9.0)),
            "close": float(last_row.get("close", 10.0))
        }
    
    def _solve_optimize(self, condition: str, target_value: float, 
                       historical_data: pd.DataFrame, price_bounds: PriceBounds,
                       initial_guess: Dict[str, float]) -> SolveResult:
        """使用数值优化方法求解价格"""
        
        def objective(x):
            """优化目标函数：最小化指标值与目标值的差异"""
            prices = {"open": x[0], "high": x[1], "low": x[2], "close": x[3]}
            
            # 构造包含新价格的完整数据
            full_data = self._build_full_data(historical_data, prices)
            
            # 计算指标值
            try:
                actual_value = self._evaluate_condition(condition, full_data)
                error = abs(actual_value - target_value)
                return error
            except Exception:
                return float('inf')
        
        def constraints(x):
            """价格合理性约束条件"""
            open_price, high, low, close = x[0], x[1], x[2], x[3]
            
            # 基本价格约束
            constraints = [
                high - low,  # 最高价 >= 最低价
                high - max(open_price, close),  # 最高价 >= 开盘价和收盘价
                min(open_price, close) - low,   # 最低价 <= 开盘价和收盘价
            ]
            
            return np.array(constraints)
        
        # 初始猜测
        x0 = [initial_guess["open"], initial_guess["high"], 
              initial_guess["low"], initial_guess["close"]]
        
        # 边界约束
        bounds = [
            (price_bounds.open_min, price_bounds.open_max),
            (price_bounds.high_min, price_bounds.high_max),
            (price_bounds.low_min, price_bounds.low_max),
            (price_bounds.close_min, price_bounds.close_max)
        ]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective, x0, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
                )
            
            if result.success:
                prices = {
                    "open": float(result.x[0]),
                    "high": float(result.x[1]),
                    "low": float(result.x[2]),
                    "close": float(result.x[3])
                }
                
                # 验证结果
                full_data = self._build_full_data(historical_data, prices)
                actual_value = self._evaluate_condition(condition, full_data)
                error = abs(actual_value - target_value)
                
                return SolveResult(
                    success=True,
                    prices=prices,
                    target_value=target_value,
                    actual_value=actual_value,
                    error=error,
                    iterations=result.nit,
                    method="optimize",
                    message=result.message
                )
            else:
                return SolveResult(
                    success=False,
                    prices=initial_guess,
                    target_value=target_value,
                    actual_value=0.0,
                    error=float('inf'),
                    iterations=result.nit,
                    method="optimize",
                    message=f"优化失败: {result.message}"
                )
                
        except Exception as e:
            return SolveResult(
                success=False,
                prices=initial_guess,
                target_value=target_value,
                actual_value=0.0,
                error=float('inf'),
                iterations=0,
                method="optimize",
                message=f"求解异常: {str(e)}"
            )
    
    def _solve_binary_search(self, condition: str, target_value: float,
                           historical_data: pd.DataFrame, price_bounds: PriceBounds,
                           initial_guess: Dict[str, float]) -> SolveResult:
        """使用二分搜索方法求解价格（适用于单调指标）"""
        
        # 对于二分搜索，我们主要调整收盘价，其他价格按比例调整
        base_prices = initial_guess.copy()
        
        def evaluate_at_close(close_price: float) -> float:
            """在指定收盘价下计算技术指标值"""
            # 按比例调整其他价格
            ratio = close_price / base_prices["close"]
            prices = {
                "open": base_prices["open"] * ratio,
                "high": base_prices["high"] * ratio,
                "low": base_prices["low"] * ratio,
                "close": close_price
            }
            
            full_data = self._build_full_data(historical_data, prices)
            return self._evaluate_condition(condition, full_data)
        
        # 二分搜索
        low = price_bounds.close_min
        high = price_bounds.close_max
        iterations = 0
        
        while high - low > self.tolerance and iterations < self.max_iterations:
            mid = (low + high) / 2
            try:
                actual_value = evaluate_at_close(mid)
                if actual_value < target_value:
                    low = mid
                else:
                    high = mid
                iterations += 1
            except Exception:
                break
        
        # 使用最终结果
        final_close = (low + high) / 2
        ratio = final_close / base_prices["close"]
        prices = {
            "open": base_prices["open"] * ratio,
            "high": base_prices["high"] * ratio,
            "low": base_prices["low"] * ratio,
            "close": final_close
        }
        
        full_data = self._build_full_data(historical_data, prices)
        actual_value = self._evaluate_condition(condition, full_data)
        error = abs(actual_value - target_value)
        
        return SolveResult(
            success=error < self.tolerance,
            prices=prices,
            target_value=target_value,
            actual_value=actual_value,
            error=error,
            iterations=iterations,
            method="binary_search",
            message="二分搜索完成"
        )
    
    def _solve_grid_search(self, condition: str, target_value: float,
                          historical_data: pd.DataFrame, price_bounds: PriceBounds,
                          initial_guess: Dict[str, float]) -> SolveResult:
        """使用网格搜索方法求解价格（适用于复杂条件）"""
        
        # 在收盘价附近进行网格搜索
        base_close = initial_guess["close"]
        search_range = base_close * 0.2  # 搜索范围：±20%
        
        close_prices = np.linspace(
            max(price_bounds.close_min, base_close - search_range),
            min(price_bounds.close_max, base_close + search_range),
            50
        )
        
        best_error = float('inf')
        best_prices = initial_guess
        best_actual_value = 0.0
        iterations = 0
        
        for close_price in close_prices:
            try:
                # 按比例调整其他价格
                ratio = close_price / base_close
                prices = {
                    "open": initial_guess["open"] * ratio,
                    "high": initial_guess["high"] * ratio,
                    "low": initial_guess["low"] * ratio,
                    "close": close_price
                }
                
                full_data = self._build_full_data(historical_data, prices)
                actual_value = self._evaluate_condition(condition, full_data)
                error = abs(actual_value - target_value)
                
                if error < best_error:
                    best_error = error
                    best_prices = prices
                    best_actual_value = actual_value
                
                iterations += 1
                
            except Exception:
                continue
        
        return SolveResult(
            success=best_error < self.tolerance,
            prices=best_prices,
            target_value=target_value,
            actual_value=best_actual_value,
            error=best_error,
            iterations=iterations,
            method="grid_search",
            message="网格搜索完成"
        )
    
    def _build_full_data(self, historical_data: pd.DataFrame, 
                        new_prices: Dict[str, float]) -> pd.DataFrame:
        """构造包含新价格的完整历史数据"""
        full_data = historical_data.copy()
        
        # 构造模拟明日数据行
        new_row = {
            "ts_code": historical_data.iloc[-1]["ts_code"] if not historical_data.empty else "000001.SZ",
            "trade_date": "20241201",  # 占位日期
            "open": new_prices["open"],
            "high": new_prices["high"],
            "low": new_prices["low"],
            "close": new_prices["close"],
            "vol": historical_data.iloc[-1]["vol"] if not historical_data.empty else 1000000
        }
        
        new_df = pd.DataFrame([new_row])
        full_data = pd.concat([full_data, new_df], ignore_index=True)
        
        return full_data
    
    def _evaluate_condition(self, condition: str, data: pd.DataFrame) -> float:
        """评估技术指标条件，返回指标数值"""
        try:
            if hasattr(tdx, 'evaluate') and ind:
                indicator_name = self._extract_indicator_name(condition)
                
                if indicator_name and indicator_name in ind.REGISTRY:
                    meta = ind.REGISTRY[indicator_name]
                    if meta.py_func:
                        result = meta.py_func(data, **(meta.kwargs or {}))
                        
                        if isinstance(result, pd.Series):
                            return float(result.iloc[-1])
                        elif isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                if pd.api.types.is_numeric_dtype(result[col]):
                                    return float(result[col].iloc[-1])
                        elif isinstance(result, (int, float)):
                            return float(result)
                
                return self._calculate_simple_indicator(condition, data)
            else:
                return self._calculate_simple_indicator(condition, data)
                
        except Exception as e:
            logger.debug(f"指标计算失败: {e}")
            return 0.0
    
    def _extract_indicator_name(self, condition: str) -> Optional[str]:
        """从条件表达式中提取指标名称"""
        condition = condition.strip().lower()
        
        indicator_map = {
            'j': 'kdj',
            'kdj': 'kdj', 
            'rsi': 'rsi',
            'ma': 'ma',
            'macd': 'macd',
            'diff': 'macd',
            'boll': 'boll',
            'cci': 'cci'
        }
        
        return indicator_map.get(condition, condition)
    
    def _calculate_simple_indicator(self, condition: str, data: pd.DataFrame) -> float:
        """计算简单技术指标"""
        try:
            if 'j' in condition.lower():
                return self._calculate_kdj_j(data)
            elif 'rsi' in condition.lower():
                return self._calculate_rsi(data)
            elif 'ma' in condition.lower():
                return self._calculate_ma(data)
            elif 'diff' in condition.lower():
                return self._calculate_diff(data)
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"简单指标计算失败: {e}")
            return 0.0
    
    def _calculate_kdj_j(self, data: pd.DataFrame) -> float:
        """计算KDJ-J值"""
        try:
            from indicators import kdj
            j_values = kdj(data)
            return float(j_values.iloc[-1]) if not j_values.empty else 50.0
        except Exception as e:
            logger.debug(f"KDJ-J计算失败: {e}")
            return 50.0
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """计算RSI值"""
        try:
            from indicators import rsi
            rsi_values = rsi(data['close'], period=period)
            return float(rsi_values.iloc[-1]) if not rsi_values.empty else 50.0
        except Exception as e:
            logger.debug(f"RSI计算失败: {e}")
            return 50.0
    
    def _calculate_ma(self, data: pd.DataFrame, period: int = 20) -> float:
        """计算移动平均"""
        try:
            from tdx_compat import MA
            ma_values = MA(data['close'], period)
            return float(ma_values.iloc[-1]) if not ma_values.empty else float(data['close'].iloc[-1])
        except Exception as e:
            logger.debug(f"MA计算失败: {e}")
            return float(data['close'].iloc[-1])
    
    def _calculate_diff(self, data: pd.DataFrame) -> float:
        """计算DIFF指标"""
        try:
            from indicators import macd_diff
            diff_values = macd_diff(data['close'])
            return float(diff_values.iloc[-1]) if not diff_values.empty else 0.0
        except Exception as e:
            logger.debug(f"DIFF计算失败: {e}")
            return 0.0


@dataclass
class Scenario:
    """
    明日情景：支持统一默认 + 个股覆盖。
    所有百分比传入 **百分比单位**（+2.5 表示 +2.5%）。
    """
    # 价格假设
    mode: Literal["close_pct", "open_pct", "gap_then_close_pct", "limit_up", "limit_down", "flat", "reverse_indicator"] = "close_pct"
    pct: float = 0.0                 # 涨跌幅（%），用于 close_pct / open_pct / gap_then_close_pct 的收盘段
    gap_pct: float = 0.0             # 缺口（%），用于 gap_then_close_pct：开盘=昨收*(1+gap_pct)
    # 高低点生成
    hl_mode: Literal["follow", "atr_like", "range_pct"] = "follow"
    range_pct: float = 1.5           # 当日高低振幅（%），仅当 hl_mode="range_pct" 生效
    atr_mult: float = 1.0            # "类 ATR"倍数（从近 N 日高低均值估略），hl_mode="atr_like"
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
    
    # 反推模式参数（当 mode="reverse_indicator" 时生效）
    reverse_indicator: str = ""           # 指标名称，如 "j", "rsi", "ma"
    reverse_target_value: float = 0.0     # 目标指标值
    reverse_method: str = "optimize"      # 求解方法：optimize, binary_search, grid_search
    reverse_tolerance: float = 1e-6       # 求解精度
    reverse_max_iterations: int = 1000    # 最大迭代次数

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
    
    # 优先使用缓存读取（仅对单只股票）
    if len(codes) == 1:
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            df = query_stock_data(
                db_path=db_path,
                ts_code=codes[0],
                start_date=start,
                end_date=ref_date,
                adj_type="qfq"
            )
            if not df.empty:
                df["trade_date"] = df["trade_date"].astype(str)
                return df
        except:
            pass
    
    if scan_with_duckdb:
        df = scan_with_duckdb(root, None, start, ref_date, columns=cols)
    else:
        if read_range is None:
            raise RuntimeError("缺少 scan_with_duckdb/read_range 任一读取函数")
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
                
            if ref_date:
                conditions.append("trade_date <= ?")
                params.append(ref_date)
                
            if adj:
                conditions.append("adj_type = ?")
                params.append(adj)
            
            # 构建SQL查询
            select_cols = "*" if not cols else ", ".join(cols)
            sql = f"SELECT {select_cols} FROM stock_data"
            
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            
            sql += " ORDER BY ts_code, trade_date"
            
            # 执行查询
            logger.info(f"[数据库连接] 开始获取数据库管理器实例 (读取股票数据用于预测: codes={len(codes) if codes else 'all'}, {start}~{ref_date})")
            manager = get_database_manager()
            df = manager.execute_sync_query(db_path, sql, params, timeout=120.0)
        except Exception as e:
            logger.error(f"读取数据范围失败: {e}")
            df = pd.DataFrame()
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
    """基于上一日 latest 行（Series），生成"模拟日"一行"""
    prev_close = float(latest["close"])
    prev_open  = float(latest["open"])
    
    # 反推模式：使用价格求解器
    if scen.mode == "reverse_indicator":
        return _apply_reverse_scenario_row(latest, scen)
    
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


def _apply_reverse_scenario_row(latest: pd.Series, scen: Scenario) -> Dict[str, Any]:
    """反推模式：根据指标条件反推价格"""
    try:
        # 检查反推参数
        if not scen.reverse_indicator:
            # 如果没有反推指标，回退到普通模式
            return _apply_scenario_row(latest, Scenario(mode="close_pct", pct=0.0))
        
        # 构造历史数据（这里简化处理，实际应该传入完整历史数据）
        historical_data = pd.DataFrame([{
            "ts_code": latest.get("ts_code", "000001.SZ"),
            "trade_date": latest.get("trade_date", "20241201"),
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "vol": float(latest.get("vol", 1000000))
        }])
        
        # 设置价格约束
        prev_close = float(latest["close"])
        price_bounds = PriceBounds(
            open_min=prev_close * 0.5,
            open_max=prev_close * 2.0,
            high_min=prev_close * 0.5,
            high_max=prev_close * 2.0,
            low_min=prev_close * 0.5,
            low_max=prev_close * 2.0,
            close_min=prev_close * 0.5,
            close_max=prev_close * 2.0
        )
        
        # 创建求解器
        solver = PriceSolver(
            max_iterations=scen.reverse_max_iterations,
            tolerance=scen.reverse_tolerance,
            verbose=False
        )
        
        # 求解价格 - 简化：直接使用指标名称
        result = solver.solve_price(
            condition=scen.reverse_indicator,  # 直接使用指标名称
            target_value=scen.reverse_target_value,
            historical_data=historical_data,
            price_bounds=price_bounds,
            method=scen.reverse_method
        )
        
        if result.success:
            # 使用求解出的价格
            O = result.prices["open"]
            H = result.prices["high"]
            L = result.prices["low"]
            C = result.prices["close"]
        else:
            # 求解失败，回退到普通模式
            _emit("pred_warning", message=f"反推求解失败: {result.message}，回退到普通模式")
            return _apply_scenario_row(latest, Scenario(mode="close_pct", pct=0.0))
        
        # 应用约束
        if scen.lock_higher_than_open:
            C = max(C, O)
        
        if scen.lock_inside_day:
            H = max(H, O, C)
            L = min(L, O, C)
        
        # 成交量处理（保持原有逻辑）
        V_prev = float(latest.get("vol", 0.0) or 0.0)
        if scen.vol_mode == "same":
            V = V_prev
        elif scen.vol_mode == "pct":
            V = V_prev * (1.0 + float(scen.vol_arg)/100.0)
        else:  # mult
            vmult = (float(scen.vol_arg) if float(scen.vol_arg)!=0 else 1.0)
            V = V_prev * vmult
        
        return {"open": float(O), "high": float(H), "low": float(L), "close": float(C), "vol": float(max(V, 0.0))}
        
    except Exception as e:
        # 其他异常，回退到普通模式
        _emit("pred_warning", message=f"反推模式异常: {str(e)}，回退到普通模式")
        return _apply_scenario_row(latest, Scenario(mode="close_pct", pct=0.0))


def _apply_reverse_scenario_row_with_history(latest: pd.Series, scen: Scenario, historical_data: pd.DataFrame) -> Dict[str, Any]:
    """反推模式：使用完整历史数据根据指标条件反推价格"""
    try:
        # 检查反推参数
        if not scen.reverse_indicator:
            # 如果没有反推指标，回退到普通模式
            return _apply_scenario_row(latest, Scenario(mode="close_pct", pct=0.0))
        
        # 设置价格约束
        prev_close = float(latest["close"])
        price_bounds = PriceBounds(
            open_min=prev_close * 0.5,
            open_max=prev_close * 2.0,
            high_min=prev_close * 0.5,
            high_max=prev_close * 2.0,
            low_min=prev_close * 0.5,
            low_max=prev_close * 2.0,
            close_min=prev_close * 0.5,
            close_max=prev_close * 2.0
        )
        
        # 创建求解器
        solver = PriceSolver(
            max_iterations=scen.reverse_max_iterations,
            tolerance=scen.reverse_tolerance,
            verbose=False
        )
        
        # 求解价格 - 简化：直接使用指标名称
        result = solver.solve_price(
            condition=scen.reverse_indicator,  # 直接使用指标名称
            target_value=scen.reverse_target_value,
            historical_data=historical_data,
            price_bounds=price_bounds,
            method=scen.reverse_method
        )
        
        if result.success:
            # 使用求解出的价格
            O = result.prices["open"]
            H = result.prices["high"]
            L = result.prices["low"]
            C = result.prices["close"]
        else:
            # 求解失败，回退到普通模式
            _emit("pred_warning", message=f"反推求解失败: {result.message}，回退到普通模式")
            return _apply_scenario_row(latest, Scenario(mode="close_pct", pct=0.0))
        
        # 应用约束
        if scen.lock_higher_than_open:
            C = max(C, O)
        
        if scen.lock_inside_day:
            H = max(H, O, C)
            L = min(L, O, C)
        
        # 成交量处理（保持原有逻辑）
        V_prev = float(latest.get("vol", 0.0) or 0.0)
        if scen.vol_mode == "same":
            V = V_prev
        elif scen.vol_mode == "pct":
            V = V_prev * (1.0 + float(scen.vol_arg)/100.0)
        else:  # mult
            vmult = (float(scen.vol_arg) if float(scen.vol_arg)!=0 else 1.0)
            V = V_prev * vmult
        
        return {"open": float(O), "high": float(H), "low": float(L), "close": float(C), "vol": float(max(V, 0.0))}
        
    except Exception as e:
        # 其他异常，回退到普通模式
        _emit("pred_warning", message=f"反推模式异常: {str(e)}，回退到普通模式")
        return _apply_scenario_row(latest, Scenario(mode="close_pct", pct=0.0))


# ---------------- 主流程函数 ----------------
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
    模拟明日股票数据并返回完整结果
    
    Args:
        ref_date: 参考日期
        universe_codes: 股票代码列表
        scenario: 模拟情景配置
        per_stock: 个股参数覆盖
        base: 数据根目录
        adj: 复权方式
        asset: 资产类型
        recompute_indicators: 指标重算配置
        out_dir: 输出目录
        cache: 缓存后端
        indicator_runner: 自定义指标计算器
    
    Returns:
        SimResult: 模拟结果
    """
    base = base or getattr(cfg, "DATA_ROOT", ".")
    adj  = adj or getattr(cfg, "API_ADJ", "qfq")
    per_stock = per_stock or PerStockOverride()

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
    
    logger.info(f"开始明日模拟: 参考日={ref_date}, 股票数={len(universe_codes)}, 情景={scenario.mode}")
    log_data_processing("predict_core", "明日模拟", (len(universe_codes),), None, True)

    # 尝试读取缓存
    if cache is not None:
        try:
            cached = cache.get(cache_key)
            if cached is not None and not cached.empty:
                df_all = cached.copy()
                df_sim = df_all[df_all["trade_date"].astype(str) == str(sim_date)].copy()
                logger.info(f"从缓存加载模拟结果: 行数={len(df_sim)}")
                return SimResult(ref_date=ref_date, sim_date=sim_date, df_sim=df_sim, df_concat=df_all)
        except Exception as e:
            logger.debug(f"缓存读取失败: {e}")

    # 加载历史数据
    logger.info(f"加载历史数据: 参考日={ref_date}, 预热天数={scenario.warmup_days}")
    df_hist = _load_hist_window(base, adj, asset, universe_codes, ref_date, scenario.warmup_days)
    
    if df_hist.empty:
        logger.warning("历史数据为空，返回空结果")
        empty = pd.DataFrame(columns=["ts_code","trade_date","open","high","low","close","vol"])
        return SimResult(ref_date=ref_date, sim_date=sim_date, df_sim=empty, df_concat=empty)

    # 生成模拟数据
    last_rows = df_hist.groupby("ts_code").tail(1).reset_index(drop=True)
    total_stocks = len(last_rows)
    recs: List[Dict[str, Any]] = []
    
    logger.info(f"开始生成模拟数据: 股票数={total_stocks}")
    
    for i, (_, row) in enumerate(last_rows.iterrows()):
        ts = str(row["ts_code"])
        scen = per_stock.for_ts(ts, scenario)
        
        try:
            if scen.mode == "reverse_indicator":
                stock_hist = df_hist[df_hist["ts_code"].astype(str) == ts].copy()
                vals = _apply_reverse_scenario_row_with_history(row, scen, stock_hist)
            else:
                vals = _apply_scenario_row(row, scen)
                
            recs.append({
                "ts_code": ts,
                "trade_date": sim_date,
                **vals
            })
            
            if (i + 1) % 100 == 0:
                logger.debug(f"模拟进度: {i+1}/{total_stocks} ({ts})")
                
        except Exception as e:
            logger.error(f"股票{ts}模拟失败: {e}", exc_info=True)
            # 使用默认值
            recs.append({
                "ts_code": ts,
                "trade_date": sim_date,
                "open": float(row["close"]),
                "high": float(row["close"]),
                "low": float(row["close"]),
                "close": float(row["close"]),
                "vol": float(row.get("vol", 0))
            })
    
    df_sim = pd.DataFrame.from_records(recs, columns=["ts_code","trade_date","open","high","low","close","vol"])
    logger.info(f"模拟数据生成完成: 行数={len(df_sim)}")

    # 拼接历史数据和模拟数据
    df_all = pd.concat([df_hist, df_sim], ignore_index=True)
    df_all = df_all.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    logger.debug(f"数据拼接完成: 总行数={len(df_all)}")

    # 重算技术指标
    if indicator_runner is not None:
        logger.info("使用自定义指标计算器")
        df_all = indicator_runner(df_all.copy(), recompute_indicators)
    elif recompute_indicators != "none" and hasattr(ind, "REGISTRY"):
        if recompute_indicators == "all":
            names = [k for k,v in getattr(ind, "REGISTRY", {}).items() if getattr(v, "py_func", None)]
        else:
            names = [n for n in recompute_indicators if n in getattr(ind, "REGISTRY", {}) and getattr(ind.REGISTRY[n], "py_func", None)]
        
        if names:
            logger.info(f"开始重算指标: {', '.join(names)}")
            parts: List[pd.DataFrame] = []
            stock_groups = list(df_all.groupby("ts_code"))
            total_stocks = len(stock_groups)
            
            for i, (ts, sub) in enumerate(stock_groups):
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
                    except Exception as e:
                        # 指标计算失败影响数据完整性，直接抛出异常
                        error_msg = f"指标{name}计算失败（股票{ts}）: {e}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg) from e
                parts.append(sub)
                
                if (i + 1) % 100 == 0:
                    logger.debug(f"指标计算进度: {i+1}/{total_stocks} ({ts})")
            
            df_all = pd.concat(parts, ignore_index=True).sort_values(["ts_code","trade_date"]).reset_index(drop=True)
            logger.info(f"指标重算完成: 总行数={len(df_all)}")

    # 保存结果文件
    if out_dir:
        out_base = Path(out_dir) / sim_date
        out_base.mkdir(parents=True, exist_ok=True)
        fpath = out_base / f"sim_{sim_date}.csv"
        try:
            df_all.to_csv(fpath, index=False, encoding="utf-8-sig")
            logger.info(f"结果已保存: {fpath}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}", exc_info=True)

    # 写入缓存
    if cache is not None:
        try:
            cache.set(cache_key, df_all)
            logger.debug("结果已写入缓存")
        except Exception as e:
            logger.debug(f"缓存写入失败: {e}")

    logger.info(f"明日模拟完成: 模拟日={sim_date}, 股票数={len(df_sim)}")
    return SimResult(ref_date=ref_date, sim_date=sim_date, df_sim=df_sim, df_concat=df_all)

# ---------------- 规则判定接口 ----------------

def _build_eval_ctx(sub: pd.DataFrame) -> pd.DataFrame:
    """构造表达式计算上下文：注入所有数值列和指标数据"""
    ctx = {}
    skip = {"ts_code", "trade_date"}
    
    # 添加数值列到上下文
    for col in sub.columns:
        if col in skip:
            continue
        s = pd.to_numeric(sub[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        arr = s.values
        name = str(col)
        up = name.upper()
        if name not in ctx:
            ctx[name] = arr
        if up not in ctx:
            ctx[up] = arr
    
    # 添加价格简写别名
    alias = {"O": "open", "H": "high", "L": "low", "C": "close", "V": "vol"}
    for k, base in alias.items():
        if k not in ctx and base in sub.columns:
            ctx[k] = pd.to_numeric(sub[base], errors="coerce").values
    
    # 确保KDJ指标可用
    def _ensure_alias(dst_name: str, cand_names: list[str]):
        if dst_name in ctx:
            return
        for c in cand_names:
            if c in sub.columns:
                arr = pd.to_numeric(sub[c], errors="coerce").values
                ctx[dst_name] = arr
                ctx[dst_name.lower()] = arr
                ctx[dst_name.upper()] = arr
                return
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
    评估股票是否满足指定条件
    
    Args:
        df_concat: 历史+模拟完整数据
        sim_date: 目标判断日期
        exprs: 规则表达式字典
        for_ts_codes: 指定股票代码列表
        extra_ctx: 额外上下文变量
    
    Returns:
        DataFrame: 规则判定结果
    """
    if not hasattr(tdx, "evaluate"):
        raise RuntimeError("tdx_compat.evaluate 不可用")

    codes = sorted(set(for_ts_codes or df_concat["ts_code"].astype(str).unique().tolist()))
    rows: List[Dict[str, Any]] = []
    
    logger.debug(f"开始规则判定: 股票数={len(codes)}, 规则数={len(exprs)}")
    
    for ts in codes:
        sub = df_concat[df_concat["ts_code"].astype(str) == ts].copy()
        sub = sub.sort_values("trade_date")
        if str(sim_date) not in set(sub["trade_date"].astype(str)):
            continue
            
        ctx_df = _build_eval_ctx(sub)
        tdx.EXTRA_CONTEXT.update({"TS": ts, "REF_DATE": str(sim_date)})
        if extra_ctx:
            tdx.EXTRA_CONTEXT.update(extra_ctx)
            
        res = {}
        for name, code in exprs.items():
            try:
                extra_ctx = {}
                for col in ctx_df.columns:
                    if col.isidentifier():
                        extra_ctx[col] = ctx_df[col]
                        extra_ctx[col.upper()] = ctx_df[col]
                
                out = tdx.evaluate(code, ctx_df, extra_context=extra_ctx)
                sig = out.get("sig") if isinstance(out, dict) else None
                if sig is None and isinstance(out, dict):
                    sig = out.get("SIG") or out.get("last_expr")
                if sig is None and isinstance(out, dict):
                    for v in out.values():
                        if isinstance(v, (list, np.ndarray, pd.Series)) and len(v)==len(ctx_df):
                            sig = v
                            break
                
                ok = False
                if isinstance(sig, (list, np.ndarray, pd.Series)) and len(sig)>0:
                    ok = bool(pd.Series(sig).iloc[-1])
                res[name] = ok
            except Exception as e:
                logger.debug(f"规则{name}计算失败: {e}")
                res[name] = False
        
        res["ts_code"] = ts
        rows.append(res)

    out_df = pd.DataFrame(rows).set_index("ts_code") if rows else pd.DataFrame(columns=list(exprs.keys()))
    logger.debug(f"规则判定完成: 结果行数={len(out_df)}")
    return out_df

# ---------------- KDJ指标计算 ----------------
def _fallback_kdj(df: pd.DataFrame) -> pd.DataFrame:
    """KDJ指标兜底计算"""
    try:
        from indicators import kdj
        j_values = kdj(df)
        return pd.DataFrame({
            "k": j_values / 3,
            "d": j_values / 3,
            "j": j_values
        })
    except Exception as e:
        logger.debug(f"KDJ计算失败: {e}")
        return pd.DataFrame({
            "k": pd.Series(0.0, index=df.index),
            "d": pd.Series(0.0, index=df.index),
            "j": pd.Series(0.0, index=df.index)
        })


def _extract_last_j_for_each(df_concat: pd.DataFrame, sim_date: str) -> pd.Series:
    """提取各股票在指定日期的J值"""
    out: Dict[str, float] = {}
    for ts, sub in df_concat.groupby("ts_code"):
        sub = sub.sort_values("trade_date")
        cols = [c for c in sub.columns if str(c).lower() in {"j","kdj_j"}]
        if cols:
            s = pd.to_numeric(sub.loc[sub["trade_date"].astype(str)==str(sim_date), cols[0]], errors="coerce")
            if not s.empty and pd.notna(s.iloc[-1]):
                out[str(ts)] = float(s.iloc[-1])
                continue
        kdj = _fallback_kdj(sub)
        out[str(ts)] = float(kdj["j"].iloc[-1])
    return pd.Series(out, name="J")

# ---------------- 主入口函数 ----------------
@dataclass
class PredictionInput:
    """预测输入参数"""
    ref_date: str
    universe: List[str]                       # 股票代码列表
    scenario: Scenario                        # 模拟情景配置
    rules: Optional[List[Dict[str, Any]]] = None   # 预测规则列表
    expr: Optional[str] = None
    use_rule_scenario: bool = False          # 使用规则自带情景
    recompute_indicators: Iterable[str] | Literal["all","none"] = ("kdj",)   # 重算指标
    cache_dir: Optional[str] = None          # 缓存目录


def run_prediction(inp: PredictionInput) -> pd.DataFrame:
    """
    执行股票预测分析
    
    Returns:
        DataFrame: 包含ts_code, rule_name, J, ref_date, sim_date, scenario_id, score, tiebreak_j, rank
    """
    results = []
    
    if inp.rules:
        logger.info(f"开始规则预测: 规则数={len(inp.rules)}")
        for rule in inp.rules:
            name = str(rule.get("name", "") or "").strip() or "<unnamed>"
            chk  = str(rule.get("check", "") or "").strip()
            scen = inp.scenario
            
            if inp.use_rule_scenario and isinstance(rule.get("scenario"), dict):
                scen = Scenario(**{**scen.__dict__, **dict(rule["scenario"])})
            
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
                
                if name in h.columns:
                    h = h[h[name] == True]  # noqa: E712
                results.append(h[["ts_code","rule_name","J","ref_date","sim_date","scenario_id"]])
            else:
                js = _extract_last_j_for_each(sim.df_concat, sim.sim_date)
                h = pd.DataFrame({"ts_code": sorted(set(sim.df_concat["ts_code"].astype(str)))})
                h["ref_date"] = sim.ref_date
                h["sim_date"] = sim.sim_date
                h["J"] = h["ts_code"].map(js.to_dict())
                h["scenario_id"] = scenario_hash(scen)
                h["rule_name"] = name
                results.append(h[["ts_code","rule_name","J","ref_date","sim_date","scenario_id"]])
                
            logger.debug(f"规则{name}处理完成")
    else:
        logger.info("开始表达式预测")
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
            out = pd.DataFrame({"ts_code": sorted(set(sim.df_concat["ts_code"].astype(str)))})
            out["rule_name"] = ""

        out["ref_date"] = sim.ref_date
        out["sim_date"] = sim.sim_date
        out["J"] = out["ts_code"].map(js.to_dict())
        out["scenario_id"] = scenario_hash(scen)
        results.append(out[["ts_code","rule_name","J","ref_date","sim_date","scenario_id"]])

    if not results:
        logger.warning("没有预测结果")
        return pd.DataFrame(columns=["ts_code","rule_name","J","ref_date","sim_date","scenario_id","score","tiebreak_j","rank"])
    
    df = pd.concat(results, ignore_index=True)
    
    if "rule_name" not in df.columns:
        df["rule_name"] = ""
        df = df[["ts_code","rule_name","J","ref_date","sim_date","scenario_id"]]
    
    # 合并得分数据
    try:
        score_path = Path("output/score/all") / f"score_all_{inp.ref_date}.csv"
        if score_path.exists() and score_path.stat().st_size > 0:
            df_score = pd.read_csv(score_path, dtype={"ts_code": str})
            df = df.merge(df_score[["ts_code", "score", "tiebreak_j", "rank"]], on="ts_code", how="left")
            df = df.sort_values(["score", "tiebreak_j", "ts_code"], 
                              ascending=[False, True, True], 
                              na_position="last").reset_index(drop=True)
            logger.debug(f"已合并得分数据: 行数={len(df)}")
        else:
            df["score"] = None
            df["tiebreak_j"] = None
            df["rank"] = None
    except Exception as e:
        logger.warning(f"读取得分文件失败: {e}")
        df["score"] = None
        df["tiebreak_j"] = None
        df["rank"] = None
    
    logger.info(f"预测完成: 结果行数={len(df)}")
    return df

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
        base = getattr(cfg, "DATA_ROOT", ".")
        adj = getattr(cfg, "API_ADJ", "qfq")
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

# ---------------- 内部：批量判定（明日模拟用） ----------------
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
    """加载持仓策略"""
    mod = _load_repo(py_path)
    if mod and hasattr(mod, "POSITION_POLICIES"):
        val = getattr(mod, "POSITION_POLICIES")
        if isinstance(val, list):
            logger.debug(f"加载持仓策略: {len(val)}条")
            return val
    return []


def load_opportunity_policies(py_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """加载机会策略"""
    mod = _load_repo(py_path)
    if mod and hasattr(mod, "OPPORTUNITY_POLICIES"):
        val = getattr(mod, "OPPORTUNITY_POLICIES")
        if isinstance(val, list):
            logger.debug(f"加载机会策略: {len(val)}条")
            return val
    return []

# 便捷函数
def solve_price_for_condition(condition: str,
                            target_value: float,
                            historical_data: pd.DataFrame,
                            method: str = "optimize",
                            **kwargs) -> SolveResult:
    """根据指标条件反推价格的便捷函数"""
    solver = PriceSolver(**kwargs)
    return solver.solve_price(condition, target_value, historical_data, method=method)


# 兼容性别名
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
    "PriceSolver",
    "PriceBounds",
    "SolveResult",
    "solve_price_for_condition",
]

# ---------------- 进度回调（已废弃，使用新日志系统） ----------------
_progress_handler = None
def set_progress_handler(fn):
    """设置进度处理器（已废弃，使用新的日志系统）"""
    global _progress_handler
    _progress_handler = fn
    # 不再输出警告，因为已经迁移到新的日志系统


def _emit(phase, current=None, total=None, message=None, **kw):
    """发送进度事件（已废弃）"""
    # 使用新的日志系统替代废弃的进度处理器
    if message:
        if total and current is not None:
            logger.info(f"[{phase}] {message} ({current}/{total})")
        else:
            logger.info(f"[{phase}] {message}")
    else:
        if total and current is not None:
            logger.info(f"[{phase}] 进度: {current}/{total}")
        else:
            logger.info(f"[{phase}] 执行中...")


def drain_progress_events():
    """清空进度事件队列（已废弃）"""
    pass
