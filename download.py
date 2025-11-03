#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载模块 - 基于database_manager的重构版本
弃用data_reader，只使用database_manager进行数据管理

新的下载流程：
1. 从database_manager获取数据库状态
2. 得到需要下载的数据参数传给tushare接口
3. 按照原样下载原始数据到内存数据库
4. 计算好指标，增量计算指标要warmup
5. 按照database_manager的规范打包数据并由database_manager统一写入
"""

from __future__ import annotations

import os
import sys
import time
import threading
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from tqdm import tqdm

# 忽略tushare的FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="tushare.pro.data_pro",
    message=".*fillna.*method.*deprecated.*"
)

# 导入配置和工具
from config import *
from log_system import get_logger
from database_manager import get_database_manager, DatabaseManager
from indicators import compute, warmup_for, REGISTRY, get_all_indicator_names, outputs_for
from utils import normalize_trade_date, normalize_ts
from tdx_compat import evaluate as tdx_eval

# 初始化日志
logger = get_logger("download")

# 导入tushare
try:
    import tushare as ts
except ImportError:
    logger.error("未安装 tushare: pip install tushare")
    sys.exit(1)

# ================= 配置和常量 =================

@dataclass
class DownloadConfig:
    """下载配置"""
    start_date: str
    end_date: str
    adj_type: str = "qfq"
    asset_type: str = "stock"  # "stock" or "index"
    threads: int = 8
    enable_warmup: bool = True
    retry_times: int = 3
    batch_size: int = 100
    rate_limit_calls_per_min: int = 500
    safe_calls_per_min: int = 490
    enable_adaptive_rate_limit: bool = True
    rate_limit_burst_capacity: int = 10
    warmup_batch_size: int = 50  # warmup查询的批处理大小

@dataclass
class DownloadStats:
    """下载统计"""
    total_stocks: int = 0
    success_count: int = 0
    skip_count: int = 0
    error_count: int = 0
    empty_count: int = 0
    failed_stocks: List[Tuple[str, str]] = None  # (ts_code, error_msg)
    
    def __post_init__(self):
        if self.failed_stocks is None:
            self.failed_stocks = []

# ================= 限频器 =================

class RateLimiter:
    """高级限频器，支持多种限频策略"""
    
    def __init__(self, calls_per_min: int = 500, safe_calls_per_min: int = 490, 
                 adaptive: bool = True, burst_capacity: int = 10):
        self.calls_per_min = calls_per_min
        self.safe_calls_per_min = safe_calls_per_min
        self.adaptive = adaptive
        self.burst_capacity = burst_capacity
        
        # 调用时间记录
        self._call_times = []
        self._lock = threading.Lock()
        
        # 自适应参数
        self._current_limit = safe_calls_per_min
        self._error_count = 0
        self._success_count = 0
        self._last_adjustment = time.time()
        
        # 统计信息
        self._total_calls = 0
        self._total_wait_time = 0.0
        self._rate_limit_hits = 0
    
    def wait_if_needed(self):
        """检查是否需要等待，如果需要则等待"""
        with self._lock:
            now = time.time()
            
            # 清理1分钟前的调用记录
            self._call_times = [t for t in self._call_times if now - t < 60]
            
            # 检查是否需要等待
            if len(self._call_times) >= self._current_limit:
                # 计算需要等待的时间
                oldest_call = self._call_times[0]
                sleep_time = 60 - (now - oldest_call) + 0.1
                
                if sleep_time > 0:
                    self._total_wait_time += sleep_time
                    self._rate_limit_hits += 1
                    time.sleep(sleep_time)
                    
                    # 重新清理
                    now = time.time()
                    self._call_times = [t for t in self._call_times if now - t < 60]
            
            # 记录本次调用
            self._call_times.append(now)
            self._total_calls += 1
    
    def record_success(self):
        """记录成功调用"""
        with self._lock:
            self._success_count += 1
            if self.adaptive:
                self._adjust_limit_after_success()
    
    def record_error(self, error_type: str = "unknown"):
        """记录错误调用"""
        with self._lock:
            self._error_count += 1
            if self.adaptive:
                self._adjust_limit_after_error(error_type)
    
    def _adjust_limit_after_success(self):
        """成功调用后调整限频"""
        now = time.time()
        
        # 每30秒调整一次
        if now - self._last_adjustment < 30:
            return
        
        # 如果最近成功率很高，可以适当提高限频
        if self._success_count > 50 and self._error_count < 5:
            if self._current_limit < self.calls_per_min * 0.95:
                self._current_limit = min(self._current_limit + 5, int(self.calls_per_min * 0.95))
                logger.debug(f"限频器自适应调整: 提高限频到 {self._current_limit}")
        
        self._last_adjustment = now
    
    def _adjust_limit_after_error(self, error_type: str):
        """错误调用后调整限频"""
        now = time.time()
        
        # 每30秒调整一次
        if now - self._last_adjustment < 30:
            return
        
        # 如果是限频错误，降低限频
        if "limit" in error_type.lower() or "quota" in error_type.lower():
            self._current_limit = max(self._current_limit - 10, int(self.safe_calls_per_min * 0.8))
            logger.warning(f"限频器自适应调整: 降低限频到 {self._current_limit}")
        
        self._last_adjustment = now
    
    def get_stats(self) -> Dict[str, Any]:
        """获取限频器统计信息"""
        with self._lock:
            now = time.time()
            recent_calls = len([t for t in self._call_times if now - t < 60])
            
            return {
                "current_limit": self._current_limit,
                "recent_calls_per_min": recent_calls,
                "total_calls": self._total_calls,
                "success_count": self._success_count,
                "error_count": self._error_count,
                "total_wait_time": self._total_wait_time,
                "rate_limit_hits": self._rate_limit_hits,
                "success_rate": self._success_count / max(self._total_calls, 1) * 100
            }
    
    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self._total_calls = 0
            self._success_count = 0
            self._error_count = 0
            self._total_wait_time = 0.0
            self._rate_limit_hits = 0

# ================= Tushare接口管理 =================

class TushareManager:
    """Tushare接口管理器"""
    
    def __init__(self, token: str = None, rate_limiter: RateLimiter = None):
        self.token = token or TOKEN
        self._pro = None
        self.rate_limiter = rate_limiter or RateLimiter()
        
        if not self.token or self.token.startswith("在这里"):
            raise ValueError("Tushare Pro 未配置：请在 config.TOKEN 中设置有效 token")
        
        # 设置token
        ts.set_token(self.token)
        self._pro = ts.pro_api()
        
        # 设置代理
        self._setup_proxy()
    
    def _setup_proxy(self):
        """设置代理配置"""
        # 不走系统代理：仅对本进程生效
        for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            os.environ.pop(k, None)
        
        # 设置直连域名
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,api.tushare.pro,github.com,raw.githubusercontent.com,pypi.org,files.pythonhosted.org,huggingface.co,cdn-lfs.huggingface.co")
        os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
    
    def _make_api_call(self, call_func, *args, **kwargs):
        """统一的API调用方法，包含限频和错误处理"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            self.rate_limiter.wait_if_needed()
            
            try:
                result = call_func(*args, **kwargs)
                self.rate_limiter.record_success()
                return result
            except Exception as e:
                error_msg = str(e)
                error_lower = error_msg.lower()
                
                # 详细记录错误信息
                logger.debug(f"Tushare API调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                logger.debug(f"调用参数: {kwargs}")
                
                # 分类错误类型
                if "limit" in error_lower or "quota" in error_lower or "频率" in error_lower:
                    self.rate_limiter.record_error("rate_limit")
                    logger.warning(f"API调用频率限制: {error_msg}")
                    # 频率限制错误，等待更长时间
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # 指数退避
                        logger.info(f"频率限制，等待 {wait_time:.1f} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                elif "token" in error_lower or "auth" in error_lower or "1002" in error_msg:
                    self.rate_limiter.record_error("auth_error")
                    logger.error(f"Token认证失败: {error_msg}")
                    # 认证错误不重试
                    break
                elif "network" in error_lower or "timeout" in error_lower or "connection" in error_lower:
                    self.rate_limiter.record_error("network_error")
                    logger.warning(f"网络连接问题: {error_msg}")
                    # 网络错误可以重试
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.info(f"网络问题，等待 {wait_time:.1f} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                else:
                    self.rate_limiter.record_error("api_error")
                    logger.error(f"API调用失败: {error_msg}")
                    # 其他错误可以重试
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.info(f"API错误，等待 {wait_time:.1f} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                
                # 如果所有重试都失败了，抛出异常
                if attempt == max_retries - 1:
                    logger.error(f"API调用最终失败，已重试 {max_retries} 次")
                    raise
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        try:
            return self._make_api_call(
                self._pro.stock_basic,
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,list_date'
            )
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise
    
    def get_stock_daily(self, ts_code: str, start_date: str, end_date: str, 
                       adj: str = None) -> pd.DataFrame:
        """获取股票日线数据（使用pro_bar接口获取原始数据）"""
        try:
            # 构建参数字典
            params = {
                'ts_code': ts_code,
                'start_date': start_date,
                'end_date': end_date,
                'freq': 'D',
                'asset': 'E'
            }
            
            # 只有当adj不为None且不是'raw'时才添加adj参数
            if adj and adj != 'raw':
                params['adj'] = adj
            
            return self._make_api_call(
                ts.pro_bar,
                **params
            )
        except Exception as e:
            logger.error(f"获取股票日线数据失败 {ts_code}: {e}")
            raise
    
    def get_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数日线数据（使用pro_bar接口获取原始数据）"""
        try:
            # 构建参数字典
            params = {
                'ts_code': ts_code,
                'start_date': start_date,
                'end_date': end_date,
                'freq': 'D',
                'asset': 'I'  # 指数
            }
            
            return self._make_api_call(
                ts.pro_bar,
                **params
            )
        except Exception as e:
            logger.error(f"获取指数日线数据失败 {ts_code}: {e}")
            raise
    
    def get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表（使用数据库管理器的统一方法）"""
        try:
            # 使用数据库管理器的统一方法
            return self.db_manager.get_trade_dates_from_tushare(start_date, end_date)
        except Exception as e:
            logger.error(f"获取交易日列表失败: {e}")
            raise
    
    def get_smart_end_date(self, end_date_config: str) -> str:
        """
        智能获取结束日期，考虑市场开盘时间和休盘情况（使用数据库管理器的统一方法）
        
        Args:
            end_date_config: 结束日期配置 ("today" 或具体日期)
            
        Returns:
            处理后的结束日期字符串
        """
        try:
            # 使用数据库管理器的统一方法
            return self.db_manager.get_smart_end_date(end_date_config)
        except Exception as e:
            logger.error(f"智能获取结束日期失败: {e}")
            # 如果智能获取失败，返回原配置或今天
            if end_date_config.lower() == "today":
                return datetime.now().strftime("%Y%m%d")
            return end_date_config

# ================= 数据处理器 =================

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.indicator_names = get_all_indicator_names()
    
    def clean_stock_data(self, df: pd.DataFrame, ts_code: str = None) -> pd.DataFrame:
        """清理股票数据"""
        if df is None or df.empty:
            return df
        
        # 复制数据
        df = df.copy()
        
        # 删除不需要的列
        columns_to_drop = ["pre_close", "change", "pct_chg"]
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # 数值化处理
        numeric_columns = ["open", "high", "low", "close", "vol", "amount"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 精度控制
        price_columns = ["open", "high", "low", "close", "vol", "amount"]
        for col in price_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(float).round(2)
        
        # 排序和去重
        df = df.sort_values("trade_date")
        df = df.drop_duplicates("trade_date", keep="last")
        
        # 规范化日期
        df = normalize_trade_date(df)
        
        return df
    
    def compute_indicators_with_warmup(self, df: pd.DataFrame, ts_code: str, 
                                     adj_type: str) -> pd.DataFrame:
        """计算指标（包含warmup机制）"""
        if df is None or df.empty:
            return df
        
        try:
            # 获取历史数据用于warmup
            historical_data = self._get_historical_data_for_warmup(ts_code, adj_type, df)
            
            if historical_data is not None and not historical_data.empty:
                # 合并历史数据和新增数据
                combined_data = pd.concat([historical_data, df], ignore_index=True)
                combined_data = combined_data.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
                
                # 计算指标
                df_with_indicators = compute(combined_data, self.indicator_names)
                
                # 只保留新增数据的部分
                df_with_indicators = df_with_indicators[
                    df_with_indicators['trade_date'] >= df['trade_date'].min()
                ].copy()
            else:
                # 没有历史数据，直接计算
                df_with_indicators = compute(df, self.indicator_names)
            
            # 精度控制
            from indicators import outputs_for
            decs = outputs_for(self.indicator_names)
            for col, n in decs.items():
                if col in df_with_indicators.columns and pd.api.types.is_numeric_dtype(df_with_indicators[col]):
                    df_with_indicators[col] = df_with_indicators[col].round(n)
            
            return df_with_indicators
            
        except Exception as e:
            # 指标计算失败影响数据完整性，直接抛出异常
            error_msg = f"指标计算失败 {ts_code}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _get_historical_data_for_warmup(self, ts_code: str, adj_type: str, 
                                       new_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """获取历史数据用于warmup"""
        try:
            # 计算需要的warmup天数
            warmup_days = warmup_for(self.indicator_names)
            if warmup_days <= 0:
                return None
            
            # 获取新数据的最早日期
            min_date = new_data['trade_date'].min()
            
            # 计算历史数据的结束日期（新数据开始日期的前一天）
            from datetime import datetime, timedelta
            min_date_obj = datetime.strptime(str(min_date), '%Y%m%d')
            end_date_obj = min_date_obj - timedelta(days=1)
            end_date = end_date_obj.strftime('%Y%m%d')
            
            # 计算历史数据的开始日期 - 减少查询范围
            start_date_obj = min_date_obj - timedelta(days=warmup_days)  # 只取必要的warmup数据
            start_date = start_date_obj.strftime('%Y%m%d')
            
            # 添加超时和重试机制
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    # 从数据库获取历史数据
                    historical_df = self.db_manager.query_stock_data(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        adj_type=adj_type
                    )
                    
                    if historical_df.empty:
                        return None
                    
                    # 确保数据格式一致
                    historical_df = self.clean_stock_data(historical_df, ts_code)
                    
                    logger.debug(f"获取历史数据用于warmup {ts_code}: {len(historical_df)} 行")
                    return historical_df
                    
                except Exception as e:
                    if attempt < max_retries and "timeout" in str(e).lower():
                        logger.warning(f"warmup查询超时 {ts_code} (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                        time.sleep(1.0 * (attempt + 1))  # 指数退避
                        continue
                    else:
                        logger.debug(f"获取历史数据失败 {ts_code}: {e}")
                        return None
            
        except Exception as e:
            logger.debug(f"获取历史数据失败 {ts_code}: {e}")
            return None
    
    def prepare_data_for_database(self, df: pd.DataFrame, adj_type: str) -> pd.DataFrame:
        """准备数据用于数据库写入"""
        if df is None or df.empty:
            return df
        
        # 添加复权类型列
        df = df.copy()
        df['adj_type'] = adj_type
        
        # 确保列顺序正确
        base_columns = ['ts_code', 'trade_date', 'adj_type', 'open', 'high', 'low', 'close', 'vol', 'amount']
        indicator_columns = [col for col in df.columns if col not in base_columns]
        df = df[base_columns + indicator_columns]
        
        return df

# ================= 下载器 =================

class StockDownloader:
    """股票数据下载器"""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        logger.info("[数据库连接] 开始获取数据库管理器实例 (初始化股票下载器)")
        self.db_manager = get_database_manager()
        
        # 创建限频器
        rate_limiter = RateLimiter(
            calls_per_min=config.rate_limit_calls_per_min,
            safe_calls_per_min=config.safe_calls_per_min,
            adaptive=config.enable_adaptive_rate_limit,
            burst_capacity=config.rate_limit_burst_capacity
        )
        
        self.tushare_manager = TushareManager(rate_limiter=rate_limiter)
        self.data_processor = DataProcessor(self.db_manager)
        self.stats = DownloadStats()
        self._lock = threading.Lock()
    
    def download_stock_data(self, ts_code: str) -> Tuple[str, str]:
        """下载单只股票数据"""
        try:
            # 下载原始数据
            logger.debug(f"开始下载 {ts_code} 的原始数据...")
            raw_data = self.tushare_manager.get_stock_daily(
                ts_code=ts_code,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                adj=self.config.adj_type if self.config.adj_type != "raw" else None
            )
            
            if raw_data is None or raw_data.empty:
                logger.debug(f"{ts_code} 无数据")
                return ts_code, "empty"
            
            logger.debug(f"{ts_code} 获取到 {len(raw_data)} 条原始数据")
            
            # 清理数据
            logger.debug(f"清理 {ts_code} 数据...")
            cleaned_data = self.data_processor.clean_stock_data(raw_data, ts_code)
            
            # 计算指标
            if self.config.enable_warmup:
                logger.debug(f"计算 {ts_code} 指标（包含warmup）...")
                data_with_indicators = self.data_processor.compute_indicators_with_warmup(
                    cleaned_data, ts_code, self.config.adj_type
                )
            else:
                logger.debug(f"计算 {ts_code} 指标...")
                data_with_indicators = compute(cleaned_data, self.data_processor.indicator_names)
            
            # 准备数据用于数据库写入
            logger.debug(f"准备 {ts_code} 数据写入数据库...")
            final_data = self.data_processor.prepare_data_for_database(
                data_with_indicators, self.config.adj_type
            )
            
            # 写入数据库
            logger.debug(f"写入 {ts_code} 数据到数据库...")
            self.db_manager.receive_stock_data(
                source_module="download",
                data=final_data,
                mode="upsert"  # 使用upsert模式避免重复数据
            )
            
            with self._lock:
                self.stats.success_count += 1
            
            logger.debug(f"{ts_code} 下载完成，共 {len(final_data)} 条记录")
            return ts_code, "success"
            
        except Exception as e:
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            
            # 如果是指标计算或数据库连接/写入错误，影响数据完整性，直接抛出异常
            if any(keyword in error_msg_lower for keyword in [
                "指标计算失败", "indicator", "计算失败", "compute",
                "数据库连接", "database connection", "connection", "connect",
                "数据库写入", "database write", "写入失败", "write",
                "数据库错误", "database error", "database", "query",
                "无法获取", "获取失败", "访问失败"
            ]):
                logger.error(f"下载股票数据失败（影响数据完整性）{ts_code}: {error_msg}")
                raise RuntimeError(f"下载股票数据失败（影响数据完整性）{ts_code}: {error_msg}") from e
            
            # 其他错误（如下载失败等）可以返回错误状态，不影响其他股票下载
            with self._lock:
                self.stats.error_count += 1
                self.stats.failed_stocks.append((ts_code, error_msg))
            
            logger.error(f"下载股票数据失败 {ts_code}: {error_msg}")
            return ts_code, f"error: {error_msg}"
    
    def download_all_stocks(self) -> DownloadStats:
        """下载所有股票数据"""
        logger.info("开始下载股票数据...")
        
        # 获取股票列表
        stock_list = self.tushare_manager.get_stock_list()
        stock_codes = stock_list['ts_code'].tolist()
        
        self.stats.total_stocks = len(stock_codes)
        logger.info(f"准备下载 {len(stock_codes)} 只股票的数据")
        
        # 创建进度条
        progress_bar = tqdm(
            total=len(stock_codes),
            desc="下载股票数据",
            unit="只",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # 并发下载
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(self.download_stock_data, ts_code): ts_code 
                for ts_code in stock_codes
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_code):
                ts_code = future_to_code[future]
                try:
                    result_code, status = future.result()
                    if status == "empty":
                        with self._lock:
                            self.stats.empty_count += 1
                    elif status == "success":
                        logger.debug(f"成功下载 {result_code}")
                    else:
                        logger.warning(f"下载失败 {result_code}: {status}")
                except Exception as e:
                    logger.error(f"处理任务异常 {ts_code}: {e}")
                    with self._lock:
                        self.stats.error_count += 1
                        self.stats.failed_stocks.append((ts_code, str(e)))
                
                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({
                    '成功': self.stats.success_count,
                    '空数据': self.stats.empty_count,
                    '失败': self.stats.error_count
                })
        
        # 关闭进度条
        progress_bar.close()
        
        # 打印统计信息
        logger.info(f"下载完成 - 成功: {self.stats.success_count}, 空数据: {self.stats.empty_count}, 失败: {self.stats.error_count}")
        
        # 打印限频器统计信息
        rate_stats = self.tushare_manager.rate_limiter.get_stats()
        logger.info(f"限频器统计 - 总调用: {rate_stats['total_calls']}, 成功率: {rate_stats['success_rate']:.1f}%, 等待时间: {rate_stats['total_wait_time']:.1f}s")
        
        return self.stats

class IndexDownloader:
    """指数数据下载器"""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        logger.info("[数据库连接] 开始获取数据库管理器实例 (初始化指数下载器)")
        self.db_manager = get_database_manager()
        
        # 创建限频器
        rate_limiter = RateLimiter(
            calls_per_min=config.rate_limit_calls_per_min,
            safe_calls_per_min=config.safe_calls_per_min,
            adaptive=config.enable_adaptive_rate_limit,
            burst_capacity=config.rate_limit_burst_capacity
        )
        
        self.tushare_manager = TushareManager(rate_limiter=rate_limiter)
        self.stats = DownloadStats()
        self._lock = threading.Lock()
    
    def download_index_data(self, ts_code: str) -> Tuple[str, str]:
        """下载单只指数数据"""
        try:
            # 下载原始数据
            logger.debug(f"开始下载 {ts_code} 的原始数据...")
            raw_data = self.tushare_manager.get_index_daily(
                ts_code=ts_code,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            
            if raw_data is None or raw_data.empty:
                logger.debug(f"{ts_code} 无数据")
                return ts_code, "empty"
            
            logger.debug(f"{ts_code} 获取到 {len(raw_data)} 条原始数据")
            
            # 清理数据
            logger.debug(f"清理 {ts_code} 数据...")
            df = raw_data.copy()
            
            # 删除不需要的列
            columns_to_drop = ["pre_close", "change", "pct_chg"]
            for col in columns_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            # 数值化处理
            numeric_columns = ["open", "high", "low", "close", "vol", "amount"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # 精度控制
            price_columns = ["open", "high", "low", "close", "vol", "amount"]
            for col in price_columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype(float).round(2)
            
            # 排序和去重
            df = df.sort_values("trade_date")
            df = df.drop_duplicates("trade_date", keep="last")
            
            # 规范化日期
            df = normalize_trade_date(df)
            
            # 添加复权类型（指数使用ind）
            df['adj_type'] = 'ind'
            
            # 确保列顺序正确
            base_columns = ['ts_code', 'trade_date', 'adj_type', 'open', 'high', 'low', 'close', 'vol', 'amount']
            df = df[base_columns]
            
            # 写入数据库
            logger.debug(f"写入 {ts_code} 数据到数据库...")
            self.db_manager.receive_stock_data(
                source_module="download",
                data=df,
                mode="upsert"
            )
            
            with self._lock:
                self.stats.success_count += 1
            
            logger.debug(f"{ts_code} 下载完成，共 {len(df)} 条记录")
            return ts_code, "success"
            
        except Exception as e:
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            
            # 如果是数据库连接/写入错误，影响数据完整性，直接抛出异常
            if any(keyword in error_msg_lower for keyword in [
                "数据库连接", "database connection", "connection", "connect",
                "数据库写入", "database write", "写入失败", "write",
                "数据库错误", "database error", "database", "query",
                "无法获取", "获取失败", "访问失败"
            ]):
                logger.error(f"下载指数数据失败（影响数据完整性）{ts_code}: {error_msg}")
                raise RuntimeError(f"下载指数数据失败（影响数据完整性）{ts_code}: {error_msg}") from e
            
            # 其他错误（如下载失败等）可以返回错误状态，不影响其他指数下载
            with self._lock:
                self.stats.error_count += 1
                self.stats.failed_stocks.append((ts_code, error_msg))
            
            logger.error(f"下载指数数据失败 {ts_code}: {error_msg}")
            return ts_code, f"error: {error_msg}"
    
    def download_all_indices(self, whitelist: List[str] = None) -> DownloadStats:
        """下载所有指数数据"""
        logger.info("开始下载指数数据...")
        
        # 使用白名单或默认指数列表
        if whitelist is None:
            whitelist = INDEX_WHITELIST
        
        self.stats.total_stocks = len(whitelist)
        logger.info(f"准备下载 {len(whitelist)} 只指数的数据")
        
        # 创建进度条
        progress_bar = tqdm(
            total=len(whitelist),
            desc="下载指数数据",
            unit="只",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # 并发下载
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(self.download_index_data, ts_code): ts_code 
                for ts_code in whitelist
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_code):
                ts_code = future_to_code[future]
                try:
                    result_code, status = future.result()
                    if status == "empty":
                        with self._lock:
                            self.stats.empty_count += 1
                    elif status == "success":
                        logger.debug(f"成功下载 {result_code}")
                    else:
                        logger.warning(f"下载失败 {result_code}: {status}")
                except Exception as e:
                    logger.error(f"处理任务异常 {ts_code}: {e}")
                    with self._lock:
                        self.stats.error_count += 1
                        self.stats.failed_stocks.append((ts_code, str(e)))
                
                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({
                    '成功': self.stats.success_count,
                    '空数据': self.stats.empty_count,
                    '失败': self.stats.error_count
                })
        
        # 关闭进度条
        progress_bar.close()
        
        # 打印统计信息
        logger.info(f"下载完成 - 成功: {self.stats.success_count}, 空数据: {self.stats.empty_count}, 失败: {self.stats.error_count}")
        
        # 打印限频器统计信息
        rate_stats = self.tushare_manager.rate_limiter.get_stats()
        logger.info(f"限频器统计 - 总调用: {rate_stats['total_calls']}, 成功率: {rate_stats['success_rate']:.1f}%, 等待时间: {rate_stats['total_wait_time']:.1f}s")
        
        return self.stats

# ================= 主下载器 =================

class DownloadManager:
    """下载管理器"""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        logger.info("[数据库连接] 开始获取数据库管理器实例 (初始化下载管理器)")
        self.db_manager = get_database_manager()
        self.tushare_manager = TushareManager()
        
        # 在数据库初始化之前判断是否为首次下载
        self.is_first_download = self._check_if_first_download()
        self._ensure_database_initialized()
    
    def _check_if_first_download(self) -> bool:
        """检查是否为首次下载（在数据库初始化之前调用）"""
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            # 如果数据库文件不存在，则为首次下载
            if not os.path.exists(db_path):
                logger.info("数据库不存在，判断为首次下载")
                return True
            
            # 使用数据库管理器获取最新日期
            latest_date = self.db_manager.get_latest_trade_date(db_path)
            if latest_date:
                logger.info(f"数据库存在且有数据，最新日期: {latest_date}，判断为增量下载")
                return False
            else:
                logger.info("数据库存在但无数据，判断为首次下载")
                return True
                
        except Exception as e:
            logger.warning(f"检查首次下载状态失败: {e}，假设为首次下载")
            return True
    
    def _ensure_database_initialized(self):
        """确保数据库已初始化"""
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            # 如果数据库不存在，初始化表结构
            if not os.path.exists(db_path):
                logger.info("数据库不存在，正在初始化...")
                self.db_manager.init_stock_data_tables(db_path, "duckdb")
                logger.info("数据库初始化完成")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def get_database_status(self) -> Dict[str, Any]:
        """获取数据库状态"""
        return self.db_manager.get_data_source_status()
    
    def get_smart_end_date(self, end_date_config: str) -> str:
        """
        智能获取结束日期，考虑市场开盘时间和休盘情况（使用数据库管理器的统一方法）
        
        Args:
            end_date_config: 结束日期配置 ("today" 或具体日期)
            
        Returns:
            处理后的结束日期字符串
        """
        return self.db_manager.get_smart_end_date(end_date_config)
    
    def determine_download_strategy(self) -> Dict[str, Any]:
        """确定下载策略"""
        # 使用预先判断的首次下载状态
        is_first_download = self.is_first_download
        
        # 获取最新数据日期
        latest_date = None
        if not is_first_download:
            try:
                # 从数据库获取最新日期
                latest_date = self.db_manager.get_latest_trade_date()
            except Exception as e:
                logger.warning(f"获取最新日期失败: {e}")
        else:
            # 首次下载，不需要查询数据库
            logger.info("首次下载，跳过数据库查询")
        
        # 确定实际下载的日期范围
        actual_start_date = self.config.start_date
        if not is_first_download and latest_date:
            # 增量下载：从最新日期的下一天开始
            from datetime import datetime, timedelta
            try:
                latest_date_obj = datetime.strptime(str(latest_date), '%Y%m%d')
                next_date_obj = latest_date_obj + timedelta(days=1)
                actual_start_date = next_date_obj.strftime('%Y%m%d')
                
                # 如果计算出的开始日期晚于配置的结束日期，则跳过下载
                if actual_start_date > self.config.end_date:
                    logger.info(f"数据已是最新，无需下载 (最新日期: {latest_date}, 配置结束日期: {self.config.end_date})")
                    return {"skip_download": True}
            except Exception as e:
                logger.warning(f"计算增量下载日期失败: {e}")
        
        strategy = {
            "is_first_download": is_first_download,
            "latest_date": latest_date,
            "start_date": actual_start_date,
            "end_date": self.config.end_date,
            "adj_type": self.config.adj_type,
            "asset_type": self.config.asset_type,
            "skip_download": False
        }
        
        return strategy
    
    def download_stocks(self, strategy: Dict[str, Any]) -> DownloadStats:
        """下载股票数据"""
        if strategy.get("skip_download", False):
            logger.info("跳过股票数据下载")
            return DownloadStats()
        
        stock_config = DownloadConfig(
            start_date=strategy["start_date"],
            end_date=strategy["end_date"],
            adj_type=strategy["adj_type"],
            asset_type="stock",
            threads=self.config.threads,
            enable_warmup=self.config.enable_warmup
        )
        
        downloader = StockDownloader(stock_config)
        return downloader.download_all_stocks()
    
    def download_indices(self, strategy: Dict[str, Any], whitelist: List[str] = None) -> DownloadStats:
        """下载指数数据"""
        if strategy.get("skip_download", False):
            logger.info("跳过指数数据下载")
            return DownloadStats()
        
        index_config = DownloadConfig(
            start_date=strategy["start_date"],
            end_date=strategy["end_date"],
            adj_type="ind",  # 指数使用ind
            asset_type="index",
            threads=self.config.threads,
            enable_warmup=False  # 指数不需要指标计算
        )
        
        downloader = IndexDownloader(index_config)
        return downloader.download_all_indices(whitelist)
    
    def run_download(self, assets: List[str] = None, index_whitelist: List[str] = None) -> Dict[str, DownloadStats]:
        """运行下载任务"""
        if assets is None:
            assets = ["stock"]
        
        if index_whitelist is None:
            index_whitelist = INDEX_WHITELIST
        
        # 确定下载策略
        strategy = self.determine_download_strategy()
        
        # 添加智能日期判断的详细说明
        if self.config.end_date.lower() == "today":
            smart_end_date = self.get_smart_end_date("today")
            logger.info("=" * 60)
            logger.info("智能日期判断说明")
            logger.info("=" * 60)
            logger.perf(f"• 配置的结束日期: today (今日)")
            logger.perf(f"• 智能判断后的实际结束日期: {smart_end_date}")
            logger.info("• 智能判断逻辑:")
            logger.info("  - 如果今天是交易日且当前时间 < 15:00，使用前一个交易日")
            logger.info("  - 如果今天是交易日且当前时间 >= 15:00，使用今天")
            logger.info("  - 如果今天不是交易日，使用最近的交易日")
            logger.info("  - 如果无法获取交易日历，使用今天")
            logger.info("=" * 60)
        
        # 显示中文下载策略信息
        if self.config.end_date.lower() == "today":
            smart_end_date = self.get_smart_end_date("today")
            logger.info(f"下载策略: 首次下载={strategy['is_first_download']}, 最新日期={strategy['latest_date']}, 开始日期={strategy['start_date']}, 结束日期={smart_end_date}, 复权类型={strategy['adj_type']}, 资产类型={strategy['asset_type']}, 跳过下载={strategy['skip_download']}")
        else:
            logger.info(f"下载策略: 首次下载={strategy['is_first_download']}, 最新日期={strategy['latest_date']}, 开始日期={strategy['start_date']}, 结束日期={strategy['end_date']}, 复权类型={strategy['adj_type']}, 资产类型={strategy['asset_type']}, 跳过下载={strategy['skip_download']}")
        
        # 检查是否需要跳过下载
        if strategy.get("skip_download", False):
            logger.info("数据已是最新，跳过下载")
            return {"stock": DownloadStats(), "index": DownloadStats()}
        
        results = {}
        total_assets = len(assets)
        current_asset = 0
        
        # 创建总体进度条
        overall_progress = tqdm(
            total=total_assets,
            desc="总体下载进度",
            unit="类",
            ncols=100,
            position=0,
            leave=True
        )
        
        try:
            # 下载股票数据
            if "stock" in assets:
                current_asset += 1
                overall_progress.set_description(f"下载股票数据 ({strategy['start_date']} - {strategy['end_date']})")
                logger.perf(f"开始下载股票数据: {strategy['start_date']} - {strategy['end_date']}")
                results["stock"] = self.download_stocks(strategy)
                overall_progress.update(1)
                overall_progress.set_postfix({
                    '股票成功': results["stock"].success_count,
                    '股票失败': results["stock"].error_count
                })
            
            # 下载指数数据
            if "index" in assets:
                current_asset += 1
                overall_progress.set_description(f"下载指数数据 ({strategy['start_date']} - {strategy['end_date']})")
                logger.perf(f"开始下载指数数据: {strategy['start_date']} - {strategy['end_date']}")
                results["index"] = self.download_indices(strategy, index_whitelist)
                overall_progress.update(1)
                overall_progress.set_postfix({
                    '指数成功': results["index"].success_count,
                    '指数失败': results["index"].error_count
                })
            
            # 显示最终统计
            overall_progress.set_description("下载完成")
            total_success = sum(stats.success_count for stats in results.values())
            total_error = sum(stats.error_count for stats in results.values())
            total_empty = sum(stats.empty_count for stats in results.values())
            
            overall_progress.set_postfix({
                '总成功': total_success,
                '总失败': total_error,
                '总空数据': total_empty
            })
            
        finally:
            overall_progress.close()
        
        # 打印最终统计信息
        logger.perf("=== 下载任务完成 ===")
        for asset_type, stats in results.items():
            logger.perf(f"{asset_type}: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
            if stats.failed_stocks:
                logger.warning(f"{asset_type} 失败股票: {[code for code, _ in stats.failed_stocks[:5]]}")
        
        logger.perf(f"总计: 成功={total_success}, 空数据={total_empty}, 失败={total_error}")
        
        return results

# ================= 便捷函数 =================

def download_data(start_date: str, end_date: str, adj_type: str = "qfq", 
                 assets: List[str] = None, threads: int = 8, 
                 enable_warmup: bool = True, enable_adaptive_rate_limit: bool = True) -> Dict[str, DownloadStats]:
    """
    便捷下载函数
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        adj_type: 复权类型 ("qfq", "hfq", "raw")
        assets: 资产类型列表 (["stock"], ["index"], ["stock", "index"])
        threads: 并发线程数
        enable_warmup: 是否启用指标warmup
        enable_adaptive_rate_limit: 是否启用自适应限频
    
    Returns:
        下载结果统计
    """
    if assets is None:
        assets = ["stock"]
    
    config = DownloadConfig(
        start_date=start_date,
        end_date=end_date,
        adj_type=adj_type,
        threads=threads,
        enable_warmup=enable_warmup,
        enable_adaptive_rate_limit=enable_adaptive_rate_limit
    )
    
    manager = DownloadManager(config)
    return manager.run_download(assets)

def main():
    """主函数"""
    # 获取配置
    start_date = START_DATE
    end_date = END_DATE
    adj_type = API_ADJ
    assets = ASSETS
    threads = STOCK_INC_THREADS
    
    logger.perf("=" * 60)
    logger.perf("开始下载数据")
    logger.perf("=" * 60)
    logger.perf(f"日期范围: {start_date} - {end_date}")
    logger.perf(f"复权类型: {adj_type}")
    logger.perf(f"资产类型: {', '.join(assets)}")
    logger.perf("=" * 60)
    
    try:
        # 执行下载
        results = download_data(
            start_date=start_date,
            end_date=end_date,
            adj_type=adj_type,
            assets=assets,
            threads=threads,
            enable_warmup=True
        )
        
        # 打印结果
        for asset_type, stats in results.items():
            logger.perf(f"{asset_type} 下载结果: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
            if stats.failed_stocks:
                logger.warning(f"{asset_type} 失败股票: {[code for code, _ in stats.failed_stocks[:10]]}")
        
        logger.perf("下载任务完成")
        
    except Exception as e:
        logger.error(f"下载任务失败: {e}")
        raise

def main_cli():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='股票数据下载工具')
    parser.add_argument('--start', type=str, default=START_DATE, help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end', type=str, default=END_DATE, help='结束日期 (YYYYMMDD)')
    parser.add_argument('--adj', type=str, default=API_ADJ, choices=['qfq', 'hfq', 'raw'], help='复权类型')
    parser.add_argument('--assets', nargs='+', default=ASSETS, choices=['stock', 'index'], help='资产类型')
    parser.add_argument('--threads', type=int, default=STOCK_INC_THREADS, help='并发线程数')
    parser.add_argument('--no-warmup', action='store_true', help='禁用指标warmup')
    parser.add_argument('--no-adaptive-rate-limit', action='store_true', help='禁用自适应限频')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    
    args = parser.parse_args()
    
    if args.interactive:
        main_interactive()
        return
    
    # 执行下载
    results = download_data(
        start_date=args.start,
        end_date=args.end,
        adj_type=args.adj,
        assets=args.assets,
        threads=args.threads,
        enable_warmup=not args.no_warmup,
        enable_adaptive_rate_limit=not args.no_adaptive_rate_limit
    )
    
    # 打印结果
    print("\n" + "=" * 30)
    print("下载结果")
    print("=" * 30)
    for asset_type, stats in results.items():
        print(f"{asset_type}: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
        if stats.failed_stocks:
            print(f"失败股票: {[code for code, _ in stats.failed_stocks[:5]]}")
    print("=" * 60)

def main_interactive():
    """交互式模式"""
    print("=" * 60)
    print("股票数据下载工具")
    print("=" * 60)
    print("基于database_manager的下载模块")
    print()
    
    # 获取用户输入
    start_date = input(f"开始日期 (默认: {START_DATE}): ").strip() or START_DATE
    end_date = input(f"结束日期 (默认: {END_DATE}): ").strip() or END_DATE
    
    # 如果用户输入了today，给出智能判断说明
    if end_date.lower() == "today":
        print("\n 智能日期判断说明:")
        print("   • 如果今天是交易日且当前时间 < 15:00，使用前一个交易日")
        print("   • 如果今天是交易日且当前时间 >= 15:00，使用今天")
        print("   • 如果今天不是交易日，使用最近的交易日")
        print("   • 如果无法获取交易日历，使用今天")
        print()
    
    adj_type = input(f"复权类型 (默认: {API_ADJ}): ").strip() or API_ADJ
    threads = input(f"并发线程数 (默认: {STOCK_INC_THREADS}): ").strip()
    threads = int(threads) if threads.isdigit() else STOCK_INC_THREADS
    
    # 限频器选项
    adaptive_rate_limit = input("启用自适应限频? (Y/n, 默认: Y): ").strip().lower()
    enable_adaptive_rate_limit = adaptive_rate_limit != 'n'
    
    # 选择资产类型
    print("\n资产类型选择:")
    print("1. 股票数据")
    print("2. 指数数据")
    print("3. 股票+指数")
    
    asset_choice = input("请选择 (1-3, 默认: 1): ").strip() or "1"
    assets = []
    if asset_choice in ['1', '3']:
        assets.append('stock')
    if asset_choice in ['2', '3']:
        assets.append('index')
    
    # 确认配置
    print(f"\n配置确认:")
    print(f"  开始日期: {start_date}")
    print(f"  结束日期: {end_date}")
    if end_date.lower() == "today":
        # 显示智能判断后的实际日期
        try:
            from database_manager import get_database_manager
            logger.info("[数据库连接] 开始获取数据库管理器实例 (获取智能结束日期)")
            db_manager = get_database_manager()
            smart_end_date = db_manager.get_smart_end_date("today")
            print(f"  智能判断后实际结束日期: {smart_end_date}")
        except:
            pass
    print(f"  复权类型: {adj_type}")
    print(f"  并发线程: {threads}")
    print(f"  自适应限频: {'是' if enable_adaptive_rate_limit else '否'}")
    print(f"  资产类型: {', '.join(assets)}")
    
    confirm = input("\n确认开始下载? (y/N): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    # 执行下载
    print("\n开始下载...")
    try:
        results = download_data(
            start_date=start_date,
            end_date=end_date,
            adj_type=adj_type,
            assets=assets,
            threads=threads,
            enable_warmup=True,
            enable_adaptive_rate_limit=enable_adaptive_rate_limit
        )
        
        # 打印结果
        print("\n=== 下载结果 ===")
        for asset_type, stats in results.items():
            print(f"{asset_type}: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
            if stats.failed_stocks:
                print(f"{asset_type} 失败股票: {[code for code, _ in stats.failed_stocks[:5]]}")
        
        print("\n下载完成!")
        
    except Exception as e:
        print(f"\n下载失败: {e}")
        logger.error(f"交互式下载失败: {e}")

# ================= 交易日相关函数 =================

def list_trade_dates(root: str) -> List[str]:
    """
    兼容现有系统的 list_trade_dates 函数
    
    Args:
        root: 根目录路径
        
    Returns:
        交易日期列表
    """
    try:
        if not os.path.exists(root):
            return []
        
        dates = []
        for item in os.listdir(root):
            if item.startswith("trade_date="):
                date_str = item.replace("trade_date=", "")
                if date_str.isdigit() and len(date_str) == 8:
                    dates.append(date_str)
        
        return sorted(dates)
        
    except Exception as e:
        logger.error(f"list_trade_dates 失败: {root}, 错误: {e}")
        return []

def get_trade_dates_from_tushare(start_date: str, end_date: str) -> List[str]:
    """
    从Tushare获取交易日列表的便捷函数（使用数据库管理器的统一方法）
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        
    Returns:
        交易日期列表
    """
    try:
        logger.info(f"[数据库连接] 开始获取数据库管理器实例 (从Tushare获取交易日列表: {start_date}~{end_date})")
        db_manager = get_database_manager()
        return db_manager.get_trade_dates_from_tushare(start_date, end_date)
    except Exception as e:
        logger.error(f"获取交易日列表失败: {e}")
        return []

def get_trade_dates_from_database(db_path: str = None) -> List[str]:
    """
    从数据库获取交易日列表的便捷函数（使用数据库管理器的统一方法）
    
    Args:
        db_path: 数据库路径，None时使用默认路径
        
    Returns:
        交易日期列表
    """
    try:
        logger.info(f"[数据库连接] 开始获取数据库管理器实例 (从数据库获取交易日列表: {db_path})")
        db_manager = get_database_manager()
        return db_manager.get_trade_dates(db_path)
    except Exception as e:
        logger.error(f"从数据库获取交易日列表失败: {e}")
        return []

def get_smart_end_date(end_date_config: str) -> str:
    """
    智能获取结束日期的便捷函数（使用数据库管理器的统一方法）
    
    Args:
        end_date_config: 结束日期配置 ("today" 或具体日期)
        
    Returns:
        处理后的结束日期字符串
    """
    try:
        logger.info(f"[数据库连接] 开始获取数据库管理器实例 (智能获取结束日期: {end_date_config})")
        db_manager = get_database_manager()
        return db_manager.get_smart_end_date(end_date_config)
    except Exception as e:
        logger.error(f"智能获取结束日期失败: {e}")
        # 如果智能获取失败，返回原配置或今天
        if end_date_config.lower() == "today":
            return datetime.now().strftime("%Y%m%d")
        return end_date_config

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_cli()
    else:
        main()
