#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1. 从database_manager获取数据库状态
2. 得到需要下载的数据参数传给tushare接口
3. 按照原样下载原始数据到内存数据库
4. 计算好指标，增量计算指标要warmup
5. 按照database_manager的规范打包数据并由database_manager统一写入
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
import importlib

# 在单独运行时自动切换到虚拟环境（支持 Linux/Windows）
def _bootstrap_venv():
    if __name__ != "__main__":
        return
    # 已在虚拟环境中则不处理
    if hasattr(sys, "real_prefix") or sys.prefix != sys.base_prefix:
        return

    project_root = Path(__file__).resolve().parent
    try:
        cfg = importlib.import_module("config")
        venv_name = getattr(cfg, "VENV_NAME", "venv") or "venv"
    except Exception:
        venv_name = "venv"
    venv_path = project_root / venv_name
    if sys.platform.startswith("win"):
        venv_bin = venv_path / "Scripts"
        venv_python = venv_bin / "python.exe"
    else:
        venv_bin = venv_path / "bin"
        venv_python = venv_bin / "python"
    if not venv_python.exists():
        return
    # 让后续子进程与当前进程都感知到虚拟环境
    os.environ["VIRTUAL_ENV"] = str(venv_path)
    bin_path = str(venv_bin)
    os.environ["PATH"] = f"{bin_path}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.pop("PYTHONHOME", None)
    print(f"检测到未激活虚拟环境，自动切换到 {venv_name} ...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()

# 第三方与项目内导入放在虚拟环境激活逻辑之后
import time
import threading
import random
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from utils import stock_list_cache_path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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
PROJECT_ROOT = Path(__file__).resolve().parent
CONCEPT_DATA_FILES = [
    PROJECT_ROOT / "stock_data" / "concepts" / "em" / "stock_concepts.csv",
    PROJECT_ROOT / "stock_data" / "concepts" / "ths" / "stock_concepts.csv",
]


def _tqdm_ncols(default: int = 100) -> int:
    """根据终端宽度动态压缩进度条，防止终端过窄时换行刷屏。"""
    try:
        env_override = os.environ.get("TQDM_COLS")
        if env_override:
            return max(40, int(env_override))
        width = shutil.get_terminal_size(fallback=(default, 20)).columns
        return max(50, min(width - 2, 140))
    except Exception:
        return default


def _tqdm_kwargs(**extra) -> Dict[str, Any]:
    """统一 tqdm 参数，强制输出（即使非 TTY），可用 TQDM_DISABLE=1 关闭。"""
    disable_env = os.environ.get("TQDM_DISABLE", "").strip()
    disable = disable_env in {"1", "true", "yes"}
    base = dict(
        ncols=_tqdm_ncols(),
        dynamic_ncols=True,
        mininterval=0.1,
        maxinterval=1.0,
        disable=disable,
    )
    base.update(extra)
    return base


def _refresh_postfix(bar: tqdm, data: Dict[str, Any]) -> None:
    """安全刷新 tqdm 的后缀，确保控制台实时显示进度计数"""
    try:
        bar.set_postfix(data, refresh=True)
    except Exception:
        try:
            bar.set_postfix(data)
        except Exception:
            return
    try:
        bar.refresh()
    except Exception:
        pass


def _concept_data_exists() -> bool:
    """检查概念数据是否已存在（任一文件存在且非空）。"""
    for path in CONCEPT_DATA_FILES:
        try:
            if path.exists() and path.stat().st_size > 0:
                return True
        except Exception as e:
            logger.debug("检查概念数据文件失败 %s: %s", path, e)
    return False


def _confirm_concept_download() -> bool:
    """缺失概念数据时，询问是否下载。非交互环境默认继续以保持行为一致。"""
    if not sys.stdin.isatty():
        logger.info("未检测到概念数据，非交互环境默认继续下载。")
        return True
    try:
        choice = input("未检测到概念数据，是否现在下载？[y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        logger.info("未检测到概念数据，用户取消操作。")
        return False
    return choice == "y"

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
    refresh_stock_list_on_download: bool = True
    token: Optional[str] = None
    download_tor: bool = False  # 是否下载换手率因子（tor）

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

class TokenBucketRateLimiter:
    """基于令牌桶的限频器，支持自动优化"""
    
    def __init__(self, 
                 calls_per_min: int = 500,
                 safe_calls_per_min: int = 490,
                 bucket_capacity: int = None,
                 refill_rate: float = None,
                 min_wait: float = None,
                 extra_delay: float = None,
                 extra_delay_threshold: int = None,
                 adaptive: bool = True):
        """
        初始化令牌桶限频器
        
        Args:
            calls_per_min: 每分钟最大调用次数
            safe_calls_per_min: 安全调用次数（留出安全边距）
            bucket_capacity: 令牌桶容量（从config读取，默认8）
            refill_rate: 令牌补充速率（次/秒，从config读取，默认8.0）
            min_wait: 最小等待时间（秒，从config读取，默认0.05）
            extra_delay: 接近限频阈值时的额外延迟（秒，从config读取，默认0.5）
            extra_delay_threshold: 触发额外延迟的调用次数阈值（从config读取，默认480）
            adaptive: 是否启用自适应优化
        """
        self.calls_per_min = calls_per_min
        self.safe_calls_per_min = safe_calls_per_min
        self.adaptive = adaptive
        self._user_bucket_capacity = bucket_capacity
        self._user_refill_rate = refill_rate
        
        # 从config读取令牌桶参数，如果没有则使用默认值
        try:
            from config import (
                RATE_BUCKET_CAPACITY, RATE_BUCKET_REFILL_RATE,
                RATE_BUCKET_MIN_WAIT, RATE_BUCKET_EXTRA_DELAY,
                RATE_BUCKET_EXTRA_DELAY_THRESHOLD
            )
            self.bucket_capacity = bucket_capacity or RATE_BUCKET_CAPACITY
            # 如果没有指定refill_rate，使用最大限频（calls_per_min / 60.0）
            # 先按照最大限频来，快到限频的时候减速，试探上限
            if refill_rate is None:
                self.refill_rate = self.calls_per_min / 60.0  # 使用最大限频
            else:
                self.refill_rate = refill_rate
            self.min_wait = min_wait or RATE_BUCKET_MIN_WAIT
            self.extra_delay = extra_delay or RATE_BUCKET_EXTRA_DELAY
            self.extra_delay_threshold = extra_delay_threshold or RATE_BUCKET_EXTRA_DELAY_THRESHOLD
        except ImportError:
            # 如果config中没有这些配置，使用默认值
            self.bucket_capacity = bucket_capacity or 8
            # 如果没有指定refill_rate，使用最大限频（calls_per_min / 60.0）
            if refill_rate is None:
                self.refill_rate = self.calls_per_min / 60.0  # 使用最大限频
            else:
                self.refill_rate = refill_rate
            self.min_wait = min_wait or 0.05
            self.extra_delay = extra_delay or 0.5
            self.extra_delay_threshold = extra_delay_threshold or 480
        
        # 保守模式（多进程/多实例时使用）：进一步收紧安全阈值
        try:
            env_factor = float(os.environ.get("TUSHARE_RATE_SAFETY_FACTOR", "1.0"))
            if env_factor < 1.0:
                self.safe_calls_per_min = int(self.safe_calls_per_min * env_factor)
                self.calls_per_min = int(self.calls_per_min * env_factor)
                logger.warning(f"启用保守限频模式：安全阈值调整为 {self.safe_calls_per_min}/min，最大 {self.calls_per_min}/min")
        except Exception:
            pass

        # 如果未显式设置补充速率，按缩放后的限频重算
        if not self._user_refill_rate:
            self.refill_rate = self.calls_per_min / 60.0

        # 动态计算令牌桶容量：低限频更细粒度，高限频允许适度突发
        def _auto_capacity(limit_per_min: int) -> int:
            # 约 2% 的每分钟额度作为容量，最低 2，最高 12，避免固定 8 粒度
            return max(2, min(12, int(limit_per_min * 0.02)))

        if not self._user_bucket_capacity:
            self.bucket_capacity = _auto_capacity(self.calls_per_min)

        # 动态放大 min_wait，限频越低越需要拉开线程间距
        min_gap = 30.0 / max(1, self.calls_per_min)  # roughly spread 30 calls across 1 min
        self.min_wait = max(self.min_wait, min_gap)
        
        # 令牌桶状态
        self._tokens = float(self.bucket_capacity)  # 当前令牌数
        self._last_refill = time.time()  # 上次补充令牌的时间
        self._lock = threading.Lock()
        
        # 自适应优化参数
        self._current_refill_rate = self.refill_rate  # 当前补充速率
        self._current_capacity = self.bucket_capacity  # 当前容量
        self._error_count = 0
        self._success_count = 0
        self._rate_limit_errors = 0  # 限频错误次数
        self._last_adjustment = time.time()
        self._adjustment_interval = 30.0  # 调整间隔（秒）
        
        # 统计信息
        self._total_calls = 0
        self._total_wait_time = 0.0
        self._rate_limit_hits = 0
        self._recent_calls = []  # 最近1分钟的调用时间记录（用于统计）
    
    def _refill_tokens(self, now: float):
        """补充令牌"""
        elapsed = now - self._last_refill
        if elapsed > 0:
            # 根据当前补充速率补充令牌
            tokens_to_add = elapsed * self._current_refill_rate
            self._tokens = min(self._current_capacity, self._tokens + tokens_to_add)
            self._last_refill = now
    
    def wait_if_needed(self):
        """检查是否需要等待，如果需要则等待
        策略：先按照最大限频来，快到限频的时候减速，试探上限
        """
        with self._lock:
            now = time.time()
            
            # 补充令牌
            self._refill_tokens(now)
            
            # 清理1分钟前的调用记录
            self._recent_calls = [t for t in self._recent_calls if now - t < 60]
            
            # 计算需要等待的时间
            wait_time = 0.0
            
            # 如果令牌不足，需要等待
            if self._tokens < 1.0:
                # 计算需要等待的时间以获取一个令牌
                tokens_needed = 1.0 - self._tokens
                wait_time = tokens_needed / self._current_refill_rate
                wait_time = max(self.min_wait, wait_time)  # 至少等待最小时间
            
            # 动态减速：根据接近限频的程度动态调整等待时间
            recent_calls_count = len(self._recent_calls)
            max_calls_per_min = self.calls_per_min
            
            # 计算接近限频的比例
            if recent_calls_count > 0:
                usage_ratio = recent_calls_count / max_calls_per_min
                
                # 如果接近限频阈值（92%以上），开始减速
                if usage_ratio >= 0.92:
                    # 根据接近程度计算额外的减速延迟
                    # 越接近限频，延迟越大
                    slowdown_factor = (usage_ratio - 0.92) / 0.08  # 0.92到1.0之间，映射到0到1
                    extra_wait = self.extra_delay * slowdown_factor
                    wait_time += extra_wait
                    
                    # 动态降低当前补充速率（试探上限）
                    if usage_ratio >= 0.98:
                        # 非常接近限频，适度降低速率（保持90%以上）
                        target_rate = max_calls_per_min / 60.0 * 0.92  # 降低到92%
                        if self._current_refill_rate > target_rate:
                            self._current_refill_rate = target_rate
                            logger.debug(f"接近限频阈值 ({recent_calls_count}/{max_calls_per_min})，降低速率到 {self._current_refill_rate:.2f} 次/秒")
                    elif usage_ratio >= 0.96:
                        # 接近限频，适度降低速率
                        target_rate = max_calls_per_min / 60.0 * 0.95  # 降低到95%
                        if self._current_refill_rate > target_rate:
                            self._current_refill_rate = target_rate
                            logger.debug(f"接近限频阈值 ({recent_calls_count}/{max_calls_per_min})，适度降低速率到 {self._current_refill_rate:.2f} 次/秒")
                    
                    if extra_wait > 0:
                        logger.debug(f"接近限频阈值 ({recent_calls_count}/{max_calls_per_min}, {usage_ratio*100:.1f}%)，添加减速延迟 {extra_wait:.2f}秒")
            
            # 如果需要等待
            if wait_time > 0:
                self._total_wait_time += wait_time
                self._rate_limit_hits += 1
                time.sleep(wait_time)
                
                # 等待后重新补充令牌
                now = time.time()
                self._refill_tokens(now)
            
            # 消耗一个令牌
            self._tokens -= 1.0
            
            # 记录本次调用
            self._recent_calls.append(now)
            self._total_calls += 1

    
    def record_success(self):
        """记录成功调用"""
        with self._lock:
            self._success_count += 1
            if self.adaptive:
                self._adjust_after_success()
    
    def record_error(self, error_type: str = "unknown"):
        """记录错误调用"""
        with self._lock:
            self._error_count += 1
            if "limit" in error_type.lower() or "quota" in error_type.lower() or "频率" in error_type.lower():
                self._rate_limit_errors += 1
            if self.adaptive:
                self._adjust_after_error(error_type)
    
    def _adjust_after_success(self):
        """成功调用后自适应调整
        策略：试探上限，如果没有限频错误，逐步提高速率
        """
        now = time.time()
        
        # 每30秒调整一次
        if now - self._last_adjustment < self._adjustment_interval:
            return
        
        max_rate = self.calls_per_min / 60.0  # 理论最大速率
        
        # 如果最近成功率很高且没有限频错误，试探上限，逐步提高速率
        if self._success_count > 25 and self._error_count < 5 and self._rate_limit_errors == 0:
            # 试探上限：逐步提高补充速率（最多到理论最大值）
            if self._current_refill_rate < max_rate:
                old_rate = self._current_refill_rate
                # 小步试探，每次提高2%
                self._current_refill_rate = min(self._current_refill_rate * 1.02, max_rate)
                logger.info(f"令牌桶试探上限: 提高补充速率 {old_rate:.2f} -> {self._current_refill_rate:.2f} 次/秒 (目标: {max_rate:.2f})")
            
            # 适当增加容量（允许更多突发）
            if self._current_capacity < self.bucket_capacity * 1.5:
                old_capacity = self._current_capacity
                self._current_capacity = min(self._current_capacity + 1, int(self.bucket_capacity * 1.5))
                logger.debug(f"令牌桶自适应调整: 增加容量 {old_capacity:.1f} -> {self._current_capacity:.1f}")
        elif self._rate_limit_errors == 0 and self._current_refill_rate < max_rate * 0.98:
            # 即使成功率不是特别高，如果没有限频错误，也可以小幅试探
            old_rate = self._current_refill_rate
            self._current_refill_rate = min(self._current_refill_rate * 1.02, max_rate * 0.98)
            logger.debug(f"令牌桶小幅试探: 提高补充速率 {old_rate:.2f} -> {self._current_refill_rate:.2f} 次/秒")
        
        # 重置统计
        self._success_count = 0
        self._error_count = 0
        self._rate_limit_errors = 0
        self._last_adjustment = now
    
    def _adjust_after_error(self, error_type: str):
        """错误调用后自适应调整
        策略：遇到限频错误时，适度降低速率，但不要降得太低，以便继续试探上限
        """
        now = time.time()
        
        # 每30秒调整一次
        if now - self._last_adjustment < self._adjustment_interval:
            return
        
        max_rate = self.calls_per_min / 60.0  # 理论最大速率
        
        # 如果是限频错误，适度降低速率（试探上限）
        if "limit" in error_type.lower() or "quota" in error_type.lower() or "频率" in error_type.lower():
            # 降低补充速率，但不要降得太低（降低5%，以便继续试探）
            old_rate = self._current_refill_rate
            # 降低到当前速率的95%，但不低于最大速率的85%（继续试探）
            target_rate = max(self._current_refill_rate * 0.95, max_rate * 0.85)
            self._current_refill_rate = target_rate
            logger.warning(f"令牌桶试探上限: 遇到限频错误，降低补充速率 {old_rate:.2f} -> {self._current_refill_rate:.2f} 次/秒（继续试探）")
            
            # 适度降低容量
            old_capacity = self._current_capacity
            self._current_capacity = max(int(self._current_capacity * 0.95), self.bucket_capacity)
            logger.debug(f"令牌桶自适应调整: 降低容量 {old_capacity:.1f} -> {self._current_capacity:.1f}（限频错误）")
        
        # 重置统计
        self._success_count = 0
        self._error_count = 0
        self._rate_limit_errors = 0
        self._last_adjustment = now
    
    def get_stats(self) -> Dict[str, Any]:
        """获取限频器统计信息"""
        with self._lock:
            now = time.time()
            recent_calls = len([t for t in self._recent_calls if now - t < 60])
            
            return {
                "current_refill_rate": self._current_refill_rate,
                "current_capacity": self._current_capacity,
                "current_tokens": self._tokens,
                "recent_calls_per_min": recent_calls,
                "total_calls": self._total_calls,
                "success_count": self._success_count,
                "error_count": self._error_count,
                "rate_limit_errors": self._rate_limit_errors,
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
            self._rate_limit_errors = 0
            self._total_wait_time = 0.0
            self._rate_limit_hits = 0
            self._recent_calls = []

# 为了向后兼容，保留RateLimiter作为TokenBucketRateLimiter的别名
RateLimiter = TokenBucketRateLimiter

# ================= Token 处理 =================
def _is_placeholder_token(token: Optional[str]) -> bool:
    """检查 token 是否为空或占位符。"""
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


def resolve_tushare_token(token: Optional[str] = None, *, allow_prompt: bool = True) -> str:
    """
    获取有效的 Tushare token，优先顺序：
    1) 显式传入
    2) 环境变量 TUSHARE_TOKEN / TS_TOKEN
    3) config.TOKEN
    4) 交互式输入（仅在 tty 且允许提示时）
    """
    candidates = [
        token,
        os.environ.get("TUSHARE_TOKEN"),
        os.environ.get("TS_TOKEN"),
        TOKEN,
    ]
    for cand in candidates:
        if not _is_placeholder_token(cand):
            return str(cand).strip()

    if allow_prompt and sys.stdin.isatty():
        try:
            user_token = input("请输入 Tushare Pro Token（不会保存，直接使用本次运行）: ").strip()
            if user_token:
                return user_token
        except Exception:
            pass

    raise ValueError("Tushare Pro 未配置：请在 config.TOKEN 中设置有效 token，或设置环境变量 TUSHARE_TOKEN")


# ================= Tushare接口管理 =================
class TushareManager:
    """Tushare接口管理器"""
    
    def __init__(self, token: str = None, rate_limiter: TokenBucketRateLimiter = None,
                 allow_prompt: bool = True, db_manager: Optional[DatabaseManager] = None):
        self.token = resolve_tushare_token(token, allow_prompt=allow_prompt)
        self._pro = None
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()
        # 持有数据库管理器，延迟创建以避免不必要的连接
        self.db_manager = db_manager

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

    def _get_db_manager(self) -> DatabaseManager:
        """延迟获取数据库管理器，保证单例复用。"""
        if self.db_manager is None:
            self.db_manager = get_database_manager()
        return self.db_manager
    
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
                        base_wait_time = retry_delay * (2 ** attempt)  # 指数退避
                        # 添加±20%的随机抖动，避免多个请求同时重试
                        jitter = random.uniform(-0.2, 0.2) * base_wait_time
                        wait_time = max(0.1, base_wait_time + jitter)  # 确保等待时间至少0.1秒
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
                        base_wait_time = retry_delay * (attempt + 1)
                        # 添加±20%的随机抖动，避免多个请求同时重试
                        jitter = random.uniform(-0.2, 0.2) * base_wait_time
                        wait_time = max(0.1, base_wait_time + jitter)  # 确保等待时间至少0.1秒
                        logger.info(f"网络问题，等待 {wait_time:.1f} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                else:
                    self.rate_limiter.record_error("api_error")
                    logger.error(f"API调用失败: {error_msg}")
                    # 其他错误可以重试
                    if attempt < max_retries - 1:
                        base_wait_time = retry_delay * (attempt + 1)
                        # 添加±20%的随机抖动，避免多个请求同时重试
                        jitter = random.uniform(-0.2, 0.2) * base_wait_time
                        wait_time = max(0.1, base_wait_time + jitter)  # 确保等待时间至少0.1秒
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
                       adj: str = None,
                       include_tor: bool = False) -> pd.DataFrame:
        """获取股票日线数据（使用pro_bar接口获取原始数据）"""
        try:
            # 构建参数字典
            params = {
                'ts_code': ts_code,
                'start_date': start_date,
                'end_date': end_date,
                'freq': 'D',
                'asset': 'E',
            }
            if include_tor:
                params['factors'] = ['tor']  # 需要换手率字段
            
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
    
    def get_stock_daily_by_date(self, trade_date: str, fields: str = None) -> pd.DataFrame:
        """按交易日拉取全市场日线（不含换手率）。"""
        try:
            return self._make_api_call(
                self._pro.daily,
                trade_date=trade_date,
                fields=fields,
            )
        except Exception as e:
            logger.error(f"获取日线失败 trade_date={trade_date}: {e}")
            raise

    def get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表（使用数据库管理器的统一方法）"""
        try:
            # 使用数据库管理器的统一方法
            return self._get_db_manager().get_trade_dates_from_tushare(start_date, end_date)
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
            return self._get_db_manager().get_smart_end_date(end_date_config)
        except Exception as e:
            logger.error(f"智能获取结束日期失败: {e}")
            # 如果智能获取失败，返回原配置或今天
            if end_date_config.lower() == "today":
                return datetime.now().strftime("%Y%m%d")
            return end_date_config

    def get_daily_basic_by_date(self, trade_date: str, fields: str) -> pd.DataFrame:
        """按交易日获取 daily_basic（一次性全市场）。"""
        try:
            return self._make_api_call(
                self._pro.daily_basic,
                trade_date=trade_date,
                fields=fields,
            )
        except Exception as e:
            logger.error(f"获取 daily_basic 失败 trade_date={trade_date}: {e}")
            raise

# ================= 数据处理器 =================
class DataProcessor:
    """数据处理器"""
    
    def __init__(self, db_manager: Optional[DatabaseManager]):
        self.db_manager = db_manager
        self.indicator_names = get_all_indicator_names()
    
    def clean_stock_data(self, df: pd.DataFrame, ts_code: str = None) -> pd.DataFrame:
        """清理股票数据"""
        if df is None or df.empty:
            return df
        
        # 复制数据
        df = df.copy()

        # 重命名字段：换手率列使用短名，便于后续存储/计算
        rename_map = {
            "turnover_rate": "tor",
        }
        df = df.rename(columns=rename_map)
        # tor 是基础列，缺失时补 NaN，避免后续取列报错
        if "tor" not in df.columns:
            df["tor"] = np.nan
        
        # 删除不需要的列
        columns_to_drop = ["pre_close", "change", "pct_chg"]
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # 数值化处理
        numeric_columns = ["open", "high", "low", "close", "vol", "amount", "tor"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 精度控制
        price_columns = ["open", "high", "low", "close", "vol", "amount", "tor"]
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
                                     adj_type: str, warmup_cache: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """计算指标（包含warmup机制）
        
        Args:
            df: 新下载的数据
            ts_code: 股票代码
            adj_type: 复权类型
            warmup_cache: 预加载的历史数据缓存（增量下载时使用）
        """
        if df is None or df.empty:
            return df
        
        try:
            historical_data = None
            
            # 优先使用预加载的缓存数据
            if warmup_cache is not None and ts_code in warmup_cache:
                historical_data = warmup_cache[ts_code].copy()
                logger.debug(f"使用预加载的历史数据 {ts_code}: {len(historical_data)} 行")
            else:
                # 如果没有缓存，按需从数据库获取（兼容旧逻辑）
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
                # 没有历史数据，直接计算（首次下载的情况）
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
        # 如果没有db_manager，无法获取历史数据
        if self.db_manager is None:
            return None
            
        try:
            # 计算需要的warmup天数
            warmup_days = warmup_for(self.indicator_names)
            if warmup_days <= 0:
                return None

            # 获取新数据的最早日期，用于界定历史数据的上限
            min_date = new_data['trade_date'].min()
            from datetime import datetime, timedelta
            min_date_obj = datetime.strptime(str(min_date), '%Y%m%d')
            end_date_obj = min_date_obj - timedelta(days=1)
            end_date = end_date_obj.strftime('%Y%m%d')

            # 按条数取 warmup 所需行数，额外多取一些作为缓冲
            need_rows = max(warmup_days + 10, warmup_days)
            
            # 添加超时和重试机制
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    # 从数据库获取历史数据
                    historical_df = self.db_manager.query_stock_data(
                        ts_code=ts_code,
                        end_date=end_date,
                        adj_type=adj_type,
                        limit=need_rows,
                        order="desc"  # 取最新 N 行
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
        if "tor" not in df.columns:
            df["tor"] = np.nan
        
        # 确保列顺序正确
        base_columns = ['ts_code', 'trade_date', 'adj_type', 'open', 'high', 'low', 'close', 'vol', 'amount', 'tor']
        indicator_columns = [col for col in df.columns if col not in base_columns]
        df = df[base_columns + indicator_columns]
        
        return df

# ================= 下载器 =================
def _compute_indicators_task(args):
    """子进程任务：合并warmup后计算指标并准备写库。"""
    ts_code, df, adj_type, warmup_df, indicator_names, decs = args
    try:
        base_df = df
        if warmup_df is not None and not warmup_df.empty:
            base_df = pd.concat([warmup_df, df], ignore_index=True)
            base_df = base_df.sort_values('trade_date').drop_duplicates('trade_date', keep='last')
        res = compute(base_df, indicator_names)
        min_date = df['trade_date'].min()
        res = res[res['trade_date'] >= min_date].copy()
        for col, n in decs.items():
            if col in res.columns and pd.api.types.is_numeric_dtype(res[col]):
                res[col] = res[col].round(n)
        res['adj_type'] = adj_type
        if 'tor' not in res.columns:
            res['tor'] = np.nan
        base_columns = ['ts_code', 'trade_date', 'adj_type', 'open', 'high', 'low', 'close', 'vol', 'amount', 'tor']
        indicator_columns = [c for c in res.columns if c not in base_columns]
        res = res[base_columns + indicator_columns]
        return ts_code, res, None
    except Exception as e:
        return ts_code, None, str(e)


class StockDownloader:
    """股票数据下载器"""
    
    def __init__(self, config: DownloadConfig, is_first_download: bool = False, batch_write_once: bool = False):
        self.config = config
        self.is_first_download = is_first_download  # 是否为首次下载
        # 是否在下载完成后批量一次性写入（适用于重试阶段避免多次写盘）
        self.batch_write_once = batch_write_once
        # 不在初始化时创建数据库连接，避免子线程创建连接
        
        # 创建限频器（使用令牌桶算法）
        calls_per_min = config.rate_limit_calls_per_min
        safe_calls_per_min = config.safe_calls_per_min
        if config.download_tor:
            # daily_basic 限频 200/min，tor 会触发
            calls_per_min = min(200, calls_per_min)
            safe_calls_per_min = min(190, safe_calls_per_min, calls_per_min - 1 if calls_per_min > 1 else calls_per_min)
            logger.warning(f"检测到 daily_basic 需求（tor），限频强制调整为 {calls_per_min}/min（safe={safe_calls_per_min}）以适配 daily_basic 限额")
        rate_limiter = TokenBucketRateLimiter(
            calls_per_min=calls_per_min,
            safe_calls_per_min=safe_calls_per_min,
            adaptive=config.enable_adaptive_rate_limit
        )
        
        self.tushare_manager = TushareManager(
            token=config.token,
            rate_limiter=rate_limiter,
        )
        # 延迟创建data_processor，只在需要时创建（主线程中）
        self._data_processor = None
        # 预加载的历史数据缓存（增量下载时使用）
        self._warmup_cache: Dict[str, pd.DataFrame] = {}  # {ts_code: historical_data}
        self.stats = DownloadStats()
        self._lock = threading.Lock()
        # 数据库管理器仅在主线程中按需创建
        self.db_manager: Optional[DatabaseManager] = None

    def _detect_preclose_mismatch(self, ts_code: str, raw_df: pd.DataFrame) -> Optional[Tuple[str, float, float]]:
        """
        用 warmup 数据校验“增量段首日”的 pre_close，一旦发现不一致，认为可能发生复权。
        仅在增量下载且存在 warmup 缓存时检查，不访问数据库。
        返回 (trade_date, pre_close, expected_prev_close) 或 None
        """
        try:
            if self.is_first_download or not self.config.enable_warmup:
                return None
            if raw_df is None or raw_df.empty or "pre_close" not in raw_df.columns:
                return None
            warmup_df = self._warmup_cache.get(ts_code)
            if warmup_df is None or warmup_df.empty or "close" not in warmup_df.columns:
                return None
            # 增量首日（新段最早日期）
            first_row = raw_df.sort_values("trade_date").iloc[0]
            trade_date = str(first_row.get("trade_date"))
            pre_close = first_row.get("pre_close")
            if pd.isna(pre_close):
                return None

            # warmup 最近一日（增量前一交易日）
            warmup_latest = warmup_df.sort_values("trade_date").iloc[-1]
            expected = warmup_latest.get("close")
            expected_date = str(warmup_latest.get("trade_date"))

            if pd.isna(expected):
                return None
            try:
                pre_c = float(pre_close)
                exp_c = float(expected)
            except Exception:
                return None

            if abs(pre_c - exp_c) > 1e-6:
                logger.debug(f"pre_close 校验 {ts_code}: 首日 {trade_date} pre_close={pre_c}, warmup 最新 {expected_date} close={exp_c}")
                return trade_date, pre_c, exp_c
            return None
        except Exception as e:
            logger.debug(f"pre_close 校验失败 {ts_code}: {e}")
            return None

    def _persist_stock_list_cache(self, df: pd.DataFrame) -> None:
        """把股票列表缓存到数据库目录下的 stock_list.csv，供其他模块复用。"""
        try:
            if df is None or df.empty or "ts_code" not in df.columns:
                return
            cache_path = stock_list_cache_path()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cols = [c for c in ["ts_code", "symbol", "name", "area", "industry", "list_date",
                                "trade_date", "total_share", "float_share", "total_mv", "circ_mv"]
                    if c in df.columns]
            df[cols].to_csv(cache_path, index=False, encoding="utf-8-sig")
            logger.info(f"股票列表已缓存：{cache_path}（保留，不再清理）")
        except Exception as e:
            logger.warning(f"写入股票列表缓存失败：{e}")
    
    def _load_stock_list_from_cache(self) -> List[str]:
        """兜底：从缓存或数据库获取股票列表"""
        try:
            if self.db_manager is None:
                self.db_manager = get_database_manager()
            db_manager = self.db_manager
        except Exception as e:
            logger.error(f"获取数据库管理器失败，无法加载股票缓存: {e}")
            return []
        
        # 1) stock_list.csv 缓存
        try:
            cached_codes = db_manager.get_stock_list_from_cache()
            if cached_codes:
                logger.warning(f"使用缓存股票列表兜底，共 {len(cached_codes)} 只")
                return cached_codes
        except Exception as e:
            logger.debug(f"读取股票列表缓存失败: {e}")
        
        # 2) 数据库已有股票列表
        try:
            db_codes = db_manager.get_stock_list(adj_type=self.config.adj_type)
            if db_codes:
                logger.warning(f"使用数据库股票列表兜底，共 {len(db_codes)} 只")
                return db_codes
        except Exception as e:
            logger.debug(f"读取数据库股票列表失败: {e}")
        
        logger.error("缓存和数据库都无法提供股票列表，兜底失败")
        return []

    def _attach_total_share_to_list(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """在刷新名单后，一次性补充最新交易日的总股本等字段。"""
        if stock_list is None or stock_list.empty or "ts_code" not in stock_list.columns:
            return stock_list
        trade_date = None
        try:
            trade_date = self.tushare_manager.get_smart_end_date("today")
        except Exception as e:
            logger.warning(f"获取最新交易日失败，跳过总股本补充：{e}")
        if not trade_date:
            return stock_list
        try:
            fields = "ts_code,trade_date,total_share,float_share,total_mv,circ_mv"
            basic_df = self.tushare_manager.get_daily_basic_by_date(trade_date=trade_date, fields=fields)
            if basic_df is None or basic_df.empty:
                logger.warning(f"daily_basic 在 {trade_date} 无数据，跳过总股本补充")
                return stock_list
            basic_df = basic_df.drop_duplicates(subset=["ts_code"])
            # 数值化
            for col in ["total_share", "float_share", "total_mv", "circ_mv"]:
                if col in basic_df.columns:
                    basic_df[col] = pd.to_numeric(basic_df[col], errors="coerce")
            merged = stock_list.merge(basic_df, on="ts_code", how="left")
            logger.info(f"已为股票列表补充总股本等字段（trade_date={trade_date}）")
            return merged
        except Exception as e:
            logger.warning(f"补充总股本失败，继续使用原名单：{e}")
            return stock_list
    
    def _get_data_processor(self):
        """获取数据处理器（延迟创建，只在主线程中创建）"""
        if self._data_processor is None:
            # 只在主线程中创建数据库连接
            logger.info("[数据库连接] 开始获取数据库管理器实例 (创建数据处理器)")
            db_manager = get_database_manager()
            self._data_processor = DataProcessor(db_manager)
        return self._data_processor
    
    def _preload_warmup_data(self, db_manager: DatabaseManager, stock_codes: List[str], 
                             data_processor: DataProcessor):
        """批量预加载历史数据到内存（用于增量下载的warmup）"""
        try:
            from indicators import warmup_for
            warmup_days = warmup_for(data_processor.indicator_names)
            if warmup_days <= 0:
                logger.info("指标不需要warmup，跳过预加载")
                return
            need_rows = max(warmup_days + 10, warmup_days)  # 按条数取，额外多取一些做缓冲

            # 计算历史数据的结束日期（新数据开始日期前一天），只用作上界避免覆盖增量段
            from datetime import datetime, timedelta
            start_date_obj = datetime.strptime(self.config.start_date, '%Y%m%d')
            load_end_obj = start_date_obj - timedelta(days=1)
            load_end = load_end_obj.strftime('%Y%m%d')
            
            logger.info(f"预加载历史数据按条数获取: 每只股票 {need_rows} 条，截止 {load_end} (warmup天数: {warmup_days})")
            
            # 批量查询所有股票的历史数据
            logger.info(f"开始批量查询 {len(stock_codes)} 只股票的历史数据...")
            preload_progress = tqdm(
                total=len(stock_codes),
                desc="预加载历史数据",
                unit="只",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
                leave=False,
                **_tqdm_kwargs()
            )
            
            loaded_count = 0
            for ts_code in stock_codes:
                try:
                    # 从数据库查询历史数据
                    historical_df = db_manager.query_stock_data(
                        ts_code=ts_code,
                        end_date=load_end,
                        adj_type=self.config.adj_type,
                        limit=need_rows,
                        order="desc"  # 直接取最新 N 行，避开停牌日期空洞
                    )
                    
                    if not historical_df.empty:
                        # 清理数据
                        historical_df = data_processor.clean_stock_data(historical_df, ts_code)
                        # 存入缓存
                        self._warmup_cache[ts_code] = historical_df
                        loaded_count += 1
                except Exception as e:
                    logger.debug(f"预加载历史数据失败 {ts_code}: {e}")
                    # 预加载失败不影响下载，继续处理下一只股票
                
                preload_progress.update(1)
                _refresh_postfix(preload_progress, {"完成": preload_progress.n})
            
            preload_progress.close()
            logger.info(f"预加载完成：成功加载 {loaded_count}/{len(stock_codes)} 只股票的历史数据")
            
        except Exception as e:
            logger.warning(f"批量预加载历史数据失败: {e}，将使用按需加载模式")
            # 预加载失败不影响下载，将使用原来的按需加载模式
        finally:
            try:
                # 预加载完成后主动清理连接池，避免长时间占用读连接
                db_manager.clear_connections_only()
            except Exception as e:
                logger.debug(f"预加载后清理数据库连接失败: {e}")
    
    def download_stock_data(self, ts_code: str, data_processor: DataProcessor = None) -> Tuple[str, str, Optional[pd.DataFrame]]:
        """下载单只股票数据，边下载边计算指标并返回处理后的数据
        
        Args:
            ts_code: 股票代码
            data_processor: 数据处理器，用于计算指标和warmup。如果为None，只下载不计算指标
        
        Returns:
            Tuple[ts_code, status, data]:
                - ts_code: 股票代码
                - status: 状态 ("success", "empty", "error:xxx")
                - data: 处理后的数据（包含指标），如果失败则为None
        """
        max_retries = self.config.retry_times
        retry_delay = 1.0
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # 下载原始数据
                logger.debug(f"开始下载 {ts_code} 的原始数据... (尝试 {attempt + 1}/{max_retries})")
                raw_data = self.tushare_manager.get_stock_daily(
                    ts_code=ts_code,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    adj=self.config.adj_type if self.config.adj_type != "raw" else None,
                    include_tor=self.config.download_tor
                )
                
                if raw_data is None or raw_data.empty:
                    logger.debug(f"{ts_code} 无数据")
                    return ts_code, "empty", None
                
                logger.debug(f"{ts_code} 获取到 {len(raw_data)} 条原始数据")

                # pre_close 与 warmup 数据校验，发现复权则标记失败，交由完整重试
                mismatch = self._detect_preclose_mismatch(ts_code, raw_data)
                if mismatch:
                    trade_date, pre_c, exp_c = mismatch
                    msg = (f"{ts_code} {trade_date} pre_close={pre_c} 与上一交易日收盘 {exp_c} 不一致，"
                           f"疑似发生复权，标记失败以触发完整重试")
                    logger.warning(msg)
                    return ts_code, f"error: {msg}", None
                
                # 清理数据（不依赖数据库连接）
                logger.debug(f"清理 {ts_code} 数据...")
                # 使用临时的DataProcessor进行清理（不需要数据库连接）
                temp_processor = DataProcessor(None)  # 传入None，因为清理操作不需要数据库
                cleaned_data = temp_processor.clean_stock_data(raw_data, ts_code)
                
                # 如果提供了data_processor，立即计算指标
                if data_processor is not None:
                    # 首次下载不需要warmup，增量下载才需要
                    need_warmup = self.config.enable_warmup and not self.is_first_download
                    
                    if need_warmup:
                        logger.debug(f"计算 {ts_code} 指标（包含warmup）...")
                        try:
                            # 计算指标（包含warmup，使用预加载的缓存数据）
                            data_with_indicators = data_processor.compute_indicators_with_warmup(
                                cleaned_data,
                                ts_code,
                                self.config.adj_type,
                                warmup_cache=self._warmup_cache
                            )
                            if data_with_indicators is None:
                                logger.warning(f"{ts_code} warmup数据不足，跳过输出")
                                with self._lock:
                                    self.stats.empty_count += 1
                                return ts_code, "empty", None
                            # 准备数据用于数据库
                            final_data = data_processor.prepare_data_for_database(
                                data_with_indicators, self.config.adj_type
                            )
                            logger.debug(f"{ts_code} 指标计算完成，共 {len(final_data)} 条记录，{len(final_data.columns)} 列")
                            return ts_code, "success", final_data
                        except Exception as e:
                            logger.error(f"{ts_code} 指标计算失败: {e}")
                            raise
                    else:
                        # 首次下载或未启用warmup，直接计算指标
                        logger.debug(f"计算 {ts_code} 指标（不包含warmup）...")
                        try:
                            from indicators import compute, get_all_indicator_names
                            data_with_indicators = compute(cleaned_data, get_all_indicator_names())
                            final_data = data_processor.prepare_data_for_database(
                                data_with_indicators, self.config.adj_type
                            )
                            logger.debug(f"{ts_code} 指标计算完成，共 {len(final_data)} 条记录")
                            return ts_code, "success", final_data
                        except Exception as e:
                            logger.error(f"{ts_code} 指标计算失败: {e}")
                            raise
                else:
                    # 没有提供data_processor，只返回清理后的原始数据
                    logger.debug(f"{ts_code} 数据清理完成，共 {len(cleaned_data)} 条记录（未计算指标）")
                    return ts_code, "success", cleaned_data
                
            except Exception as e:
                error_msg = str(e)
                error_msg_lower = error_msg.lower()
                last_error = e
                
                # 如果是指标计算错误，影响数据完整性，直接抛出异常，不重试
                if any(keyword in error_msg_lower for keyword in [
                    "指标计算失败", "indicator", "计算失败", "compute"
                ]):
                    logger.error(f"下载股票数据失败（影响数据完整性）{ts_code}: {error_msg}")
                    raise RuntimeError(f"下载股票数据失败（影响数据完整性）{ts_code}: {error_msg}") from e
                
                # 认证错误不重试
                if "token" in error_msg_lower or "auth" in error_msg_lower or "1002" in error_msg:
                    logger.error(f"Token认证失败 {ts_code}: {error_msg}，不重试")
                    with self._lock:
                        self.stats.error_count += 1
                        self.stats.failed_stocks.append((ts_code, error_msg))
                    return ts_code, f"error: {error_msg}", None
                
                # 其他错误可以重试
                if attempt < max_retries - 1:
                    base_wait_time = retry_delay * (2 ** attempt)  # 指数退避
                    # 添加±20%的随机抖动，避免多个请求同时重试
                    jitter = random.uniform(-0.2, 0.2) * base_wait_time
                    wait_time = max(0.1, base_wait_time + jitter)  # 确保等待时间至少0.1秒
                    logger.warning(f"下载股票数据失败 {ts_code} (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                    logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 所有重试都失败了
                    logger.error(f"下载股票数据最终失败 {ts_code} (已重试 {max_retries} 次): {error_msg}")
                    with self._lock:
                        self.stats.error_count += 1
                        self.stats.failed_stocks.append((ts_code, error_msg))
                    return ts_code, f"error: {error_msg}", None
        
        # 如果所有重试都失败，返回错误
        if last_error:
            error_msg = str(last_error)
            logger.error(f"下载股票数据失败 {ts_code} (已重试 {max_retries} 次): {error_msg}")
            with self._lock:
                self.stats.error_count += 1
                self.stats.failed_stocks.append((ts_code, error_msg))
            return ts_code, f"error: {error_msg}", None
        
        return ts_code, "error: 未知错误", None
    
    def download_all_stocks(self, stock_codes: Optional[List[str]] = None) -> DownloadStats:
        """下载股票数据，由主线程统一写入数据库
        
        Args:
            stock_codes: 如果提供，则仅下载指定股票；否则下载全市场
        """
        logger.info("开始下载股票数据...")

        # 在主线程中创建数据库连接，供缓存/兜底使用
        if self.db_manager is None:
            logger.info("[数据库连接] 开始获取数据库管理器实例 (股票下载器)")
            self.db_manager = get_database_manager()
        
        # 获取股票列表
        if stock_codes is None:
            stock_list = None
            # 下载新数据时：可选择主动刷新名单
            if getattr(self.config, "refresh_stock_list_on_download", True):
                try:
                    stock_list = self.tushare_manager.get_stock_list()
                except Exception as e:
                    logger.warning(f"刷新股票列表失败（继续尝试缓存/数据库）: {e}")
                    stock_list = None
                if stock_list is not None and not stock_list.empty and "ts_code" in stock_list.columns:
                    stock_list = self._attach_total_share_to_list(stock_list)
                    self._persist_stock_list_cache(stock_list)
                    stock_codes = stock_list["ts_code"].astype(str).tolist()

            if stock_codes is None:
                # 优先使用本地缓存/数据库，只有完全没有名单时才调用 Tushare
                try:
                    cached_codes = self.db_manager.get_stock_list_from_cache()
                except Exception as e:
                    logger.debug(f"读取缓存股票列表失败: {e}")
                    cached_codes = []

                if cached_codes:
                    logger.info(f"使用缓存股票列表，共 {len(cached_codes)} 只")
                    stock_codes = cached_codes
                else:
                    try:
                        db_codes = self.db_manager.get_stock_list(adj_type=self.config.adj_type)
                    except Exception as e:
                        logger.debug(f"读取数据库股票列表失败: {e}")
                        db_codes = []

                    if db_codes:
                        logger.info(f"使用数据库股票列表，共 {len(db_codes)} 只")
                        stock_codes = db_codes
                    else:
                        try:
                            stock_list = self.tushare_manager.get_stock_list()
                        except Exception as e:
                            logger.error(f"获取股票列表时发生异常: {e}")
                            import traceback
                            logger.error(f"异常详情: {traceback.format_exc()}")
                            stock_list = None

                        if stock_list is None or stock_list.empty:
                            logger.error("获取股票列表失败：返回结果为空")
                            stock_codes = self._load_stock_list_from_cache()
                        elif 'ts_code' not in stock_list.columns:
                            logger.error(f"股票列表缺少 'ts_code' 列，可用列: {stock_list.columns.tolist()}")
                            stock_codes = self._load_stock_list_from_cache()
                        else:
                            self._persist_stock_list_cache(stock_list)
                            stock_codes = stock_list['ts_code'].tolist()
            # 最后兜底再清洗一遍
            if not stock_codes:
                stock_codes = self._load_stock_list_from_cache()
        else:
            logger.info(f"使用外部提供的股票列表，共 {len(stock_codes)} 只")
            normalized_codes: List[str] = []
            seen_codes: Set[str] = set()
            for code in stock_codes:
                if not code:
                    continue
                norm_code = normalize_ts(code) or str(code).strip()
                if not norm_code:
                    continue
                if norm_code in seen_codes:
                    continue
                seen_codes.add(norm_code)
                normalized_codes.append(norm_code)
            stock_codes = normalized_codes
        
        if not stock_codes:
            logger.error("股票列表为空，无法下载")
            return self.stats
        
        self.stats.total_stocks = len(stock_codes)
        logger.info(f"准备下载 {len(stock_codes)} 只股票的数据，日期范围: {self.config.start_date} - {self.config.end_date}")

        # 在主线程中创建数据库连接和数据处理器
        db_manager = self.db_manager or get_database_manager()
        logger.info("[数据库连接] 获取数据库管理器实例成功 (主线程统一写入)")
        data_processor = DataProcessor(db_manager)
        
        # 如果是增量下载且启用了warmup，批量预加载历史数据到内存
        if not self.is_first_download and self.config.enable_warmup:
            logger.info("增量下载模式：开始批量预加载历史数据用于warmup...")
            self._preload_warmup_data(db_manager, stock_codes, data_processor)
        
        # 创建进度条
        progress_bar = tqdm(
            total=len(stock_codes),
            desc="下载并计算指标",
            unit="只",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            **_tqdm_kwargs()
        )
        
        # 用于收集需要批量写入的数据（如果启用批量写入模式）
        collected_data = []  # 存储所有成功下载并处理的数据
        
        # 边下载边计算指标边写入
        logger.info("开始边下载边计算指标边写入...")
        
        # 并发下载（边下载边计算指标）
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            # 提交所有任务（传入data_processor以便计算指标）
            future_to_code = {
                executor.submit(self.download_stock_data, ts_code, data_processor): ts_code 
                for ts_code in stock_codes
            }
            
            # 用于批量写入的锁和计数器
            write_lock = threading.Lock()
            write_count = 0
            
            # 处理完成的任务
            for future in as_completed(future_to_code):
                ts_code = future_to_code[future]
                try:
                    result_code, status, data = future.result()
                    if status == "empty":
                        with self._lock:
                            self.stats.empty_count += 1
                    elif status == "success" and data is not None:
                        if self.batch_write_once:
                            # 收集数据，稍后统一写入
                            with write_lock:
                                collected_data.append(data)
                                self.stats.success_count += 1
                                write_count += 1
                            logger.debug(f"✓ {result_code} 下载、计算指标完成，已收集待批量写入（当前 {write_count} 只）")
                        else:
                            # 立即写入数据库（边下载边写入）
                            try:
                                logger.debug(f"开始写入 {result_code} 的数据到数据库（{len(data)} 条记录）...")
                                
                                # 使用回调函数在写入完成后更新状态
                                write_completed = threading.Event()
                                write_result = {"success": False, "error": None}
                                
                                def update_status_callback(response):
                                    try:
                                        if hasattr(response, 'success') and response.success:
                                            write_result["success"] = True
                                            logger.debug(f"{result_code} 数据已成功写入数据库，导入 {response.rows_imported} 条记录")
                                        else:
                                            write_result["success"] = False
                                            write_result["error"] = getattr(response, 'error', '未知错误')
                                            logger.error(f"{result_code} 数据写入数据库失败: {write_result['error']}")
                                    except Exception as e:
                                        logger.warning(f"更新状态失败 {result_code}: {e}")
                                    finally:
                                        write_completed.set()
                                
                                # 提交写入请求（异步，不阻塞）
                                request_id = db_manager.receive_stock_data(
                                    source_module="download",
                                    data=data,
                                    mode="upsert",
                                    callback=update_status_callback
                                )
                                
                                # 等待写入完成（最多等待60秒，因为可能需要从数据库获取历史数据进行warmup）
                                if write_completed.wait(timeout=60):
                                    if not write_result["success"]:
                                        logger.error(f"{result_code} 写入失败: {write_result['error']}")
                                        with self._lock:
                                            self.stats.error_count += 1
                                            self.stats.failed_stocks.append(
                                                (result_code, write_result.get("error") or "写入失败")
                                            )
                                        continue
                                else:
                                    logger.warning(f"{result_code} 写入请求超时（60秒），数据可能仍在后台处理中，继续处理下一只股票")
                                
                                # 写入成功，更新统计
                                with self._lock:
                                    self.stats.success_count += 1
                                    write_count += 1
                                
                                logger.debug(f"✓ {result_code} 下载、计算指标、写入完成")
                                
                            except Exception as e:
                                logger.error(f"{result_code} 写入数据库失败: {e}")
                                with self._lock:
                                    self.stats.error_count += 1
                                    self.stats.failed_stocks.append((result_code, str(e)))
                    else:
                        logger.warning(f"下载失败 {result_code}: {status}")
                        with self._lock:
                            self.stats.error_count += 1
                            self.stats.failed_stocks.append((result_code, status))
                except Exception as e:
                    logger.error(f"处理任务异常 {ts_code}: {e}")
                    with self._lock:
                        self.stats.error_count += 1
                        self.stats.failed_stocks.append((ts_code, str(e)))
                
                # 更新进度条
                progress_bar.update(1)
                _refresh_postfix(progress_bar, {
                    '成功': self.stats.success_count,
                    '空数据': self.stats.empty_count,
                    '失败': self.stats.error_count
                })
        
        # 关闭进度条
        progress_bar.close()
        
        # 批量写入模式：合并所有成功数据后一次性写库
        if self.batch_write_once:
            if collected_data:
                final_df = pd.concat(collected_data, ignore_index=True)
                logger.info(f"批量写入模式：合并 {len(final_df)} 行，覆盖 {write_count} 只股票，一次性写库")
                
                write_completed = threading.Event()
                write_result = {"success": False, "error": None}
                
                def update_status_callback(response):
                    try:
                        if hasattr(response, 'success') and response.success:
                            write_result["success"] = True
                            logger.info(f"批量写入成功，导入 {response.rows_imported} 条，耗时 {response.execution_time:.2f}s")
                        else:
                            write_result["success"] = False
                            write_result["error"] = getattr(response, 'error', '未知错误')
                            logger.error(f"批量写入失败: {write_result['error']}")
                    finally:
                        write_completed.set()
                
                try:
                    request_id = db_manager.receive_stock_data(
                        source_module="download",
                        data=final_df,
                        mode="upsert",
                        callback=update_status_callback
                    )
                    logger.info(f"批量写入请求已提交 (ID: {request_id})，等待后台线程处理...")
                    if not write_completed.wait(timeout=300):
                        logger.warning("批量写入超时（5分钟），数据可能仍在后台处理中")
                    elif not write_result["success"]:
                        self.stats.error_count = max(self.stats.error_count, 1)
                except Exception as e:
                    logger.error(f"批量写入失败: {e}")
                    self.stats.error_count = max(self.stats.error_count, 1)
            else:
                logger.warning("批量写入模式下无成功数据可写入")
        
        # 所有数据写入后更新状态文件
        if self.stats.success_count > 0:
            try:
                from database_manager import update_stock_data_status
                update_stock_data_status()
                logger.info("状态文件已更新")
            except Exception as e:
                logger.warning(f"更新状态文件失败: {e}")
        
        # 打印统计信息
        logger.info(f"下载完成 - 成功: {self.stats.success_count}, 空数据: {self.stats.empty_count}, 失败: {self.stats.error_count}")
        
        # 打印限频器统计信息
        rate_stats = self.tushare_manager.rate_limiter.get_stats()
        logger.info(f"令牌桶限频器统计 - 总调用: {rate_stats['total_calls']}, 成功率: {rate_stats['success_rate']:.1f}%, "
                   f"补充速率: {rate_stats['current_refill_rate']:.2f}次/秒, 容量: {rate_stats['current_capacity']:.1f}, "
                   f"当前令牌: {rate_stats['current_tokens']:.1f}, 等待时间: {rate_stats['total_wait_time']:.1f}s, "
                   f"限频命中: {rate_stats['rate_limit_hits']}")
        
        return self.stats


class DailyStockDownloader(StockDownloader):
    """按交易日增量下载器：先按日拉取全市场，再统一计算指标并写库。"""

    def __init__(self, config: DownloadConfig):
        super().__init__(config, is_first_download=False)

    def _fetch_daily_frames(self, trade_dates: List[str]) -> List[pd.DataFrame]:
        frames = []
        progress = tqdm(
            total=len(trade_dates),
            desc='按日拉取',
            unit='日',
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            **_tqdm_kwargs()
        )
        for td in trade_dates:
            try:
                daily_df = self.tushare_manager.get_stock_daily_by_date(
                    trade_date=td,
                    fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
                )
                if daily_df is None or daily_df.empty:
                    progress.update(1)
                    continue
                merged = daily_df
                if self.config.download_tor:
                    try:
                        basic_df = self.tushare_manager.get_daily_basic_by_date(
                            trade_date=td,
                            fields='ts_code,trade_date,turnover_rate'
                        )
                        if basic_df is not None and not basic_df.empty:
                            merged = daily_df.merge(
                                basic_df[['ts_code', 'trade_date', 'turnover_rate']],
                                on=['ts_code', 'trade_date'],
                                how='left'
                            )
                    except Exception as e:
                        logger.warning(f"{td} 获取换手率失败，继续：{e}")
                frames.append(merged)
            except Exception as e:
                logger.warning(f"按日拉取失败 {td}: {e}")
            finally:
                progress.update(1)
        progress.close()
        return frames

    def _check_preclose_mismatches_by_dates(self, all_data: pd.DataFrame,
                                            trade_dates: List[str]) -> Dict[str, List[Tuple[str, float, float]]]:
        """
        按交易日逐对检查 pre_close 是否等于上一交易日 close。
        返回 {ts_code: [(trade_date, pre_close, prev_close), ...]}
        """
        mismatches: Dict[str, List[Tuple[str, float, float]]] = {}
        if all_data is None or all_data.empty or not trade_dates:
            return mismatches

        pairs = [(trade_dates[i - 1], trade_dates[i]) for i in range(1, len(trade_dates))]
        if not pairs:
            return mismatches

        check_bar = tqdm(
            total=len(pairs),
            desc="pre_close 校验",
            unit="段",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            **_tqdm_kwargs()
        )
        try:
            for prev_date, curr_date in pairs:
                prev_df = all_data[all_data["trade_date"] == prev_date][["ts_code", "close"]]
                curr_df = all_data[all_data["trade_date"] == curr_date][["ts_code", "pre_close"]]
                if prev_df.empty or curr_df.empty:
                    check_bar.update(1)
                    _refresh_postfix(check_bar, {"区间": f"{prev_date}->{curr_date}"})
                    continue
                merged = curr_df.merge(prev_df, on="ts_code", how="inner")
                if merged.empty:
                    check_bar.update(1)
                    _refresh_postfix(check_bar, {"区间": f"{prev_date}->{curr_date}"})
                    continue
                merged = merged.dropna(subset=["pre_close", "close"])
                if merged.empty:
                    check_bar.update(1)
                    _refresh_postfix(check_bar, {"区间": f"{prev_date}->{curr_date}"})
                    continue
                diff = (merged["pre_close"].astype(float) - merged["close"].astype(float)).abs()
                bad = merged[diff > 1e-6]
                if not bad.empty:
                    for row in bad.itertuples(index=False):
                        ts_code = str(row.ts_code)
                        mismatches.setdefault(ts_code, []).append(
                            (str(curr_date), float(row.pre_close), float(row.close))
                        )
                check_bar.update(1)
                _refresh_postfix(check_bar, {"区间": f"{prev_date}->{curr_date}"})
        finally:
            check_bar.close()

        return mismatches

    def _check_preclose_mismatch_with_warmup(self, all_data: pd.DataFrame) -> Dict[str, List[Tuple[str, float, float]]]:
        """
        对每只股票的增量首日，用 warmup 最新 close 进行 pre_close 校验。
        返回 {ts_code: [(trade_date, pre_close, warmup_close), ...]}
        """
        mismatches: Dict[str, List[Tuple[str, float, float]]] = {}
        if all_data is None or all_data.empty:
            return mismatches
        for ts_code, group in all_data.groupby("ts_code"):
            if group is None or group.empty:
                continue
            warmup_df = self._warmup_cache.get(ts_code)
            if warmup_df is None or warmup_df.empty or "close" not in warmup_df.columns:
                continue
            first_row = group.sort_values("trade_date").iloc[0]
            pre_close = first_row.get("pre_close")
            trade_date = str(first_row.get("trade_date"))
            if pd.isna(pre_close):
                continue
            warmup_latest = warmup_df.sort_values("trade_date").iloc[-1]
            expected = warmup_latest.get("close")
            if pd.isna(expected):
                continue
            try:
                pre_c = float(pre_close)
                exp_c = float(expected)
            except Exception:
                continue
            if abs(pre_c - exp_c) > 1e-6:
                mismatches.setdefault(str(ts_code), []).append((trade_date, pre_c, exp_c))
        return mismatches

    def download_all_stocks(self, stock_codes: Optional[List[str]] = None) -> DownloadStats:
        logger.info("按交易日增量下载模式：按日拉取后统一计算指标")
        if self.db_manager is None:
            logger.info("[数据库连接] 开始获取数据库管理器实例 (按日增量)")
            self.db_manager = get_database_manager()
        data_processor = DataProcessor(self.db_manager)

        trade_dates = self.tushare_manager.get_trade_dates(self.config.start_date, self.config.end_date)
        if not trade_dates:
            logger.warning(f"指定区间内无交易日: {self.config.start_date} - {self.config.end_date}")
            return self.stats

        frames = self._fetch_daily_frames(trade_dates)
        if not frames:
            logger.warning("按日拉取无数据")
            return self.stats

        all_data = pd.concat(frames, ignore_index=True)
        if all_data.empty:
            logger.warning("按日拉取后为空")
            return self.stats

        if stock_codes:
            whitelist = {normalize_ts(c) or str(c).strip() for c in stock_codes if c}
            if whitelist:
                all_data = all_data[all_data['ts_code'].isin(whitelist)]

        # 预载 warmup（需要股票列表）
        codes = sorted(set(all_data['ts_code'].astype(str).tolist()))
        self.stats.total_stocks = len(codes)
        if not codes:
            logger.warning("无可处理股票")
            return self.stats

        if self.config.enable_warmup:
            self._preload_warmup_data(self.db_manager, codes, data_processor)

        # pre_close 校验：逐日检查本次按日下载区间，并补充 warmup 边界检查
        date_mismatches = self._check_preclose_mismatches_by_dates(all_data, trade_dates)
        warmup_mismatches = self._check_preclose_mismatch_with_warmup(all_data)
        for ts_code, rows in warmup_mismatches.items():
            date_mismatches.setdefault(ts_code, []).extend(rows)
        if date_mismatches:
            total_issues = sum(len(v) for v in date_mismatches.values())
            sample = []
            for code, rows in list(date_mismatches.items())[:5]:
                td, pre_c, exp_c = rows[0]
                sample.append(f"{code}:{td} pre_close={pre_c} != prev_close={exp_c}")
            logger.warning(
                f"pre_close 校验发现不一致: 股票 {len(date_mismatches)} 只, 记录 {total_issues} 条; 示例: {sample}"
            )

        indicator_names = data_processor.indicator_names
        decs = outputs_for(indicator_names)

        tasks = []
        for ts_code, group in all_data.groupby('ts_code'):
            if group is None or group.empty:
                self.stats.empty_count += 1
                continue
            if ts_code in date_mismatches:
                trade_date, pre_c, exp_c = date_mismatches[ts_code][0]
                msg = f"{ts_code} {trade_date} pre_close={pre_c} 与上一交易日收盘 {exp_c} 不一致"
                self.stats.error_count += 1
                self.stats.failed_stocks.append((ts_code, msg))
                continue
            cleaned = data_processor.clean_stock_data(group, ts_code)
            if cleaned is None or cleaned.empty:
                self.stats.empty_count += 1
                continue
            tasks.append((ts_code, cleaned, self.config.adj_type, self._warmup_cache.get(ts_code), indicator_names, decs))

        if not tasks:
            logger.warning("按日拉取后无可计算的数据")
            return self.stats

        workers = max(2, min(self.config.threads, os.cpu_count() or 2))
        results = []
        compute_bar = tqdm(
            total=len(tasks),
            desc='指标计算',
            unit='只',
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            **_tqdm_kwargs()
        )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_code = {pool.submit(_compute_indicators_task, t): t[0] for t in tasks}
            for fut in as_completed(future_to_code):
                ts_code = future_to_code[fut]
                try:
                    code, data, err = fut.result()
                    if err or data is None or data.empty:
                        self.stats.error_count += 1
                        self.stats.failed_stocks.append((ts_code, err or 'compute_empty'))
                    else:
                        results.append(data)
                        self.stats.success_count += 1
                    compute_bar.update(1)
                    _refresh_postfix(compute_bar, {'成功': self.stats.success_count, '失败': self.stats.error_count})
                except Exception as e:
                    self.stats.error_count += 1
                    self.stats.failed_stocks.append((ts_code, str(e)))
                    compute_bar.update(1)
        compute_bar.close()

        # 过滤掉空或全 NA 的结果，避免 pandas FutureWarning
        filtered_results = []
        for df in results:
            if df is None or df.empty:
                continue
            trimmed = df.dropna(axis=1, how="all")
            if trimmed.empty or trimmed.isna().all().all():
                continue
            filtered_results.append(trimmed)
        if not filtered_results:
            logger.warning("指标计算结果为空")
            return self.stats

        final_df = pd.concat(filtered_results, ignore_index=True)
        logger.info(f"按日模式合并后准备写入 {len(final_df)} 行，覆盖 {self.stats.success_count} 只股票")

        write_completed = threading.Event()
        write_result = {'success': False, 'error': None}

        def update_status_callback(response):
            try:
                if hasattr(response, 'success') and response.success:
                    write_result['success'] = True
                    logger.info(f"按日增量写入成功，导入 {response.rows_imported} 条，耗时 {response.execution_time:.2f}s")
                    from database_manager import update_stock_data_status
                    update_stock_data_status()
                else:
                    write_result['success'] = False
                    write_result['error'] = getattr(response, 'error', '未知错误')
                    logger.error(f"按日增量写入失败: {write_result['error']}")
            finally:
                write_completed.set()

        try:
            request_id = self.db_manager.receive_stock_data(
                source_module='download',
                data=final_df,
                mode='upsert',
                callback=update_status_callback
            )
            logger.info(f"写入请求已提交 (ID: {request_id})，等待后台线程处理...")
            if not write_completed.wait(timeout=300):
                logger.warning("按日增量写入超时（5分钟），数据可能仍在后台处理中")
            elif not write_result['success']:
                self.stats.error_count = max(self.stats.error_count, 1)
        except Exception as e:
            logger.error(f"按日增量写入失败: {e}")
            self.stats.error_count = max(self.stats.error_count, 1)
        finally:
            rate_stats = self.tushare_manager.rate_limiter.get_stats()
            logger.info(
                f"令牌桶限频器统计 - 总调用: {rate_stats['total_calls']}, 成功率: {rate_stats['success_rate']:.1f}%, "
                f"补充速率: {rate_stats['current_refill_rate']:.2f}次/秒, 容量: {rate_stats['current_capacity']:.1f}, "
                f"当前令牌: {rate_stats['current_tokens']:.1f}, 等待时间: {rate_stats['total_wait_time']:.1f}s, "
                f"限频命中: {rate_stats['rate_limit_hits']}"
            )

        return self.stats


class IndexDownloader:
    """指数数据下载器"""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        # 不在初始化时创建数据库连接，避免子线程创建连接
        
        # 创建限频器（使用令牌桶算法）
        rate_limiter = TokenBucketRateLimiter(
            calls_per_min=config.rate_limit_calls_per_min,
            safe_calls_per_min=config.safe_calls_per_min,
            adaptive=config.enable_adaptive_rate_limit
        )
        
        self.tushare_manager = TushareManager(
            token=config.token,
            rate_limiter=rate_limiter,
        )
        self.stats = DownloadStats()
        self._lock = threading.Lock()
    
    def download_index_data(self, ts_code: str) -> Tuple[str, str, Optional[pd.DataFrame]]:
        """下载单只指数数据，返回数据而不直接写入数据库
        
        Returns:
            Tuple[ts_code, status, data]:
                - ts_code: 指数代码
                - status: 状态 ("success", "empty", "error:xxx")
                - data: 准备好的数据（DataFrame），如果失败则为None
        """
        max_retries = self.config.retry_times
        retry_delay = 1.0
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # 下载原始数据
                logger.debug(f"开始下载 {ts_code} 的原始数据... (尝试 {attempt + 1}/{max_retries})")
                raw_data = self.tushare_manager.get_index_daily(
                    ts_code=ts_code,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date
                )
                
                if raw_data is None or raw_data.empty:
                    logger.debug(f"{ts_code} 无数据")
                    return ts_code, "empty", None
                
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
                
                logger.debug(f"{ts_code} 下载完成，共 {len(df)} 条记录")
                return ts_code, "success", df
                
            except Exception as e:
                error_msg = str(e)
                error_msg_lower = error_msg.lower()
                last_error = e
                
                # 认证错误不重试
                if "token" in error_msg_lower or "auth" in error_msg_lower or "1002" in error_msg:
                    logger.error(f"Token认证失败 {ts_code}: {error_msg}，不重试")
                    with self._lock:
                        self.stats.error_count += 1
                        self.stats.failed_stocks.append((ts_code, error_msg))
                    return ts_code, f"error: {error_msg}", None
                
                # 其他错误可以重试
                if attempt < max_retries - 1:
                    base_wait_time = retry_delay * (2 ** attempt)  # 指数退避
                    # 添加±20%的随机抖动，避免多个请求同时重试
                    jitter = random.uniform(-0.2, 0.2) * base_wait_time
                    wait_time = max(0.1, base_wait_time + jitter)  # 确保等待时间至少0.1秒
                    logger.warning(f"下载指数数据失败 {ts_code} (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                    logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 所有重试都失败了
                    logger.error(f"下载指数数据最终失败 {ts_code} (已重试 {max_retries} 次): {error_msg}")
                    with self._lock:
                        self.stats.error_count += 1
                        self.stats.failed_stocks.append((ts_code, error_msg))
                    return ts_code, f"error: {error_msg}", None
        
        # 如果所有重试都失败，返回错误
        if last_error:
            error_msg = str(last_error)
            logger.error(f"下载指数数据失败 {ts_code} (已重试 {max_retries} 次): {error_msg}")
            with self._lock:
                self.stats.error_count += 1
                self.stats.failed_stocks.append((ts_code, error_msg))
            return ts_code, f"error: {error_msg}", None
        
        return ts_code, "error: 未知错误", None
    
    def download_all_indices(self, whitelist: List[str] = None) -> DownloadStats:
        """下载所有指数数据，由主线程统一写入数据库"""
        logger.info("开始下载指数数据...")
        
        # 使用白名单或默认指数列表
        if whitelist is None:
            whitelist = INDEX_WHITELIST
        
        if not whitelist:
            logger.error("指数白名单为空，无法下载")
            return self.stats
        
        self.stats.total_stocks = len(whitelist)
        logger.info(f"准备下载 {len(whitelist)} 只指数的数据，日期范围: {self.config.start_date} - {self.config.end_date}")
        
        # 在主线程中创建数据库连接
        logger.info("[数据库连接] 开始获取数据库管理器实例 (主线程统一写入指数)")
        db_manager = get_database_manager()
        
        # 创建进度条
        progress_bar = tqdm(
            total=len(whitelist),
            desc="下载指数数据",
            unit="只",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            **_tqdm_kwargs()
        )
        
        # 收集所有下载的数据
        collected_data = []  # 存储所有成功下载的数据
        
        # 并发下载（不写入数据库）
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
                    result_code, status, data = future.result()
                    if status == "empty":
                        with self._lock:
                            self.stats.empty_count += 1
                    elif status == "success" and data is not None:
                        # 收集数据，稍后统一写入
                        collected_data.append(data)
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
                _refresh_postfix(progress_bar, {
                    '成功': len(collected_data),
                    '空数据': self.stats.empty_count,
                    '失败': self.stats.error_count
                })
        
        # 关闭进度条
        progress_bar.close()
        
        # 在主线程中统一写入数据库
        if collected_data:
            logger.info(f"开始统一写入 {len(collected_data)} 只指数的数据到数据库...")
            try:
                # 合并所有数据
                all_data = pd.concat(collected_data, ignore_index=True)
                logger.info(f"合并后共 {len(all_data)} 条记录")
                
                # 统一写入数据库
                logger.info(f"提交写入数据库请求，共 {len(all_data)} 条记录...")
                
                # 使用回调函数在写入完成后更新状态文件
                write_completed = threading.Event()
                write_result = {"success": False, "error": None}
                
                def update_status_callback(response):
                    try:
                        if hasattr(response, 'success') and response.success:
                            write_result["success"] = True
                            logger.info(f"数据已成功写入数据库，导入 {response.rows_imported} 条记录，耗时 {response.execution_time:.2f} 秒")
                            from database_manager import update_stock_data_status
                            update_stock_data_status()
                        else:
                            write_result["success"] = False
                            write_result["error"] = getattr(response, 'error', '未知错误')
                            logger.error(f"数据写入数据库失败: {write_result['error']}")
                    except Exception as e:
                        logger.warning(f"更新状态文件失败: {e}")
                    finally:
                        write_completed.set()
                
                request_id = db_manager.receive_stock_data(
                    source_module="download",
                    data=all_data,
                    mode="upsert",  # 使用upsert模式避免重复数据
                    callback=update_status_callback
                )
                
                logger.info(f"写入请求已提交 (ID: {request_id})，等待后台线程处理...")
                
                # 等待写入完成（最多等待5分钟）
                if write_completed.wait(timeout=300):
                    if not write_result["success"]:
                        raise RuntimeError(f"数据写入失败: {write_result['error']}")
                else:
                    logger.warning("写入请求超时（5分钟），数据可能仍在后台处理中")
                
                with self._lock:
                    self.stats.success_count = len(collected_data)
                
                logger.info(f"统一写入完成，共 {len(all_data)} 条记录")
            except Exception as e:
                logger.error(f"统一写入数据库失败: {e}")
                raise
        
        # 打印统计信息
        logger.info(f"下载完成 - 成功: {self.stats.success_count}, 空数据: {self.stats.empty_count}, 失败: {self.stats.error_count}")
        
        # 打印限频器统计信息
        rate_stats = self.tushare_manager.rate_limiter.get_stats()
        logger.info(f"令牌桶限频器统计 - 总调用: {rate_stats['total_calls']}, 成功率: {rate_stats['success_rate']:.1f}%, "
                   f"补充速率: {rate_stats['current_refill_rate']:.2f}次/秒, 容量: {rate_stats['current_capacity']:.1f}, "
                   f"当前令牌: {rate_stats['current_tokens']:.1f}, 等待时间: {rate_stats['total_wait_time']:.1f}s, "
                   f"限频命中: {rate_stats['rate_limit_hits']}")
        
        return self.stats

# ================= 主下载器 =================
class DownloadManager:
    """下载管理器"""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        logger.info("[数据库连接] 开始获取数据库管理器实例 (初始化下载管理器)")
        self.db_manager = get_database_manager()
        self.tushare_manager = TushareManager(token=config.token, db_manager=self.db_manager)
        
        # 在数据库初始化之前判断是否为首次下载
        self.is_first_download = self._check_if_first_download()
        self._ensure_database_initialized()

    def _persist_stock_list_cache(self, df: pd.DataFrame) -> None:
        """把股票列表缓存到数据库目录下的 stock_list.csv，供其他模块复用。"""
        try:
            if df is None or df.empty or "ts_code" not in df.columns:
                return
            cache_path = stock_list_cache_path()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cols = [c for c in ["ts_code", "symbol", "name", "area", "industry", "list_date"] if c in df.columns]
            df[cols].to_csv(cache_path, index=False, encoding="utf-8-sig")
            logger.info(f"股票列表已缓存：{cache_path}（保留，不再清理）")
        except Exception as e:
            logger.warning(f"写入股票列表缓存失败：{e}")
    
    def _check_if_first_download(self) -> bool:
        """检查是否为首次下载（使用状态文件，不读取数据库）"""
        try:
            from database_manager import get_database_manager
            
            # 检查状态文件（load_status_file内部已经处理了所有异常，返回None或字典）
            manager = get_database_manager()
            status = manager.load_status_file()
            
            # 状态文件不存在或读取失败，判断为首次下载
            if status is None:
                logger.info("状态文件不存在或读取失败，判断为首次下载")
                return True
            
            # 检查状态文件格式是否正确
            if not isinstance(status, dict):
                logger.warning(f"状态文件格式错误（不是字典），判断为首次下载")
                return True
            
            # 检查股票数据状态
            stock_data_status = status.get("stock_data", {})
            if not isinstance(stock_data_status, dict):
                logger.warning(f"状态文件中stock_data格式错误，判断为首次下载")
                return True
                
            if not stock_data_status.get("database_exists", False):
                logger.info("状态文件显示数据库不存在，判断为首次下载")
                return True
            
            # 检查是否有数据
            total_records = stock_data_status.get("total_records", 0)
            if total_records == 0:
                logger.info("状态文件显示数据库无数据，判断为首次下载")
                return True
            
            # 有数据，判断为增量下载
            max_date = None
            adj_types = stock_data_status.get("adj_types", {})
            if isinstance(adj_types, dict):
                for adj_type, adj_status in adj_types.items():
                    if isinstance(adj_status, dict):
                        adj_max_date = adj_status.get("max_date")
                        if adj_max_date and (max_date is None or adj_max_date > max_date):
                            max_date = adj_max_date
            
            if max_date:
                logger.info(f"状态文件显示数据库有数据，最新日期: {max_date}，判断为增量下载")
            else:
                logger.info("状态文件显示数据库有数据，判断为增量下载")
            
            return False
                
        except Exception as e:
            # 只有在获取DatabaseManager失败时才报错，这种情况下不应该读数据库
            # 因为如果没有db_manager，也无法读数据库
            logger.error(f"检查首次下载状态失败（获取DatabaseManager失败）: {e}，判断为首次下载")
            import traceback
            logger.debug(f"异常详情: {traceback.format_exc()}")
            # 如果获取DatabaseManager都失败了，无法读数据库，直接判断为首次下载
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

    def _get_reference_stock_codes(self) -> List[str]:
        """
        获取用于完整性校验的股票代码列表
        
        优先顺序：
        1. 本地缓存 stock_list.csv（避免频繁访问API）
        2. 实时调用 Tushare 获取（与下载流程保持一致）
        3. 数据库现有股票列表（按配置的复权类型，兜底）
        """
        # 1) 读取缓存
        try:
            cached_codes = self.db_manager.get_stock_list_from_cache()
            if cached_codes:
                logger.debug(f"使用缓存股票列表进行完整性校验，共 {len(cached_codes)} 只")
                return cached_codes
        except Exception as e:
            logger.debug(f"读取缓存股票列表失败，尝试其他来源: {e}")
        
        # 2) 调用Tushare（当缓存不存在时必须拉取）
        try:
            stock_list_df = self.tushare_manager.get_stock_list()
            if stock_list_df is not None and not stock_list_df.empty and "ts_code" in stock_list_df.columns:
                self._persist_stock_list_cache(stock_list_df)
                codes = stock_list_df["ts_code"].astype(str).tolist()
                logger.info(f"使用Tushare股票列表进行完整性校验，共 {len(codes)} 只")
                return codes
            logger.warning("Tushare返回的股票列表为空，无法进行完整性校验")
        except Exception as e:
            logger.error(f"调用Tushare获取股票列表失败，无法进行完整性校验: {e}")
        
        # 3) 从数据库读取（兜底）
        try:
            db_codes = self.db_manager.get_stock_list(adj_type=self.config.adj_type)
            if db_codes:
                logger.debug(f"使用数据库股票列表进行完整性校验，共 {len(db_codes)} 只")
                return db_codes
        except Exception as e:
            logger.debug(f"从数据库获取股票列表失败，兜底失败: {e}")
        
        return []

    def _check_latest_date_completeness(self, trade_date: str) -> Dict[str, Any]:
        """
        校验指定日期的股票数据是否完整
        
        Returns:
            {
                "trade_date": str,
                "can_check": bool,
                "expected_count": int,
                "actual_count": int,
                "missing_codes": List[str]
            }
        """
        result = {
            "trade_date": trade_date,
            "can_check": False,
            "expected_count": 0,
            "actual_count": 0,
            "missing_codes": []
        }
        
        if not trade_date:
            logger.warning("未提供需要校验的日期，无法检查数据完整性")
            return result
        
        reference_codes = self._get_reference_stock_codes()
        if not reference_codes:
            logger.warning("无法获取股票列表，跳过最新日期数据完整性检查")
            return result
        
        result["expected_count"] = len(reference_codes)
        result["can_check"] = True
        
        try:
            df = self.db_manager.query_stock_data(
                start_date=trade_date,
                end_date=trade_date,
                columns=["ts_code"],
                adj_type=self.config.adj_type
            )
            
            if df.empty or "ts_code" not in df.columns:
                logger.warning(f"数据库中未查询到日期 {trade_date} 的股票数据，将视为全部缺失")
                result["actual_count"] = 0
                result["missing_codes"] = reference_codes.copy()
                return result
            
            actual_codes: Set[str] = set(df["ts_code"].astype(str).tolist())
            result["actual_count"] = len(actual_codes)
            
            missing_codes = [code for code in reference_codes if code and code not in actual_codes]
            result["missing_codes"] = missing_codes
            
            if missing_codes:
                logger.warning(
                    f"检测到最新日期 {trade_date} 缺失 {len(missing_codes)} 只股票数据 "
                    f"(实际: {result['actual_count']}, 期望: {result['expected_count']})"
                )
            else:
                logger.info(
                    f"最新日期 {trade_date} 数据完整 "
                    f"(股票数: {result['actual_count']}/{result['expected_count']})"
                )
            
            return result

        except Exception as e:
            logger.warning(f"检查日期 {trade_date} 的股票数据完整性失败: {e}")
            result["can_check"] = False
            result["missing_codes"] = []
            return result

    def _prompt_latest_date_action(self, latest_date: str, end_date: str,
                                   coverage_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """当数据库已覆盖到结束日期时，提供交互式选项。"""
        if not sys.stdin.isatty():
            return None

        missing_count = len(coverage_info.get("missing_codes") or [])
        can_check = bool(coverage_info.get("can_check", False))
        if can_check:
            missing_hint = f"缺失 {missing_count} 只" if missing_count else "无缺失"
        else:
            missing_hint = "无法校验完整性"

        print("\n数据库最新日期已覆盖到结束日期。")
        print(f"最新日期: {latest_date}，结束日期: {end_date}，完整性: {missing_hint}")
        print("选项:")
        print("1. 正常补齐")
        print("2. 强制重建（输入重建多少日数据，增量模式重建）")
        print("3. 退出")

        choice = input("请选择 (1-3, 默认: 1): ").strip() or "1"
        if choice == "2":
            while True:
                days_str = input("请输入重建的交易日数量 (>=1): ").strip()
                if not days_str:
                    continue
                if days_str.isdigit() and int(days_str) > 0:
                    return {"action": "rebuild", "days": int(days_str)}
                print("输入无效，请输入正整数。")
        if choice == "3":
            return {"action": "exit"}
        return {"action": "normal"}

    def _calc_rebuild_start_date(self, end_date: str, days: int) -> str:
        """根据交易日历计算强制重建的开始日期。"""
        if not end_date or days <= 1:
            return end_date
        try:
            trade_dates = self.db_manager.get_trade_dates()
            if trade_dates:
                trade_dates = [d for d in trade_dates if d and str(d) <= end_date]
                if trade_dates:
                    idx = max(0, len(trade_dates) - days)
                    return str(trade_dates[idx])
        except Exception as e:
            logger.warning(f"获取交易日历失败，回退到自然日计算: {e}")
        try:
            from datetime import datetime, timedelta
            end_obj = datetime.strptime(str(end_date), "%Y%m%d")
            start_obj = end_obj - timedelta(days=days - 1)
            return start_obj.strftime("%Y%m%d")
        except Exception:
            return end_date

    def determine_download_strategy(self) -> Dict[str, Any]:
        """确定下载策略"""
        # 使用预先判断的首次下载状态
        is_first_download = self.is_first_download
        
        # 获取最新数据日期
        latest_date = None
        if not is_first_download:
            try:
                # 从数据库获取最新日期（使用与_check_if_first_download相同的路径）
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                logger.debug(f"[determine_download_strategy] 使用数据库路径: {db_path}")
                
                # 从数据库获取最新日期（确保使用相同的路径）
                latest_date = self.db_manager.get_latest_trade_date(db_path)
                logger.debug(f"[determine_download_strategy] 获取到最新日期: {latest_date}")
                
                # 如果获取不到最新日期，说明数据库不可读或为空
                if not latest_date:
                    status = self.db_manager.load_status_file()
                    status_db_exists = False
                    if isinstance(status, dict):
                        stock_status = status.get("stock_data", {})
                        if isinstance(stock_status, dict):
                            status_db_exists = bool(stock_status.get("database_exists", False))
                    if not status_db_exists:
                        logger.info("状态文件显示数据库不存在，重新判断为首次下载")
                        is_first_download = True
                    else:
                        raise RuntimeError("状态文件显示数据库存在但无法读取最新日期，可能被占用，请关闭占用后重试。")
            except Exception as e:
                logger.error(f"获取最新日期失败: {e}")
                import traceback
                logger.debug(f"获取最新日期失败详情: {traceback.format_exc()}")
                raise
        else:
            # 首次下载，不需要查询数据库
            logger.info("首次下载，跳过数据库查询")
        
        # 处理结束日期：如果配置为"today"，先获取智能结束日期
        actual_end_date = self.config.end_date
        if self.config.end_date.lower() == "today":
            actual_end_date = self.get_smart_end_date("today")
            logger.info(f"结束日期配置为'today'，智能判断后实际结束日期: {actual_end_date}")
        
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
                if actual_start_date > actual_end_date:
                    coverage_info = self._check_latest_date_completeness(latest_date)
                    missing_codes = coverage_info.get("missing_codes", []) if isinstance(coverage_info, dict) else []
                    can_check = coverage_info.get("can_check", False) if isinstance(coverage_info, dict) else False

                    action = self._prompt_latest_date_action(latest_date, actual_end_date, coverage_info)
                    if action:
                        if action.get("action") == "exit":
                            logger.info("用户选择退出下载流程")
                            return {
                                "skip_download": True,
                                "user_exit": True,
                                "is_first_download": is_first_download,
                                "latest_date": latest_date,
                                "start_date": actual_start_date,
                                "end_date": actual_end_date,
                                "adj_type": self.config.adj_type,
                                "asset_type": self.config.asset_type,
                                "stock_whitelist": None,
                                "latest_date_completeness": coverage_info,
                                "fill_missing_latest_date": False
                            }
                        if action.get("action") == "rebuild":
                            rebuild_days = int(action.get("days", 1))
                            rebuild_start = self._calc_rebuild_start_date(actual_end_date, rebuild_days)
                            logger.warning(f"用户选择强制重建最近 {rebuild_days} 日数据: {rebuild_start} - {actual_end_date}")
                            return {
                                "skip_download": False,
                                "is_first_download": False,
                                "latest_date": latest_date,
                                "start_date": rebuild_start,
                                "end_date": actual_end_date,
                                "adj_type": self.config.adj_type,
                                "asset_type": self.config.asset_type,
                                "stock_whitelist": None,
                                "latest_date_completeness": coverage_info,
                                "fill_missing_latest_date": False,
                                "rebuild_days": rebuild_days
                            }
                    
                    if can_check and missing_codes:
                        logger.warning(
                            f"最新日期 {latest_date} 虽然与目标结束日期一致，但检测到 {len(missing_codes)} 只股票缺失数据，"
                            f"将强制补齐该日期的缺失股票"
                        )
                        return {
                            "skip_download": False,
                            "is_first_download": is_first_download,
                            "latest_date": latest_date,
                            "start_date": latest_date,
                            "end_date": latest_date,
                            "adj_type": self.config.adj_type,
                            "asset_type": self.config.asset_type,
                            "stock_whitelist": missing_codes,
                            "latest_date_completeness": coverage_info,
                            "fill_missing_latest_date": True
                        }
                    
                    if not can_check:
                        logger.warning(
                            f"无法校验最新日期 {latest_date} 的数据完整性，默认认为数据已最新，跳过下载"
                        )
                    else:
                        logger.info(
                            f"数据已是最新，无需下载 (最新日期: {latest_date}, 计算出的开始日期: {actual_start_date}, "
                            f"实际结束日期: {actual_end_date})"
                        )
                    
                    logger.info(f"跳过下载原因: 开始日期({actual_start_date}) > 结束日期({actual_end_date})")
                    return {
                        "skip_download": True,
                        "is_first_download": is_first_download,
                        "latest_date": latest_date,
                        "start_date": actual_start_date,
                        "end_date": actual_end_date,
                        "adj_type": self.config.adj_type,
                        "asset_type": self.config.asset_type,
                        "stock_whitelist": None,
                        "latest_date_completeness": coverage_info,
                        "fill_missing_latest_date": False
                    }
            except Exception as e:
                logger.warning(f"计算增量下载日期失败: {e}")
        
        strategy = {
            "is_first_download": is_first_download,
            "latest_date": latest_date,
            "start_date": actual_start_date,
            "end_date": actual_end_date,
            "adj_type": self.config.adj_type,
            "asset_type": self.config.asset_type,
            "skip_download": False,
            "stock_whitelist": None,
            "latest_date_completeness": None,
            "fill_missing_latest_date": False
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
            enable_warmup=self.config.enable_warmup,
            token=self.config.token,
            rate_limit_calls_per_min=self.config.rate_limit_calls_per_min,
            safe_calls_per_min=self.config.safe_calls_per_min,
            enable_adaptive_rate_limit=self.config.enable_adaptive_rate_limit,
            download_tor=self.config.download_tor,
        )
        
        # 传递是否为首次下载的信息；增量重试或补齐场景可强制回退到逐股模式
        is_first_download = strategy.get("is_first_download", False)
        force_per_stock_retry = strategy.get("force_per_stock_retry", False)
        batch_write_once = strategy.get("batch_write_once", False)
        # 补齐最新日期缺失数据时也使用逐股下载，避免按日模式遗漏
        fill_missing_latest_date = strategy.get("fill_missing_latest_date", False)
        per_stock_mode = is_first_download or force_per_stock_retry or fill_missing_latest_date
        if per_stock_mode:
            downloader = StockDownloader(
                stock_config,
                is_first_download=is_first_download,
                batch_write_once=batch_write_once
            )
        else:
            downloader = DailyStockDownloader(stock_config)
        
        stock_whitelist = strategy.get("stock_whitelist")
        if stock_whitelist:
            logger.warning(
                f"本次仅下载 {len(stock_whitelist)} 只股票以补齐最新日期 {strategy.get('end_date')}"
            )
        
        return downloader.download_all_stocks(stock_whitelist)
    
    def download_indices(self, strategy: Dict[str, Any], whitelist: List[str] = None) -> DownloadStats:
        """下载指数数据"""
        if strategy.get("skip_download", False):
            logger.info("跳过指数数据下载")
            return DownloadStats()
        
        # 指数也做一次增量检查，避免重复全量下载
        start_date = strategy["start_date"]
        end_date = strategy["end_date"]
        index_latest = None
        try:
            status = self.db_manager.get_stock_data_status(use_cache=False)
            index_latest = (status.get("adj_types", {}) or {}).get("ind", {}).get("max_date")
        except Exception as e:
            logger.warning(f"获取指数最新日期失败，按原配置下载: {e}")
        
        if index_latest:
            try:
                from datetime import datetime, timedelta
                next_date = (datetime.strptime(str(index_latest), "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
                # 如果指数已经覆盖到结束日期，跳过下载
                if next_date > end_date:
                    logger.info(f"指数数据已覆盖到 {index_latest}，结束日期为 {end_date}，跳过指数下载")
                    stats = DownloadStats()
                    stats.skip_count = len(whitelist) if whitelist else 0
                    return stats
                # 否则从最新日期的下一天开始增量
                if next_date > start_date:
                    logger.info(f"指数最新日期 {index_latest}，增量开始日期调整为 {next_date}")
                    start_date = next_date
            except Exception as e:
                logger.warning(f"解析指数最新日期失败，按原配置下载: {e}")
        
        index_config = DownloadConfig(
            start_date=start_date,
            end_date=end_date,
            adj_type="ind",  # 指数使用ind
            asset_type="index",
            threads=self.config.threads,
            enable_warmup=False,  # 指数不需要指标计算
            token=self.config.token,
        )
        
        downloader = IndexDownloader(index_config)
        return downloader.download_all_indices(whitelist)
    
    def run_download(self, assets: List[str] = None, index_whitelist: List[str] = None,
                     retry_on_failure: bool = True) -> Dict[str, DownloadStats]:
        """运行下载任务"""
        if assets is None:
            assets = ["stock"]
        
        if index_whitelist is None:
            index_whitelist = INDEX_WHITELIST
        
        # 确定下载策略（内部已经处理了智能结束日期）
        strategy = self.determine_download_strategy()
        
        def _collect_totals(res: Dict[str, DownloadStats]) -> Tuple[int, int, int]:
            total_success = sum(stats.success_count for stats in res.values())
            total_error = sum(stats.error_count for stats in res.values())
            total_empty = sum(stats.empty_count for stats in res.values())
            return total_success, total_error, total_empty
        
        def _log_results(res: Dict[str, DownloadStats], heading: str) -> Tuple[int, int, int]:
            total_success, total_error, total_empty = _collect_totals(res)
            logger.info(heading)
            for asset_type, stats in res.items():
                logger.info(f"{asset_type}: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
                if stats.failed_stocks:
                    failed_codes = [code for code, _ in stats.failed_stocks if code]
                    logger.warning(f"{asset_type} 失败股票: {failed_codes}")
            logger.info(f"总计: 成功={total_success}, 空数据={total_empty}, 失败={total_error}")
            return total_success, total_error, total_empty
        
        def _unique_failed_codes(failed_list: List[Tuple[str, str]]) -> List[str]:
            codes: List[str] = []
            seen: Set[str] = set()
            for code, _ in failed_list or []:
                if code and code not in seen:
                    seen.add(code)
                    codes.append(code)
            return codes
        
        def _merge_stats(original: DownloadStats, retry_stats: DownloadStats) -> DownloadStats:
            """合并重试结果，保留首次统计中的总量与跳过信息。"""
            merged = DownloadStats()
            merged.total_stocks = original.total_stocks
            merged.skip_count = original.skip_count
            merged.empty_count = original.empty_count + retry_stats.empty_count
            merged.success_count = original.success_count + retry_stats.success_count
            merged.error_count = retry_stats.error_count
            merged.failed_stocks = retry_stats.failed_stocks or []
            return merged
        
        # 检查是否需要跳过下载（必须在访问其他键之前检查）
        if strategy.get("skip_download", False):
            if strategy.get("user_exit"):
                logger.info("用户选择退出下载流程")
                return {"stock": DownloadStats(), "index": DownloadStats()}
            logger.warning("=" * 30)
            logger.warning("数据已是最新，跳过下载")
            logger.warning("=" * 30)
            logger.warning(f"跳过原因: 最新日期={strategy.get('latest_date', 'N/A')}, 计算出的开始日期={strategy.get('start_date', 'N/A')}, 实际结束日期={strategy.get('end_date', 'N/A')}")
            logger.warning(f"如果最新日期 >= 结束日期，说明数据已经是最新的，无需下载")
            logger.warning("=" * 30)
            return {"stock": DownloadStats(), "index": DownloadStats()}
        
        if strategy.get("fill_missing_latest_date"):
            missing_count = len(strategy.get("stock_whitelist") or [])
            logger.warning(
                f"触发最新日期补齐流程: 需要补齐 {missing_count} 只股票 (日期 {strategy.get('end_date')})"
            )
        if strategy.get("rebuild_days"):
            logger.warning(
                f"强制重建模式: 最近 {strategy.get('rebuild_days')} 日 "
                f"({strategy.get('start_date')} - {strategy.get('end_date')})"
            )
        
        # 添加智能日期判断的详细说明（如果配置为"today"）
        if self.config.end_date.lower() == "today":
            # strategy['end_date'] 已经是智能处理后的日期
            smart_end_date = strategy['end_date']
            logger.info("=" * 30)
            logger.info("智能日期判断说明")
            logger.info("=" * 30)
            logger.info(f"• 配置的结束日期: today (今日)")
            logger.info(f"• 智能判断后的实际结束日期: {smart_end_date}")
            logger.info("• 智能判断逻辑:")
            logger.info("  - 如果今天是交易日且当前时间 < 15:00，使用前一个交易日")
            logger.info("  - 如果今天是交易日且当前时间 >= 15:00，使用今天")
            logger.info("  - 如果今天不是交易日，使用最近的交易日")
            logger.info("  - 如果无法获取交易日历，使用今天")
            logger.info("=" * 30)
        
        # 显示中文下载策略信息（使用strategy中已经智能处理后的结束日期）
        logger.info(f"下载策略: 首次下载={strategy['is_first_download']}, 最新日期={strategy['latest_date']}, 开始日期={strategy['start_date']}, 结束日期={strategy['end_date']}, 复权类型={strategy['adj_type']}, 资产类型={strategy['asset_type']}, 跳过下载={strategy['skip_download']}")
        
        results = {}
        total_assets = len(assets)
        current_asset = 0
        
        # 创建总体进度条
        overall_progress = tqdm(
            total=total_assets,
            desc="总体下载进度",
            unit="类",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            **_tqdm_kwargs(position=0, leave=True)
        )
        
        try:
            # 下载股票数据
            if "stock" in assets:
                current_asset += 1
                overall_progress.set_description(f"下载股票数据 ({strategy['start_date']} - {strategy['end_date']})")
                logger.info(f"开始下载股票数据: {strategy['start_date']} - {strategy['end_date']}")
                results["stock"] = self.download_stocks(strategy)
                overall_progress.update(1)
                _refresh_postfix(overall_progress, {
                    '股票成功': results["stock"].success_count,
                    '股票失败': results["stock"].error_count
                })
            
            # 下载指数数据
            if "index" in assets:
                current_asset += 1
                overall_progress.set_description(f"下载指数数据 ({strategy['start_date']} - {strategy['end_date']})")
                logger.info(f"开始下载指数数据: {strategy['start_date']} - {strategy['end_date']}")
                results["index"] = self.download_indices(strategy, index_whitelist)
                overall_progress.update(1)
                _refresh_postfix(overall_progress, {
                    '指数成功': results["index"].success_count,
                    '指数失败': results["index"].error_count
                })
            
            # 显示最终统计
            overall_progress.set_description("下载完成")
            total_success = sum(stats.success_count for stats in results.values())
            total_error = sum(stats.error_count for stats in results.values())
            total_empty = sum(stats.empty_count for stats in results.values())
            
            _refresh_postfix(overall_progress, {
                '总成功': total_success,
                '总失败': total_error,
                '总空数据': total_empty
            })
            
        finally:
            overall_progress.close()
        
        # 打印首次尝试的统计信息
        total_success, total_error, total_empty = _log_results(results, "=== 下载任务完成（首次尝试） ===")
        
        # 如果有失败并且允许自动重试，针对失败项再尝试一次
        if retry_on_failure and total_error > 0:
            retry_results: Dict[str, DownloadStats] = {}
            retry_stock_codes: List[str] = []
            retry_index_codes: List[str] = []
            
            if "stock" in assets and "stock" in results:
                retry_stock_codes = _unique_failed_codes(results["stock"].failed_stocks)
            if "index" in assets and "index" in results:
                retry_index_codes = _unique_failed_codes(results["index"].failed_stocks)
            
            if retry_stock_codes or retry_index_codes:
                logger.warning(f"检测到 {total_error} 个失败，开始自动重试一次...")
                if retry_stock_codes:
                    retry_strategy = dict(strategy)
                    retry_strategy["stock_whitelist"] = retry_stock_codes
                    retry_strategy["skip_download"] = False
                    retry_strategy["start_date"] = self.config.start_date
                    # 重试时强制回退到逐股下载，避免按日模式再失败
                    retry_strategy["force_per_stock_retry"] = True
                    # 重试阶段按需求一次性写盘，避免多次写入
                    retry_strategy["batch_write_once"] = True
                    logger.info(f"股票重试列表共 {len(retry_stock_codes)} 只")
                    retry_results["stock"] = self.download_stocks(retry_strategy)
                
                if retry_index_codes:
                    retry_index_strategy = dict(strategy)
                    retry_index_strategy["skip_download"] = False
                    retry_index_strategy["start_date"] = self.config.start_date
                    logger.info(f"指数重试列表共 {len(retry_index_codes)} 只")
                    retry_results["index"] = self.download_indices(retry_index_strategy, retry_index_codes)
                
                # 合并重试结果
                if retry_results:
                    merged_results: Dict[str, DownloadStats] = {}
                    for asset_type in set(list(results.keys()) + list(retry_results.keys())):
                        if asset_type in retry_results and asset_type in results:
                            merged_results[asset_type] = _merge_stats(results[asset_type], retry_results[asset_type])
                        elif asset_type in retry_results:
                            merged_results[asset_type] = retry_results[asset_type]
                        else:
                            merged_results[asset_type] = results[asset_type]
                    results = merged_results
                    total_success, total_error, total_empty = _log_results(results, "=== 自动重试后结果 ===")
        
        return results

# ================= 便捷函数 =================
def download_data(start_date: str, end_date: str, adj_type: str = "qfq", 
                 assets: List[str] = None, threads: int = 8, 
                 enable_warmup: bool = True, enable_adaptive_rate_limit: bool = True,
                 token: Optional[str] = None) -> Dict[str, DownloadStats]:
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
        token: Tushare token（为空时自动从环境/config读取，交互式终端会提示输入）
    
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
        enable_adaptive_rate_limit=enable_adaptive_rate_limit,
        token=token,
        rate_limit_calls_per_min=CALLS_PER_MIN,
        safe_calls_per_min=SAFE_CALLS_PER_MIN,
        download_tor=DOWNLOAD_TOR,
    )
    
    manager = DownloadManager(config)
    try:
        results = manager.run_download(assets)
        # 下载完股票后抓取概念（爬虫）
        if "stock" in assets:
            try:
                if not _concept_data_exists() and not _confirm_concept_download():
                    logger.info("缺少概念数据，用户选择跳过抓取。")
                else:
                    from scrape_concepts import main as scrape_concepts
                    logger.info("开始抓取概念数据（爬虫,可能失败）...")
                    scrape_concepts()
                    logger.info("概念抓取完成，输出目录 stock_data/concepts")
            except Exception as e:
                logger.warning(f"概念抓取失败，已跳过：{e}")
        return results
    finally:
        # 确保后台数据库线程被优雅关闭，避免进程悬挂
        try:
            manager.db_manager.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"下载结束时关闭数据库管理器失败: {e}")


def wait_for_exit(prompt: str = "下载完成，按任意键退出...") -> None:
    """在交互式终端中等待用户按键后退出，便于查看输出。"""
    if not sys.stdin.isatty():
        return
    try:
        input(prompt)
    except (EOFError, KeyboardInterrupt):
        pass


def main():
    """主函数"""
    # 获取配置
    start_date = START_DATE
    end_date = END_DATE
    adj_type = API_ADJ
    assets = ASSETS
    threads = STOCK_INC_THREADS
    
    logger.info("=" * 30)
    logger.info("开始下载数据")
    logger.info("=" * 30)
    logger.info(f"日期范围: {start_date} - {end_date}")
    logger.info(f"复权类型: {adj_type}")
    logger.info(f"资产类型: {', '.join(assets)}")
    logger.info("=" * 30)
    
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
            logger.info(f"{asset_type} 下载结果: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
            if stats.failed_stocks:
                failed_codes = [code for code, _ in stats.failed_stocks if code]
                logger.warning(f"{asset_type} 失败股票: {failed_codes}")
        
        logger.info("下载任务完成，若出现失败可重新运行以补齐数据")
        
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
    parser.add_argument('--token', type=str, default=None, help='Tushare token（留空则读取环境/config，交互式时会提示输入）')
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
        enable_adaptive_rate_limit=not args.no_adaptive_rate_limit,
        token=args.token,
    )
    
    # 打印结果
    print("\n" + "=" * 30)
    print("下载结果")
    print("=" * 30)
    for asset_type, stats in results.items():
        print(f"{asset_type}: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
        if stats.failed_stocks:
            failed_codes = [code for code, _ in stats.failed_stocks if code]
            print(f"失败股票: {failed_codes}")
    print("=" * 30)


def main_interactive():
    """交互式模式"""
    print("=" * 30)
    print("股票数据下载工具")
    print("=" * 30)
    print("基于database_manager的下载模块")
    print()
    
    # 获取用户输入
    start_date = input(f"开始日期 (默认: {START_DATE}): ").strip() or START_DATE
    end_date = input(f"结束日期 (默认: {END_DATE}): ").strip() or END_DATE
    token_default = "" if _is_placeholder_token(TOKEN) else str(TOKEN)
    token_input = input("Tushare Token (留空使用 config/环境变量): ").strip()
    token_use = token_input or token_default
    
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
    print(f"  Token 来源: {'自定义输入' if token_input else ('环境变量/config' if token_use else '未设置')}")
    
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
            enable_adaptive_rate_limit=enable_adaptive_rate_limit,
            token=token_use or None,
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
    
    wait_for_exit()
