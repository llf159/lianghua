#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按日期截面重写的下载模块（进行中）
本文件首先搭好虚拟环境自举、日志、以及 Tushare 连接骨架，后续步骤将逐步填充业务逻辑。
"""

from __future__ import annotations

import os
import sys
import importlib
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
import argparse


# 在单独运行时自动切换到虚拟环境（支持 Linux/Windows）
def _bootstrap_venv() -> None:
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
import warnings
import random
import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    import tushare as ts
except ImportError:
    print("未安装 tushare: pip install tushare")
    sys.exit(1)

from config import (
    TOKEN,
    START_DATE,
    END_DATE,
    API_ADJ,
    DOWNLOAD_TOR,
    CALLS_PER_MIN,
    SAFE_CALLS_PER_MIN,
    STOCK_INC_THREADS,
    ASSETS,
    INDEX_WHITELIST,
)
from log_system import get_logger
from database_manager import get_database_manager, DatabaseManager
from indicators import compute, warmup_for, get_all_indicator_names, outputs_for
from utils import normalize_trade_date, stock_list_cache_path, normalize_ts

# 忽略 tushare 的 FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="tushare.pro.data_pro",
    message=".*fillna.*method.*deprecated.*",
)

logger = get_logger("download_manager")
PROJECT_ROOT = Path(__file__).resolve().parent
CONCEPT_DATA_FILES = [
    PROJECT_ROOT / "stock_data" / "concepts" / "em" / "stock_concepts.csv",
    PROJECT_ROOT / "stock_data" / "concepts" / "ths" / "stock_concepts.csv",
]


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
        logger.perf("未检测到概念数据，非交互环境默认继续下载。")
        return True
    try:
        choice = input("未检测到概念数据，是否现在下载？[y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        logger.perf("未检测到概念数据，用户取消操作。")
        return False
    return choice == "y"


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
    """安全刷新 tqdm 的后缀，确保控制台实时显示进度计数。"""
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


class TokenBucketRateLimiter:
    """基于令牌桶的限频器，支持自动优化（完整复刻 download.py）。"""

    def __init__(
        self,
        calls_per_min: int = 500,
        safe_calls_per_min: int = 490,
        bucket_capacity: int = None,
        refill_rate: float = None,
        min_wait: float = None,
        extra_delay: float = None,
        extra_delay_threshold: int = None,
        adaptive: bool = True,
    ):
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

        try:
            from config import (
                RATE_BUCKET_CAPACITY,
                RATE_BUCKET_REFILL_RATE,
                RATE_BUCKET_MIN_WAIT,
                RATE_BUCKET_EXTRA_DELAY,
                RATE_BUCKET_EXTRA_DELAY_THRESHOLD,
            )
            self.bucket_capacity = bucket_capacity or RATE_BUCKET_CAPACITY
            if refill_rate is None:
                self.refill_rate = self.calls_per_min / 60.0
            else:
                self.refill_rate = refill_rate
            self.min_wait = min_wait or RATE_BUCKET_MIN_WAIT
            self.extra_delay = extra_delay or RATE_BUCKET_EXTRA_DELAY
            self.extra_delay_threshold = extra_delay_threshold or RATE_BUCKET_EXTRA_DELAY_THRESHOLD
        except ImportError:
            self.bucket_capacity = bucket_capacity or 8
            if refill_rate is None:
                self.refill_rate = self.calls_per_min / 60.0
            else:
                self.refill_rate = refill_rate
            self.min_wait = min_wait or 0.05
            self.extra_delay = extra_delay or 0.5
            self.extra_delay_threshold = extra_delay_threshold or 480

        try:
            env_factor = float(os.environ.get("TUSHARE_RATE_SAFETY_FACTOR", "1.0"))
            if env_factor < 1.0:
                self.safe_calls_per_min = int(self.safe_calls_per_min * env_factor)
                self.calls_per_min = int(self.calls_per_min * env_factor)
                logger.warning(
                    f"启用保守限频模式：安全阈值调整为 {self.safe_calls_per_min}/min，最大 {self.calls_per_min}/min"
                )
        except Exception:
            pass

        if not self._user_refill_rate:
            self.refill_rate = self.calls_per_min / 60.0

        def _auto_capacity(limit_per_min: int) -> int:
            return max(2, min(12, int(limit_per_min * 0.02)))

        if not self._user_bucket_capacity:
            self.bucket_capacity = _auto_capacity(self.calls_per_min)

        min_gap = 30.0 / max(1, self.calls_per_min)
        self.min_wait = max(self.min_wait, min_gap)

        self._tokens = float(self.bucket_capacity)
        self._last_refill = time.time()
        self._lock = threading.Lock()

        self._current_refill_rate = self.refill_rate
        self._current_capacity = self.bucket_capacity
        self._error_count = 0
        self._success_count = 0
        self._rate_limit_errors = 0
        self._last_adjustment = time.time()
        self._adjustment_interval = 30.0

        self._total_calls = 0
        self._total_wait_time = 0.0
        self._rate_limit_hits = 0
        self._recent_calls: List[float] = []

    def _refill_tokens(self, now: float):
        """补充令牌"""
        elapsed = now - self._last_refill
        if elapsed > 0:
            tokens_to_add = elapsed * self._current_refill_rate
            self._tokens = min(self._current_capacity, self._tokens + tokens_to_add)
            self._last_refill = now

    def wait_if_needed(self):
        """检查是否需要等待，如果需要则等待。策略：先按最大限频，接近阈值时减速试探。"""
        with self._lock:
            now = time.time()
            self._refill_tokens(now)
            self._recent_calls = [t for t in self._recent_calls if now - t < 60]

            wait_time = 0.0
            if self._tokens < 1.0:
                tokens_needed = 1.0 - self._tokens
                wait_time = tokens_needed / self._current_refill_rate
                wait_time = max(self.min_wait, wait_time)

            recent_calls_count = len(self._recent_calls)
            max_calls_per_min = self.calls_per_min
            if recent_calls_count > 0:
                usage_ratio = recent_calls_count / max_calls_per_min
                if usage_ratio >= 0.92:
                    slowdown_factor = (usage_ratio - 0.92) / 0.08
                    extra_wait = self.extra_delay * slowdown_factor
                    wait_time += extra_wait

                    if usage_ratio >= 0.98:
                        target_rate = max_calls_per_min / 60.0 * 0.92
                        if self._current_refill_rate > target_rate:
                            self._current_refill_rate = target_rate
                            logger.debug(
                                f"接近限频阈值 ({recent_calls_count}/{max_calls_per_min})，降低速率到 {self._current_refill_rate:.2f} 次/秒"
                            )
                    elif usage_ratio >= 0.96:
                        target_rate = max_calls_per_min / 60.0 * 0.95
                        if self._current_refill_rate > target_rate:
                            self._current_refill_rate = target_rate
                            logger.debug(
                                f"接近限频阈值 ({recent_calls_count}/{max_calls_per_min})，适度降低速率到 {self._current_refill_rate:.2f} 次/秒"
                            )

                    if extra_wait > 0:
                        logger.debug(
                            f"接近限频阈值 ({recent_calls_count}/{max_calls_per_min}, {usage_ratio*100:.1f}%)，添加减速延迟 {extra_wait:.2f}秒"
                        )

            if wait_time > 0:
                self._total_wait_time += wait_time
                self._rate_limit_hits += 1
                time.sleep(wait_time)
                now = time.time()
                self._refill_tokens(now)

            self._tokens -= 1.0
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
        """成功调用后自适应调整"""
        now = time.time()
        if now - self._last_adjustment < self._adjustment_interval:
            return

        max_rate = self.calls_per_min / 60.0
        if self._success_count > 25 and self._error_count < 5 and self._rate_limit_errors == 0:
            if self._current_refill_rate < max_rate:
                old_rate = self._current_refill_rate
                self._current_refill_rate = min(self._current_refill_rate * 1.02, max_rate)
                logger.info(
                    f"令牌桶试探上限: 提高补充速率 {old_rate:.2f} -> {self._current_refill_rate:.2f} 次/秒 (目标: {max_rate:.2f})"
                )
            if self._current_capacity < self.bucket_capacity * 1.5:
                old_capacity = self._current_capacity
                self._current_capacity = min(self._current_capacity + 1, int(self.bucket_capacity * 1.5))
                logger.debug(f"令牌桶自适应调整: 增加容量 {old_capacity:.1f} -> {self._current_capacity:.1f}")
        elif self._rate_limit_errors == 0 and self._current_refill_rate < max_rate * 0.98:
            old_rate = self._current_refill_rate
            self._current_refill_rate = min(self._current_refill_rate * 1.02, max_rate * 0.98)
            logger.debug(f"令牌桶小幅试探: 提高补充速率 {old_rate:.2f} -> {self._current_refill_rate:.2f} 次/秒")

        self._success_count = 0
        self._error_count = 0
        self._rate_limit_errors = 0
        self._last_adjustment = now

    def _adjust_after_error(self, error_type: str):
        """错误调用后自适应调整"""
        now = time.time()
        if now - self._last_adjustment < self._adjustment_interval:
            return

        max_rate = self.calls_per_min / 60.0
        if "limit" in error_type.lower() or "quota" in error_type.lower() or "频率" in error_type.lower():
            old_rate = self._current_refill_rate
            target_rate = max(self._current_refill_rate * 0.95, max_rate * 0.85)
            self._current_refill_rate = target_rate
            logger.warning(
                f"令牌桶试探上限: 遇到限频错误，降低补充速率 {old_rate:.2f} -> {self._current_refill_rate:.2f} 次/秒（继续试探）"
            )

            old_capacity = self._current_capacity
            self._current_capacity = max(int(self._current_capacity * 0.95), self.bucket_capacity)
            logger.debug(f"令牌桶自适应调整: 降低容量 {old_capacity:.1f} -> {self._current_capacity:.1f}（限频错误）")

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
                "success_rate": self._success_count / max(self._total_calls, 1) * 100,
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


RateLimiter = TokenBucketRateLimiter


class TushareManager:
    """Tushare 接口管理：token 解析、代理设置、基础调用封装。"""

    def __init__(
        self,
        token: Optional[str] = None,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.token = resolve_tushare_token(token)
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()
        self._pro = None
        self.db_manager = db_manager

        # 设置 token 并获取 Pro API
        ts.set_token(self.token)
        self._pro = ts.pro_api()

        # 配置代理
        self._setup_proxy()

    def _setup_proxy(self) -> None:
        """设置代理配置，默认直连 Tushare。"""
        for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            os.environ.pop(k, None)
        os.environ.setdefault(
            "NO_PROXY",
            "127.0.0.1,localhost,api.tushare.pro,github.com,raw.githubusercontent.com,"
            "pypi.org,files.pythonhosted.org,huggingface.co,cdn-lfs.huggingface.co",
        )
        os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])

    def _get_db_manager(self) -> DatabaseManager:
        """延迟获取数据库管理器，保证单例复用。"""
        if self.db_manager is None:
            self.db_manager = get_database_manager()
        return self.db_manager

    def _make_api_call(self, call_func, *args, **kwargs):
        """统一 API 调用入口，带限频与简单重试。"""
        max_retries = 3
        retry_delay = 1.0
        for attempt in range(max_retries):
            self.rate_limiter.wait_if_needed()
            try:
                result = call_func(*args, **kwargs)
                self.rate_limiter.record_success()
                return result
            except Exception as e:
                err = str(e)
                if attempt < max_retries - 1:
                    base_wait = retry_delay * (2 ** attempt)
                    jitter = random.uniform(-0.2, 0.2) * base_wait
                    wait_time = max(0.1, base_wait + jitter)
                    logger.warning(f"Tushare 调用失败（尝试 {attempt + 1}/{max_retries}）: {err}，等待 {wait_time:.1f}s 重试")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Tushare 调用最终失败: {err}")
                self.rate_limiter.record_error("api_error")
                raise

    @property
    def pro(self):
        return self._pro

    def get_stock_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        adj: Optional[str] = None,
        include_tor: bool = False,
    ) -> pd.DataFrame:
        """获取股票日线数据（pro_bar）"""
        params = {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            "freq": "D",
            "asset": "E",
        }
        if include_tor:
            params["factors"] = ["tor"]
        if adj and adj != "raw":
            params["adj"] = adj
        return self._make_api_call(ts.pro_bar, **params)

    def get_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数日线数据（pro_bar, asset=I）。"""
        params = {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            "freq": "D",
            "asset": "I",
        }
        return self._make_api_call(ts.pro_bar, **params)

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表。"""
        return self._make_api_call(
            self._pro.stock_basic,
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,list_date",
        )

    def get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表，复用 database_manager 的统一方法。"""
        return self._get_db_manager().get_trade_dates_from_tushare(start_date, end_date)

    def get_smart_end_date(self, end_date_config: str) -> str:
        """智能获取结束日期，兼容 today 语义。"""
        try:
            return self._get_db_manager().get_smart_end_date(end_date_config)
        except Exception as e:
            logger.error(f"智能获取结束日期失败: {e}")
            if isinstance(end_date_config, str) and end_date_config.lower() == "today":
                return datetime.now().strftime("%Y%m%d")
            return end_date_config

    def get_daily_basic_by_date(self, trade_date: str, fields: str) -> pd.DataFrame:
        """按交易日获取 daily_basic。"""
        return self._make_api_call(
            self._pro.daily_basic,
            trade_date=trade_date,
            fields=fields,
        )


# ================= 配置与数据结构 =================
@dataclass
class DownloadConfig:
    """下载配置（精简版，默认按日期截面）"""

    start_date: str
    end_date: str
    adj_type: str = "qfq"
    asset_type: str = "stock"
    download_tor: bool = False
    threads: int = 8
    retry_times: int = 3
    batch_size: int = 100
    cross_section_limit: int = 12000
    include_latest_for_check: bool = True
    latest_date: Optional[str] = None
    enable_warmup: bool = True
    compute_indicators: bool = True
    refresh_stock_list_on_download: bool = True
    enable_adaptive_rate_limit: bool = True
    rate_limit_calls_per_min: int = 500
    safe_calls_per_min: int = 490
    rate_limit_burst_capacity: int = 10
    warmup_batch_size: int = 50
    token: Optional[str] = None
    use_cross_section: bool = True  # 允许按股票模式回退


@dataclass
class DownloadStats:
    total_stocks: int = 0
    success_count: int = 0
    skip_count: int = 0
    empty_count: int = 0
    error_count: int = 0
    failed_stocks: List[Tuple[str, str]] = None
    stock_list_updated: bool = False

    def __post_init__(self):
        if self.failed_stocks is None:
            self.failed_stocks = []


@dataclass
class IndicatorBatchResult:
    """独立指标计算结果"""
    computed: Dict[str, pd.DataFrame]
    failed: List[str]
    empty: List[str]
    success_count: int = 0
    error_count: int = 0
    empty_count: int = 0

class TempStore:
    """
    临时内存数据库：集中缓存各模块数据，统一落盘。
    当前仅缓存 stock_data，按 (ts_code, adj_type) 归并并去重。
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._stock_data: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._lock = threading.Lock()

    def add_stock_data(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        required = {"ts_code", "trade_date", "adj_type"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"临时缓存缺少必要列: {missing}")
        df = df.copy()
        df["ts_code"] = df["ts_code"].astype(str)
        df["trade_date"] = df["trade_date"].astype(str)
        for (code, adj), part in df.groupby(["ts_code", "adj_type"]):
            key = (code, adj)
            part = part.sort_values("trade_date")
            part = part.drop_duplicates("trade_date", keep="last")
            with self._lock:
                if key in self._stock_data:
                    merged = pd.concat([self._stock_data[key], part], ignore_index=True)
                    merged = merged.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
                    self._stock_data[key] = merged
                else:
                    self._stock_data[key] = part

    def flush_stock_data(self, mode: str = "upsert") -> Tuple[int, List[str]]:
        """将缓存的股票数据统一落盘，返回成功数与错误列表。"""
        errors: List[str] = []
        success = 0
        items = list(self._stock_data.items())
        for (code, adj), df in items:
            if df is None or df.empty:
                continue
            write_completed = threading.Event()
            write_result = {"success": False, "error": None}

            def update_status_callback(response):
                try:
                    if hasattr(response, "success") and response.success:
                        write_result["success"] = True
                    else:
                        write_result["success"] = False
                        write_result["error"] = getattr(response, "error", "未知错误")
                finally:
                    write_completed.set()

            try:
                self.db_manager.receive_stock_data(
                    source_module="download_manager_tempstore",
                    data=df,
                    mode=mode,
                    callback=update_status_callback,
                )
                if write_completed.wait(timeout=60):
                    if write_result["success"]:
                        success += 1
                    else:
                        errors.append(f"{code}:{write_result.get('error') or '写入失败'}")
                else:
                    errors.append(f"{code}:写入超时")
            except Exception as e:
                errors.append(f"{code}:{e}")
        return success, errors


# ================= 数据处理器 =================
class DataProcessor:
    """数据清洗与指标计算"""

    def __init__(self, db_manager: Optional[DatabaseManager]):
        self.db_manager = db_manager
        self.indicator_names = get_all_indicator_names()

    def clean_stock_data(self, df: pd.DataFrame, ts_code: str = None) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        df = df.rename(columns={"turnover_rate": "tor"})
        if "tor" not in df.columns:
            df["tor"] = np.nan
        for col in ["pre_close", "change", "pct_chg"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        numeric_columns = ["open", "high", "low", "close", "vol", "amount", "tor"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["open", "high", "low", "close", "vol", "amount", "tor"]:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(float).round(2)
        df = df.sort_values("trade_date")
        df = df.drop_duplicates("trade_date", keep="last")
        df = normalize_trade_date(df)
        return df

    def _get_historical_data_for_warmup(self, ts_code: str, adj_type: str, new_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.db_manager is None:
            return None
        try:
            warmup_days = warmup_for(self.indicator_names)
            if warmup_days <= 0:
                return None
            min_date = new_data["trade_date"].min()
            min_date_obj = datetime.strptime(str(min_date), "%Y%m%d")
            end_date_obj = min_date_obj - timedelta(days=1)
            end_date = end_date_obj.strftime("%Y%m%d")
            need_rows = max(warmup_days + 10, warmup_days)
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    historical_df = self.db_manager.query_stock_data(
                        ts_code=ts_code,
                        end_date=end_date,
                        adj_type=adj_type,
                        limit=need_rows,
                        order="desc",
                    )
                    if historical_df.empty:
                        return None
                    historical_df = self.clean_stock_data(historical_df, ts_code)
                    return historical_df
                except Exception as e:
                    if attempt < max_retries and "timeout" in str(e).lower():
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    return None
        except Exception:
            return None

    def compute_indicators_with_warmup(
        self,
        df: pd.DataFrame,
        ts_code: str,
        adj_type: str,
        warmup_cache: Dict[str, pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        try:
            historical_data = None
            if warmup_cache is not None and ts_code in warmup_cache:
                historical_data = warmup_cache[ts_code].copy()
            else:
                historical_data = self._get_historical_data_for_warmup(ts_code, adj_type, df)
            if historical_data is not None and not historical_data.empty:
                combined_data = pd.concat([historical_data, df], ignore_index=True)
                combined_data = combined_data.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
                df_with_indicators = compute(combined_data, self.indicator_names)
                df_with_indicators = df_with_indicators[df_with_indicators["trade_date"] >= df["trade_date"].min()].copy()
            else:
                df_with_indicators = compute(df, self.indicator_names)
            decs = outputs_for(self.indicator_names)
            for col, n in decs.items():
                if col in df_with_indicators.columns and pd.api.types.is_numeric_dtype(df_with_indicators[col]):
                    df_with_indicators[col] = df_with_indicators[col].round(n)
            return df_with_indicators
        except Exception as e:
            error_msg = f"指标计算失败 {ts_code}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def prepare_data_for_database(self, df: pd.DataFrame, adj_type: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        df["adj_type"] = adj_type
        if "tor" not in df.columns:
            df["tor"] = np.nan
        base_columns = ["ts_code", "trade_date", "adj_type", "open", "high", "low", "close", "vol", "amount", "tor"]
        indicator_columns = [col for col in df.columns if col not in base_columns]
        df = df[base_columns + indicator_columns]
        return df


class LatestPriceValidator:
    """
    收盘价校验与重拉（独立模块）。
    通用化：调用方可指定校验日期（例如开始日期前一天），默认回退到 config.latest_date 或数据库最新交易日。
    """

    def __init__(
        self,
        db_provider,
        stock_fetcher,
        data_processor: DataProcessor,
        config: DownloadConfig,
    ):
        self._db_provider = db_provider
        self._stock_fetcher = stock_fetcher
        self._data_processor = data_processor
        self._config = config

    def validate_close(self, df: pd.DataFrame, check_date: Optional[str] = None, enabled: Optional[bool] = None) -> pd.DataFrame:
        enabled_flag = self._config.include_latest_for_check if enabled is None else enabled
        if (
            df is None
            or df.empty
            or not enabled_flag
            or "trade_date" not in df.columns
            or "ts_code" not in df.columns
        ):
            return df

        # 确定校验日期：调用方传入 > config.latest_date > 数据库最新交易日
        target_date = check_date or self._config.latest_date
        if not target_date:
            try:
                target_date = self._db_provider().get_latest_trade_date()
            except Exception as e:
                logger.warning(f"获取数据库最新日数据失败，跳过复权校验: {e}")
                return df
        if not target_date:
            return df
        target_date = str(target_date)

        try:
            db_df = self._db_provider().query_stock_data(
                start_date=target_date,
                end_date=target_date,
                columns=["ts_code", "close"],
                adj_type=self._config.adj_type,
            )
            if db_df.empty:
                return df
            existing_close = {
                str(row.ts_code): float(row.close)
                for row in db_df.itertuples()
                if pd.notna(getattr(row, "close", None))
            }
        except Exception as e:
            logger.warning(f"获取数据库最新日数据失败，跳过复权校验: {e}")
            return df

        latest_new = df[df["trade_date"] == target_date]
        mismatched_codes: List[str] = []
        for row in latest_new.itertuples():
            code = str(row.ts_code)
            new_close = getattr(row, "close", None)
            if code in existing_close and pd.notna(new_close):
                try:
                    if not np.isclose(float(existing_close[code]), float(new_close), atol=1e-4):
                        mismatched_codes.append(code)
                except Exception:
                    mismatched_codes.append(code)

        if not mismatched_codes:
            return df

        logger.warning(f"检测到 {len(mismatched_codes)} 只股票在 {target_date} 收盘价不一致，准备单独重拉")
        refetched_frames: List[pd.DataFrame] = []
        for code in mismatched_codes:
            try:
                refetched = self._stock_fetcher(code, target_date, target_date)
                if refetched is None or refetched.empty:
                    continue
                ref_cleaned = self._data_processor.clean_stock_data(refetched, code)
                refetched_frames.append(ref_cleaned)
            except Exception as e:
                logger.warning(f"重拉 {code} 失败: {e}")
        filtered = df[
            ~((df["trade_date"] == target_date) & (df["ts_code"].isin(mismatched_codes)))
        ]
        if refetched_frames:
            filtered = pd.concat([filtered] + refetched_frames, ignore_index=True)
        return filtered


class IndexDownloader:
    """指数数据下载器"""

    def __init__(self, config: DownloadConfig):
        self.config = config
        # 不在初始化时创建数据库连接，避免子线程创建连接

        # 创建限频器（使用令牌桶算法）
        rate_limiter = TokenBucketRateLimiter(
            calls_per_min=config.rate_limit_calls_per_min,
            safe_calls_per_min=config.safe_calls_per_min,
            adaptive=config.enable_adaptive_rate_limit,
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
                    ts_code=ts_code, start_date=self.config.start_date, end_date=self.config.end_date
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
                df["adj_type"] = "ind"

                # 确保列顺序正确
                base_columns = ["ts_code", "trade_date", "adj_type", "open", "high", "low", "close", "vol", "amount"]
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
                    base_wait_time = retry_delay * (2**attempt)  # 指数退避
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
            **_tqdm_kwargs(),
        )

        # 收集所有下载的数据
        collected_data = []  # 存储所有成功下载的数据

        # 并发下载（不写入数据库）
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            # 提交所有任务
            future_to_code = {executor.submit(self.download_index_data, ts_code): ts_code for ts_code in whitelist}

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
                _refresh_postfix(
                    progress_bar,
                    {
                        "成功": self.stats.success_count,
                        "空数据": self.stats.empty_count,
                        "失败": self.stats.error_count,
                    },
                )

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
                        if hasattr(response, "success") and response.success:
                            write_result["success"] = True
                            logger.info(
                                f"数据已成功写入数据库，导入 {response.rows_imported} 条记录，耗时 {response.execution_time:.2f} 秒"
                            )
                            from database_manager import update_stock_data_status

                            update_stock_data_status()
                        else:
                            write_result["success"] = False
                            write_result["error"] = getattr(response, "error", "未知错误")
                            logger.error(f"数据写入数据库失败: {write_result['error']}")
                    except Exception as e:
                        logger.warning(f"更新状态文件失败: {e}")
                    finally:
                        write_completed.set()

                request_id = db_manager.receive_stock_data(
                    source_module="download",
                    data=all_data,
                    mode="upsert",  # 使用upsert模式避免重复数据
                    callback=update_status_callback,
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
        logger.info(
            f"下载完成 - 成功: {self.stats.success_count}, 空数据: {self.stats.empty_count}, 失败: {self.stats.error_count}"
        )

        # 打印限频器统计信息
        rate_stats = self.tushare_manager.rate_limiter.get_stats()
        logger.info(
            f"令牌桶限频器统计 - 总调用: {rate_stats['total_calls']}, 成功率: {rate_stats['success_rate']:.1f}%, "
            f"补充速率: {rate_stats['current_refill_rate']:.2f}次/秒, 容量: {rate_stats['current_capacity']:.1f}, "
            f"当前令牌: {rate_stats['current_tokens']:.1f}, 等待时间: {rate_stats['total_wait_time']:.1f}s, "
            f"限频命中: {rate_stats['rate_limit_hits']}"
        )

        return self.stats


def _process_one_stock_task(args) -> Tuple[str, str, Optional[pd.DataFrame]]:
    """
    进程内处理单只股票：计算指标并写库。
    返回 (status, message)
    """
    code, df_code, cfg, warmup_df = args
    try:
        if df_code is None or df_code.empty:
            return "empty", "", None
        db_manager = get_database_manager()
        processor = DataProcessor(db_manager)
        df_code = df_code.sort_values("trade_date")
        if cfg.get("enable_warmup"):
            cache = {code: warmup_df} if warmup_df is not None else None
            computed = processor.compute_indicators_with_warmup(df_code, code, cfg.get("adj_type"), warmup_cache=cache)
        else:
            computed = compute(df_code, processor.indicator_names)
        if computed is None or computed.empty:
            return "empty", "", None
        final_data = processor.prepare_data_for_database(computed, cfg.get("adj_type"))
        return "success", "", final_data
    except Exception as e:
        return "error", str(e), None


class IndicatorCalculator:
    """
    独立的指标计算模块，多进程优化，计算细节沿用 download.py / DataProcessor 逻辑。
    仅负责计算（含可选 warmup），不写库，由调用方决定后续处理。
    """

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers

    def compute(
        self,
        data: pd.DataFrame,
        adj_type: str,
        *,
        warmup_cache: Optional[Dict[str, pd.DataFrame]] = None,
        enable_warmup: bool = True,
        is_first_download: bool = False,
        max_workers: Optional[int] = None,
    ) -> IndicatorBatchResult:
        if data is None or data.empty:
            return IndicatorBatchResult(computed={}, failed=[], empty=[], success_count=0, error_count=0, empty_count=0)

        data = data.copy()
        if "ts_code" not in data.columns:
            raise ValueError("计算指标需要 ts_code 列")
        data["ts_code"] = data["ts_code"].astype(str)
        stock_codes = sorted(data["ts_code"].dropna().unique().tolist())
        if not stock_codes:
            return IndicatorBatchResult(computed={}, failed=[], empty=[], success_count=0, error_count=0, empty_count=0)

        cfg = {
            "adj_type": adj_type,
            "enable_warmup": enable_warmup and not is_first_download,
            "is_first_download": is_first_download,
        }
        tasks = []
        for code in stock_codes:
            df_code = data[data["ts_code"] == code].copy()
            warmup_df = warmup_cache.get(code) if warmup_cache else None
            tasks.append((code, df_code, cfg, warmup_df))

        results: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []
        empty_codes: List[str] = []
        success_count = error_count = empty_count = 0

        workers = max_workers or self.max_workers or os.cpu_count() or 4
        progress_bar = tqdm(
            **_tqdm_kwargs(
                total=len(tasks),
                desc="指标计算",
                unit="只",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_code = {executor.submit(_process_one_stock_task, t): t[0] for t in tasks}
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    status, message, df = future.result()
                    if status == "success":
                        if df is not None and not df.empty:
                            results[code] = df
                            success_count += 1
                        else:
                            empty_count += 1
                            empty_codes.append(code)
                    elif status == "empty":
                        empty_count += 1
                        empty_codes.append(code)
                    else:
                        error_count += 1
                        failed.append(f"{code}:{message}")
                except Exception as e:
                    error_count += 1
                    failed.append(f"{code}:{e}")
                finally:
                    progress_bar.update(1)
        progress_bar.close()

        return IndicatorBatchResult(
            computed=results,
            failed=failed,
            empty=empty_codes,
            success_count=success_count,
            error_count=error_count,
            empty_count=empty_count,
        )


class DateDownloader:
    """按日期截面下载模块。"""

    def __init__(self, config: DownloadConfig, tushare: "TushareManager"):
        self.config = config
        self.tushare = tushare

    def _fetch_cross_section_daily(self, trade_date: str) -> pd.DataFrame:
        limit = self.config.cross_section_limit or 5000
        fields = "ts_code,trade_date,open,high,low,close,vol,amount"
        frames: List[pd.DataFrame] = []
        offset = 0
        while True:
            df = self.tushare._make_api_call(
                self.tushare.pro.daily,
                trade_date=trade_date,
                fields=fields,
                limit=limit,
                offset=offset,
            )
            if df is None or df.empty:
                break
            frames.append(df)
            if len(df) < limit:
                break
            offset += limit
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    def _fetch_cross_section_basic(self, trade_date: str) -> pd.DataFrame:
        if not self.config.download_tor:
            return pd.DataFrame()
        limit = self.config.cross_section_limit or 5000
        fields = "ts_code,trade_date,turnover_rate"
        frames: List[pd.DataFrame] = []
        offset = 0
        while True:
            df = self.tushare._make_api_call(
                self.tushare.pro.daily_basic,
                trade_date=trade_date,
                fields=fields,
                limit=limit,
                offset=offset,
            )
            if df is None or df.empty:
                break
            frames.append(df)
            if len(df) < limit:
                break
            offset += limit
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    def _merge_daily_and_basic(self, daily_df: pd.DataFrame, basic_df: pd.DataFrame) -> pd.DataFrame:
        """将日线与 basic 数据合并（若 basic 为空则原样返回日线）。"""
        if daily_df is None or daily_df.empty:
            return pd.DataFrame()
        if basic_df is None or basic_df.empty:
            return daily_df
        return daily_df.merge(
            basic_df[["ts_code", "trade_date", "turnover_rate"]],
            on=["ts_code", "trade_date"],
            how="left",
        )

    def collect_frames(self, trade_dates: List[str]) -> List[pd.DataFrame]:
        """批量拉取截面数据并合并 basic，返回非空数据块列表。"""
        frames: List[pd.DataFrame] = []
        progress_bar = tqdm(
            **_tqdm_kwargs(
                total=len(trade_dates),
                desc="按日拉取",
                unit="日",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        )
        for td in trade_dates:
            try:
                daily_df = self._fetch_cross_section_daily(td)
                if daily_df is None or daily_df.empty:
                    continue
                basic_df = self._fetch_cross_section_basic(td)
                merged = self._merge_daily_and_basic(daily_df, basic_df)
                if merged is not None and not merged.empty:
                    frames.append(merged)
            except Exception as e:
                logger.error(f"{td} 截面获取失败: {e}")
            finally:
                progress_bar.update(1)
        progress_bar.close()
        return frames


class StockDownloader:
    """股票数据下载器"""

    def __init__(self, config: DownloadConfig, is_first_download: bool = False):
        self.config = config
        self.is_first_download = is_first_download  # 是否为首次下载
        # 不在初始化时创建数据库连接，避免子线程创建连接

        # 创建限频器（使用令牌桶算法）
        calls_per_min = config.rate_limit_calls_per_min
        safe_calls_per_min = config.safe_calls_per_min
        if config.download_tor:
            # daily_basic 限频 200/min，tor 会触发
            calls_per_min = min(200, calls_per_min)
            safe_calls_per_min = min(190, safe_calls_per_min, calls_per_min - 1 if calls_per_min > 1 else calls_per_min)
            logger.warning(
                f"检测到 daily_basic 需求（tor），限频强制调整为 {calls_per_min}/min（safe={safe_calls_per_min}）以适配 daily_basic 限额"
            )
        rate_limiter = TokenBucketRateLimiter(
            calls_per_min=calls_per_min,
            safe_calls_per_min=safe_calls_per_min,
            adaptive=config.enable_adaptive_rate_limit,
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

    def _persist_stock_list_cache(self, df: pd.DataFrame) -> None:
        """把股票列表缓存到数据库目录下的 stock_list.csv，供其他模块复用。"""
        try:
            if df is None or df.empty or "ts_code" not in df.columns:
                return
            cache_path = stock_list_cache_path()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cols = [
                c
                for c in [
                    "ts_code",
                    "symbol",
                    "name",
                    "area",
                    "industry",
                    "list_date",
                    "trade_date",
                    "total_share",
                    "float_share",
                    "total_mv",
                    "circ_mv",
                ]
                if c in df.columns
            ]
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

    def _preload_warmup_data(self, db_manager: DatabaseManager, stock_codes: List[str], data_processor: DataProcessor):
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

            start_date_obj = datetime.strptime(self.config.start_date, "%Y%m%d")
            load_end_obj = start_date_obj - timedelta(days=1)
            load_end = load_end_obj.strftime("%Y%m%d")

            logger.info(f"预加载历史数据按条数获取: 每只股票 {need_rows} 条，截止 {load_end} (warmup天数: {warmup_days})")

            # 批量查询所有股票的历史数据
            logger.info(f"开始批量查询 {len(stock_codes)} 只股票的历史数据...")
            preload_progress = tqdm(
                total=len(stock_codes),
                desc="预加载历史数据",
                unit="只",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
                leave=False,
                **_tqdm_kwargs(),
            )

            loaded_count = 0
            for ts_code in stock_codes:
                try:
                    # 从数据库查询历史数据
                    historical_df = db_manager.query_stock_data(
                        ts_code=ts_code, end_date=load_end, adj_type=self.config.adj_type, limit=need_rows, order="desc"
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

    def download_stock_data(
        self, ts_code: str, data_processor: DataProcessor = None
    ) -> Tuple[str, str, Optional[pd.DataFrame]]:
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
                    include_tor=self.config.download_tor,
                )

                if raw_data is None or raw_data.empty:
                    logger.debug(f"{ts_code} 无数据")
                    return ts_code, "empty", None

                logger.debug(f"{ts_code} 获取到 {len(raw_data)} 条原始数据")

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
                                cleaned_data, ts_code, self.config.adj_type, warmup_cache=self._warmup_cache
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
                if any(keyword in error_msg_lower for keyword in ["指标计算失败", "indicator", "计算失败", "compute"]):
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
                    base_wait_time = retry_delay * (2**attempt)  # 指数退避
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
                        elif "ts_code" not in stock_list.columns:
                            logger.error(f"股票列表缺少 'ts_code' 列，可用列: {stock_list.columns.tolist()}")
                            stock_codes = self._load_stock_list_from_cache()
                        else:
                            self._persist_stock_list_cache(stock_list)
                            stock_codes = stock_list["ts_code"].tolist()
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
            **_tqdm_kwargs(),
        )

        # 用于收集需要批量写入的数据（如果启用批量写入模式）
        collected_data = []  # 存储所有成功下载并处理的数据

        # 边下载边计算指标边写入
        logger.info("开始边下载边计算指标边写入...")

        # 并发下载（边下载边计算指标）
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            # 提交所有任务（传入data_processor以便计算指标）
            future_to_code = {executor.submit(self.download_stock_data, ts_code, data_processor): ts_code for ts_code in stock_codes}

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
                        # 立即写入数据库（边下载边写入）
                        try:
                            logger.debug(f"开始写入 {result_code} 的数据到数据库（{len(data)} 条记录）...")

                            # 使用回调函数在写入完成后更新状态
                            write_completed = threading.Event()
                            write_result = {"success": False, "error": None}

                            def update_status_callback(response):
                                try:
                                    if hasattr(response, "success") and response.success:
                                        write_result["success"] = True
                                        logger.debug(
                                            f"{result_code} 数据已成功写入数据库，导入 {response.rows_imported} 条记录"
                                        )
                                    else:
                                        write_result["success"] = False
                                        write_result["error"] = getattr(response, "error", "未知错误")
                                        logger.error(f"{result_code} 数据写入数据库失败: {write_result['error']}")
                                except Exception as e:
                                    logger.warning(f"更新状态失败 {result_code}: {e}")
                                finally:
                                    write_completed.set()

                            # 提交写入请求（异步，不阻塞）
                            request_id = db_manager.receive_stock_data(
                                source_module="download", data=data, mode="upsert", callback=update_status_callback
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
                _refresh_postfix(
                    progress_bar,
                    {
                        "成功": self.stats.success_count,
                        "空数据": self.stats.empty_count,
                        "失败": self.stats.error_count,
                    },
                )

        # 关闭进度条
        progress_bar.close()

        # 所有数据已经边下载边写入，更新状态文件
        if self.stats.success_count > 0:
            try:
                from database_manager import update_stock_data_status

                update_stock_data_status()
                logger.info("状态文件已更新")
            except Exception as e:
                logger.warning(f"更新状态文件失败: {e}")

        # 打印统计信息
        logger.info(
            f"下载完成 - 成功: {self.stats.success_count}, 空数据: {self.stats.empty_count}, 失败: {self.stats.error_count}"
        )

        # 打印限频器统计信息
        rate_stats = self.tushare_manager.rate_limiter.get_stats()
        logger.info(
            f"令牌桶限频器统计 - 总调用: {rate_stats['total_calls']}, 成功率: {rate_stats['success_rate']:.1f}%, "
            f"补充速率: {rate_stats['current_refill_rate']:.2f}次/秒, 容量: {rate_stats['current_capacity']:.1f}, "
            f"当前令牌: {rate_stats['current_tokens']:.1f}, 等待时间: {rate_stats['total_wait_time']:.1f}s, "
            f"限频命中: {rate_stats['rate_limit_hits']}"
        )

        return self.stats


# ================= 下载管理器 =================
class DownloadManager:
    """下载管理器（复刻 download.py 的流程，边下载边计算写库）"""

    def __init__(self, config: DownloadConfig):
        self.config = config
        logger.info("[数据库连接] 开始获取数据库管理器实例 (初始化下载管理器)")
        self.db_manager = get_database_manager()
        self.tushare_manager = TushareManager(token=config.token, db_manager=self.db_manager)
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
            manager = get_database_manager()
            status = manager.load_status_file()
            if status is None:
                logger.info("状态文件不存在或读取失败，判断为首次下载")
                return True
            if not isinstance(status, dict):
                logger.warning("状态文件格式错误（不是字典），判断为首次下载")
                return True
            stock_data_status = status.get("stock_data", {})
            if not isinstance(stock_data_status, dict):
                logger.warning("状态文件中stock_data格式错误，判断为首次下载")
                return True
            if not stock_data_status.get("database_exists", False):
                logger.info("状态文件显示数据库不存在，判断为首次下载")
                return True
            total_records = stock_data_status.get("total_records", 0)
            if total_records == 0:
                logger.info("状态文件显示数据库无数据，判断为首次下载")
                return True
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
            logger.error(f"检查首次下载状态失败（获取DatabaseManager失败）: {e}，判断为首次下载")
            import traceback
            logger.debug(f"异常详情: {traceback.format_exc()}")
            return True

    def _ensure_database_initialized(self):
        """确保数据库已初始化"""
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
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
        """智能获取结束日期"""
        return self.db_manager.get_smart_end_date(end_date_config)

    def _get_reference_stock_codes(self) -> List[str]:
        """获取用于完整性校验的股票代码列表"""
        try:
            cached_codes = self.db_manager.get_stock_list_from_cache()
            if cached_codes:
                logger.debug(f"使用缓存股票列表进行完整性校验，共 {len(cached_codes)} 只")
                return cached_codes
        except Exception as e:
            logger.debug(f"读取缓存股票列表失败，尝试其他来源: {e}")
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
        try:
            db_codes = self.db_manager.get_stock_list(adj_type=self.config.adj_type)
            if db_codes:
                logger.debug(f"使用数据库股票列表进行完整性校验，共 {len(db_codes)} 只")
                return db_codes
        except Exception as e:
            logger.debug(f"从数据库获取股票列表失败，兜底失败: {e}")
        return []

    def _check_latest_date_completeness(self, trade_date: str) -> Dict[str, Any]:
        """校验指定日期的股票数据是否完整"""
        result = {"trade_date": trade_date, "can_check": False, "expected_count": 0, "actual_count": 0, "missing_codes": []}
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
                start_date=trade_date, end_date=trade_date, columns=["ts_code"], adj_type=self.config.adj_type
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
                    f"最新日期 {trade_date} 数据完整 (股票数: {result['actual_count']}/{result['expected_count']})"
                )
            return result
        except Exception as e:
            logger.warning(f"检查日期 {trade_date} 的股票数据完整性失败: {e}")
            result["can_check"] = False
            result["missing_codes"] = []
            return result

    def determine_download_strategy(self) -> Dict[str, Any]:
        """确定下载策略"""
        is_first_download = self.is_first_download
        latest_date = None
        if not is_first_download:
            try:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                logger.debug(f"[determine_download_strategy] 使用数据库路径: {db_path}")
                latest_date = self.db_manager.get_latest_trade_date(db_path)
                logger.debug(f"[determine_download_strategy] 获取到最新日期: {latest_date}")
                if not latest_date:
                    logger.info("数据库存在但无数据，重新判断为首次下载")
                    is_first_download = True
            except Exception as e:
                logger.warning(f"获取最新日期失败: {e}，重新判断为首次下载")
                import traceback
                logger.debug(f"获取最新日期失败详情: {traceback.format_exc()}")
                is_first_download = True
        else:
            logger.info("首次下载，跳过数据库查询")

        actual_end_date = self.config.end_date
        if self.config.end_date.lower() == "today":
            actual_end_date = self.get_smart_end_date("today")
            logger.info(f"结束日期配置为'today'，智能判断后实际结束日期: {actual_end_date}")

        actual_start_date = self.config.start_date
        if not is_first_download and latest_date:
            from datetime import datetime, timedelta
            try:
                latest_date_obj = datetime.strptime(str(latest_date), "%Y%m%d")
                next_date_obj = latest_date_obj + timedelta(days=1)
                actual_start_date = next_date_obj.strftime("%Y%m%d")
                if actual_start_date > actual_end_date:
                    coverage_info = self._check_latest_date_completeness(latest_date)
                    missing_codes = coverage_info.get("missing_codes", []) if isinstance(coverage_info, dict) else []
                    can_check = coverage_info.get("can_check", False) if isinstance(coverage_info, dict) else False
                    if can_check and missing_codes:
                        logger.warning(
                            f"最新日期 {latest_date} 虽然与目标结束日期一致，但检测到 {len(missing_codes)} 只股票缺失数据，将强制补齐"
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
                            "fill_missing_latest_date": True,
                        }
                    if not can_check:
                        logger.warning(f"无法校验最新日期 {latest_date} 的数据完整性，默认认为数据已最新，跳过下载")
                    else:
                        logger.info(
                            f"数据已是最新，无需下载 (最新日期: {latest_date}, 计算出的开始日期: {actual_start_date}, 实际结束日期: {actual_end_date})"
                        )
                    logger.perf(f"跳过下载原因: 开始日期({actual_start_date}) > 结束日期({actual_end_date})")
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
                        "fill_missing_latest_date": False,
                    }
            except Exception as e:
                logger.warning(f"计算增量下载日期失败: {e}")

        return {
            "is_first_download": is_first_download,
            "latest_date": latest_date,
            "start_date": actual_start_date,
            "end_date": actual_end_date,
            "adj_type": self.config.adj_type,
            "asset_type": self.config.asset_type,
            "skip_download": False,
            "stock_whitelist": None,
            "latest_date_completeness": None,
            "fill_missing_latest_date": False,
        }

    def download_stocks(self, strategy: Dict[str, Any]) -> DownloadStats:
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
        downloader = StockDownloader(stock_config, is_first_download=strategy.get("is_first_download", False))
        stock_whitelist = strategy.get("stock_whitelist")
        if stock_whitelist:
            logger.warning(f"本次仅下载 {len(stock_whitelist)} 只股票以补齐最新日期 {strategy.get('end_date')}")
        return downloader.download_all_stocks(stock_whitelist)

    def download_indices(self, strategy: Dict[str, Any], whitelist: List[str] = None) -> DownloadStats:
        if strategy.get("skip_download", False):
            logger.info("跳过指数数据下载")
            return DownloadStats()
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
                if next_date > end_date:
                    logger.info(f"指数数据已覆盖到 {index_latest}，结束日期为 {end_date}，跳过指数下载")
                    stats = DownloadStats()
                    stats.skip_count = len(whitelist) if whitelist else 0
                    return stats
                if next_date > start_date:
                    logger.info(f"指数最新日期 {index_latest}，增量开始日期调整为 {next_date}")
                    start_date = next_date
            except Exception as e:
                logger.warning(f"解析指数最新日期失败，按原配置下载: {e}")
        index_config = DownloadConfig(
            start_date=start_date,
            end_date=end_date,
            adj_type="ind",
            asset_type="index",
            threads=self.config.threads,
            enable_warmup=False,
            token=self.config.token,
        )
        downloader = IndexDownloader(index_config)
        return downloader.download_all_indices(whitelist)

    def run_download(self, assets: List[str] = None, index_whitelist: List[str] = None, retry_on_failure: bool = True) -> Dict[str, DownloadStats]:
        if assets is None:
            assets = ["stock"]
        if index_whitelist is None:
            index_whitelist = INDEX_WHITELIST
        strategy = self.determine_download_strategy()

        def _collect_totals(res: Dict[str, DownloadStats]) -> Tuple[int, int, int]:
            total_success = sum(stats.success_count for stats in res.values())
            total_error = sum(stats.error_count for stats in res.values())
            total_empty = sum(stats.empty_count for stats in res.values())
            return total_success, total_error, total_empty

        def _log_results(res: Dict[str, DownloadStats], heading: str) -> Tuple[int, int, int]:
            total_success, total_error, total_empty = _collect_totals(res)
            logger.perf(heading)
            for asset_type, stats in res.items():
                logger.perf(f"{asset_type}: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
                if stats.failed_stocks:
                    logger.warning(f"{asset_type} 失败股票: {[code for code, _ in stats.failed_stocks[:5]]}")
            logger.perf(f"总计: 成功={total_success}, 空数据={total_empty}, 失败={total_error}")
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
            merged = DownloadStats()
            merged.total_stocks = original.total_stocks
            merged.skip_count = original.skip_count
            merged.empty_count = original.empty_count + retry_stats.empty_count
            merged.success_count = original.success_count + retry_stats.success_count
            merged.error_count = retry_stats.error_count
            merged.failed_stocks = retry_stats.failed_stocks or []
            return merged

        if strategy.get("skip_download", False):
            logger.warning("=" * 30)
            logger.warning("数据已是最新，跳过下载")
            logger.warning("=" * 30)
            logger.warning(
                f"跳过原因: 最新日期={strategy.get('latest_date', 'N/A')}, 计算出的开始日期={strategy.get('start_date', 'N/A')}, 实际结束日期={strategy.get('end_date', 'N/A')}"
            )
            logger.warning("如果最新日期 >= 结束日期，说明数据已经是最新的，无需下载")
            logger.warning("=" * 30)
            return {"stock": DownloadStats(), "index": DownloadStats()}

        if strategy.get("fill_missing_latest_date"):
            missing_count = len(strategy.get("stock_whitelist") or [])
            logger.warning(f"触发最新日期补齐流程: 需要补齐 {missing_count} 只股票 (日期 {strategy.get('end_date')})")

        if self.config.end_date.lower() == "today":
            smart_end_date = strategy["end_date"]
            logger.info("=" * 30)
            logger.info("智能日期判断说明")
            logger.info("=" * 30)
            logger.perf("• 配置的结束日期: today (今日)")
            logger.perf(f"• 智能判断后的实际结束日期: {smart_end_date}")
            logger.info("• 智能判断逻辑:")
            logger.info("  - 如果今天是交易日且当前时间 < 15:00，使用前一个交易日")
            logger.info("  - 如果今天是交易日且当前时间 >= 15:00，使用今天")
            logger.info("  - 如果今天不是交易日，使用最近的交易日")
            logger.info("  - 如果无法获取交易日历，使用今天")
            logger.info("=" * 30)

        logger.info(
            f"下载策略: 首次下载={strategy['is_first_download']}, 最新日期={strategy['latest_date']}, 开始日期={strategy['start_date']}, 结束日期={strategy['end_date']}, 复权类型={strategy['adj_type']}, 资产类型={strategy['asset_type']}, 跳过下载={strategy['skip_download']}"
        )

        results: Dict[str, DownloadStats] = {}
        total_assets = len(assets)
        overall_progress = tqdm(
            total=total_assets,
            desc="总体下载进度",
            unit="类",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            **_tqdm_kwargs(position=0, leave=True),
        )
        try:
            if "stock" in assets:
                overall_progress.set_description(f"下载股票数据 ({strategy['start_date']} - {strategy['end_date']})")
                logger.perf(f"开始下载股票数据: {strategy['start_date']} - {strategy['end_date']}")
                results["stock"] = self.download_stocks(strategy)
                overall_progress.update(1)
                _refresh_postfix(
                    overall_progress, {"股票成功": results["stock"].success_count, "股票失败": results["stock"].error_count}
                )
            if "index" in assets:
                overall_progress.set_description(f"下载指数数据 ({strategy['start_date']} - {strategy['end_date']})")
                logger.perf(f"开始下载指数数据: {strategy['start_date']} - {strategy['end_date']}")
                results["index"] = self.download_indices(strategy, index_whitelist)
                overall_progress.update(1)
                _refresh_postfix(
                    overall_progress, {"指数成功": results["index"].success_count, "指数失败": results["index"].error_count}
                )
            overall_progress.set_description("下载完成")
            total_success = sum(stats.success_count for stats in results.values())
            total_error = sum(stats.error_count for stats in results.values())
            total_empty = sum(stats.empty_count for stats in results.values())
            _refresh_postfix(overall_progress, {"总成功": total_success, "总失败": total_error, "总空数据": total_empty})
        finally:
            overall_progress.close()

        total_success, total_error, total_empty = _log_results(results, "=== 下载任务完成（首次尝试） ===")

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
                    logger.perf(f"股票重试列表共 {len(retry_stock_codes)} 只")
                    retry_results["stock"] = self.download_stocks(retry_strategy)
                if retry_index_codes:
                    logger.perf(f"指数重试列表共 {len(retry_index_codes)} 只")
                    retry_results["index"] = self.download_indices(strategy, retry_index_codes)
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
def download_data(
    start_date: str,
    end_date: str,
    adj_type: str = "qfq",
    assets: List[str] = None,
    threads: int = 8,
    enable_warmup: bool = True,
    enable_adaptive_rate_limit: bool = True,
    token: Optional[str] = None,
) -> Dict[str, DownloadStats]:
    """便捷下载函数"""
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
        if "stock" in assets:
            try:
                if not _concept_data_exists() and not _confirm_concept_download():
                    logger.perf("缺少概念数据，用户选择跳过抓取。")
                else:
                    from scrape_concepts import main as scrape_concepts
                    logger.perf("开始抓取概念数据（爬虫,可能失败）...")
                    scrape_concepts()
                    logger.perf("概念抓取完成，输出目录 stock_data/concepts")
            except Exception as e:
                logger.warning(f"概念抓取失败，已跳过：{e}")
        return results
    finally:
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
    start_date = START_DATE
    end_date = END_DATE
    adj_type = API_ADJ
    assets = ASSETS
    threads = STOCK_INC_THREADS

    logger.perf("=" * 30)
    logger.perf("开始下载数据")
    logger.perf("=" * 30)
    logger.perf(f"日期范围: {start_date} - {end_date}")
    logger.perf(f"复权类型: {adj_type}")
    logger.perf(f"资产类型: {', '.join(assets)}")
    logger.perf("=" * 30)

    try:
        results = download_data(
            start_date=start_date,
            end_date=end_date,
            adj_type=adj_type,
            assets=assets,
            threads=threads,
            enable_warmup=True,
        )
        for asset_type, stats in results.items():
            logger.perf(f"{asset_type} 下载结果: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
            if stats.failed_stocks:
                logger.warning(f"{asset_type} 失败股票: {[code for code, _ in stats.failed_stocks[:10]]}")
        logger.perf("下载任务完成，若出现失败可重新运行以补齐数据")
    except Exception as e:
        logger.error(f"下载任务失败: {e}")
        raise


def main_cli():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="股票数据下载工具")
    parser.add_argument("--start", type=str, default=START_DATE, help="开始日期 (YYYYMMDD)")
    parser.add_argument("--end", type=str, default=END_DATE, help="结束日期 (YYYYMMDD)")
    parser.add_argument("--adj", type=str, default=API_ADJ, choices=["qfq", "hfq", "raw"], help="复权类型")
    parser.add_argument("--assets", nargs="+", default=ASSETS, choices=["stock", "index"], help="资产类型")
    parser.add_argument("--threads", type=int, default=STOCK_INC_THREADS, help="并发线程数")
    parser.add_argument("--no-warmup", action="store_true", help="禁用指标warmup")
    parser.add_argument("--no-adaptive-rate-limit", action="store_true", help="禁用自适应限频")
    parser.add_argument("--token", type=str, default=None, help="Tushare token（留空则读取环境/config，交互式时会提示输入）")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")

    args = parser.parse_args()

    if args.interactive:
        main_interactive()
        return

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

    print("\n" + "=" * 30)
    print("下载结果")
    print("=" * 30)
    for asset_type, stats in results.items():
        print(f"{asset_type}: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
        if stats.failed_stocks:
            print(f"失败股票: {[code for code, _ in stats.failed_stocks[:5]]}")
    print("=" * 30)


def main_interactive():
    """交互式模式"""
    print("=" * 30)
    print("股票数据下载工具")
    print("=" * 30)
    print("基于database_manager的下载模块")
    print()
    start_date = input(f"开始日期 (默认: {START_DATE}): ").strip() or START_DATE
    end_date = input(f"结束日期 (默认: {END_DATE}): ").strip() or END_DATE
    token_default = "" if _is_placeholder_token(TOKEN) else str(TOKEN)
    token_input = input("Tushare Token (留空使用 config/环境变量): ").strip()
    token_use = token_input or token_default
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
    adaptive_rate_limit = input("启用自适应限频? (Y/n, 默认: Y): ").strip().lower()
    enable_adaptive_rate_limit = adaptive_rate_limit != "n"
    print("\n资产类型选择:")
    print("1. 股票数据")
    print("2. 指数数据")
    print("3. 股票+指数")
    asset_choice = input("请选择 (1-3, 默认: 1): ").strip() or "1"
    assets = []
    if asset_choice in ["1", "3"]:
        assets.append("stock")
    if asset_choice in ["2", "3"]:
        assets.append("index")
    results = download_data(
        start_date=start_date,
        end_date=end_date,
        adj_type=adj_type,
        assets=assets,
        threads=threads,
        enable_warmup=True,
        enable_adaptive_rate_limit=enable_adaptive_rate_limit,
        token=token_use,
    )
    print("\n" + "=" * 30)
    print("下载结果")
    print("=" * 30)
    for asset_type, stats in results.items():
        print(f"{asset_type}: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
        if stats.failed_stocks:
            print(f"失败股票: {[code for code, _ in stats.failed_stocks[:5]]}")
    print("=" * 30)
    wait_for_exit()


if __name__ == "__main__":
    main()
