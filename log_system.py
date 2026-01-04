#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一日志系统模块
提供完整的debug日志系统，支持多级别日志记录、性能监控和错误追踪
包含日志初始化、配置管理和便捷函数

特性：
- 异步文件写入：使用后台线程异步写入日志文件，提高性能，避免阻塞主线程
- 多进程支持：子进程日志通过进程队列发送到主进程，统一异步写入
- 多级别日志：支持DEBUG、INFO、PREP、WARNING、ERROR、CRITICAL和性能日志
- 日志轮转：自动轮转日志文件，防止文件过大
"""

import os
import sys
import logging
import logging.handlers
import queue
import shutil
# 避免logging内部报错打断业务
logging.raiseExceptions = False

# 检查是否在 streamlit 环境中运行
def _is_streamlit_env() -> bool:
    """检查是否在 streamlit 环境中运行"""
    try:
        # 优先检查环境变量（最可靠，streamlit 启动时会设置）
        streamlit_env_vars = [
            'STREAMLIT_SERVER_PORT',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS',
            'STREAMLIT_SERVER_ADDRESS',
            'STREAMLIT_SERVER_HEADLESS'
        ]
        if any(os.environ.get(var) for var in streamlit_env_vars):
            return True
        # 检查是否已经导入了 streamlit
        if 'streamlit' in sys.modules:
            return True
        # 检查命令行参数是否包含 streamlit 相关命令或脚本
        # streamlit run 会启动一个 Python 进程，argv[0] 通常是脚本路径
        # 但更可靠的是检查 sys.executable 和模块路径
        for arg in sys.argv:
            arg_lower = arg.lower()
            if 'streamlit' in arg_lower:
                return True
    except Exception:
        pass
    return False

# 在Windows上设置控制台编码为UTF-8，避免中文乱码
# 注意：在 streamlit 环境下不重定向 stdout/stderr，避免干扰 streamlit 的启动消息
if sys.platform == 'win32' and not _is_streamlit_env():
    try:
        # 尝试设置控制台代码页为UTF-8
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True, check=False)
    except Exception:
        pass
    
    try:
        # 确保标准输出使用UTF-8编码
        if hasattr(sys.stdout, 'buffer'):
            import io
            # 检查当前编码，如果不是UTF-8则重新配置
            # 仅在非 streamlit 环境下重定向
            if not hasattr(sys.stdout, '_encoding') or (hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8'):
                try:
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
                    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
                except (AttributeError, ValueError):
                    # 某些环境可能不支持，使用备用方案
                    pass
    except Exception:
        # 如果设置失败，忽略错误（某些环境可能不支持）
        pass
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import traceback
import functools
import time
import threading
from contextlib import contextmanager
import multiprocessing
import multiprocessing.managers

try:
    import config
except Exception:
    config = None

# ========== 自定义等级：PREP（介于 INFO 与 WARNING 之间） ==========
PREP_LEVEL_NUM = logging.INFO + 5  # 25
logging.addLevelName(PREP_LEVEL_NUM, "PREP")

def _resolve_file_log_level() -> int:
    """从配置解析最低落盘等级"""
    level_name = "INFO"
    try:
        if config and hasattr(config, "LOG_FILE_LEVEL"):
            level_name = str(getattr(config, "LOG_FILE_LEVEL", "INFO")).upper()
    except Exception:
        pass
    level = logging._nameToLevel.get(level_name, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO
    return level


FILE_LOG_MIN_LEVEL = _resolve_file_log_level()

def _is_main_process() -> bool:
    try:
        return multiprocessing.current_process().name == "MainProcess"
    except Exception:
        return True


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'PREP': '\033[34m',     # 蓝色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 确保消息使用UTF-8编码处理
        try:
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                # 确保消息是UTF-8编码
                if isinstance(record.msg, bytes):
                    record.msg = record.msg.decode('utf-8', errors='replace')
        except Exception:
            pass
        
        if hasattr(record, 'levelname') and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        try:
            result = super().format(record)
            # 确保返回的字符串是有效的UTF-8
            if isinstance(result, bytes):
                result = result.decode('utf-8', errors='replace')
            return result
        except UnicodeEncodeError:
            # 如果编码失败，尝试使用错误替换
            try:
                result = super().format(record)
                return result.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            except Exception:
                # 最后的备用方案
                return f"{record.levelname} [{record.name}] {record.getMessage()}"


class PerformanceLogger:
    """性能监控日志记录器"""
    
    def __init__(self, logger: logging.Logger, debug_logger: Optional['DebugLogger'] = None):
        self.logger = logger
        self.debug_logger = debug_logger  # 保存 DebugLogger 引用，用于调用 performance 方法
        self._timers: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def __call__(self, message: str, **kwargs):
        """使 PerformanceLogger 可以像函数一样被调用"""
        # 直接按 INFO 级别记录，去掉 [PERF] 前缀以减少日志噪音
        extra = kwargs.pop("extra", None) or {}
        merged_extra = {"is_perf": True}
        if isinstance(extra, dict):
            merged_extra.update(extra)
        kwargs["extra"] = merged_extra
        self.logger.info(message, **kwargs)
    
    def start_timer(self, operation: str) -> None:
        """开始计时"""
        with self._lock:
            self._timers[operation] = time.time()
            self.logger.debug(f"开始执行: {operation}")
    
    def end_timer(self, operation: str, log_level: int = logging.INFO, **kwargs) -> float:
        """结束计时并记录"""
        with self._lock:
            if operation in self._timers:
                duration = time.time() - self._timers[operation]
                del self._timers[operation]
                # 使用 performance() 方法记录性能日志，确保写入性能日志文件
                if self.debug_logger is not None:
                    # 如果有 DebugLogger 引用，使用它的 performance 方法
                    self.debug_logger.performance(f"完成执行: {operation} (耗时: {duration:.3f}s)")
                else:
                    # 使用指定的级别记录（默认 INFO）
                    extra = kwargs.pop("extra", None) or {}
                    merged_extra = {"is_perf": True}
                    if isinstance(extra, dict):
                        merged_extra.update(extra)
                    kwargs["extra"] = merged_extra
                    self.logger.log(
                        max(log_level, logging.INFO),
                        f"完成执行: {operation} (耗时: {duration:.3f}s)",
                        **kwargs
                    )
                return duration
            return 0.0
    
    @contextmanager
    def timer(self, operation: str, log_level: int = logging.INFO):
        """上下文管理器形式的计时器"""
        self.start_timer(operation)
        try:
            yield
        finally:
            self.end_timer(operation, log_level)


class DebugLogger:
    """增强的调试日志记录器"""
    # 类级别集合，跟踪已清除日志的logger（每个进程独立）
    _cleared_loggers = set()
    
    def __init__(self, name: str, log_dir: str = "log", run_dir: Optional[Path] = None):
        self.name = name
        self.base_log_dir = Path(log_dir)
        self.logger_dir = self.base_log_dir / self.name
        self.logger_dir.mkdir(parents=True, exist_ok=True)
        self._owner_pid = os.getpid()  # 记录创建该logger的进程
        if run_dir is not None:
            # 子进程复用父进程的运行目录，实现同一批日志输出
            self.run_dir = Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.run_timestamp = self.run_dir.name
        else:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self._create_run_dir()
        # 兼容旧字段命名，后续写文件统一落在 run_dir 下
        self.log_dir = self.run_dir
        
        # 创建主日志记录器
        self.logger = logging.getLogger(name)
        # logger级别取落盘最低级别与WARNING的较小值，确保控制台WARNING可用
        self.logger.setLevel(min(FILE_LOG_MIN_LEVEL, logging.WARNING))
        self.logger.propagate = False
        
        # 异步写入队列和监听器（用于文件异步写入）
        self._async_queue: Optional[queue.Queue] = None
        self._async_listener: Optional[logging.handlers.QueueListener] = None
        
        # 在主进程中清除之前的日志文件（每个logger只清除一次）
        if _is_main_process() and name not in DebugLogger._cleared_loggers:
            self._clear_old_logs()
            DebugLogger._cleared_loggers.add(name)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()
        
        # 性能监控器（传入 self 引用，用于调用 performance 方法）
        self.perf = PerformanceLogger(self.logger, debug_logger=self)
        
        # 错误统计
        self._error_count = 0
        self._warning_count = 0
        self._lock = threading.Lock()
    
    def _create_run_dir(self) -> Path:
        """为本次运行创建独立日志目录，名称为时间戳。"""
        base_name = self.run_timestamp
        run_dir = self.logger_dir / base_name
        suffix = 1
        while run_dir.exists():
            run_dir = self.logger_dir / f"{base_name}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _clear_old_logs(self):
        """清理旧日志运行目录，保留最近 N 次运行"""
        try:
            max_keep = 3  # 最多保留的运行目录数
            run_dirs = [
                d for d in self.logger_dir.iterdir()
                if d.is_dir()
            ]
            run_dirs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            deleted_count = 0
            for old_dir in run_dirs[max_keep:]:
                try:
                    shutil.rmtree(old_dir, ignore_errors=True)
                    deleted_count += 1
                except Exception:
                    pass

            # 保持静默清理，避免在控制台输出冗余信息
        except Exception:
            # 清除失败不影响日志系统初始化
            pass
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 控制台处理器（所有进程都使用）
        console_handler = logging.StreamHandler(sys.stdout)
        # 设为 WARNING：控制台只显示 WARNING/ERROR/CRITICAL，性能日志已降级为 INFO 不会刷屏
        console_handler.setLevel(logging.WARNING)
        console_formatter = ColoredFormatter(
            '%(asctime)s %(levelname)s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # 文件格式化器
        file_formatter = logging.Formatter(
            '%(asctime)s %(levelname)s [%(name)s] [PID:%(process)d] %(filename)s:%(lineno)d %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 多进程日志处理：使用队列统一写入
        log_queue = _get_log_queue()
        is_main = _is_main_process()
        
        # 如果是子进程且有队列，使用 QueueHandler；否则直接使用文件处理器
        if not is_main and log_queue is not None:
            # 子进程：使用 QueueHandler 发送日志到队列
            queue_handler = logging.handlers.QueueHandler(log_queue)
            self.logger.addHandler(queue_handler)
            # 控制台输出仍然保留
            self.logger.addHandler(console_handler)
            return  # 子进程只使用队列，不直接写文件

        # ---- 只通过"某个精确等级"的过滤器 ----
        class _LevelOnly(logging.Filter):
            def __init__(self, levelno: int):
                super().__init__()
                self.levelno = levelno
            def filter(self, record: logging.LogRecord) -> bool:
                return record.levelno == self.levelno

        # 性能日志现在也是 INFO，保留过滤器以兼容旧格式（当前不做过滤）
        class _ExcludePerf(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return True

        # ---- 按等级分别落盘（仅主进程） ----
        # 主进程：创建文件处理器并设置监听器
        handlers = []
        
        if FILE_LOG_MIN_LEVEL <= logging.DEBUG:
            debug_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.name}_debug.log",
                maxBytes=10*1024*1024, backupCount=5,
                encoding='utf-8', delay=True
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.addFilter(_LevelOnly(logging.DEBUG))
            debug_handler.addFilter(_ExcludePerf())
            debug_handler.setFormatter(file_formatter)
            handlers.append(debug_handler)

        if FILE_LOG_MIN_LEVEL <= logging.INFO:
            info_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.name}_info.log",
                maxBytes=10*1024*1024, backupCount=5,
                encoding='utf-8', delay=True
            )
            info_handler.setLevel(logging.INFO)
            info_handler.addFilter(_LevelOnly(logging.INFO))
            info_handler.addFilter(_ExcludePerf())
            info_handler.setFormatter(file_formatter)
            handlers.append(info_handler)

            prep_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.name}_prep.log",
                maxBytes=10*1024*1024, backupCount=5,
                encoding='utf-8', delay=True
            )
            prep_handler.setLevel(PREP_LEVEL_NUM)
            prep_handler.addFilter(_LevelOnly(PREP_LEVEL_NUM))
            prep_handler.addFilter(_ExcludePerf())
            prep_handler.setFormatter(file_formatter)
            handlers.append(prep_handler)

        if FILE_LOG_MIN_LEVEL <= logging.WARNING:
            warning_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.name}_warning.log",
                maxBytes=10*1024*1024, backupCount=5,
                encoding='utf-8', delay=True
            )
            warning_handler.setLevel(logging.WARNING)
            warning_handler.addFilter(_LevelOnly(logging.WARNING))
            warning_handler.addFilter(_ExcludePerf())
            warning_handler.setFormatter(file_formatter)
            handlers.append(warning_handler)

        if FILE_LOG_MIN_LEVEL <= logging.ERROR:
            error_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.name}_error.log",
                maxBytes=10*1024*1024, backupCount=5,
                encoding='utf-8', delay=True
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.addFilter(_LevelOnly(logging.ERROR))
            error_handler.addFilter(_ExcludePerf())
            error_handler.setFormatter(file_formatter)
            handlers.append(error_handler)

        critical_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_critical.log",
            maxBytes=10*1024*1024, backupCount=5,
            encoding='utf-8', delay=True
        )
        critical_handler.setLevel(logging.CRITICAL)
        critical_handler.addFilter(_LevelOnly(logging.CRITICAL))
        critical_handler.addFilter(_ExcludePerf())
        critical_handler.setFormatter(file_formatter)
        handlers.append(critical_handler)

        # 性能文件处理器
        perf_formatter = logging.Formatter(
            '%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_performance.log",
            maxBytes=10*1024*1024, backupCount=5,
            encoding='utf-8', delay=True
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(perf_formatter)
        # 仅写入标记为性能日志的记录
        class _PerfOnly(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    return bool(getattr(record, "is_perf", False))
                except Exception:
                    return False
        perf_handler.addFilter(_PerfOnly())
        handlers.append(perf_handler)

        # 主进程：使用异步写入队列（提高性能，避免阻塞）
        # 子进程：已经通过队列发送，不执行到这里
        if is_main:
            # 创建线程队列用于异步写入文件
            self._async_queue = queue.Queue(-1)  # 无限制队列
            
            # 为每个handler添加logger名称过滤器（用于多进程日志）
            class LoggerNameFilter(logging.Filter):
                def __init__(self, logger_name):
                    super().__init__()
                    self.logger_name = logger_name
                def filter(self, record):
                    return record.name == self.logger_name
            
            filtered_handlers = []
            for handler in handlers:
                handler.addFilter(LoggerNameFilter(self.logger.name))
                filtered_handlers.append(handler)
            
            # 创建异步写入监听器（后台线程处理文件写入）
            self._async_listener = logging.handlers.QueueListener(
                self._async_queue,
                *filtered_handlers,
                respect_handler_level=True
            )
            self._async_listener.start()
            
            # 使用QueueHandler将日志放入异步队列
            async_handler = logging.handlers.QueueHandler(self._async_queue)
            self.logger.addHandler(async_handler)
            
            # 主进程：启动全局队列监听器，避免多个监听器竞争同一队列
            _ensure_global_queue_listener()
        else:
            # 子进程：直接添加文件处理器（虽然不会执行到这里，但保留以防万一）
            for handler in handlers:
                self.logger.addHandler(handler)
        
        # 所有进程都添加控制台处理器（同步输出，保持实时性）
        self.logger.addHandler(console_handler)

    
    def debug(self, message: str, *args, **kwargs):
        """调试日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """信息日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """警告日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        with self._lock:
            self._warning_count += 1
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, exc_info: bool = True, *args, **kwargs):
        """错误日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        with self._lock:
            self._error_count += 1
        self.logger.error(message, exc_info=exc_info, *args, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, *args, **kwargs):
        """严重错误日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        with self._lock:
            self._error_count += 1
        self.logger.critical(message, exc_info=exc_info, *args, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """异常日志（自动包含异常信息）"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        with self._lock:
            self._error_count += 1
        self.logger.exception(message, **kwargs)
    
    def performance(self, message: str, **kwargs):
        """性能日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        # 作为 INFO 级别记录，去掉 [PERF] 前缀以减少噪音
        extra = kwargs.pop("extra", None) or {}
        merged_extra = {"is_perf": True}
        if isinstance(extra, dict):
            merged_extra.update(extra)
        kwargs["extra"] = merged_extra
        self.logger.info(message, **kwargs)

    # 新增：PREP 等级便捷方法
    def prep(self, message: str, **kwargs):
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        self.logger.log(PREP_LEVEL_NUM, message, **kwargs)
    
    def isEnabledFor(self, level: int) -> bool:
        """检查是否启用了指定级别的日志"""
        return self.logger.isEnabledFor(level)
    
    def log_function_call(self, func_name: str, args: tuple = (), kwargs: dict = None):
        """记录函数调用"""
        kwargs = kwargs or {}
        args_str = ", ".join([str(arg) for arg in args])
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        self.debug(f"调用函数: {func_name}({params})")
    
    def log_function_result(self, func_name: str, result: Any, duration: float = None):
        """记录函数返回结果"""
        duration_str = f" (耗时: {duration:.3f}s)" if duration else ""
        self.debug(f"函数返回: {func_name} -> {type(result).__name__}{duration_str}")
    
    def log_data_info(self, data_name: str, data: Any):
        """记录数据信息"""
        if hasattr(data, 'shape'):
            self.debug(f"数据信息: {data_name} shape={data.shape}")
        elif hasattr(data, '__len__'):
            self.debug(f"数据信息: {data_name} length={len(data)}")
        else:
            self.debug(f"数据信息: {data_name} type={type(data).__name__}")
    
    def get_stats(self) -> Dict[str, int]:
        """获取日志统计信息"""
        with self._lock:
            return {
                'error_count': self._error_count,
                'warning_count': self._warning_count
            }
    
    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self._error_count = 0
            self._warning_count = 0
    
    def stop(self):
        """停止异步监听器（应在程序退出时调用）"""
        # 停止异步写入监听器
        if self._async_listener is not None:
            try:
                self._async_listener.stop()
            except Exception:
                pass
            self._async_listener = None


def create_logger(name: str, log_dir: str = "log", run_dir: Optional[Path] = None) -> DebugLogger:
    """创建调试日志记录器"""
    return DebugLogger(name, log_dir, run_dir=run_dir)


def log_function_calls(logger: DebugLogger):
    """装饰器：记录函数调用"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.log_function_call(func.__name__, args, kwargs)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_function_result(func.__name__, result, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"函数执行失败: {func.__name__} (耗时: {duration:.3f}s)", exc_info=True)
                raise
        return wrapper
    return decorator


def log_performance(logger: DebugLogger, operation: str, log_level: int = logging.INFO):
    """装饰器：记录性能"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with logger.perf.timer(f"{operation}.{func.__name__}", log_level):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# 全局日志记录器实例
_global_loggers: Dict[str, DebugLogger] = {}
_shared_run_dirs: Dict[str, Path] = {}  # 记录父进程的 run_dir，供无队列时子进程复用

# 多进程日志队列和监听器
_log_queue: Optional[multiprocessing.Queue] = None
_log_listener: Optional[logging.handlers.QueueListener] = None
_log_queue_lock = threading.Lock()
_log_listener_lock = threading.Lock()
_log_queue_init_failed = False
_log_manager: Optional[multiprocessing.managers.SyncManager] = None
_log_manager_clients: list = []
_LOG_QUEUE_ADDR_ENV = "LOG_QUEUE_MANAGER_ADDR"
_LOG_QUEUE_AUTH_ENV = "LOG_QUEUE_MANAGER_AUTH"
_SHARED_RUN_DIR_PREFIX = "LOG_SHARED_RUN_DIR_"


def _export_shared_run_dir(name: str, run_dir: Optional[Path]) -> None:
    """将 run_dir 写入环境变量，spawn 子进程可复用"""
    if run_dir is None:
        return
    try:
        os.environ[f"{_SHARED_RUN_DIR_PREFIX}{name}"] = str(run_dir)
    except Exception:
        pass


def _read_shared_run_dir(name: str) -> Optional[Path]:
    """从环境变量读取父进程的 run_dir"""
    try:
        value = os.environ.get(f"{_SHARED_RUN_DIR_PREFIX}{name}")
        if value:
            return Path(value)
    except Exception:
        return None
    return None


def _start_log_manager() -> Optional[multiprocessing.managers.SyncManager]:
    """在主进程启动管理器，暴露统一队列供子进程连接"""
    global _log_manager
    if _log_manager is not None:
        return _log_manager

    class _LogQueueManager(multiprocessing.managers.SyncManager):
        pass

    _shared = {}

    def _get_or_create_queue():
        q = _shared.get("queue")
        if q is None:
            # 使用线程安全队列，由管理器进程持有，避免某些环境的信号量限制
            q = queue.Queue(-1)
            _shared["queue"] = q
        return q

    _LogQueueManager.register("get_log_queue", callable=_get_or_create_queue)
    try:
        authkey = os.urandom(16)
        _log_manager = _LogQueueManager(address=("127.0.0.1", 0), authkey=authkey)
        _log_manager.start()
        addr = _log_manager.address
        if addr:
            os.environ[_LOG_QUEUE_ADDR_ENV] = f"{addr[0]}:{addr[1]}"
            os.environ[_LOG_QUEUE_AUTH_ENV] = authkey.hex()
        return _log_manager
    except Exception:
        _log_manager = None
        return None


def _connect_existing_manager(addr_str: str, auth_hex: str):
    """子进程根据环境变量连接到主进程队列管理器"""
    try:
        host, port_str = addr_str.split(":")
        port = int(port_str)
        class _LogQueueManager(multiprocessing.managers.SyncManager):
            pass
        _LogQueueManager.register("get_log_queue")
        manager = _LogQueueManager(address=(host, port), authkey=bytes.fromhex(auth_hex))
        manager.connect()
        _log_manager_clients.append(manager)
        return manager.get_log_queue()
    except Exception:
        return None

def _get_log_queue() -> Optional[multiprocessing.Queue]:
    """获取全局日志队列（仅在主进程中创建）"""
    # 某些受限环境（如CI/沙箱）可能不允许创建多进程锁/队列，允许通过环境变量禁用
    if os.environ.get("LOG_DISABLE_MP_QUEUE", "").lower() in ("1", "true", "yes"):
        return None
    global _log_queue, _log_queue_init_failed
    if _log_queue_init_failed:
        return None
    if _log_queue is None:
        if _is_main_process():
            with _log_queue_lock:
                if _log_queue is None:
                    try:
                        manager = _start_log_manager()
                        if manager is not None:
                            _log_queue = manager.get_log_queue()
                        else:
                            _log_queue = multiprocessing.Queue(-1)
                    except Exception as e:
                        # 在不支持信号量/共享内存的沙箱环境下，降级为无多进程队列模式
                        _log_queue_init_failed = True
                        print(f"创建日志进程队列失败，已禁用多进程日志队列: {e}")
                        return None
        else:
            addr = os.environ.get(_LOG_QUEUE_ADDR_ENV)
            auth = os.environ.get(_LOG_QUEUE_AUTH_ENV)
            if addr and auth:
                queue_obj = _connect_existing_manager(addr, auth)
                if queue_obj is not None:
                    _log_queue = queue_obj
    return _log_queue


def _ensure_global_queue_listener():
    """确保只有一个全局队列监听器，避免多个监听器竞争同一个队列导致日志丢失"""
    global _log_listener
    # 仅主进程负责消费子进程发送的日志
    if not _is_main_process():
        return
    log_queue = _get_log_queue()
    if log_queue is None:
        return
    with _log_listener_lock:
        if _log_listener is not None:
            return

        class _DispatchToLogger(logging.Handler):
            """将进程队列中的日志转发到对应的 logger 管道"""
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    target = _global_loggers.get(record.name)
                    if target is not None:
                        # 复用目标 logger 已有的异步/文件/控制台处理链
                        target.logger.handle(record)
                    else:
                        # 如果目标 logger 尚未创建，降级交给同名 logger 处理
                        logging.getLogger(record.name).handle(record)
                except Exception:
                    self.handleError(record)

        _log_listener = logging.handlers.QueueListener(
            log_queue,
            _DispatchToLogger(),
            respect_handler_level=True
        )
        _log_listener.start()

# 全局仅创建一个队列监听器，避免多个监听器同时消费同一个队列导致日志丢失
# 每条日志都会根据 record.name 转发到对应的 logger 管道

def _stop_queue_listener():
    """停止队列监听器（应在程序退出时调用）"""
    global _log_listener, _log_manager
    with _log_listener_lock:
        if _log_listener is not None:
            try:
                _log_listener.stop()
            except Exception:
                pass
            _log_listener = None
    if _log_manager is not None:
        try:
            _log_manager.shutdown()
        except Exception:
            pass
        _log_manager = None


def stop_all_loggers():
    """停止所有日志记录器的异步监听器（应在程序退出时调用）"""
    for logger in _global_loggers.values():
        try:
            logger.stop()
        except Exception:
            pass
    _stop_queue_listener()


def get_logger(name: str, log_dir: str = "log") -> DebugLogger:
    """获取全局日志记录器"""
    # 队列会在创建logger时自动初始化（通过 _get_log_queue()）
    existing = _global_loggers.get(name)
    log_queue = _get_log_queue()
    # 进程切换时（如子进程 fork 后），需要为当前进程重新创建 logger，避免沿用父进程的 handler/线程
    if existing is None or getattr(existing, "_owner_pid", None) != os.getpid():
        # 记录父进程 run_dir，便于无队列时子进程复用
        if existing is not None:
            _shared_run_dirs[name] = getattr(existing, "run_dir", None) or _shared_run_dirs.get(name)
            _export_shared_run_dir(name, _shared_run_dirs.get(name))
            try:
                existing.stop()
            except Exception:
                pass
        # 清理根 logger 的旧 handler，避免重复输出
        base_logger = logging.getLogger(name)
        for handler in list(base_logger.handlers):
            try:
                base_logger.removeHandler(handler)
            except Exception:
                pass
        # 队列不可用时，子进程复用父进程的 run_dir；否则仍按正常流程创建
        reuse_run_dir = None
        if log_queue is None:
            reuse_run_dir = _shared_run_dirs.get(name) or _read_shared_run_dir(name)
        _global_loggers[name] = create_logger(name, log_dir, run_dir=reuse_run_dir)
        if name not in _shared_run_dirs:
            _shared_run_dirs[name] = getattr(_global_loggers[name], "run_dir", None)
        _export_shared_run_dir(name, _shared_run_dirs.get(name))
    return _global_loggers[name]


def cleanup_old_logs(log_dir: str = "log", days: int = 30):
    """清理旧日志文件/目录"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    deleted_count = 0
    
    for path in log_path.iterdir():
        try:
            if path.is_file() and path.suffix.startswith(".log"):
                if path.stat().st_mtime < cutoff_time:
                    path.unlink()
                    deleted_count += 1
                    print(f"已删除旧日志文件: {path}")
            elif path.is_dir():
                for run_dir in path.iterdir():
                    if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff_time:
                        shutil.rmtree(run_dir, ignore_errors=True)
                        deleted_count += 1
                        print(f"已删除旧日志目录: {run_dir}")
        except Exception as e:
            print(f"删除日志失败: {path}, 错误: {e}")
    
    if deleted_count > 0:
        print(f"清理完成，共删除 {deleted_count} 个旧日志")

# ==================== 日志初始化功能 ====================
# 确保log目录存在
log_dir = Path("log")
log_dir.mkdir(exist_ok=True)

def init_all_loggers():
    """初始化所有模块的日志记录器"""
    loggers = {}
    
    # 主要模块
    modules = [
        "scoring_core",
        "score_ui", 
        "predict_core",
        "data_reader",
        "download",
        # "stats_core",  # 已移除，功能已移到 score_ui.py
        "tdx_compat",
        "indicators",
        "utils"
    ]
    
    for module in modules:
        try:
            loggers[module] = get_logger(module)
        except Exception as e:
            print(f"初始化日志记录器失败 {module}: {e}")
    
    return loggers


def cleanup_logs():
    """清理超过30天的旧日志文件"""
    try:
        cleanup_old_logs("log", days=30)
    except Exception as e:
        print(f"清理旧日志文件失败: {e}")


def get_module_logger(module_name: str):
    """获取指定模块的日志记录器"""
    return get_logger(module_name)

# ==================== 便捷函数 ====================
def debug_log(message: str, logger_name: str = "default"):
    """快速调试日志"""
    logger = get_logger(logger_name)
    logger.debug(message)


def info_log(message: str, logger_name: str = "default"):
    """快速信息日志"""
    logger = get_logger(logger_name)
    logger.info(message)


def error_log(message: str, logger_name: str = "default", exc_info: bool = True):
    """快速错误日志"""
    logger = get_logger(logger_name)
    logger.error(message, exc_info=exc_info)


def warning_log(message: str, logger_name: str = "default"):
    """快速警告日志"""
    logger = get_logger(logger_name)
    logger.warning(message)


def critical_log(message: str, logger_name: str = "default", exc_info: bool = True):
    """快速严重错误日志"""
    logger = get_logger(logger_name)
    logger.critical(message, exc_info=exc_info)


def performance_log(message: str, logger_name: str = "default"):
    """快速性能日志"""
    logger = get_logger(logger_name)
    logger.performance(message)


def prep_log(message: str, logger_name: str = "default"):
    """快速 PREP 级别日志"""
    logger = get_logger(logger_name)
    logger.prep(message)


def log_data_processing(logger_name: str, operation: str, data_shape: tuple = None, 
                       duration: float = None, success: bool = True):
    """记录数据处理操作"""
    logger = get_logger(logger_name)
    status = "成功" if success else "失败"
    shape_info = f", 数据形状: {data_shape}" if data_shape else ""
    duration_info = f", 耗时: {duration:.3f}s" if duration else ""
    logger.info(f"[数据处理] {operation} {status}{shape_info}{duration_info}")


def log_database_operation(logger_name: str, operation: str, table: str = None, 
                          rows_affected: int = None, duration: float = None, success: bool = True):
    """记录数据库操作"""
    logger = get_logger(logger_name)
    status = "成功" if success else "失败"
    table_info = f", 表: {table}" if table else ""
    rows_info = f", 影响行数: {rows_affected}" if rows_affected is not None else ""
    duration_info = f", 耗时: {duration:.3f}s" if duration else ""
    logger.info(f"[数据库] {operation} {status}{table_info}{rows_info}{duration_info}")


def log_file_operation(logger_name: str, operation: str, file_path: str, 
                      file_size: int = None, duration: float = None, success: bool = True):
    """记录文件操作"""
    logger = get_logger(logger_name)
    status = "成功" if success else "失败"
    size_info = f", 文件大小: {file_size} bytes" if file_size else ""
    duration_info = f", 耗时: {duration:.3f}s" if duration else ""
    logger.info(f"[文件操作] {operation} {status}, 路径: {file_path}{size_info}{duration_info}")


def log_algorithm_execution(logger_name: str, algorithm: str, input_params: dict = None, 
                           output_shape: tuple = None, duration: float = None, success: bool = True):
    """记录算法执行"""
    logger = get_logger(logger_name)
    status = "成功" if success else "失败"
    params_info = f", 参数: {input_params}" if input_params else ""
    output_info = f", 输出形状: {output_shape}" if output_shape else ""
    duration_info = f", 耗时: {duration:.3f}s" if duration else ""
    logger.info(f"[算法执行] {algorithm} {status}{params_info}{output_info}{duration_info}")


def get_all_logger_stats() -> Dict[str, Dict[str, int]]:
    """获取所有日志记录器的统计信息"""
    stats = {}
    for name, logger in _global_loggers.items():
        stats[name] = logger.get_stats()
    return stats


def reset_all_logger_stats():
    """重置所有日志记录器的统计信息"""
    for logger in _global_loggers.values():
        logger.reset_stats()

# 导出主要类和函数
__all__ = [
    'DebugLogger',
    'PerformanceLogger',
    'create_logger',
    'get_logger',
    'log_function_calls',
    'log_performance',
    'cleanup_old_logs',
    'init_all_loggers',
    'cleanup_logs',
    'get_module_logger',
    'stop_all_loggers',
    'debug_log',
    'info_log',
    'error_log',
    'warning_log',
    'critical_log',
    'performance_log',
    'prep_log',
    'log_data_processing',
    'log_database_operation',
    'log_file_operation',
    'log_algorithm_execution',
    'get_all_logger_stats',
    'reset_all_logger_stats'
]

if __name__ == "__main__":
    # 测试日志系统
    print("初始化日志系统...")
    loggers = init_all_loggers()
    print(f"已初始化 {len(loggers)} 个日志记录器")
    
    # 测试日志记录
    test_logger = get_logger("test")
    test_logger.info("日志系统测试成功")
    test_logger.debug("这是调试信息")
    test_logger.warning("这是警告信息")
    test_logger.error("这是错误信息", exc_info=False)
    test_logger.critical("这是严重错误", exc_info=False)
    
    # 测试性能日志
    test_logger.performance("这是性能测试日志")
    test_logger.performance("操作耗时: 1.234秒")
    
    # 测试性能监控器
    with test_logger.perf.timer("测试操作", log_level=logging.WARNING):
        time.sleep(0.1)
    
    print("日志系统测试完成")
