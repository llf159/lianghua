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

# ========== 自定义等级：PREP（介于 INFO 与 WARNING 之间） ==========
PREP_LEVEL_NUM = logging.INFO + 5  # 25
logging.addLevelName(PREP_LEVEL_NUM, "PREP")

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
        # 仍用 WARNING 级别，让控制台能看到性能提醒，但普通等级文件已排除 [PERF]
        self.logger.warning(f"[PERF] {message}", **kwargs)
    
    def start_timer(self, operation: str) -> None:
        """开始计时"""
        with self._lock:
            self._timers[operation] = time.time()
            self.logger.debug(f"开始执行: {operation}")
    
    def end_timer(self, operation: str, log_level: int = logging.DEBUG) -> float:
        """结束计时并记录"""
        with self._lock:
            if operation in self._timers:
                duration = time.time() - self._timers[operation]
                del self._timers[operation]
                # 使用 performance() 方法记录性能日志，确保写入性能日志文件
                if self.debug_logger is not None:
                    # 如果有 DebugLogger 引用，使用它的 performance 方法
                    self.debug_logger.performance(f"完成执行: {operation} (耗时: {duration:.3f}s)")
                elif log_level >= logging.INFO:
                    # 如果 log_level >= INFO，使用指定的级别记录
                    self.logger.log(log_level, f"完成执行: {operation} (耗时: {duration:.3f}s)")
                else:
                    # 默认使用 WARNING 级别，直接添加 [PERF] 前缀，会被性能日志过滤器捕获
                    self.logger.warning(f"[PERF] 完成执行: {operation} (耗时: {duration:.3f}s)")
                return duration
            return 0.0
    
    @contextmanager
    def timer(self, operation: str, log_level: int = logging.DEBUG):
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
    
    def __init__(self, name: str, log_dir: str = "log"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建主日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)  # 设置最低级别为INFO，不记录DEBUG
        self.logger.propagate = False
        
        # 队列监听器引用（仅主进程使用，用于多进程日志）
        self._queue_listener: Optional[logging.handlers.QueueListener] = None
        
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
    
    def _clear_old_logs(self):
        """清除之前的日志文件（包括备份文件）"""
        try:
            # 清除所有与该logger相关的日志文件
            log_patterns = [
                f"{self.name}_info.log*",
                f"{self.name}_prep.log*",
                f"{self.name}_warning.log*",
                f"{self.name}_error.log*",
                f"{self.name}_critical.log*",
                f"{self.name}_performance.log*",
                f"{self.name}_debug.log*"
            ]
            
            deleted_count = 0
            for pattern in log_patterns:
                for log_file in self.log_dir.glob(pattern):
                    try:
                        log_file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        # 如果文件被占用或其他错误，忽略继续处理其他文件
                        pass
            
            if deleted_count > 0:
                print(f"已清除 {deleted_count} 个旧日志文件: {self.name}")
        except Exception as e:
            # 清除失败不影响日志系统初始化
            pass
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 控制台处理器（所有进程都使用）
        console_handler = logging.StreamHandler(sys.stdout)
        # 设为 WARNING：控制台显示 WARNING/ERROR/CRITICAL（包括带[PERF]前缀的性能日志）
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

        # 排除 [PERF] 到普通等级日志文件，避免重复
        class _ExcludePerf(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    return not str(record.getMessage()).startswith("[PERF]")
                except Exception:
                    return True

        # ---- 按等级分别落盘（仅主进程） ----
        # 主进程：创建文件处理器并设置监听器
        handlers = []
        
        # DEBUG 日志已关闭（不写入文件）
        # debug_handler = logging.handlers.RotatingFileHandler(
        #     self.log_dir / f"{self.name}_debug.log",
        #     maxBytes=10*1024*1024, backupCount=5,
        #     encoding='utf-8', delay=True
        # )
        # debug_handler.setLevel(logging.DEBUG)
        # debug_handler.addFilter(_LevelOnly(logging.DEBUG))
        # debug_handler.addFilter(_ExcludePerf())
        # debug_handler.setFormatter(file_formatter)
        # handlers.append(debug_handler)

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
            '%(asctime)s [PERF] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_performance.log",
            maxBytes=10*1024*1024, backupCount=5,
            encoding='utf-8', delay=True
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(perf_formatter)
        # 仅写入 [PERF] 开头的记录
        class _PerfOnly(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    return str(record.getMessage()).startswith("[PERF]")
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
            
            # 主进程：启动监听器处理子进程发送到进程队列的日志
            log_queue = _get_log_queue()
            if log_queue is not None and self._queue_listener is None:
                # 子进程的日志也需要异步写入，所以也发送到异步队列
                # 创建一个特殊的handler，将进程队列的日志转发到异步队列
                class ProcessQueueToAsyncHandler(logging.Handler):
                    def __init__(self, async_queue):
                        super().__init__()
                        self.async_queue = async_queue
                    def emit(self, record):
                        try:
                            self.async_queue.put(record)
                        except Exception:
                            self.handleError(record)
                
                process_handler = ProcessQueueToAsyncHandler(self._async_queue)
                process_handler.addFilter(LoggerNameFilter(self.logger.name))
                
                listener = logging.handlers.QueueListener(
                    log_queue,
                    process_handler,
                    respect_handler_level=True
                )
                listener.start()
                self._queue_listener = listener
        else:
            # 子进程：直接添加文件处理器（虽然不会执行到这里，但保留以防万一）
            for handler in handlers:
                self.logger.addHandler(handler)
        
        # 所有进程都添加控制台处理器（同步输出，保持实时性）
        self.logger.addHandler(console_handler)

    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        with self._lock:
            self._warning_count += 1
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """错误日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        with self._lock:
            self._error_count += 1
        self.logger.error(message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """严重错误日志"""
        # 确保消息是UTF-8编码的字符串
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        with self._lock:
            self._error_count += 1
        self.logger.critical(message, exc_info=exc_info, **kwargs)
    
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
        # 仍用 WARNING 级别，让控制台能看到性能提醒，但普通等级文件已排除 [PERF]
        self.logger.warning(f"[PERF] {message}", **kwargs)

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
        
        # 停止多进程队列监听器
        if self._queue_listener is not None:
            try:
                self._queue_listener.stop()
            except Exception:
                pass
            self._queue_listener = None


def create_logger(name: str, log_dir: str = "log") -> DebugLogger:
    """创建调试日志记录器"""
    return DebugLogger(name, log_dir)


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


def log_performance(logger: DebugLogger, operation: str, log_level: int = logging.DEBUG):
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

# 多进程日志队列和监听器
_log_queue: Optional[multiprocessing.Queue] = None
_log_listener: Optional[logging.handlers.QueueListener] = None
_log_queue_lock = threading.Lock()

def _get_log_queue() -> Optional[multiprocessing.Queue]:
    """获取全局日志队列（仅在主进程中创建）"""
    global _log_queue
    if _log_queue is None and _is_main_process():
        with _log_queue_lock:
            if _log_queue is None:
                _log_queue = multiprocessing.Queue(-1)  # 无限制队列
    return _log_queue

# 注意：每个logger创建自己的监听器，监听同一个队列
# QueueListener会从队列中取出日志，并发送到所有提供的handler
# 由于handler会根据logger名称过滤，多个监听器可能导致日志丢失
# 实际方案：每个logger创建监听器，但只监听自己logger的日志
# 这是通过handler的filter机制实现的

def _stop_queue_listener():
    """停止队列监听器（应在程序退出时调用）"""
    global _log_listener
    if _log_listener is not None:
        try:
            _log_listener.stop()
        except Exception:
            pass
        _log_listener = None


def stop_all_loggers():
    """停止所有日志记录器的异步监听器（应在程序退出时调用）"""
    for logger in _global_loggers.values():
        try:
            logger.stop()
        except Exception:
            pass


def get_logger(name: str, log_dir: str = "log") -> DebugLogger:
    """获取全局日志记录器"""
    # 队列会在创建logger时自动初始化（通过 _get_log_queue()）
    if name not in _global_loggers:
        _global_loggers[name] = create_logger(name, log_dir)
    return _global_loggers[name]


def cleanup_old_logs(log_dir: str = "log", days: int = 30):
    """清理旧日志文件"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    deleted_count = 0
    
    for log_file in log_path.glob("*.log*"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                deleted_count += 1
                print(f"已删除旧日志文件: {log_file}")
            except Exception as e:
                print(f"删除日志文件失败: {log_file}, 错误: {e}")
    
    if deleted_count > 0:
        print(f"清理完成，共删除 {deleted_count} 个旧日志文件")

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
