#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一日志系统模块
提供完整的debug日志系统，支持多级别日志记录、性能监控和错误追踪
包含日志初始化、配置管理和便捷函数
"""

import os
import sys
import logging
import logging.handlers
# 避免logging内部报错打断业务
logging.raiseExceptions = False
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import traceback
import functools
import time
import threading
from contextlib import contextmanager
import multiprocessing

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
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        if hasattr(record, 'levelname') and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

class PerformanceLogger:
    """性能监控日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def start_timer(self, operation: str) -> None:
        """开始计时"""
        with self._lock:
            self._timers[operation] = time.time()
            self.logger.debug(f"开始执行: {operation}")
    
    def end_timer(self, operation: str, log_level: int = logging.WARNING) -> float:
        """结束计时并记录"""
        with self._lock:
            if operation in self._timers:
                duration = time.time() - self._timers[operation]
                del self._timers[operation]
                if isinstance(log_level, int):
                    self.logger.log(log_level, f"完成执行: {operation} (耗时: {duration:.3f}s)")
                else:
                    self.logger.debug(f"完成执行: {operation} (耗时: {duration:.3f}s)")
                return duration
            return 0.0
    
    @contextmanager
    def timer(self, operation: str, log_level: int = logging.WARNING):
        """上下文管理器形式的计时器"""
        self.start_timer(operation)
        try:
            yield
        finally:
            self.end_timer(operation, log_level)

class DebugLogger:
    """增强的调试日志记录器"""
    
    def __init__(self, name: str, log_dir: str = "log"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建主日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()
        
        # 性能监控器
        self.perf = PerformanceLogger(self.logger)
        
        # 错误统计
        self._error_count = 0
        self._warning_count = 0
        self._lock = threading.Lock()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_formatter = ColoredFormatter(
            '%(asctime)s %(levelname)s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # 进程后缀（子进程写独立文件，避免与主进程轮转冲突）
        pid_suffix = "" if _is_main_process() else f".{os.getpid()}"

        # 文件处理器 - 所有级别
        if _is_main_process():
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.name}.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8',
                delay=True
            )
        else:
            file_handler = logging.FileHandler(
                self.log_dir / f"{self.name}{pid_suffix}.log",
                mode="a",
                encoding='utf-8',
                delay=True
            )
        file_handler.setLevel(logging.WARNING)
        file_formatter = logging.Formatter(
            '%(asctime)s %(levelname)s [%(name)s] %(filename)s:%(lineno)d %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # 错误文件处理器
        if _is_main_process():
            error_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.name}_error.log",
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3,
                encoding='utf-8',
                delay=True
            )
        else:
            error_handler = logging.FileHandler(
                self.log_dir / f"{self.name}_error{pid_suffix}.log",
                mode="a",
                encoding='utf-8',
                delay=True
            )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        # 性能文件处理器
        if _is_main_process():
            perf_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.name}_performance.log",
                maxBytes=5*1024*1024,  # 5MB
                backupCount=2,
                encoding='utf-8',
                delay=True
            )
        else:
            perf_handler = logging.FileHandler(
                self.log_dir / f"{self.name}_performance{pid_suffix}.log",
                mode="a",
                encoding='utf-8',
                delay=True
            )
        perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter(
            '%(asctime)s [PERF] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)

        # 添加处理器（先清空，避免重复添加）
        self.logger.handlers = []
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(perf_handler)

    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        with self._lock:
            self._warning_count += 1
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """错误日志"""
        with self._lock:
            self._error_count += 1
        self.logger.error(message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """严重错误日志"""
        with self._lock:
            self._error_count += 1
        self.logger.critical(message, exc_info=exc_info, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """异常日志（自动包含异常信息）"""
        with self._lock:
            self._error_count += 1
        self.logger.exception(message, **kwargs)
    
    def performance(self, message: str, **kwargs):
        """性能日志"""
        # 使用 WARNING 级别，与控制台输出保持一致
        self.logger.warning(f"[PERF] {message}", **kwargs)
    
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

def log_performance(logger: DebugLogger, operation: str, log_level: int = logging.WARNING):
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

def get_logger(name: str, log_dir: str = "log") -> DebugLogger:
    """获取全局日志记录器"""
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
        "stats_core",
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
    'debug_log',
    'info_log',
    'error_log',
    'warning_log',
    'critical_log',
    'performance_log',
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
    test_logger.error("这是错误信息")
    
    print("日志系统测试完成")
