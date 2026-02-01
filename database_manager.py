# -*- coding: utf-8 -*-
"""
database_manager.py — 统一数据库管理器
负责所有数据库连接、读写操作的统一管理
支持队列功能，确保并发读写有序进行
仅在需要时独立连接，避免多余进程占据数据库
"""
from __future__ import annotations

import os
import time
import queue
import threading
import logging
import json
import uuid
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 多进程环境检测
try:
    import multiprocessing
    _MULTIPROCESSING_AVAILABLE = True
    _get_current_process = multiprocessing.current_process
except ImportError:
    _MULTIPROCESSING_AVAILABLE = False
    def _get_current_process():
        class MockProcess:
            name = "MainProcess"
        return MockProcess()

# 配置日志 - 使用统一的日志系统
try:
    from log_system import get_logger
    logger = get_logger("database_manager")
except ImportError:
    # 回退到标准logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

class OperationType(Enum):
    """操作类型枚举"""
    READ = "read"
    WRITE = "write"
    BATCH_READ = "batch_read"
    BATCH_WRITE = "batch_write"
    DATA_IMPORT = "data_import"
    BATCH_IMPORT = "batch_import"

@dataclass
class DatabaseRequest:
    """数据库请求"""
    request_id: str
    operation_type: OperationType
    db_path: str
    sql: str
    params: Optional[List[Any]] = None
    priority: int = 0
    retry_count: int = 0
    callback: Optional[Callable] = None
    timeout: float = 30.0
    
    def __lt__(self, other):
        """支持优先级队列排序"""
        if not isinstance(other, DatabaseRequest):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.request_id < other.request_id

@dataclass
class DatabaseResponse:
    """数据库响应"""
    request_id: str
    success: bool
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    execution_time: float = 0.0

# 全局配置缓存，确保所有连接使用完全相同的配置
_DUCKDB_CONFIG = None
_DUCKDB_CONFIG_LOCK = threading.Lock()

def _abs_norm(p: str) -> str:
    """标准化路径为绝对路径（统一使用/分隔符）"""
    return os.path.abspath(p).replace("\\", "/")

# 检测直接使用duckdb.connect的警告
def _warn_direct_duckdb_usage():
    """警告直接使用duckdb.connect的情况"""
    import traceback
    stack = traceback.extract_stack()
    for frame in stack[:-1]:  # 排除当前函数
        if 'duckdb.connect' in frame.line:
            logger.warning(f"检测到直接使用duckdb.connect: {frame.filename}:{frame.lineno}")
            logger.warning("建议使用get_database_manager()或get_connection()统一管理连接")
            break


class DatabaseConnectionConfigManager:
    """统一的数据库连接配置管理器
    
    确保所有数据库连接使用相同的配置参数，避免配置不一致导致的错误。
    采用单例模式，保证全局唯一配置。
    """
    _instance = None
    _lock = threading.Lock()
    _config = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            # 检测进程信息（用于日志）
            process_name = "MainProcess"
            process_id = os.getpid()
            if _MULTIPROCESSING_AVAILABLE:
                try:
                    current_process = _get_current_process()
                    process_name = current_process.name
                except:
                    pass
            
            self._config = self._load_config()
            self._initialized = True
            
            # 记录配置信息，确保子进程也使用相同的配置
            config_info = f"threads={self._config.get('read', {}).get('threads')}, " \
                         f"memory_limit={self._config.get('read', {}).get('memory_limit')}, " \
                         f"temp_directory={self._config.get('read', {}).get('temp_directory')}"
            
            logger.info(
                f"数据库连接配置管理器初始化完成 "
                f"(进程: {process_name}, PID: {process_id}) - 配置: {config_info}"
            )
            
            # 如果是子进程，额外记录配置信息以便对比验证
            if process_name != "MainProcess":
                logger.info(
                    f"子进程 {process_name} (PID: {process_id}) 配置验证: "
                    f"只读配置={self._config.get('read', {})}, "
                    f"读写配置={self._config.get('write', {})}"
                )
    
    def _load_config(self) -> Dict[str, Dict[str, Any]]:
        """从config.py加载数据库配置，分别管理只读和读写配置"""
        logger.info("开始加载数据库连接配置...")
        try:
            from config import (
                DUCKDB_THREADS,
                DUCKDB_MEMORY_LIMIT,
                DUCKDB_TEMP_DIR,
                DB_QUERY_TIMEOUT,
                DB_ENABLE_INDEXES,
                DB_BATCH_SIZE,
                DUCKDB_CLEAR_DAILY_BEFORE,
            )
            
            logger.info(
                f"从config.py成功导入配置项: "
                f"DUCKDB_THREADS={DUCKDB_THREADS}, "
                f"DUCKDB_MEMORY_LIMIT={DUCKDB_MEMORY_LIMIT}, "
                f"DUCKDB_TEMP_DIR={DUCKDB_TEMP_DIR}, "
                f"DB_QUERY_TIMEOUT={DB_QUERY_TIMEOUT}, "
                f"DB_ENABLE_INDEXES={DB_ENABLE_INDEXES}, "
                f"DB_BATCH_SIZE={DB_BATCH_SIZE}, "
                f"DUCKDB_CLEAR_DAILY_BEFORE={DUCKDB_CLEAR_DAILY_BEFORE}"
            )
            
            # 注意：DuckDB 要求连接到同一个数据库文件的所有连接必须使用相同的配置参数
            # 因此，所有连接（只读和读写）都使用相同的配置参数
            # 使用较大的配置值（只读配置），对读写操作也没有坏处
            unified_threads = DUCKDB_THREADS  # 统一使用完整线程数
            unified_memory_limit = DUCKDB_MEMORY_LIMIT
            unified_temp_dir = DUCKDB_TEMP_DIR
            
            # 只读连接配置
            read_config = {
                'threads': unified_threads,
                'memory_limit': unified_memory_limit,
                'temp_directory': unified_temp_dir,
                'query_timeout': DB_QUERY_TIMEOUT,
                'enable_indexes': DB_ENABLE_INDEXES,
                'batch_size': DB_BATCH_SIZE,
                # 只读连接不需要清理配置
            }
            
            # 读写连接配置（使用相同的配置参数，避免配置冲突）
            write_config = {
                'threads': unified_threads,  # 使用相同的线程数，避免配置冲突
                'memory_limit': unified_memory_limit,
                'temp_directory': unified_temp_dir,
                'query_timeout': DB_QUERY_TIMEOUT * 2,  # 写操作可能需要更长时间（用于应用层逻辑）
                'enable_indexes': DB_ENABLE_INDEXES,
                'batch_size': DB_BATCH_SIZE,
                'clear_daily_before': DUCKDB_CLEAR_DAILY_BEFORE,
            }
            
            config = {
                'read': read_config,
                'write': write_config,
            }
            
            logger.info(
                f"数据库配置加载成功 - "
                f"只读配置: threads={read_config['threads']}, "
                f"memory_limit={read_config['memory_limit']}, "
                f"temp_directory={read_config['temp_directory']}, "
                f"query_timeout={read_config['query_timeout']}s, "
                f"enable_indexes={read_config['enable_indexes']}, "
                f"batch_size={read_config['batch_size']}; "
                f"读写配置: threads={write_config['threads']}, "
                f"memory_limit={write_config['memory_limit']}, "
                f"temp_directory={write_config['temp_directory']}, "
                f"query_timeout={write_config['query_timeout']}s, "
                f"enable_indexes={write_config['enable_indexes']}, "
                f"batch_size={write_config['batch_size']}, "
                f"clear_daily_before={write_config['clear_daily_before']}"
            )
            logger.debug(f"完整配置字典: {config}")
            return config
            
        except ImportError as e:
            logger.warning(
                f"无法从config模块导入配置，使用默认配置。错误详情: {e} (类型: {type(e).__name__})"
            )
            import traceback
            try:
                exc_str = traceback.format_exc()
                # 确保异常堆栈跟踪正确编码
                if isinstance(exc_str, bytes):
                    exc_str = exc_str.decode('utf-8', errors='replace')
                logger.debug(f"ImportError堆栈跟踪:\n{exc_str}")
            except Exception:
                logger.debug("ImportError堆栈跟踪: (无法格式化异常信息)")
            # 使用默认配置（所有连接使用相同的配置参数）
            unified_threads = 16
            unified_memory_limit = '18GB'
            unified_temp_dir = None
            read_config = {
                'threads': unified_threads,
                'memory_limit': unified_memory_limit,
                'temp_directory': unified_temp_dir,
                'query_timeout': 30,
                'enable_indexes': True,
                'batch_size': 1000,
            }
            write_config = {
                'threads': unified_threads,  # 使用相同的线程数
                'memory_limit': unified_memory_limit,
                'temp_directory': unified_temp_dir,
                'query_timeout': 60,
                'enable_indexes': True,
                'batch_size': 1000,
                'clear_daily_before': False,
            }
            logger.info(
                f"使用默认配置 - "
                f"只读配置: threads={read_config['threads']}, "
                f"memory_limit={read_config['memory_limit']}, "
                f"temp_directory={read_config['temp_directory']}, "
                f"query_timeout={read_config['query_timeout']}s; "
                f"读写配置: threads={write_config['threads']}, "
                f"memory_limit={write_config['memory_limit']}, "
                f"query_timeout={write_config['query_timeout']}s"
            )
            return {
                'read': read_config,
                'write': write_config,
            }
        except Exception as e:
            logger.error(
                f"加载数据库配置时发生异常: {e} (类型: {type(e).__name__})"
            )
            import traceback
            try:
                exc_str = traceback.format_exc()
                # 确保异常堆栈跟踪正确编码
                if isinstance(exc_str, bytes):
                    exc_str = exc_str.decode('utf-8', errors='replace')
                logger.error(f"异常堆栈跟踪:\n{exc_str}")
            except Exception:
                logger.error("异常堆栈跟踪: (无法格式化异常信息)")
            # 使用安全的默认配置（所有连接使用相同的配置参数）
            unified_threads = 16
            unified_memory_limit = '18GB'
            unified_temp_dir = None
            read_config = {
                'threads': unified_threads,
                'memory_limit': unified_memory_limit,
                'temp_directory': unified_temp_dir,
                'query_timeout': 30,
                'enable_indexes': True,
                'batch_size': 1000,
            }
            write_config = {
                'threads': unified_threads,  # 使用相同的线程数
                'memory_limit': unified_memory_limit,
                'temp_directory': unified_temp_dir,
                'query_timeout': 60,
                'enable_indexes': True,
                'batch_size': 1000,
                'clear_daily_before': False,
            }
            logger.warning("回退到安全默认配置")
            return {
                'read': read_config,
                'write': write_config,
            }
    
    def get_connection_config(self, read_only: bool = True) -> Dict[str, Any]:
        """获取连接配置字典
        
        Args:
            read_only: 是否为只读连接
            
        Returns:
            配置字典，可直接用于duckdb.connect()或conn.execute()设置配置
        """
        if not self._config:
            logger.warning("配置未初始化，返回空配置")
            return {}
        
        # 根据只读/读写模式返回对应的配置
        if read_only:
            return self._config.get('read', {}).copy()
        else:
            return self._config.get('write', {}).copy()
    
    def apply_config_to_connection(self, conn, read_only: bool = True):
        """将配置应用到已创建的连接
        
        Args:
            conn: DuckDB连接对象
            read_only: 是否为只读连接
        """
        config = self.get_connection_config(read_only)
        mode_str = "只读" if read_only else "读写"
        logger.info(f"开始应用配置到连接 (模式: {mode_str})")
        logger.debug(f"配置项: {config}")
        
        try:
            # 应用线程配置
            if 'threads' in config and config['threads']:
                threads_value = config['threads']
                logger.debug(f"设置线程数: SET threads={threads_value}")
                conn.execute(f"SET threads={threads_value}")
                logger.debug(f"线程数设置成功: {threads_value}")
            else:
                logger.debug("跳过线程配置（配置项为空或不存在）")
            
            # 应用内存限制
            if 'memory_limit' in config and config['memory_limit']:
                memory_limit_value = config['memory_limit']
                # 转义单引号避免SQL注入（虽然这里是配置值，但为了安全也应该处理）
                memory_limit_sql = str(memory_limit_value).replace("'", "''")
                sql_cmd = f"SET memory_limit='{memory_limit_sql}'"
                logger.debug(f"设置内存限制: {sql_cmd} (原始值: {memory_limit_value})")
                conn.execute(sql_cmd)
                logger.debug(f"内存限制设置成功: {memory_limit_value}")
            else:
                logger.debug("跳过内存限制配置（配置项为空或不存在）")
            
            # 应用临时目录
            if 'temp_directory' in config and config['temp_directory']:
                temp_dir = config['temp_directory']
                logger.debug(f"处理临时目录配置: {temp_dir}")
                # 确保临时目录存在，如果不存在则创建
                if temp_dir:
                    try:
                        logger.debug(f"创建临时目录（如果不存在）: {temp_dir}")
                        os.makedirs(temp_dir, exist_ok=True)
                        # 检查目录是否存在且可写
                        if not os.path.exists(temp_dir):
                            raise RuntimeError(f"临时目录创建失败: {temp_dir}")
                        if not os.access(temp_dir, os.W_OK):
                            raise RuntimeError(f"临时目录不可写: {temp_dir}")
                        logger.debug(f"临时目录验证成功: {temp_dir}")
                        
                        # 将Windows路径的反斜杠转换为正斜杠，并转义单引号避免SQL注入
                        # DuckDB接受正斜杠路径，即使在Windows上
                        temp_dir_original = temp_dir
                        temp_dir_sql = temp_dir.replace('\\', '/').replace("'", "''")
                        sql_cmd = f"SET temp_directory='{temp_dir_sql}'"
                        logger.debug(
                            f"设置临时目录: {sql_cmd} "
                            f"(原始路径: {temp_dir_original}, SQL路径: {temp_dir_sql})"
                        )
                        conn.execute(sql_cmd)
                        logger.info(f"临时目录设置成功: {temp_dir_original} -> {temp_dir_sql}")
                    except Exception as dir_error:
                        logger.warning(
                            f"无法创建或使用临时目录 {temp_dir}: "
                            f"{dir_error} (类型: {type(dir_error).__name__})"
                        )
                        import traceback
                        try:
                            exc_str = traceback.format_exc()
                            # 确保异常堆栈跟踪正确编码
                            if isinstance(exc_str, bytes):
                                exc_str = exc_str.decode('utf-8', errors='replace')
                            logger.debug(f"临时目录错误堆栈跟踪:\n{exc_str}")
                        except Exception:
                            logger.debug("临时目录错误堆栈跟踪: (无法格式化异常信息)")
                else:
                    logger.debug("临时目录配置为空，跳过")
            else:
                logger.debug("跳过临时目录配置（配置项为空或不存在）")
            
            # 注意：DuckDB 不支持 query_timeout 配置参数
            # 查询超时应该通过应用层的超时机制来控制
            # 配置字典中的 query_timeout 保留用于应用层逻辑，但不在此处设置
            if 'query_timeout' in config:
                logger.debug(
                    f"query_timeout配置值: {config['query_timeout']}s "
                    f"(注意: DuckDB不支持此配置，由应用层控制)"
                )
            
            # 应用索引配置（如果支持）
            if 'enable_indexes' in config:
                # DuckDB默认启用索引，这里主要用于记录配置状态
                logger.debug(f"索引配置: enable_indexes={config['enable_indexes']} (DuckDB默认启用)")
            
            if 'batch_size' in config:
                logger.debug(f"批处理大小配置: batch_size={config['batch_size']} (应用层使用)")
            
            logger.info(f"配置应用到连接成功 (模式: {mode_str})")
            logger.debug(f"完整应用的配置: {config}")
            
        except Exception as e:
            # 配置应用失败必须抛出异常，避免配置不一致的连接被使用
            import traceback
            error_detail = (
                f"应用配置到连接时出错 (模式: {mode_str}): "
                f"{e} (类型: {type(e).__name__})"
            )
            logger.error(error_detail)
            try:
                exc_str = traceback.format_exc()
                # 确保异常堆栈跟踪正确编码
                if isinstance(exc_str, bytes):
                    exc_str = exc_str.decode('utf-8', errors='replace')
                logger.error(f"配置应用错误堆栈跟踪:\n{exc_str}")
            except Exception:
                logger.error("配置应用错误堆栈跟踪: (无法格式化异常信息)")
            logger.error(f"当前配置: {config}")
            raise RuntimeError(f"配置应用失败: {e}") from e
    
    def get_config_summary(self, read_only: bool = None) -> str:
        """获取配置摘要（用于日志和调试）
        
        Args:
            read_only: 如果指定，返回对应模式的配置摘要；如果为None，返回所有配置摘要
        """
        if not self._config:
            return "配置未初始化"
        
        if read_only is not None:
            config = self.get_connection_config(read_only)
            mode = "只读" if read_only else "读写"
            summary = ", ".join([f"{k}={v}" for k, v in config.items()])
            return f"数据库配置({mode}): {summary}"
        else:
            read_summary = ", ".join([f"{k}={v}" for k, v in self._config.get('read', {}).items()])
            write_summary = ", ".join([f"{k}={v}" for k, v in self._config.get('write', {}).items()])
            return f"数据库配置(只读): {read_summary} | 数据库配置(读写): {write_summary}"
    
    def reload_config(self):
        """重新加载配置（用于配置更新后刷新）"""
        with self._lock:
            old_config = self._config.copy() if self._config else None
            self._config = self._load_config()
            logger.info(f"配置已重新加载: 旧配置={old_config}, 新配置={self._config}")


# 全局配置管理器实例
_config_manager = None
_config_manager_lock = threading.Lock()

def get_config_manager() -> DatabaseConnectionConfigManager:
    """获取数据库连接配置管理器实例（单例）
    
    在多进程环境下，每个进程会创建自己的配置管理器实例。
    这是正常行为，因为进程间不能共享内存，每个进程需要独立加载配置。
    
    每个进程的配置管理器会：
    1. 从 config.py 加载相同的配置参数
    2. 确保该进程内的所有连接使用统一的配置
    3. 避免同一进程内不同连接使用不同配置导致的冲突
    """
    global _config_manager
    
    if _config_manager is None:
        with _config_manager_lock:
            if _config_manager is None:
                # 检测是否为子进程（用于日志）
                process_name = "MainProcess"
                process_id = os.getpid()
                if _MULTIPROCESSING_AVAILABLE:
                    try:
                        current_process = _get_current_process()
                        process_name = current_process.name
                    except:
                        pass
                
                _config_manager = DatabaseConnectionConfigManager()
                if process_name != "MainProcess":
                    logger.info(
                        f"子进程 {process_name} (PID: {process_id}) 已创建配置管理器实例，"
                        f"确保该子进程的所有数据库连接使用统一配置"
                    )
    
    return _config_manager


class DatabaseConnectionPool:
    """数据库连接池"""
    
    def __init__(self, db_path: str, max_connections: int = 10, 
                 read_only: bool = True, config_manager: Optional[DatabaseConnectionConfigManager] = None):
        if not db_path:
            raise ValueError("数据库路径不能为空")
        self.db_path = os.path.abspath(db_path)
        self.max_connections = max_connections
        self.read_only = read_only  # 连接池模式：只读或读写
        # 配置管理器：如果未传入则使用单例
        self._config_manager = config_manager if config_manager is not None else get_config_manager()
        self._pool = queue.Queue(maxsize=max_connections)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)  # 条件变量用于阻塞等待
        self._active_connections = set()
        self._connection_created_times = {}  # 记录连接创建时间
        self._connection_last_used = {}     # 记录连接最后使用时间
        self._connection_types = {}        # 记录连接类型
        self._max_idle_time = 300  # 5分钟空闲超时
        self._max_connection_age = 3600  # 1小时连接最大年龄
        self._stats = {
            'total_created': 0,
            'total_reused': 0,
            'active_count': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'connections_closed_idle': 0,
            'connections_closed_aged': 0,
            'connection_errors': 0,
            'wait_timeouts': 0,  # 等待超时次数
            'wait_success': 0    # 等待成功次数
        }
    
    def _create_connection(self, read_only: bool = True):
        """创建新连接
        
        注意：移除了所有重试和关闭连接解决冲突的逻辑。
        遇到连接问题（包括配置冲突、数据库锁定等）将直接抛出异常，不再兜底。
        原因：
        1. 关闭连接无法确保资源真正释放，DuckDB的文件锁由操作系统和数据库引擎控制
        2. 简单的重试和等待无法解决根本问题，反而可能掩盖问题
        3. 连接问题应该通过合理的连接池管理、连接复用和架构设计来避免，而不是依赖兜底处理
        """
        import duckdb
        
        # 根据 read_only 参数创建相应类型的连接
        use_read_only = read_only
        process_name = "MainProcess"
        process_id = os.getpid()
        
        if _MULTIPROCESSING_AVAILABLE:
            current_process = _get_current_process()
            process_name = current_process.name
            if current_process.name != "MainProcess":
                # 子进程的读取操作强制使用 read_only=True
                if read_only:
                    use_read_only = True
                    logger.debug(f"子进程 {process_name} (PID: {process_id}) 读取操作使用 read_only=True 模式")
                else:
                    # 写入操作允许使用 read_only=False
                    use_read_only = False
                    logger.debug(f"子进程 {process_name} (PID: {process_id}) 写入操作使用 read_only=False 模式")
        elif os.name == 'nt':  # Windows 单进程环境
            # Windows 上，如果是只读操作，使用 read_only=True
            # 这样可以允许多个进程同时打开同一个数据库文件
            if read_only:
                use_read_only = True
            else:
                use_read_only = False
        
        # 记录连接创建尝试的详细信息
        logger.debug(
            f"[连接创建] 进程: {process_name} (PID: {process_id}), "
            f"数据库: {self.db_path}, "
            f"请求模式: read_only={read_only}, "
            f"实际模式: read_only={use_read_only}"
        )
        
        # 直接创建连接，不进行任何重试或兜底处理
        # 如果连接失败（配置冲突、数据库锁定等），直接抛出异常
        try:
            # 使用连接池的配置管理器（已在初始化时传入或获取单例）
            # 注意：DuckDB要求所有连接到同一个数据库的连接必须使用相同的配置
            # 因此，我们使用统一的配置（基于连接池的模式），确保所有连接配置一致
            pool_read_only = self.read_only
            
            # 先获取配置，确保配置已加载
            config = self._config_manager.get_connection_config(read_only=pool_read_only)
            
            # 创建连接
            conn = duckdb.connect(self.db_path, read_only=use_read_only)
            
            # 立即应用统一配置到连接（必须在连接创建后立即应用，确保配置一致性）
            # 注意：配置应用失败会导致连接配置不一致，必须关闭连接并抛出异常
            logger.debug(
                f"准备应用配置到新创建的连接 (进程: {process_name}, PID: {process_id}, "
                f"连接池模式: {'只读' if pool_read_only else '读写'}, "
                f"数据库路径: {self.db_path})"
            )
            try:
                self._config_manager.apply_config_to_connection(conn, read_only=pool_read_only)
                # 记录进程信息，确保子进程也使用统一配置
                config_summary = self._config_manager.get_config_summary(read_only=pool_read_only)
                logger.info(
                    f"配置应用成功 (进程: {process_name}, PID: {process_id}, "
                    f"连接池模式: {'只读' if pool_read_only else '读写'}, "
                    f"数据库: {self.db_path}): {config_summary}"
                )
                logger.debug(
                    f"完整配置详情 (进程: {process_name}, PID: {process_id}): "
                    f"{config_summary}"
                )
            except Exception as config_error:
                # 配置应用失败必须关闭连接并抛出异常，避免配置不一致的连接被使用
                import traceback
                logger.error(
                    f"配置应用失败，开始关闭连接 (进程: {process_name}, PID: {process_id})"
                )
                try:
                    conn.close()
                    logger.debug("连接已关闭")
                except Exception as close_error:
                    logger.warning(f"关闭连接时出错: {close_error}")
                error_msg = (
                    f"应用配置到连接失败 -> {self.db_path}\n"
                    f"  进程: {process_name} (PID: {process_id})\n"
                    f"  连接池模式: {'只读' if pool_read_only else '读写'}\n"
                    f"  连接对象: {conn} (ID: {id(conn)})\n"
                    f"  错误类型: {type(config_error).__name__}\n"
                    f"  错误信息: {config_error}\n"
                    f"  已关闭连接以避免配置不一致"
                )
                logger.error(error_msg)
                try:
                    exc_str = traceback.format_exc()
                    # 确保异常堆栈跟踪正确编码
                    if isinstance(exc_str, bytes):
                        exc_str = exc_str.decode('utf-8', errors='replace')
                    logger.error(f"配置应用失败堆栈跟踪:\n{exc_str}")
                except Exception:
                    logger.error("配置应用失败堆栈跟踪: (无法格式化异常信息)")
                raise RuntimeError(f"配置应用失败: {config_error}") from config_error
            
            # 记录连接创建时间和类型
            conn_id = id(conn)
            current_time = time.time()
            self._connection_created_times[conn_id] = current_time
            self._connection_last_used[conn_id] = current_time
            self._connection_types[conn_id] = read_only
            
            self._stats['total_created'] += 1
            logger.info(
                f"数据库连接创建成功: {self.db_path} "
                f"(连接ID: {conn_id}, 进程: {process_name}, PID: {process_id}, "
                f"只读: {use_read_only}, 连接池模式: {'只读' if pool_read_only else '读写'})"
            )
            logger.debug(
                f"连接统计: 总创建数={self._stats['total_created']}, "
                f"当前活跃连接数={len(self._active_connections)}, "
                f"连接池大小={self.max_connections}"
            )
            return conn
            
        except Exception as e:
            # 连接创建失败，直接抛出异常，不进行重试或关闭连接等兜底操作
            self._stats['connection_errors'] += 1
            error_msg = (
                f"创建数据库连接失败 -> {self.db_path}\n"
                f"  进程: {process_name} (PID: {process_id})\n"
                f"  请求模式: read_only={read_only}\n"
                f"  实际模式: read_only={use_read_only}\n"
                f"  错误: {e}\n"
                f"  可能原因: 其他进程已使用不同的 read_only 配置打开该数据库，"
                f"或配置参数不一致导致连接冲突"
            )
            logger.error(error_msg)
            raise
    
    def _close_all_existing_connections(self):
        """关闭所有现有连接
        
        注意：此方法仅用于连接池的正常清理和关闭，不应依赖此方法来解决数据库占用问题。
        原因：
        1. 关闭连接无法确保数据库文件锁立即释放，DuckDB的文件锁由操作系统和数据库引擎控制
        2. 简单的关闭操作无法解决多进程/多线程环境下的数据库占用问题
        3. 如果遇到数据库占用问题，应该通过合理的连接池管理、连接复用和架构设计来避免，
           而不是依赖关闭连接来解决。遇到连接问题时应该直接抛出异常，让调用方处理。
        """
        try:
            # 关闭活跃连接
            for conn in list(self._active_connections):
                self._close_connection(conn)
            self._active_connections.clear()
            
            # 清空连接池
            while not self._pool.empty():
                try:
                    conn_info = self._pool.get_nowait()
                    conn, _ = conn_info
                    self._close_connection(conn)
                except queue.Empty:
                    break
            
            logger.debug("所有现有连接已关闭")
            
        except Exception as e:
            logger.warning(f"关闭现有连接时出错: {e}")
    
    def get_connection(self, read_only: bool = True, timeout: float = 30.0):
        """获取连接 - 真正的连接池复用版本，支持阻塞等待"""
        # 确保连接请求的模式与连接池一致，避免混用 read_only/read_write 配置导致 DuckDB 拒绝新连接
        if read_only != self.read_only:
            logger.debug(
                f"连接请求模式 ({'只读' if read_only else '读写'}) 与连接池模式 "
                f"({'只读' if self.read_only else '读写'}) 不一致，已使用连接池模式以保持配置一致"
            )
            read_only = self.read_only
        with self._condition:
            try:
                # 先尝试从池中获取可用连接
                if not self._pool.empty():
                    try:
                        conn_info = self._pool.get_nowait()
                        conn, conn_read_only = conn_info
                        
                        # 检查连接是否仍然有效且类型匹配
                        conn_id = id(conn)
                        if (conn_read_only == read_only and 
                            conn_id in self._connection_created_times):
                            
                            # 检查连接是否过期
                            current_time = time.time()
                            if (current_time - self._connection_last_used[conn_id] < self._max_idle_time and
                                current_time - self._connection_created_times[conn_id] < self._max_connection_age):
                                
                                # 连接有效，更新使用时间
                                self._connection_last_used[conn_id] = current_time
                                self._active_connections.add(conn)
                                self._stats['pool_hits'] += 1
                                self._stats['active_count'] = len(self._active_connections)
                                logger.debug(f"复用数据库连接: {self.db_path} (只读: {read_only})")
                                return conn
                            else:
                                # 连接过期，关闭并创建新连接
                                self._close_connection(conn)
                        else:
                            # 连接类型不匹配或无效，关闭并创建新连接
                            self._close_connection(conn)
                    except queue.Empty:
                        pass  # 池为空，继续创建新连接
                
                # 池为空或没有可用连接，检查是否达到上限
                if len(self._active_connections) >= self.max_connections:
                    # 达到上限，阻塞等待连接释放
                    logger.debug(f"连接池已满，等待连接释放 (当前活跃: {len(self._active_connections)}/{self.max_connections})")
                    
                    # 使用条件变量等待，直到有连接释放或超时
                    start_time = time.time()
                    while len(self._active_connections) >= self.max_connections:
                        remaining_time = timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            self._stats['wait_timeouts'] += 1
                            raise TimeoutError(f"获取数据库连接超时: 等待 {timeout} 秒后仍无可用连接")
                        
                        # 等待连接释放，最多等待remaining_time秒
                        if not self._condition.wait(timeout=remaining_time):
                            self._stats['wait_timeouts'] += 1
                            raise TimeoutError(f"获取数据库连接超时: 等待 {timeout} 秒后仍无可用连接")
                    
                    self._stats['wait_success'] += 1
                    logger.debug(f"等待成功，获得可用连接槽位 (等待时间: {time.time() - start_time:.2f}秒)")
                
                # 创建新连接
                conn = self._create_connection(read_only)
                self._active_connections.add(conn)
                self._stats['pool_misses'] += 1
                self._stats['active_count'] = len(self._active_connections)
                logger.debug(f"[数据库连接] 创建新连接: {self.db_path} (只读: {read_only})")
                return conn
                
            except Exception as e:
                logger.error(f"获取数据库连接失败: {e}")
                raise
    
    def return_connection(self, conn):
        """归还连接 - 真正的连接池复用版本"""
        with self._condition:
            if conn in self._active_connections:
                self._active_connections.remove(conn)
                self._stats['active_count'] = len(self._active_connections)
                
                # 检查连接是否仍然有效且未过期
                conn_id = id(conn)
                current_time = time.time()
                
                if (conn_id in self._connection_created_times and
                    current_time - self._connection_last_used[conn_id] < self._max_idle_time and
                    current_time - self._connection_created_times[conn_id] < self._max_connection_age):
                    
                    # 连接仍然有效，尝试放回池中
                    try:
                        # 获取连接类型
                        conn_read_only = self._connection_types.get(conn_id, True)
                        
                        # 如果池未满，放回池中
                        if self._pool.qsize() < self.max_connections:
                            self._pool.put_nowait((conn, conn_read_only))
                            self._stats['total_reused'] += 1
                            logger.debug(f"连接已归还到池中: {self.db_path}")
                        else:
                            # 池已满，关闭连接
                            self._close_connection(conn)
                            logger.debug(f"连接池已满，关闭连接: {self.db_path}")
                    except queue.Full:
                        # 池已满，关闭连接
                        self._close_connection(conn)
                        logger.debug(f"连接池已满，关闭连接: {self.db_path}")
                else:
                    # 连接过期或无效，关闭连接
                    self._close_connection(conn)
                    if conn_id in self._connection_created_times:
                        if current_time - self._connection_last_used[conn_id] >= self._max_idle_time:
                            self._stats['connections_closed_idle'] += 1
                        if current_time - self._connection_created_times[conn_id] >= self._max_connection_age:
                            self._stats['connections_closed_aged'] += 1
                
                # 通知等待的线程有连接可用
                self._condition.notify_all()
    
    def _close_connection(self, conn):
        """关闭连接
        
        注意：此方法仅用于正常的连接清理和连接池管理，不应依赖此方法来解决数据库占用问题。
        原因：
        1. DuckDB的文件锁是在操作系统和数据库引擎层面管理的，简单的 close() 调用
           无法确保文件锁立即释放
        2. 在多进程/多线程环境下，即使当前进程关闭了连接，其他进程的连接仍然可能
           占用数据库文件
        3. 如果遇到数据库占用问题，应该通过合理的架构设计（如连接池复用、read_only
           模式、连接管理策略）来避免，而不是依赖关闭连接来解决
        """
        try:
            conn_id = id(conn)
            # 清理连接记录
            self._connection_created_times.pop(conn_id, None)
            self._connection_last_used.pop(conn_id, None)
            self._connection_types.pop(conn_id, None)
            # 关闭连接
            try:
                # 先尝试提交或回滚任何未完成的事务
                try:
                    conn.execute("ROLLBACK")
                except:
                    pass
            except:
                pass
            # 关闭连接
            try:
                conn.close()
                # DuckDB连接需要显式关闭
                import duckdb
                if hasattr(conn, 'close'):
                    conn.close()
            except Exception as close_error:
                # 如果关闭失败，尝试强制关闭
                try:
                    del conn
                except:
                    pass
        except Exception as e:
            logger.warning(f"关闭数据库连接时出错: {e}")
    
    def close_all(self):
        """关闭所有连接"""
        with self._lock:
            # 关闭活跃连接
            for conn in list(self._active_connections):
                self._close_connection(conn)
            self._active_connections.clear()
            
            # 清空连接池
            while not self._pool.empty():
                try:
                    conn_info = self._pool.get_nowait()
                    conn, _ = conn_info
                    self._close_connection(conn)
                except queue.Empty:
                    break
            
            self._stats['active_count'] = 0
            logger.debug(f"连接池已清空: {self.db_path}")
    
    def force_close_all_connections(self):
        """强制关闭所有连接，包括可能存在的其他连接"""
        try:
            import duckdb
            # 尝试关闭所有可能的连接
            # 注意：这是一个危险操作，可能会影响其他进程
            logger.warning("强制关闭所有DuckDB连接")
            # DuckDB没有全局关闭所有连接的方法，只能通过进程重启
        except Exception as e:
            logger.warning(f"强制关闭连接失败: {e}")
    
    def get_stats(self):
        """获取统计信息"""
        stats = self._stats.copy()
        
        # 计算复用率
        total_used = stats['pool_hits'] + stats['pool_misses']
        if total_used > 0:
            stats['reuse_rate'] = (stats['pool_hits'] / total_used) * 100
        else:
            stats['reuse_rate'] = 0.0
        
        # 添加池状态信息
        stats['pool_size'] = self._pool.qsize()
        stats['pool_capacity'] = self.max_connections
        stats['pool_utilization'] = (stats['active_count'] / self.max_connections) * 100 if self.max_connections > 0 else 0
        
        # 添加连接年龄统计
        current_time = time.time()
        if self._connection_created_times:
            ages = [current_time - created_time for created_time in self._connection_created_times.values()]
            stats['avg_connection_age'] = sum(ages) / len(ages) if ages else 0
            stats['max_connection_age'] = max(ages) if ages else 0
        else:
            stats['avg_connection_age'] = 0
            stats['max_connection_age'] = 0
        
        return stats


class DatabaseManager:
    """统一数据库管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # 多进程环境检测
        self._is_multiprocess = self._detect_multiprocess_environment()
        self._process_id = os.getpid()
        self._process_name = _get_current_process().name if _MULTIPROCESSING_AVAILABLE else "MainProcess"
        
        # 两套连接池：读写分离
        self._read_pools: Dict[str, DatabaseConnectionPool] = {}  # 只读连接池
        self._write_pools: Dict[str, DatabaseConnectionPool] = {}  # 读写连接池
        # 设置队列大小限制，防止内存占用过高
        max_queue_size = 1000
        self._request_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self._response_callbacks: Dict[str, Callable] = {}
        self._worker_threads: List[threading.Thread] = []
        self._shutdown = False
        self._pause_lock = threading.Lock()
        self._paused = False
        self._max_queue_size = max_queue_size
        self._queue_full_count = 0  # 队列满的次数统计
        
        # 查询缓存机制（多进程环境下每个进程独立缓存）
        self._query_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self._cache_lock = threading.RLock()
        self._cache_max_size = 100  # 最大缓存条目数
        self._cache_ttl = 300  # 缓存生存时间（秒）
        
        # 预加载数据缓存（用于存储全市场预加载数据）
        self._preload_caches: Dict[str, Dict[str, Any]] = {}  # key: cache_name, value: cache_dict
        self._preload_cache_lock = threading.RLock()
        # 全表内存缓存（针对小体量数据库，加速查询）
        self._full_table_cache: Dict[Tuple[str, str], pd.DataFrame] = {}  # key=(db_path, adj_type)
        self._full_cache_lock = threading.RLock()
        
        # 状态文件管理（合并自 database_status）
        try:
            from config import DATA_ROOT
            self._status_file_path = os.path.join(DATA_ROOT, "database_status.json")
        except ImportError:
            self._status_file_path = None
        
        # 启动工作线程
        self._start_workers()
        
        env_info = f"进程 {self._process_name} (PID: {self._process_id})"
        if self._is_multiprocess:
            env_info += " [多进程模式]"
        logger.info(f"数据库管理器初始化完成 {env_info} (队列大小限制: {max_queue_size}, 缓存大小: {self._cache_max_size}, 读写分离: 启用)")
    
    def _detect_multiprocess_environment(self) -> bool:
        """检测是否在多进程环境下运行"""
        try:
            if not _MULTIPROCESSING_AVAILABLE:
                return False
            current_process = _get_current_process()
            # 如果不是主进程，说明在多进程环境下
            return current_process.name != "MainProcess"
        except Exception:
            return False
    
    def _get_cache_key(self, ts_codes: List[str] = None, start_date: str = None, 
                      end_date: str = None, columns: List[str] = None, adj_type: str = "qfq") -> str:
        """生成缓存键 - 使用MD5哈希优化内存占用"""
        import hashlib
        
        # 对参数进行排序，确保相同的查询生成相同的键
        ts_codes_sorted = sorted(ts_codes) if ts_codes else []
        columns_sorted = sorted(columns) if columns else []
        
        # 将大列表转换为字符串后进行MD5哈希
        ts_codes_str = ','.join(ts_codes_sorted)
        columns_str = ','.join(columns_sorted)
        
        # 计算MD5哈希值
        ts_codes_hash = hashlib.md5(ts_codes_str.encode('utf-8')).hexdigest()[:8]
        columns_hash = hashlib.md5(columns_str.encode('utf-8')).hexdigest()[:8]
        
        # 构建紧凑的缓存键
        return f"{ts_codes_hash}:{start_date}:{end_date}:{columns_hash}:{adj_type}"
    
    def _is_cache_valid(self, cache_time: float) -> bool:
        """检查缓存是否有效"""
        return time.time() - cache_time < self._cache_ttl
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        with self._cache_lock:
            expired_keys = [
                key for key, (_, cache_time) in self._query_cache.items()
                if current_time - cache_time >= self._cache_ttl
            ]
            for key in expired_keys:
                del self._query_cache[key]
            logger.debug(f"清理过期缓存: {len(expired_keys)} 个条目")
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        with self._cache_lock:
            if cache_key in self._query_cache:
                df, cache_time = self._query_cache[cache_key]
                if self._is_cache_valid(cache_time):
                    logger.debug(f"缓存命中: {cache_key}")
                    return df.copy()
                else:
                    # 缓存过期，删除
                    del self._query_cache[cache_key]
        return None
    
    def _set_cached_data(self, cache_key: str, df: pd.DataFrame):
        """设置缓存数据"""
        with self._cache_lock:
            # 如果缓存已满，清理最旧的条目
            if len(self._query_cache) >= self._cache_max_size:
                self._cleanup_cache()
                # 如果还是满，删除最旧的条目
                if len(self._query_cache) >= self._cache_max_size:
                    oldest_key = min(self._query_cache.keys(), 
                                   key=lambda k: self._query_cache[k][1])
                    del self._query_cache[oldest_key]
            
            self._query_cache[cache_key] = (df.copy(), time.time())
            logger.debug(f"缓存设置: {cache_key}")
    
    def _start_workers(self):
        """启动工作线程，线程数与评分并发 SC_MAX_WORKERS 对齐（至多 16）
        
        在多进程环境下，每个进程的工作线程数应该适当减少，避免总线程数过多
        """
        # 尝试获取 SC_MAX_WORKERS 配置，以便与评分并发对齐
        try:
            from config import SC_MAX_WORKERS
            base_worker_count = SC_MAX_WORKERS or min(os.cpu_count() or 4, 8)
        except ImportError:
            base_worker_count = min(os.cpu_count() or 4, 8)
        
        # 在多进程环境下，每个进程的工作线程数应该适当减少
        if self._is_multiprocess:
            # 在多进程环境下，每个进程使用较少的工作线程
            # 假设有 N 个进程，每个进程最多使用 base_worker_count / N 个线程
            try:
                # 尝试从环境变量获取进程数（由父进程设置）
                proc_count = int(os.environ.get('DB_PROCESS_COUNT', '1'))
                if proc_count > 1:
                    # 每个进程使用较少的线程，但至少保证有 2 个线程
                    worker_count = max(2, base_worker_count // proc_count)
                    worker_count = min(worker_count, 8)  # 每个进程最多 8 个线程
                else:
                    worker_count = min(base_worker_count, 8)
            except (ValueError, KeyError):
                # 无法确定进程数，保守估计使用较少线程
                worker_count = min(base_worker_count // 2, 8)
                worker_count = max(worker_count, 2)
            logger.info(f"多进程环境检测：每个进程使用 {worker_count} 个工作线程")
        else:
            # 单进程环境，可以使用更多线程
            worker_count = min(base_worker_count, 16)
        
        for i in range(worker_count):
            worker_name = f"DBWorker-{self._process_name}-{i}"
            # 在多进程环境下，使用daemon线程，避免阻塞子进程退出
            # 主进程使用非daemon线程，确保任务完成
            use_daemon = self._is_multiprocess
            worker = threading.Thread(
                target=self._worker_loop, 
                name=worker_name,
                daemon=use_daemon
            )
            worker.start()
            self._worker_threads.append(worker)

    def _reset_pools_for_path(self, abs_path: str, *, wait: float = 1.0, reason: str = ""):
        """自愈：关闭并移除指定路径的读/写连接池，等待锁释放"""
        logger.warning(
            f"触发连接池自愈（{abs_path}）：{reason or '未知原因'}，将关闭并重建连接池"
        )
        # 先尝试关闭读池和写池
        for label, pools in (("只读", self._read_pools), ("读写", self._write_pools)):
            pool = pools.pop(abs_path, None)
            if pool:
                try:
                    pool.close_all()
                    logger.debug(f"{label}连接池已关闭（自愈）: {abs_path}")
                except Exception as e:
                    logger.warning(f"{label}连接池关闭失败（自愈）{abs_path}: {e}")
        # 等待锁释放
        if wait and wait > 0:
            try:
                time.sleep(wait)
            except Exception:
                pass
    
    def _get_connection_pool(self, db_path: str, read_only: bool = True) -> DatabaseConnectionPool:
        """获取连接池 - 支持读写分离，不同数据库路径完全独立
        
        每个数据库路径的读取和写入连接池完全独立，互不干扰。
        例如：读取stockdata数据库和写入details数据库可以并行执行，不会冲突。
        
        注意：DuckDB 不允许同一个数据库文件同时存在 read_only=True 和 read_only=False 的连接。
        如果已经创建了读写连接池，所有连接（包括只读）都必须使用读写连接池。
        如果已经创建了只读连接池，需要读写连接时必须先关闭只读连接池。
        
        优化策略：
        - 对于需要写入的数据库（如details.db），直接使用读写连接池，避免先创建只读连接池后关闭
        - 对于只读数据库（如stock_data.db），使用只读连接池
        
        在多进程环境下，每个进程的连接池大小应该适当减少，避免总连接数过多
        """
        if not db_path:
            raise ValueError("数据库路径不能为空")
        abs_path = os.path.abspath(db_path)
        
        # 判断数据库是否需要写入操作
        # details.db 总是需要写入
        # stock_data.db 在下载场景下也需要写入，为了避免配置冲突，统一使用 read_only=False
        # 注意：下载模块总是需要写入，所以stock_data.db应该识别为写入数据库
        db_path_lower = abs_path.lower().replace('\\', '/')
        is_write_database = 'details' in db_path_lower or 'detail' in db_path_lower
        
        # stock_data.db 在下载/指标计算场景下始终会发生写入。
        # 为避免 DuckDB read_only/read_write 配置混用导致的冲突，统一使用读写连接池。
        is_stock_data_db = 'stock_data' in db_path_lower and db_path_lower.endswith('.db')
        if is_stock_data_db:
            if read_only:
                logger.debug(
                    f"数据库 {abs_path} 为 stock_data.db，覆盖只读请求并使用读写连接池，"
                    f"以避免 read_only/read_write 配置冲突"
                )
            else:
                logger.debug(
                    f"数据库 {abs_path} 需要写入（stock_data.db），"
                    f"使用读写连接池以避免配置冲突"
                )
            is_write_database = True
        
        with self._lock:
            # DuckDB 限制：同一个数据库文件的所有连接必须使用相同的 read_only 配置
            # 如果已有读写连接池，必须使用读写连接池（即使是只读操作）
            if abs_path in self._write_pools:
                if read_only:
                    # 已有读写连接池，只读操作也使用读写连接池
                    logger.debug(f"数据库 {abs_path} 已有读写连接池，只读操作使用读写连接池")
                return self._write_pools[abs_path]
            
            # 优化：对于需要写入的数据库，即使当前是只读操作，也优先创建读写连接池
            # 这样可以避免后续需要写入时关闭只读连接池的代价
            # 如果已经存在只读连接池，需要先关闭它（因为写入数据库必须使用读写连接池）
            if is_write_database:
                if abs_path in self._read_pools:
                    # 写入数据库已经存在只读连接池，需要关闭它并切换到读写连接池
                    logger.warning(
                        f"数据库 {abs_path} 已识别为写入数据库（多进程环境），但已有只读连接池，"
                        f"正在关闭只读连接池并切换到读写连接池以避免配置冲突"
                    )
                    try:
                        # 关闭只读连接池中的所有连接
                        read_pool = self._read_pools[abs_path]
                        # 检查是否有活跃连接
                        active_count = read_pool._stats.get('active_count', 0)
                        if active_count > 0:
                            logger.warning(
                                f"只读连接池有 {active_count} 个活跃连接，等待释放..."
                            )
                            # 等待活跃连接完成（最多等待3秒）
                            max_wait = 3.0
                            wait_start = time.time()
                            while active_count > 0 and (time.time() - wait_start) < max_wait:
                                time.sleep(0.2)
                                active_count = read_pool._stats.get('active_count', 0)
                            
                            # 若仍未释放，触发自愈重置
                            if active_count > 0:
                                self._reset_pools_for_path(
                                    abs_path, wait=1.5,
                                    reason="写库切换时只读池仍有活跃连接"
                                )
                                # 自愈后不再尝试关闭原池，直接进入写池创建逻辑
                                if abs_path not in self._read_pools:
                                    logger.debug(f"自愈后已移除只读池: {abs_path}")
                                    read_pool = None
                        
                        if read_pool is not None and abs_path in self._read_pools:
                            # 关闭连接池
                            read_pool.close_all()
                            # 等待更长时间，确保DuckDB连接真正关闭（至少1秒）
                            time.sleep(1.0)
                            # 移除只读连接池
                            del self._read_pools[abs_path]
                            logger.debug(f"已关闭只读连接池并切换到读写连接池: {abs_path}")
                    except Exception as e:
                        logger.warning(f"关闭只读连接池时出错: {e}")
                        # 强制重置以避免卡死
                        self._reset_pools_for_path(
                            abs_path, wait=1.5,
                            reason="写库切换关闭只读池失败"
                        )
                
                # 对于需要写入的数据库，直接创建读写连接池（即使是只读操作）
                if abs_path not in self._write_pools:
                    logger.debug(
                        f"数据库 {abs_path} 需要写入操作，只读操作也使用读写连接池，"
                        f"避免后续切换连接池的代价"
                    )
                    if self._is_multiprocess:
                        max_connections = 4
                    else:
                        max_connections = 10
                    config_manager = get_config_manager()
                    self._write_pools[abs_path] = DatabaseConnectionPool(
                        abs_path, 
                        max_connections=max_connections,
                        read_only=False,  # 读写连接池
                        config_manager=config_manager
                    )
                    logger.debug(f"创建读写连接池: {abs_path} (最大连接数: {max_connections})")
                return self._write_pools[abs_path]
            
            # 如果需要读写连接，但已有只读连接池，必须先关闭只读连接池
            if not read_only and abs_path in self._read_pools:
                read_pool = self._read_pools[abs_path]
                # 检查只读连接池是否有活跃连接
                active_count = read_pool._stats.get('active_count', 0)
                if active_count > 0:
                    logger.warning(
                        f"数据库 {abs_path} 已有只读连接池，且有 {active_count} 个活跃连接正在使用，"
                        f"需要等待这些连接释放后才能切换到读写连接池"
                    )
                    # 尝试关闭只读连接池，但如果有活跃连接，关闭可能会失败
                    # 这里等待一段时间，让活跃连接完成
                    max_wait = 5.0  # 最多等待5秒
                    wait_start = time.time()
                    while active_count > 0 and (time.time() - wait_start) < max_wait:
                        time.sleep(0.1)
                        active_count = read_pool._stats.get('active_count', 0)
                    
                    if active_count > 0:
                        # 自愈：强制重置该路径的连接池，避免长时间占用导致写入失败
                        self._reset_pools_for_path(
                            abs_path, wait=1.5,
                            reason="读写切换时只读池活跃连接未释放"
                        )
                        # 重置后尝试获取最新的read_pool引用
                        read_pool = self._read_pools.get(abs_path)
                        active_count = read_pool._stats.get('active_count', 0) if read_pool else 0
                        if active_count > 0:
                            logger.error(
                                f"数据库 {abs_path} 自愈后仍有 {active_count} 个只读活跃连接，"
                                f"无法切换到读写连接池，请稍后重试。"
                            )
                            raise RuntimeError(
                                f"无法获取读写连接：自愈后只读连接仍未释放 ({active_count})。"
                            )
                
                logger.warning(
                    f"数据库 {abs_path} 已有只读连接池，需要读写连接，"
                    f"正在关闭只读连接池并创建读写连接池"
                )
                try:
                    # 关闭只读连接池中的所有连接
                    if read_pool:
                        read_pool.close_all()
                        # 等待更长时间，确保DuckDB连接真正关闭（至少1秒）
                        time.sleep(1.0)
                        # 移除只读连接池
                        del self._read_pools[abs_path]
                        logger.debug(f"已成功关闭只读连接池并切换到读写连接池: {abs_path}")
                    else:
                        logger.debug(f"自愈后不存在只读池，直接创建读写池: {abs_path}")
                except Exception as e:
                    logger.error(f"关闭只读连接池时出错: {e}")
                    # 强制重置后再尝试写池创建
                    self._reset_pools_for_path(
                        abs_path, wait=1.5,
                        reason="读写切换关闭只读池失败"
                    )
                    read_pool = self._read_pools.get(abs_path)
                    if read_pool:
                        raise RuntimeError(f"无法关闭只读连接池以切换到读写模式: {e}")
            
            # 根据 read_only 参数选择相应的连接池
            if read_only:
                # 只读连接池：每个数据库路径独立管理
                if abs_path not in self._read_pools:
                    # 计算连接池大小
                    try:
                        from config import SC_MAX_WORKERS
                        base_max_connections = max(SC_MAX_WORKERS or 16, 16)
                    except ImportError:
                        base_max_connections = 16
                    
                    # 在多进程环境下，每个进程的连接池应该适当减少
                    if self._is_multiprocess:
                        try:
                            # 尝试从环境变量获取进程数
                            proc_count = int(os.environ.get('DB_PROCESS_COUNT', '1'))
                            if proc_count > 1:
                                # 每个进程使用较少的连接，但至少保证有 4 个连接
                                max_connections = max(4, base_max_connections // proc_count)
                                max_connections = min(max_connections, 12)  # 每个进程最多 12 个连接
                            else:
                                max_connections = min(base_max_connections, 12)
                        except (ValueError, KeyError):
                            # 无法确定进程数，保守估计使用较少连接
                            max_connections = max(4, base_max_connections // 2)
                            max_connections = min(max_connections, 12)
                        logger.debug(f"多进程环境：只读连接池 [{abs_path}] 大小调整为 {max_connections}")
                    else:
                        # 单进程环境，可以使用更多连接
                        max_connections = base_max_connections
                    
                    # 为每个数据库路径创建独立的只读连接池
                    config_manager = get_config_manager()
                    self._read_pools[abs_path] = DatabaseConnectionPool(
                        abs_path, 
                        max_connections=max_connections,
                        read_only=True,  # 只读连接池
                        config_manager=config_manager
                    )
                    logger.debug(f"创建只读连接池: {abs_path} (最大连接数: {max_connections})")
                return self._read_pools[abs_path]
            else:
                # 读写连接池：每个数据库路径独立管理
                if abs_path not in self._write_pools:
                    # 写入连接池保持较小（避免写锁冲突）
                    # 在多进程环境下，写入连接应该更少
                    if self._is_multiprocess:
                        max_connections = 4  # 多进程环境下每个进程最多 4 个写连接
                    else:
                        max_connections = 10  # 单进程环境可以使用更多写连接
                    # 为每个数据库路径创建独立的读写连接池
                    config_manager = get_config_manager()
                    self._write_pools[abs_path] = DatabaseConnectionPool(
                        abs_path, 
                        max_connections=max_connections,
                        read_only=False,  # 读写连接池
                        config_manager=config_manager
                    )
                    logger.debug(f"创建读写连接池: {abs_path} (最大连接数: {max_connections})")
                return self._write_pools[abs_path]
    
    def _worker_loop(self):
        """工作线程循环"""
        thread_name = threading.current_thread().name
        logger.debug(f"数据库工作线程启动: {thread_name}")
        
        while not self._shutdown:
            try:
                # 检查是否暂停
                with self._pause_lock:
                    if self._paused:
                        time.sleep(0.1)
                        continue
                
                # 获取请求
                try:
                    request = self._request_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # 处理请求
                self._process_request(request)
                
            except Exception as e:
                logger.error(f"数据库工作线程 {thread_name} 出错: {e}")
                # 添加短暂延迟避免错误循环
                time.sleep(0.1)
        
        logger.debug(f"数据库工作线程结束: {thread_name}")
    
    def _process_request(self, request: DatabaseRequest):
        """处理数据库请求
        
        不同数据库路径的读取和写入操作完全独立，不会互相阻塞。
        例如：读取stockdata数据库和写入details数据库可以并行执行。
        """
        start_time = time.time()
        response = None
        max_retries = 3
        retry_delay = 1.0
        
        # 获取数据库路径的规范化表示（用于日志）
        abs_db_path = os.path.abspath(request.db_path) if request.db_path else "unknown"
        
        logger.debug(
            f"开始处理数据库请求: {request.request_id} "
            f"[数据库: {abs_db_path}, 操作: {request.operation_type.value}]"
        )
        
        for attempt in range(max_retries + 1):
            try:
                # 根据操作类型选择连接池
                # 每个数据库路径的读取和写入连接池完全独立，互不干扰
                is_read_operation = request.operation_type in [OperationType.READ, OperationType.BATCH_READ]
                pool = self._get_connection_pool(request.db_path, read_only=is_read_operation)
                
                logger.debug(
                    f"使用{'只读' if is_read_operation else '读写'}连接进行 {request.operation_type.value} 操作 "
                    f"[数据库: {abs_db_path}]"
                )
                
                # 获取相应类型的连接
                conn = pool.get_connection(read_only=is_read_operation, timeout=request.timeout)
                
                # 对于写入操作，验证连接确实不是只读的
                is_write_operation = request.operation_type not in [OperationType.READ, OperationType.BATCH_READ]
                if is_write_operation:
                    try:
                        # 测试连接是否支持写入
                        conn.execute("CREATE TEMP TABLE write_test (id INTEGER)")
                        conn.execute("DROP TABLE write_test")
                        logger.debug("连接写入测试成功")
                    except Exception as e:
                        logger.warning(f"连接写入测试失败，重新创建连接: {e}")
                        pool.return_connection(conn)
                        conn = pool.get_connection(read_only=False, timeout=request.timeout)
                
                try:
                    # 执行SQL
                    if request.params:
                        result = conn.execute(request.sql, request.params).df()
                    else:
                        result = conn.execute(request.sql).df()
                    
                    # 创建响应
                    response = DatabaseResponse(
                        request_id=request.request_id,
                        success=True,
                        data=result,
                        execution_time=time.time() - start_time
                    )
                    
                    logger.debug(f"数据库请求执行成功: {request.request_id}")
                    # 成功执行，跳出重试循环
                    break
                    
                finally:
                    # 归还连接
                    pool.return_connection(conn)
                
            except Exception as e:
                error_msg = str(e)
                error_msg_lower = error_msg.lower()
                is_timeout = "timeout" in error_msg_lower or "超时" in error_msg
                is_table_not_exist = (
                    "catalog" in error_msg_lower and "does not exist" in error_msg_lower
                ) or any(keyword in error_msg_lower for keyword in [
                    "table", "does not exist", "no such table", "catalog", "relation"
                ]) and ("stock_details" in error_msg or "stock_data" in error_msg)
                
                if attempt < max_retries and is_timeout:
                    # 如果是超时错误且还有重试次数，等待后重试
                    logger.warning(f"数据库请求超时 {request.request_id} (尝试 {attempt + 1}/{max_retries + 1}): {error_msg}")
                    time.sleep(retry_delay * (attempt + 1))  # 指数退避
                    continue
                else:
                    # 创建错误响应
                    response = DatabaseResponse(
                        request_id=request.request_id,
                        success=False,
                        error=error_msg,
                        execution_time=time.time() - start_time
                    )
                    
                    if is_timeout:
                        logger.error(f"数据库请求最终超时 {request.request_id}: {error_msg}")
                    elif is_table_not_exist:
                        # 表不存在是预期情况，使用debug级别
                        logger.debug(f"数据库表不存在 {request.request_id}: {error_msg}")
                    else:
                        logger.error(f"数据库请求失败 {request.request_id}: {error_msg}")
                    break
        
        # 调用回调函数
        if request.callback and response:
            try:
                logger.debug(f"调用回调函数: {request.request_id}")
                request.callback(response)
                logger.debug(f"回调函数执行完成: {request.request_id}")
            except Exception as e:
                logger.error(f"回调函数执行失败 {request.request_id}: {e}")
        else:
            logger.warning(f"没有回调函数或响应: {request.request_id}, callback: {request.callback is not None}, response: {response is not None}")
    
    @contextmanager
    def get_connection(self, db_path: str, read_only: bool = True, timeout: float = 30.0):
        """获取数据库连接的上下文管理器
        
        不同数据库路径的读取和写入操作完全独立，互不干扰。
        例如：读取stockdata数据库和写入details数据库可以并行执行，不会冲突。
        
        每个数据库路径有独立的读取和写入连接池，确保不同数据库的操作不会互相阻塞。
        """
        # 在多进程环境下，如果是子进程且要写入details数据库，应该拒绝或强制改为只读模式
        # details数据库应该只有主进程才能写入，避免多进程写入冲突
        actual_read_only = read_only
        abs_db_path = os.path.abspath(db_path) if db_path else "unknown"
        
        if _MULTIPROCESSING_AVAILABLE:
            current_process = _get_current_process()
            if current_process.name != "MainProcess":
                # 子进程：检查是否是details数据库且要写入
                db_path_lower = abs_db_path.lower().replace('\\', '/')
                is_details_db = 'details' in db_path_lower and db_path_lower.endswith('.db')
                if is_details_db and not read_only:
                    # 子进程不允许写入details数据库，强制改为只读模式
                    actual_read_only = True
                    logger.warning(
                        f"子进程 {current_process.name} 尝试写入details数据库 {abs_db_path}，"
                        f"已强制改为只读模式。details数据库应该只有主进程才能写入。"
                    )
        
        # 获取连接池：每个数据库路径的读取和写入连接池完全独立
        # 注意：_get_connection_pool 已经处理了 DuckDB 连接配置冲突的问题
        # 如果已有读写连接池，只读操作也会使用读写连接池；如果已有只读连接池，需要读写连接时会先关闭只读连接池
        pool = self._get_connection_pool(db_path, read_only=actual_read_only)
        logger.debug(
            f"[数据库连接] 开始获取连接: {abs_db_path} "
            f"[{'只读' if actual_read_only else '读写'}模式]"
        )
        conn = None
        try:
            conn = pool.get_connection(read_only=actual_read_only, timeout=timeout)
            logger.debug(f"[数据库连接] 成功获取连接: {abs_db_path} [{'只读' if actual_read_only else '读写'}模式]")
            
            # 对于details数据库的写入操作，验证连接确实是读写连接
            if not actual_read_only:
                db_path_lower = os.path.abspath(db_path).lower().replace('\\', '/')
                is_details_db = 'details' in db_path_lower and db_path_lower.endswith('.db')
                if is_details_db:
                    # 验证连接支持写入操作
                    try:
                        # 测试连接是否支持写入
                        conn.execute("CREATE TEMP TABLE write_test (id INTEGER)")
                        conn.execute("DROP TABLE write_test")
                        logger.debug("details数据库写入连接验证成功")
                    except Exception as verify_error:
                        # 如果连接不支持写入，关闭连接并抛出异常
                        pool.return_connection(conn)
                        logger.error(f"details数据库写入连接验证失败: {verify_error}")
                        raise RuntimeError(f"无法获取details数据库的写入连接: {verify_error}")
            
            # 使用yield提供连接给调用方
            yield conn
        finally:
            if conn:
                try:
                    pool.return_connection(conn)
                    logger.debug(f"[数据库连接] 连接已归还: {abs_db_path} [{'只读' if actual_read_only else '读写'}模式]")
                except Exception as e:
                    logger.error(f"[数据库连接] 归还连接时出错: {abs_db_path}, 错误: {e}")
    
    def execute_query(self, db_path: str, sql: str, params: Optional[List[Any]] = None, 
                     timeout: float = 30.0, callback: Optional[Callable] = None) -> str:
        """异步执行查询"""
        request_id = f"query_{int(time.time() * 1000)}_{id(threading.current_thread())}"
        
        request = DatabaseRequest(
            request_id=request_id,
            operation_type=OperationType.READ,
            db_path=db_path,
            sql=sql,
            params=params,
            timeout=timeout,
            callback=callback
        )
        
        try:
            self._request_queue.put(request, timeout=5.0)  # 添加超时
        except queue.Full:
            self._queue_full_count += 1
            logger.warning(f"请求队列已满，丢弃查询请求: {request_id}")
            raise RuntimeError("数据库请求队列已满，请稍后重试")
        
        return request_id
    
    def execute_write(self, db_path: str, sql: str, params: Optional[List[Any]] = None,
                     timeout: float = 30.0, callback: Optional[Callable] = None) -> str:
        """异步执行写入"""
        request_id = f"write_{int(time.time() * 1000)}_{id(threading.current_thread())}"
        
        request = DatabaseRequest(
            request_id=request_id,
            operation_type=OperationType.WRITE,
            db_path=db_path,
            sql=sql,
            params=params,
            timeout=timeout,
            callback=callback
        )
        
        try:
            self._request_queue.put(request, timeout=5.0)  # 添加超时
        except queue.Full:
            self._queue_full_count += 1
            logger.warning(f"请求队列已满，丢弃写入请求: {request_id}")
            raise RuntimeError("数据库请求队列已满，请稍后重试")
        
        return request_id
    
    def execute_sync_query(self, db_path: str, sql: str, params: Optional[List[Any]] = None,
                          timeout: float = 30.0) -> pd.DataFrame:
        """同步执行查询 - 优化版：只读查询直接执行，但受连接池上限约束
        
        不同数据库路径的查询操作使用独立的连接池，不会互相阻塞。
        例如：读取stockdata数据库和写入details数据库可以并行执行，不会冲突。
        """
        # 检查是否为只读查询（SELECT语句）
        sql_upper = sql.strip().upper()
        is_read_only = sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')
        
        # 检查是否在当前进程的线程中调用
        current_thread = threading.current_thread()
        is_same_process = True  # 假设都在同一进程中
        
        # 如果是只读查询且在同一进程中，直接执行，但受连接池上限约束
        # 每个数据库路径的只读连接池完全独立，不会互相影响
        if is_read_only and is_same_process:
            try:
                abs_db_path = os.path.abspath(db_path) if db_path else "unknown"
                logger.debug(
                    f"直接执行只读查询，受连接池约束: {sql[:100]}... "
                    f"[数据库: {abs_db_path}]"
                )
                # 获取该数据库路径的独立只读连接池（与其他数据库路径完全独立）
                pool = self._get_connection_pool(db_path, read_only=True)
                
                # 检查连接池状态，如果接近上限则等待
                # 注意：这个连接池只针对当前数据库路径，不会影响其他数据库的操作
                if pool._stats['active_count'] >= pool.max_connections * 0.9:  # 90%阈值
                    logger.debug(
                        f"连接池接近上限 ({pool._stats['active_count']}/{pool.max_connections})，"
                        f"等待可用连接 [数据库: {abs_db_path}]"
                    )
                    # 使用连接池的阻塞等待机制，确保真正的限流
                    # 这里会触发连接池内部的阻塞等待逻辑
                
                # 从该数据库路径的独立只读连接池获取连接
                conn = pool.get_connection(read_only=True, timeout=timeout)
                
                try:
                    if params:
                        result = conn.execute(sql, params).df()
                    else:
                        result = conn.execute(sql).df()
                    logger.debug(f"直接查询执行成功，返回 {len(result)} 条记录")
                    # 去重重复列（尤其是 trade_date）
                    try:
                        dup_mask = result.columns.duplicated()
                        if dup_mask.any():
                            logger.warning(f"查询返回存在重复列: {list(result.columns[dup_mask])} -> 保留首列")
                            result = result.loc[:, ~dup_mask].copy()
                    except Exception:
                        pass
                    return result
                finally:
                    pool.return_connection(conn)
            except Exception as e:
                # 回退到原有的队列模式
                pass
        
        # 回退到原有的队列轮询模式（用于跨线程/批量/写入操作）
        result_container = {'data': None, 'error': None, 'completed': False}
        
        def callback(response: DatabaseResponse):
            logger.debug(f"同步查询回调被调用: {response.request_id}, 成功: {response.success}")
            if response.success:
                result_container['data'] = response.data
            else:
                result_container['error'] = response.error
            result_container['completed'] = True
        
        # 增加请求超时时间，给工作线程更多时间处理
        request_timeout = max(timeout, 60.0)  # 至少60秒
        request_id = self.execute_query(db_path, sql, params, request_timeout, callback)
        logger.debug(f"同步查询请求已提交: {request_id}")
        
        # 等待结果，使用更长的超时时间
        start_time = time.time()
        wait_timeout = timeout + 30.0  # 给额外30秒缓冲时间
        
        while not result_container['completed']:
            elapsed = time.time() - start_time
            if elapsed > wait_timeout:
                logger.error(f"同步查询超时: {request_id}, 队列大小: {self._request_queue.qsize()}, 已等待: {elapsed:.2f}秒")
                raise TimeoutError(f"查询超时: {request_id}")
            
            # 动态调整等待间隔
            if elapsed < 1.0:
                time.sleep(0.01)  # 前1秒快速检查
            else:
                time.sleep(0.1)   # 之后每100ms检查一次
        
        if result_container['error']:
            raise Exception(f"查询失败: {result_container['error']}")
        
        logger.debug(f"同步查询完成: {request_id}, 耗时: {time.time() - start_time:.2f}秒")
        return result_container['data']
    
    def execute_sync_write(self, db_path: str, sql: str, params: Optional[List[Any]] = None,
                          timeout: float = 30.0) -> bool:
        """同步执行写入"""
        result_container = {'success': None, 'error': None}
        
        def callback(response: DatabaseResponse):
            if response.success:
                result_container['success'] = True
            else:
                result_container['error'] = response.error
        
        request_id = self.execute_write(db_path, sql, params, timeout, callback)
        
        # 等待结果
        start_time = time.time()
        while result_container['success'] is None and result_container['error'] is None:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"写入超时: {request_id}")
            time.sleep(0.01)
        
        if result_container['error']:
            raise Exception(f"写入失败: {result_container['error']}")
        
        return result_container['success']
    
    def pause(self):
        """暂停处理"""
        with self._pause_lock:
            self._paused = True
        logger.info("数据库管理器已暂停")
    
    def resume(self):
        """恢复处理"""
        with self._pause_lock:
            self._paused = False
        logger.info("数据库管理器已恢复")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'read_pools': {},
            'write_pools': {},
            'queue_size': self._request_queue.qsize(),
            'max_queue_size': self._max_queue_size,
            'queue_full_count': self._queue_full_count,
            'queue_utilization': self._request_queue.qsize() / self._max_queue_size * 100,
            'paused': self._paused,
            'worker_count': len(self._worker_threads),
            'cache_stats': {
                'cache_size': len(self._query_cache),
                'cache_max_size': self._cache_max_size,
                'cache_utilization': len(self._query_cache) / self._cache_max_size * 100
            }
        }
        
        # 汇总连接池统计
        total_pool_hits = 0
        total_pool_misses = 0
        total_active_connections = 0
        total_created = 0
        total_reused = 0
        
        for db_path, pool in self._read_pools.items():
            pool_stats = pool.get_stats()
            stats['read_pools'][db_path] = pool_stats
            total_pool_hits += pool_stats.get('pool_hits', 0)
            total_pool_misses += pool_stats.get('pool_misses', 0)
            total_active_connections += pool_stats.get('active_count', 0)
            total_created += pool_stats.get('total_created', 0)
            total_reused += pool_stats.get('total_reused', 0)
        
        for db_path, pool in self._write_pools.items():
            pool_stats = pool.get_stats()
            stats['write_pools'][db_path] = pool_stats
            total_pool_hits += pool_stats.get('pool_hits', 0)
            total_pool_misses += pool_stats.get('pool_misses', 0)
            total_active_connections += pool_stats.get('active_count', 0)
            total_created += pool_stats.get('total_created', 0)
            total_reused += pool_stats.get('total_reused', 0)
        
        # 添加汇总统计
        total_used = total_pool_hits + total_pool_misses
        stats['summary'] = {
            'total_pool_hits': total_pool_hits,
            'total_pool_misses': total_pool_misses,
            'total_active_connections': total_active_connections,
            'total_created': total_created,
            'total_reused': total_reused,
            'overall_reuse_rate': (total_pool_hits / total_used * 100) if total_used > 0 else 0,
            'connection_efficiency': (total_reused / total_created * 100) if total_created > 0 else 0
        }
        
        # 添加DuckDB配置信息（从第一个可用连接池获取）
        try:
            # 尝试从读连接池或写连接池获取第一个数据库路径
            sample_db_path = None
            if self._read_pools:
                sample_db_path = next(iter(self._read_pools.keys()))
            elif self._write_pools:
                sample_db_path = next(iter(self._write_pools.keys()))
            
            # 不再使用配置参数，移除配置统计
        except Exception as e:
            logger.debug(f"获取连接统计失败: {e}")
        
        return stats
    
    def close_db_pools(self, db_path: str):
        """关闭指定数据库路径的所有连接池（只读和读写）"""
        if not db_path:
            return
        
        abs_path = os.path.abspath(db_path)
        logger.info(f"开始关闭数据库连接池: {abs_path}")
        
        # 关闭只读连接池
        if abs_path in self._read_pools:
            try:
                pool = self._read_pools[abs_path]
                pool.close_all()
                del self._read_pools[abs_path]
                logger.info(f"只读连接池已关闭: {abs_path}")
            except Exception as e:
                logger.error(f"关闭只读连接池失败 {abs_path}: {e}")
        
        # 关闭读写连接池
        if abs_path in self._write_pools:
            try:
                pool = self._write_pools[abs_path]
                pool.close_all()
                del self._write_pools[abs_path]
                logger.info(f"读写连接池已关闭: {abs_path}")
            except Exception as e:
                logger.error(f"关闭读写连接池失败 {abs_path}: {e}")
    
    def clear_connections_only(self):
        """仅清理连接池，不关闭工作线程 - 用于轻量级清理"""
        logger.info("开始清理数据库连接池...")
        
        # 清理只读连接池
        for db_path, pool in self._read_pools.items():
            try:
                pool.close_all()
                logger.debug(f"只读连接池已清理: {db_path}")
            except Exception as e:
                logger.error(f"清理只读连接池失败 {db_path}: {e}")
        
        # 清理读写连接池
        for db_path, pool in self._write_pools.items():
            try:
                pool.close_all()
                logger.debug(f"读写连接池已清理: {db_path}")
            except Exception as e:
                logger.error(f"清理读写连接池失败 {db_path}: {e}")
        
        logger.info("数据库连接池已清理（工作线程保持运行）")
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """优雅关闭数据库管理器"""
        logger.info("开始优雅关闭数据库管理器...")
        
        # 设置关闭标志
        self._shutdown = True
        
        # 暂停新请求
        with self._pause_lock:
            self._paused = True
        logger.info("已暂停新请求接收")
        
        if wait:
            # 等待工作线程结束
            logger.info(f"等待 {len(self._worker_threads)} 个工作线程结束...")
            start_time = time.time()
            
            for i, worker in enumerate(self._worker_threads):
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    logger.warning(f"等待工作线程超时，强制继续")
                    break
                    
                worker.join(timeout=min(remaining_time, 10.0))
                if worker.is_alive():
                    logger.warning(f"工作线程 {worker.name} 未能在超时时间内结束")
                else:
                    logger.debug(f"工作线程 {worker.name} 已结束")
        
        # 关闭所有连接池
        logger.info("关闭所有连接池...")
        
        # 关闭只读连接池
        for db_path, pool in self._read_pools.items():
            try:
                pool.close_all()
                logger.debug(f"只读连接池已关闭: {db_path}")
            except Exception as e:
                logger.error(f"关闭只读连接池失败 {db_path}: {e}")
        
        # 关闭读写连接池
        for db_path, pool in self._write_pools.items():
            try:
                pool.close_all()
                logger.debug(f"读写连接池已关闭: {db_path}")
            except Exception as e:
                logger.error(f"关闭读写连接池失败 {db_path}: {e}")
        
        # 清空连接池字典
        self._read_pools.clear()
        self._write_pools.clear()
        
        # 清空缓存
        with self._cache_lock:
            self._query_cache.clear()
        
        # 打印最终统计信息
        logger.info("数据库管理器关闭完成")
        logger.info(f"最终统计信息: {self.get_stats()}")
    
    def close_all(self):
        """关闭所有连接（向后兼容方法）"""
        self.shutdown(wait=True, timeout=30.0)
    
    # ================= 股票数据特化功能 =================
    
    def get_latest_trade_date(self, db_path: str = None) -> Optional[str]:
        """获取数据库最新交易日期（单个日期）"""
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            logger.debug(f"[get_latest_trade_date] 开始查询最新交易日，数据库路径: {db_path}")
            
            if not os.path.exists(db_path):
                logger.warning(f"[get_latest_trade_date] 数据库文件不存在: {db_path}")
                return None
            
            # 只查询最新日期，避免全表扫描
            sql = "SELECT MAX(trade_date) as max_date FROM stock_data LIMIT 1"
            logger.debug(f"[get_latest_trade_date] 执行SQL查询: {sql}")
            
            try:
                df = self.execute_sync_query(db_path, sql, [], timeout=30.0)
                logger.debug(f"[get_latest_trade_date] 查询返回DataFrame: 行数={len(df)}, 列={list(df.columns) if not df.empty else 'empty'}")
            except Exception as query_error:
                logger.error(f"[get_latest_trade_date] SQL查询执行失败: {query_error}")
                logger.error(f"[get_latest_trade_date] 查询失败详情: {type(query_error).__name__}: {str(query_error)}")
                import traceback
                try:
                    exc_str = traceback.format_exc()
                    # 确保异常堆栈跟踪正确编码
                    if isinstance(exc_str, bytes):
                        exc_str = exc_str.decode('utf-8', errors='replace')
                    logger.debug(f"[get_latest_trade_date] 查询失败堆栈: {exc_str}")
                except Exception:
                    logger.debug("[get_latest_trade_date] 查询失败堆栈: (无法格式化异常信息)")
                return None
            
            if not df.empty and "max_date" in df.columns:
                max_date_value = df["max_date"].iloc[0]
                if max_date_value is not None and pd.notna(max_date_value):
                    latest_date = str(max_date_value)
                    logger.info(f"[get_latest_trade_date] 成功获取最新交易日: {latest_date}")
                    return latest_date
                else:
                    logger.debug(f"[get_latest_trade_date] max_date值为None或NaN: {max_date_value}")
            else:
                if df.empty:
                    logger.debug(f"[get_latest_trade_date] 查询返回空DataFrame")
                else:
                    logger.debug(f"[get_latest_trade_date] DataFrame中无'max_date'列，列名: {list(df.columns)}")
            
            logger.info("[get_latest_trade_date] 数据库为空或无交易日数据")
            return None
            
        except Exception as e:
            logger.error(f"[get_latest_trade_date] 获取最新交易日失败: {e}")
            import traceback
            try:
                exc_str = traceback.format_exc()
                # 确保异常堆栈跟踪正确编码
                if isinstance(exc_str, bytes):
                    exc_str = exc_str.decode('utf-8', errors='replace')
                logger.error(f"[get_latest_trade_date] 异常堆栈: {exc_str}")
            except Exception:
                logger.error("[get_latest_trade_date] 异常堆栈: (无法格式化异常信息)")
            return None
    
    def get_trade_dates(self, db_path: str = None) -> List[str]:
        """
        获取交易日历（默认走缓存，每次调用都会检查未来覆盖是否满足15天，必要时自动刷新）。
        db_path 参数保留兼容，当前不再从数据库 distinct 读取。
        """
        try:
            cal = self.get_trade_calendar_cached(refresh_if_insufficient=True)
            if cal:
                return cal
            logger.warning("交易日历缓存为空，尝试回退数据库 distinct")
        except Exception as e:
            logger.warning(f"交易日历缓存获取失败: {e}，尝试回退数据库 distinct")
        # 兜底：旧逻辑从数据库 distinct
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            if not os.path.exists(db_path):
                logger.warning(f"数据库文件不存在: {db_path}")
                return []
            sql = "SELECT DISTINCT trade_date FROM stock_data ORDER BY trade_date"
            df = self.execute_sync_query(db_path, sql, [], timeout=30.0)
            if not df.empty and "trade_date" in df.columns:
                return df["trade_date"].astype(str).tolist()
        except Exception as e:
            logger.error(f"获取交易日列表失败: {e}")
        return []

    def get_trade_dates_from_db(self, db_path: str = None, *, table: str = "stock_data") -> List[str]:
        """
        获取数据库中实际存在的交易日期列表（按日期升序去重）。
        
        Args:
            db_path: 数据库路径；None 时使用默认路径或 details 路径
            table: 使用的表名，仅支持 'stock_data' 或 'stock_details'
        """
        try:
            table_date_col = {
                "stock_data": "trade_date",
                "stock_details": "ref_date",
            }
            date_col = table_date_col.get(table)
            if date_col is None:
                logger.error(f"不支持的表名: {table}")
                return []

            if table == "stock_details":
                db_path = get_details_db_path(db_path)
            elif db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)

            if not db_path or not os.path.exists(db_path):
                logger.warning(f"数据库文件不存在: {db_path}")
                return []

            sql = f"""
            SELECT DISTINCT {date_col} AS d
            FROM {table}
            WHERE {date_col} IS NOT NULL
            ORDER BY {date_col}
            """
            df = self.execute_sync_query(db_path, sql, [], timeout=30.0)
            if df is not None and not df.empty and "d" in df.columns:
                return df["d"].astype(str).tolist()
        except Exception as e:
            logger.error(f"从数据库获取交易日列表失败: {e}")
        return []

    def get_trade_dates_from_tushare(self, start_date: str, end_date: str) -> List[str]:
        """从Tushare获取交易日列表"""
        try:
            import tushare as ts
            from config import TOKEN
            
            if not TOKEN or TOKEN.startswith("在这里"):
                logger.error("Tushare Pro 未配置：请在 config.TOKEN 中设置有效 token")
                return []
            
            # 设置token
            ts.set_token(TOKEN)
            pro = ts.pro_api()
            
            # 设置代理
            import os
            for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                os.environ.pop(k, None)
            os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,api.tushare.pro")
            os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
            
            # 获取交易日历
            cal = pro.trade_cal(
                exchange="SSE",
                start_date=start_date,
                end_date=end_date,
                is_open=1,
                fields="cal_date"
            )
            
            if cal.empty:
                logger.debug(f'交易日历返回为空({start_date}~{end_date})')
                return []
            
            # 确保交易日列表按日期升序排列
            trading_days = cal["cal_date"].astype(str).tolist()
            trading_days.sort()
            logger.debug(f"从Tushare获取交易日列表: {len(trading_days)} 个日期")
            return trading_days
            
        except Exception as e:
            logger.error(f"从Tushare获取交易日列表失败: {e}")
            return []
    
    def get_trade_calendar_cached(self, start_date: Optional[str] = None, end_date: Optional[str] = None, refresh_if_insufficient: bool = True) -> List[str]:
        """
        从缓存获取全量交易日历；覆盖未来不足15天时自动刷新并写回缓存。
        """
        # 与 stock_list.csv 同目录（DATA_ROOT 下，与统一行情库同级）
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            cache_dir = os.path.dirname(db_path)
        except Exception:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(base_dir, "cache", "stock_list")
        cache_path = os.path.join(cache_dir, "trade_calendar.csv")
        try:
            from config import START_DATE as _CFG_START
            cfg_start = str(_CFG_START) if _CFG_START else "20000101"
        except Exception:
            cfg_start = "20000101"
        start = start_date or cfg_start
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=365)).strftime("%Y%m%d")
        need_until = (datetime.now() + timedelta(days=15)).strftime("%Y%m%d")
        need_since = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        # 过去30天窗口也作为覆盖要求；与显式 start 取更早者
        min_start_required = min(start, need_since)

        def _load_cache() -> List[str]:
            if not os.path.exists(cache_path):
                return []
            try:
                df = pd.read_csv(cache_path, dtype=str)
                if "cal_date" in df.columns:
                    vals = [str(x) for x in df["cal_date"].tolist() if pd.notna(x)]
                    vals = sorted(set(vals))
                    return vals
            except Exception as e:
                logger.debug(f"读取交易日历缓存失败: {e}")
            return []

        cached = _load_cache()
        cover_future_ok = bool(cached) and cached[-1] >= need_until
        cover_start_ok = bool(cached) and cached[0] <= min_start_required
        has_range = cover_future_ok and cover_start_ok
        if cached and has_range:
            return cached
        if cached and not refresh_if_insufficient:
            return cached

        fetched = self.get_trade_dates_from_tushare(min_start_required, end_date)
        if not fetched:
            return cached

        merged = sorted(set(cached) | set(fetched))
        try:
            os.makedirs(cache_dir, exist_ok=True)
            pd.DataFrame({"cal_date": merged}).to_csv(cache_path, index=False)
            logger.info(f"交易日历已更新缓存: {cache_path}，共 {len(merged)} 条")
        except Exception as e:
            logger.warning(f"写入交易日历缓存失败: {e}")
        return merged
    
    def get_smart_end_date(self, end_date_config: str) -> str:
        """
        智能获取结束日期，考虑市场开盘时间和休盘情况
        
        Args:
            end_date_config: 结束日期配置 ("today" 或具体日期)
            
        Returns:
            处理后的结束日期字符串
        """
        if end_date_config.lower() == "today":
            from datetime import datetime, timedelta
            now = datetime.now()
            today_str = datetime.now().strftime("%Y%m%d")
            
            # 获取最近15个交易日，用于判断今天是否开盘
            try:
                start_date = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")
                trading_days_list = self.get_trade_calendar_cached(start_date=start_date, end_date=today_str, refresh_if_insufficient=True)
                
                if trading_days_list:
                    # 检查今天是否开盘
                    today_is_trading = today_str in trading_days_list
                    logger.debug(f"[SMART] 交易日历(已排序): {trading_days_list}")
                    logger.debug(f"[SMART] 今天是否开盘: {today_is_trading}")
                    
                    if today_is_trading:
                        # 今天开盘，根据时间判断
                        # 使用15:00作为判断标准（15:00是收盘时间，15:00之前使用前一天，15:00及之后使用今天）
                        current_time = now.hour * 100 + now.minute  # 转换为HHMM格式便于比较
                        cutoff_time = 1500  # 15:00
                        
                        if current_time < cutoff_time:
                            # 15点前（即14:59及之前），使用前一个交易日
                            # 找到今天之前的最后一个交易日
                            prev_trading_day = None
                            for day in reversed(trading_days_list):
                                if day < today_str:
                                    prev_trading_day = day
                                    break
                            
                            if prev_trading_day:
                                logger.debug(f"[SMART] 收盘前运行 (当前时间: {now.hour:02d}:{now.minute:02d})，使用前一个交易日: {prev_trading_day}")
                                return prev_trading_day
                            else:
                                logger.warning(f"[SMART] 未找到前一个交易日，使用今天: {today_str}")
                                return today_str
                        else:
                            # 15:00及之后，使用今天
                            logger.debug(f"[SMART] 收盘后运行 (当前时间: {now.hour:02d}:{now.minute:02d})，使用今天: {today_str}")
                            return today_str
                    else:
                        # 今天不开盘，使用最近的交易日
                        # 仅使用今天及之前的交易日，防止缓存包含未来日期导致选到未来无数据的日子
                        past_or_today = [day for day in trading_days_list if day <= today_str]
                        if past_or_today:
                            latest_trading_day = past_or_today[-1]
                            logger.debug(f"[SMART] 今天休盘，使用最近交易日(<=today): {latest_trading_day}")
                            return latest_trading_day
                        # 兜底：若列表全是未来日期，仍用最后一个并记录警告
                        latest_trading_day = trading_days_list[-1]
                        logger.warning(f"[SMART] 交易日历仅有未来日期，兜底使用: {latest_trading_day}")
                        return latest_trading_day
                else:
                    # 无法获取交易日历，使用今天
                    logger.warning(f"[SMART] 无法获取交易日历，使用今天: {today_str}")
                    return today_str
                    
            except Exception as e:
                logger.warning(f"[SMART] 获取交易日历失败: {e}，使用今天: {today_str}")
                return today_str
        else:
            return end_date_config
    
    def get_stock_list_from_cache(self) -> List[str]:
        """从缓存获取股票列表（优化版）"""
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            cache_dir = os.path.dirname(db_path)
            cache_file = os.path.join(cache_dir, "stock_list.csv")
            
            if os.path.exists(cache_file):
                import pandas as pd
                df = pd.read_csv(cache_file, dtype=str)
                if not df.empty and "ts_code" in df.columns:
                    codes = df["ts_code"].tolist()
                    logger.debug(f"从缓存获取股票列表: {len(codes)} 只股票")
                    return codes
            
            logger.warning("股票列表缓存不存在")
            return []
            
        except Exception as e:
            logger.error(f"从缓存获取股票列表失败: {e}")
            return []
    
    def _get_trade_dates_from_parquet(self) -> List[str]:
        """从Parquet文件获取交易日列表"""
        try:
            from config import DATA_ROOT
            root = DATA_ROOT
            
            if not os.path.isdir(root):
                return []
            
            dates = []
            for name in os.listdir(root):
                if name.startswith("trade_date="):
                    try:
                        dates.append(name.split("=")[1])
                    except Exception:
                        continue
            return sorted(dates)
            
        except Exception as e:
            logger.error(f"从Parquet文件获取交易日列表失败: {e}")
            return []
    
    def get_stock_list(self, db_path: str = None, adj_type: str = "raw") -> List[str]:
        """获取股票代码列表"""
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            if not os.path.exists(db_path):
                logger.warning(f"数据库文件不存在: {db_path}")
                return []
            
            # 先检查数据库是否有数据
            count_sql = "SELECT COUNT(*) as count FROM stock_data WHERE adj_type = ?"
            count_df = self.execute_sync_query(db_path, count_sql, [adj_type], timeout=30.0)
            
            if count_df.empty or count_df["count"].iloc[0] == 0:
                logger.debug(f"数据库中没有 {adj_type} 类型的数据")
                return []
            
            sql = "SELECT DISTINCT ts_code FROM stock_data WHERE adj_type = ? ORDER BY ts_code"
            df = self.execute_sync_query(db_path, sql, [adj_type], timeout=120.0)
            
            if not df.empty and "ts_code" in df.columns:
                return df["ts_code"].tolist()
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取股票代码列表失败: {e}")
            return []
    
    def register_preload_cache(self, cache_name: str, data: pd.DataFrame, ref_date: str, 
                               start_date: str, columns: List[str]) -> None:
        """
        注册预加载数据缓存
        
        Args:
            cache_name: 缓存名称（如 "rank" 或 "screen"）
            data: 预加载的全市场数据 DataFrame
            ref_date: 参考日期（结束日期）
            start_date: 起始日期
            columns: 包含的列名列表
        """
        with self._preload_cache_lock:
            self._preload_caches[cache_name] = {
                "data": data.copy() if data is not None else None,
                "ref_date": ref_date,
                "start_date": start_date,
                "columns": columns.copy() if columns else []
            }
            logger.debug(f"注册预加载缓存: {cache_name}, ref_date={ref_date}, start_date={start_date}, columns={len(columns)}")
    
    def clear_preload_cache(self, cache_name: str = None) -> None:
        """
        清除预加载数据缓存
        
        Args:
            cache_name: 缓存名称，如果为None则清除所有缓存
        """
        with self._preload_cache_lock:
            if cache_name:
                if cache_name in self._preload_caches:
                    del self._preload_caches[cache_name]
                    logger.debug(f"清除预加载缓存: {cache_name}")
            else:
                self._preload_caches.clear()
                logger.debug("清除所有预加载缓存")
    
    def _get_data_from_preload_cache(self, ts_code: str, start_date: str, end_date: str, 
                                     columns: List[str], require_exact_start: bool = True) -> Optional[pd.DataFrame]:
        """
        从预加载缓存中获取股票数据
        
        Args:
            ts_code: 股票代码
            start_date: 起始日期
            end_date: 结束日期
            columns: 需要的列
            require_exact_start: 是否要求起始日期完全匹配
        
        Returns:
            如果命中缓存则返回DataFrame，否则返回None
        """
        with self._preload_cache_lock:
            # 按优先级检查所有缓存（先检查排名缓存，再检查筛选缓存）
            cache_priority = ["rank", "screen"]  # 排名缓存优先级更高
            
            for cache_name in cache_priority:
                if cache_name not in self._preload_caches:
                    continue
                
                cache_dict = self._preload_caches[cache_name]
                if cache_dict["data"] is None:
                    continue
                
                # 检查参考日期和列
                if cache_dict["ref_date"] != end_date:
                    continue
                
                if not all(col in cache_dict["columns"] for col in columns):
                    continue
                
                # 检查起始日期（根据类型决定是否严格匹配）
                if require_exact_start:
                    if cache_dict["start_date"] != start_date:
                        continue
                else:
                    # 筛选缓存可能包含更早的数据，只需要确保覆盖所需范围
                    if cache_dict["start_date"] > start_date:
                        continue
                
                try:
                    # 从预加载数据中筛选指定股票
                    stock_df = cache_dict["data"][cache_dict["data"]["ts_code"] == ts_code].copy()
                    if stock_df.empty:
                        continue
                    
                    # 如果不是严格匹配起始日期，需要筛选时间范围
                    if not require_exact_start:
                        mask = (stock_df["trade_date"] >= str(start_date)) & (stock_df["trade_date"] <= str(end_date))
                        stock_df = stock_df.loc[mask].copy()
                        if stock_df.empty:
                            continue
                    
                    # 确保包含所需的列（去重基础列，避免 trade_date 重复）
                    base_cols = [c for c in ["ts_code", "trade_date"] if c in stock_df.columns]
                    available_cols = [
                        col for col in columns
                        if col in stock_df.columns and col not in base_cols
                    ]
                    col_order: list[str] = []
                    for c in base_cols + available_cols:
                        if c not in col_order:
                            col_order.append(c)
                    if not col_order:
                        continue
                    
                    result_df = stock_df[col_order]
                    logger.debug(f"[{ts_code}] 使用{cache_name}预加载数据: {len(result_df)} 条记录")
                    return result_df
                    
                except Exception as e:
                    logger.warning(f"[{ts_code}] {cache_name}预加载数据筛选失败: {e}")
                    continue
        
        return None

    def _can_use_full_cache(self, db_path: str) -> bool:
        """
        判断是否启用全表内存缓存：
        - 优先 FULL_STOCK_CACHE_MAX_MB 设定绝对阈值（MB）
        - 否则按内存占比阈值判断：默认最多占用物理内存的 20%（可用 FULL_STOCK_CACHE_RATIO 调整）
        - 配置 FULL_STOCK_CACHE_ENABLED=False 或 FULL_STOCK_CACHE_DISABLE=True 时禁用
        """
        try:
            from config import (
                FULL_STOCK_CACHE_ENABLED,
                FULL_STOCK_CACHE_MAX_MB,
                FULL_STOCK_CACHE_RATIO,
                FULL_STOCK_CACHE_DISABLE,
            )
        except Exception:
            FULL_STOCK_CACHE_ENABLED = True
            FULL_STOCK_CACHE_MAX_MB = None
            FULL_STOCK_CACHE_RATIO = 0.2
            FULL_STOCK_CACHE_DISABLE = False
        
        if not FULL_STOCK_CACHE_ENABLED or FULL_STOCK_CACHE_DISABLE:
            logger.debug("[full-cache] FULL_STOCK_CACHE_ENABLED=False，禁用全表缓存")
            return False
        
        # 1) 显式 MB 阈值优先
        try:
            max_mb = float(FULL_STOCK_CACHE_MAX_MB) if FULL_STOCK_CACHE_MAX_MB is not None else None
        except Exception:
            max_mb = None
        # 2) 按内存占比计算阈值
        if max_mb is None or max_mb <= 0:
            try:
                ratio = float(FULL_STOCK_CACHE_RATIO)
                ratio = min(max(ratio, 0.01), 0.8)  # clamp 1%~80%
            except Exception:
                ratio = 0.2
            total_mem = None
            try:
                import psutil  # type: ignore
                total_mem = psutil.virtual_memory().total
            except Exception:
                try:
                    total_mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
                except Exception:
                    total_mem = None
            if total_mem:
                max_mb = (total_mem / (1024 * 1024)) * ratio
            else:
                max_mb = 2048.0  # 回退
        try:
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            return size_mb <= max_mb
        except Exception:
            return False

    def _get_full_table_cache(self, db_path: str, adj_type: str) -> Optional[pd.DataFrame]:
        """
        返回指定 adj_type 的全表内存缓存；若不存在且符合条件则加载。
        仅在单进程场景可靠，多进程会各自维护一份内存副本。
        """
        key = (db_path, adj_type or "")
        with self._full_cache_lock:
            if key in self._full_table_cache:
                return self._full_table_cache[key]
        
        if not self._can_use_full_cache(db_path):
            return None
        
        try:
            sql = "SELECT * FROM stock_data"
            params: List[Any] = []
            if adj_type:
                sql += " WHERE adj_type = ?"
                params.append(adj_type)
            sql += " ORDER BY ts_code, trade_date"
            df = self.execute_sync_query(db_path, sql, params, timeout=300.0)
            if df.empty:
                return None
            if "trade_date" in df.columns and df["trade_date"].dtype != object:
                df["trade_date"] = df["trade_date"].astype(str)
            with self._full_cache_lock:
                self._full_table_cache[key] = df
            logger.info(f"[full-cache] 已加载全表缓存: {db_path}, adj={adj_type}, 行数={len(df)}")
            return df
        except Exception as e:
            logger.warning(f"[full-cache] 加载失败: {db_path}, adj={adj_type}, err={e}")
            return None
    
    def query_stock_data(self, db_path: str = None, ts_code: str = None, start_date: str = None, 
                        end_date: str = None, columns: List[str] = None, adj_type: str = "qfq", limit: Optional[int] = None,
                        order: str = "asc") -> pd.DataFrame:
        """
        查询股票数据 - 统一接口，优先使用预加载缓存
        
        Args:
            db_path: 数据库路径
            ts_code: 股票代码（单只股票查询时，会优先检查预加载缓存）
            start_date: 起始日期
            end_date: 结束日期
            columns: 需要的列
            adj_type: 复权类型
            limit: 限制返回行数
            order: 排序方向，asc/desc（默认升序；desc 常用于取最新 N 行）
        
        Returns:
            股票数据 DataFrame
        """
        # 如果是单只股票查询，优先检查预加载缓存
        if ts_code and start_date and end_date and columns and adj_type == "qfq":
            # 先尝试从排名缓存获取（要求起始日期严格匹配）
            df = self._get_data_from_preload_cache(
                ts_code, start_date, end_date, columns, require_exact_start=True
            )
            if df is not None:
                return df
            
            # 再尝试从筛选缓存获取（起始日期可以更早）
            df = self._get_data_from_preload_cache(
                ts_code, start_date, end_date, columns, require_exact_start=False
            )
            if df is not None:
                return df
        
        # 缓存未命中，从数据库查询
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            if not os.path.exists(db_path):
                logger.warning(f"数据库文件不存在: {db_path}")
                return pd.DataFrame()

            # 生成缓存键
            cache_key = self._get_cache_key([ts_code] if ts_code else None, start_date, end_date, columns, adj_type)
            
            # 有 limit 的场景不加载全表缓存，避免增量预加载时整表进内存拖慢启动
            use_full_cache = limit is None
            if use_full_cache:
                full_df = self._get_full_table_cache(db_path, adj_type)
                if full_df is not None:
                    df_mem = full_df
                    # 条件过滤
                    mask = pd.Series(True, index=df_mem.index)
                    if ts_code:
                        mask &= df_mem["ts_code"] == ts_code
                    if start_date:
                        mask &= df_mem["trade_date"] >= str(start_date)
                    if end_date:
                        mask &= df_mem["trade_date"] <= str(end_date)
                    if adj_type:
                        mask &= df_mem["adj_type"] == adj_type
                    df_mem = df_mem.loc[mask]

                    if columns:
                        keep_cols = [c for c in columns if c in df_mem.columns]
                        base_cols = [c for c in ["ts_code", "trade_date"] if c in df_mem.columns]
                        col_order: List[str] = []
                        for c in base_cols + keep_cols:
                            if c not in col_order:
                                col_order.append(c)
                        if col_order:
                            df_mem = df_mem[col_order]

                    # 排序与 limit
                    order = (order or "asc").lower()
                    asc = False if order == "desc" else True
                    sort_cols = ["trade_date"]
                    if not ts_code:
                        sort_cols = ["ts_code", "trade_date"]
                    df_mem = df_mem.sort_values(sort_cols, ascending=asc)
                    if limit is not None:
                        try:
                            limit_int = int(limit)
                            if limit_int > 0:
                                df_mem = df_mem.head(limit_int)
                        except Exception:
                            pass

                    return df_mem.reset_index(drop=True).copy()
            
            # 构建查询条件
            conditions = []
            params = []
            
            if ts_code:
                conditions.append("ts_code = ?")
                params.append(ts_code)
            
            if start_date:
                conditions.append("trade_date >= ?")
                params.append(start_date)
                
            if end_date:
                conditions.append("trade_date <= ?")
                params.append(end_date)
                
            if adj_type:
                conditions.append("adj_type = ?")
                params.append(adj_type)
            
            # 构建SQL查询
            select_cols = "*" if not columns else ", ".join(columns)
            sql = f"SELECT {select_cols} FROM stock_data"
            
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            
            order = (order or "asc").lower()
            order = "desc" if order == "desc" else "asc"
            if ts_code:
                sql += f" ORDER BY trade_date {order}"
            else:
                sql += f" ORDER BY ts_code, trade_date {order}"

            # 限制返回行数（如提供）
            if limit is not None:
                try:
                    limit_int = int(limit)
                    if limit_int > 0:
                        sql += f" LIMIT {limit_int}"
                except Exception:
                    pass
            
            # 执行查询
            df = self.execute_sync_query(db_path, sql, params, timeout=120.0)
            
            # 优化类型转换：只在需要时转换
            if not df.empty and "trade_date" in df.columns:
                if df["trade_date"].dtype != 'object':
                    df["trade_date"] = df["trade_date"].astype(str)
                elif len(df) > 0 and not isinstance(df["trade_date"].iloc[0], str):
                    df["trade_date"] = df["trade_date"].astype(str)
            
            return df
            
        except Exception as e:
            logger.error(f"查询股票数据失败: {e}")
            return pd.DataFrame()

    def count_stock_data(self, db_path: str = None, ts_code: str = None, start_date: str = None,
                         end_date: str = None, adj_type: str = "qfq") -> int:
        """统计股票数据总行数（不受 LIMIT 影响）"""
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)

            if not os.path.exists(db_path):
                logger.warning(f"数据库文件不存在: {db_path}")
                return 0

            conditions: List[str] = []
            params: List[Any] = []

            if ts_code:
                conditions.append("ts_code = ?")
                params.append(ts_code)

            if start_date:
                conditions.append("trade_date >= ?")
                params.append(start_date)

            if end_date:
                conditions.append("trade_date <= ?")
                params.append(end_date)

            if adj_type:
                conditions.append("adj_type = ?")
                params.append(adj_type)

            sql = "SELECT COUNT(*) AS total FROM stock_data"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

            df = self.execute_sync_query(db_path, sql, params, timeout=60.0)
            if not df.empty and "total" in df.columns:
                try:
                    return int(df["total"].iloc[0])
                except Exception:
                    return 0
            return 0
        except Exception as e:
            logger.error(f"统计股票数据总行数失败: {e}")
            return 0
    
    def batch_query_stock_data(self, db_path: str = None, ts_codes: List[str] = None, 
                              start_date: str = None, end_date: str = None, 
                              columns: List[str] = None, adj_type: str = "qfq") -> pd.DataFrame:
        """批量查询多只股票数据 - 高效接口（带缓存，使用临时表+JOIN优化）"""
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            if not os.path.exists(db_path):
                logger.warning(f"数据库文件不存在: {db_path}")
                return pd.DataFrame()

            # 生成缓存键
            cache_key = self._get_cache_key(ts_codes, start_date, end_date, columns, adj_type)
            
            # 尝试使用全表内存缓存（小体量数据库）
            full_df = self._get_full_table_cache(db_path, adj_type)
            if full_df is not None:
                df_mem = full_df
                mask = pd.Series(True, index=df_mem.index)
                if ts_codes:
                    mask &= df_mem["ts_code"].isin(ts_codes)
                if start_date:
                    mask &= df_mem["trade_date"] >= str(start_date)
                if end_date:
                    mask &= df_mem["trade_date"] <= str(end_date)
                if adj_type:
                    mask &= df_mem["adj_type"] == adj_type
                df_mem = df_mem.loc[mask]

                if columns:
                    keep_cols = [c for c in columns if c in df_mem.columns]
                    base_cols = [c for c in ["ts_code", "trade_date"] if c in df_mem.columns]
                    col_order: List[str] = []
                    for c in base_cols + keep_cols:
                        if c not in col_order:
                            col_order.append(c)
                    if col_order:
                        df_mem = df_mem[col_order]

                df_mem = df_mem.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

                # 设置查询缓存，避免后续重复过滤
                if not df_mem.empty:
                    self._set_cached_data(cache_key, df_mem)
                return df_mem
            
            # 尝试从缓存获取
            cached_df = self._get_cached_data(cache_key)
            if cached_df is not None:
                return cached_df
            
            # 如果没有指定股票代码，直接查询全市场
            if not ts_codes or len(ts_codes) == 0:
                return self._batch_query_all_stocks(db_path, start_date, end_date, columns, adj_type)
            
            # 使用临时表+JOIN优化大批量查询
            if len(ts_codes) > 100:  # 超过100只股票时使用临时表优化
                return self._batch_query_with_temp_table(db_path, ts_codes, start_date, end_date, columns, adj_type, cache_key)
            else:
                # 小批量查询仍使用IN语句
                return self._batch_query_with_in_clause(db_path, ts_codes, start_date, end_date, columns, adj_type, cache_key)
            
        except Exception as e:
            logger.error(f"批量查询股票数据失败: {e}")
            return pd.DataFrame()
    
    def _batch_query_all_stocks(self, db_path: str, start_date: str, end_date: str, 
                               columns: List[str], adj_type: str) -> pd.DataFrame:
        """查询全市场股票数据"""
        conditions = []
        params = []
        
        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
            
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)
            
        if adj_type:
            conditions.append("adj_type = ?")
            params.append(adj_type)
        
        # 构建SQL查询
        select_cols = "*" if not columns else ", ".join(columns)
        sql = f"SELECT {select_cols} FROM stock_data"
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY ts_code, trade_date"
        
        # 执行查询
        df = self.execute_sync_query(db_path, sql, params, timeout=120.0)
        logger.debug(f"全市场查询完成: {len(df)} 条记录")
        return df
    
    def _batch_query_with_in_clause(self, db_path: str, ts_codes: List[str], start_date: str, 
                                   end_date: str, columns: List[str], adj_type: str, cache_key: str) -> pd.DataFrame:
        """使用IN语句进行小批量查询"""
        conditions = []
        params = []
        
        # 使用 IN 子句进行批量查询
        placeholders = ",".join(["?" for _ in ts_codes])
        conditions.append(f"ts_code IN ({placeholders})")
        params.extend(ts_codes)
        
        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
            
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)
            
        if adj_type:
            conditions.append("adj_type = ?")
            params.append(adj_type)
        
        # 构建SQL查询
        select_cols = "*" if not columns else ", ".join(columns)
        sql = f"SELECT {select_cols} FROM stock_data"
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY ts_code, trade_date"
        
        # 执行查询
        df = self.execute_sync_query(db_path, sql, params, timeout=120.0)
        
        # 缓存结果
        if not df.empty:
            self._set_cached_data(cache_key, df)
        
        logger.debug(f"IN语句批量查询完成: {len(ts_codes)} 只股票, {len(df)} 条记录")
        return df
    
    def _batch_query_with_temp_table(self, db_path: str, ts_codes: List[str], start_date: str, 
                                    end_date: str, columns: List[str], adj_type: str, cache_key: str) -> pd.DataFrame:
        """使用临时表+JOIN进行大批量查询优化"""
        import uuid
        
        # 生成唯一的临时表名
        temp_table_name = f"_codes_{uuid.uuid4().hex[:8]}"
        
        try:
            # 分批插入股票代码到临时表，避免单次插入过多数据
            batch_size = 1000
            pool = self._get_connection_pool(db_path, read_only=True)
            conn = pool.get_connection(read_only=True, timeout=120.0)
            
            try:
                # 创建临时表
                conn.execute(f"CREATE TEMP TABLE {temp_table_name}(ts_code VARCHAR)")
                
                # 分批插入股票代码
                for i in range(0, len(ts_codes), batch_size):
                    batch_codes = ts_codes[i:i + batch_size]
                    values = ",".join([f"('{code}')" for code in batch_codes])
                    conn.execute(f"INSERT INTO {temp_table_name} VALUES {values}")
                
                # 构建查询条件
                conditions = []
                params = []
                
                if start_date:
                    conditions.append("s.trade_date >= ?")
                    params.append(start_date)
                    
                if end_date:
                    conditions.append("s.trade_date <= ?")
                    params.append(end_date)
                    
                if adj_type:
                    conditions.append("s.adj_type = ?")
                    params.append(adj_type)
                
                # 构建SQL查询 - 使用JOIN替代IN
                select_cols = "*" if not columns else ", ".join([f"s.{col}" for col in columns])
                sql = f"""
                SELECT {select_cols}
                FROM stock_data s
                JOIN {temp_table_name} c USING(ts_code)
                """
                
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY s.ts_code, s.trade_date"
                
                # 执行查询
                df = conn.execute(sql, params).df()
                
                # 防御：去重重复列，避免后续 df["trade_date"] 变宽表
                try:
                    dup_mask = df.columns.duplicated()
                    if dup_mask.any():
                        logger.warning(f"JOIN查询返回存在重复列: {list(df.columns[dup_mask])} -> 保留首列")
                        df = df.loc[:, ~dup_mask].copy()
                except Exception:
                    pass
                # 缓存结果
                if not df.empty:
                    self._set_cached_data(cache_key, df)
                
                logger.debug(f"临时表JOIN批量查询完成: {len(ts_codes)} 只股票, {len(df)} 条记录")
                return df
                
            finally:
                # 清理临时表
                try:
                    conn.execute(f"DROP TABLE {temp_table_name}")
                except:
                    pass
                pool.return_connection(conn)
                
        except Exception as e:
            logger.error(f"临时表JOIN查询失败，回退到IN语句: {e}")
            # 回退到IN语句查询
            return self._batch_query_with_in_clause(db_path, ts_codes, start_date, end_date, columns, adj_type, cache_key)
    
    def _build_adj_filter(self, adj: Optional[str], asset: str, ts_code: Optional[str]) -> tuple[str, list]:
        """构建复权类型过滤条件"""
        try:
            ts_tail = (ts_code or "").upper()
        except Exception:
            ts_tail = ""
        
        if asset == "index" or (ts_tail.endswith(".SH") and ts_tail.startswith(("000", "88"))):
            return "adj_type = ?", ["ind"]
        if adj in ("raw", "qfq", "hfq", "ind"):
            return "adj_type = ?", [adj]
        return "adj_type IN (?, ?)", ["qfq", "raw"]
    
    def is_using_unified_db(self, db_path: str = None) -> bool:
        """检查是否使用统一数据库"""
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            # 检查数据库文件是否存在
            if not os.path.exists(db_path):
                return False
            
            # 检查数据库是否有数据
            try:
                count_sql = "SELECT COUNT(*) as count FROM stock_data"
                count_df = self.execute_sync_query(db_path, count_sql, [], timeout=10.0)
                
                if count_df.empty or count_df["count"].iloc[0] == 0:
                    return False  # 数据库存在但没有数据
                
                return True  # 数据库存在且有数据
            except Exception:
                return False  # 查询失败，认为不是有效数据库
                
        except Exception:
            return False
    
    def get_database_info(self, db_path: str = None, use_cache: bool = True) -> Dict[str, Any]:
        """获取数据库信息（优先使用状态文件缓存，优化版 - 只查询最新日期，不统计股票数量）"""
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            info = {
                "db_path": db_path,
                "exists": os.path.exists(db_path),
                "size": 0,
                "tables": [],
                "stock_count": 0,
                "date_range": {"start": None, "end": None}
            }
            
            if not info["exists"]:
                return info
            
            # 优先从状态文件读取
            if use_cache:
                try:
                    saved_status = self.load_status_file()
                    
                    if saved_status and "stock_data" in saved_status:
                        stock_data_status = saved_status["stock_data"]
                        # 验证数据库路径是否匹配
                        if stock_data_status.get("database_path") == db_path:
                            # 从状态文件获取信息
                            info["exists"] = stock_data_status.get("database_exists", False)
                            info["stock_count"] = stock_data_status.get("total_stocks", 0)
                            
                            # 获取日期范围（从复权类型中获取最大日期）
                            adj_types = stock_data_status.get("adj_types", {})
                            max_date = None
                            for adj_type, adj_status in adj_types.items():
                                adj_max_date = adj_status.get("max_date")
                                if adj_max_date:
                                    if max_date is None or adj_max_date > max_date:
                                        max_date = adj_max_date
                            
                            if max_date:
                                info["date_range"]["end"] = max_date
                            
                            # 获取文件大小
                            try:
                                info["size"] = os.path.getsize(db_path)
                            except Exception:
                                pass
                            
                            # 从状态文件推断表列表
                            if stock_data_status.get("database_exists"):
                                info["tables"] = ["stock_data"]  # 通常都有这个表
                            
                            logger.debug("从状态文件缓存获取数据库信息")
                            return info
                except Exception as cache_error:
                    logger.debug(f"从状态文件读取失败，回退到数据库查询: {cache_error}")
            
            # 获取文件大小
            try:
                info["size"] = os.path.getsize(db_path)
            except Exception:
                pass
            
            # 获取表信息（优化版 - 只查询最新日期）
            try:
                with self.get_connection(db_path, read_only=True) as conn:
                    # 获取表列表
                    tables_df = conn.execute("SHOW TABLES").df()
                    if not tables_df.empty:
                        info["tables"] = tables_df["name"].tolist()
                    
                    # 只查询最新日期，避免全表扫描
                    if "stock_data" in info["tables"]:
                        latest_date_df = conn.execute("SELECT MAX(trade_date) as max_date FROM stock_data LIMIT 1").df()
                        if not latest_date_df.empty and latest_date_df["max_date"].iloc[0] is not None:
                            info["date_range"]["end"] = str(latest_date_df["max_date"].iloc[0])
                            
            except Exception as e:
                logger.warning(f"获取数据库详细信息失败: {e}")
            
            return info
            
        except Exception as e:
            logger.error(f"获取数据库信息失败: {e}")
            return {"error": str(e)}
    
    def get_data_source_status(self, db_path: str = None) -> Dict[str, Any]:
        """获取数据源状态"""
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            status = {
                "use_unified_db": self.is_using_unified_db(db_path),
                "database_info": self.get_database_info(db_path),
                "manager_stats": self.get_stats()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"获取数据源状态失败: {e}")
            return {"error": str(e)}
    
    # ========== 状态管理方法（合并自 DatabaseStatusManager） ==========
    
    def get_status_file_path(self) -> Optional[str]:
        """获取状态文件路径"""
        return self._status_file_path
    
    def set_status_file_path(self, status_file_path: Optional[str] = None):
        """设置状态文件路径"""
        if status_file_path is None:
            try:
                from config import DATA_ROOT
                status_file_path = os.path.join(DATA_ROOT, "database_status.json")
            except ImportError:
                status_file_path = None
        self._status_file_path = status_file_path
    
    def get_stock_data_status(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        获取股票数据状态（优先使用状态文件缓存）
        
        Args:
            use_cache: 是否优先使用状态文件缓存，默认为True
            
        Returns:
            包含股票数据状态的字典
        """
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH, API_ADJ
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            # 优先从状态文件读取
            if use_cache and self._status_file_path:
                saved_status = self.load_status_file()
                if saved_status and "stock_data" in saved_status:
                    cached_status = saved_status["stock_data"]
                    # 验证数据库路径是否匹配
                    if cached_status.get("database_path") == db_path:
                        # 检查数据库文件是否存在（如果状态文件显示不存在，但文件实际存在，需要重新查询）
                        db_exists = os.path.exists(db_path)
                        if cached_status.get("database_exists") == db_exists:
                            logger.debug("从状态文件缓存获取股票数据状态")
                            return cached_status
            
            # 状态文件不存在或需要刷新，从数据库读取
            logger.debug("从数据库读取股票数据状态")
            status = {
                "database_path": db_path,
                "database_exists": os.path.exists(db_path),
                "adj_types": {},
                "last_update": None
            }
            
            if not status["database_exists"]:
                logger.warning(f"股票数据数据库不存在: {db_path}")
                return status
            
            # 获取所有复权类型
            sql = "SELECT DISTINCT adj_type FROM stock_data ORDER BY adj_type"
            df = self.execute_sync_query(db_path, sql, [], timeout=30.0)
            
            if df.empty:
                logger.warning("股票数据表中没有数据")
                return status
            
            adj_types = df["adj_type"].tolist()
            
            # 为每个复权类型获取详细信息
            for adj_type in adj_types:
                adj_status = self._get_adj_type_status(db_path, adj_type)
                status["adj_types"][adj_type] = adj_status
            
            # 获取整体统计信息
            total_sql = "SELECT COUNT(*) as total FROM stock_data"
            total_df = self.execute_sync_query(db_path, total_sql, [], timeout=30.0)
            if not total_df.empty:
                status["total_records"] = int(total_df["total"].iloc[0])
            
            # 获取所有股票代码
            all_stocks_sql = "SELECT DISTINCT ts_code FROM stock_data"
            all_stocks_df = self.execute_sync_query(db_path, all_stocks_sql, [], timeout=60.0)
            if not all_stocks_df.empty:
                status["total_stocks"] = len(all_stocks_df)
            
            status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return status
            
        except Exception as e:
            logger.error(f"获取股票数据状态失败: {e}")
            return {
                "database_path": None,
                "database_exists": False,
                "error": str(e),
                "adj_types": {},
                "last_update": None
            }
    
    def _get_adj_type_status(self, db_path: str, adj_type: str) -> Dict[str, Any]:
        """
        获取特定复权类型的状态
        
        Args:
            db_path: 数据库路径
            adj_type: 复权类型
            
        Returns:
            包含该复权类型状态的字典
        """
        try:
            # 获取日期范围
            date_range_sql = """
                SELECT 
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date,
                    COUNT(*) as total_records,
                    COUNT(DISTINCT ts_code) as stock_count
                FROM stock_data 
                WHERE adj_type = ?
            """
            df = self.execute_sync_query(db_path, date_range_sql, [adj_type], timeout=30.0)
            
            if df.empty:
                return {
                    "min_date": None,
                    "max_date": None,
                    "total_records": 0,
                    "stock_count": 0
                }
            
            row = df.iloc[0]
            return {
                "min_date": str(row["min_date"]) if row["min_date"] else None,
                "max_date": str(row["max_date"]) if row["max_date"] else None,
                "total_records": int(row["total_records"]),
                "stock_count": int(row["stock_count"])
            }
            
        except Exception as e:
            logger.error(f"获取复权类型 {adj_type} 状态失败: {e}")
            return {
                "min_date": None,
                "max_date": None,
                "total_records": 0,
                "stock_count": 0,
                "error": str(e)
            }
    
    def get_details_data_status(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        获取细节数据状态（优先使用状态文件缓存）
        
        Args:
            use_cache: 是否优先使用状态文件缓存，默认为True
            
        Returns:
            包含细节数据状态的字典
        """
        try:
            from config import SC_OUTPUT_DIR, SC_DETAIL_DB_TYPE, SC_DETAIL_DB_PATH
            
            if SC_DETAIL_DB_TYPE == "duckdb":
                db_path = os.path.join(SC_OUTPUT_DIR, 'details', 'details.db')
            else:
                db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
            
            db_path = os.path.abspath(db_path)
            
            # 优先从状态文件读取
            if use_cache and self._status_file_path:
                saved_status = self.load_status_file()
                if saved_status and "details_data" in saved_status:
                    cached_status = saved_status["details_data"]
                    # 验证数据库路径是否匹配
                    if cached_status.get("database_path") == db_path:
                        # 检查数据库文件是否存在（如果状态文件显示不存在，但文件实际存在，需要重新查询）
                        db_exists = os.path.exists(db_path)
                        if cached_status.get("database_exists") == db_exists:
                            logger.debug("从状态文件缓存获取细节数据状态")
                            return cached_status
            
            # 状态文件不存在或需要刷新，从数据库读取
            logger.debug("从数据库读取细节数据状态")
            status = {
                "database_path": db_path,
                "database_exists": os.path.exists(db_path),
                "min_date": None,
                "max_date": None,
                "total_records": 0,
                "stock_count": 0,
                "last_update": None
            }
            
            if not status["database_exists"]:
                logger.warning(f"细节数据数据库不存在: {db_path}")
                return status
            
            # 获取日期范围和统计信息
            try:
                stats_sql = """
                    SELECT 
                        MIN(ref_date) as min_date,
                        MAX(ref_date) as max_date,
                        COUNT(*) as total_records,
                        COUNT(DISTINCT ts_code) as stock_count
                    FROM stock_details
                """
                df = self.execute_sync_query(db_path, stats_sql, [], timeout=30.0)
                
                if not df.empty:
                    row = df.iloc[0]
                    status["min_date"] = str(row["min_date"]) if row["min_date"] else None
                    status["max_date"] = str(row["max_date"]) if row["max_date"] else None
                    status["total_records"] = int(row["total_records"])
                    status["stock_count"] = int(row["stock_count"])
            except Exception as e:
                _handle_details_db_error(e, db_path, "获取细节数据统计信息")
            
            status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return status
            
        except Exception as e:
            _handle_details_db_error(e, db_path if 'db_path' in locals() else None, "获取细节数据状态")
            return {
                "database_path": db_path if 'db_path' in locals() else None,
                "database_exists": False,
                "error": str(e),
                "min_date": None,
                "max_date": None,
                "total_records": 0,
                "stock_count": 0,
                "last_update": None
            }
    
    def generate_status_file(self) -> Dict[str, Any]:
        """
        生成状态文件（从数据库读取最新状态）
        
        Returns:
            包含完整状态的字典
        """
        if not self._status_file_path:
            raise ValueError("状态文件路径未设置")
        
        logger.info("开始生成数据库状态文件...")
        
        # 生成状态文件时，不使用缓存，从数据库读取最新状态
        status = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stock_data": self.get_stock_data_status(use_cache=False),
            "details_data": self.get_details_data_status(use_cache=False)
        }
        
        # 确保目录存在
        status_dir = os.path.dirname(self._status_file_path)
        if status_dir:
            os.makedirs(status_dir, exist_ok=True)
        
        # 写入文件
        try:
            with open(self._status_file_path, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据库状态文件已生成: {self._status_file_path}")
            return status
            
        except Exception as e:
            logger.error(f"写入状态文件失败: {e}")
            raise
    
    def load_status_file(self) -> Optional[Dict[str, Any]]:
        """
        加载状态文件
        
        Returns:
            状态字典，如果文件不存在则返回None
        """
        if not self._status_file_path or not os.path.exists(self._status_file_path):
            logger.warning(f"状态文件不存在: {self._status_file_path}")
            return None
        
        try:
            with open(self._status_file_path, 'r', encoding='utf-8') as f:
                status = json.load(f)
            
            logger.info(f"状态文件已加载: {self._status_file_path}")
            return status
            
        except Exception as e:
            logger.error(f"读取状态文件失败: {e}")
            return None
    
    def check_status(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        检查当前数据库状态与状态文件的差异（优先使用状态文件）
        
        Args:
            use_cache: 是否优先使用状态文件缓存，默认为True
            
        Returns:
            包含检查结果的字典
        """
        logger.info("开始检查数据库状态...")
        
        # 加载已保存的状态
        saved_status = self.load_status_file()
        
        if saved_status is None:
            # 状态文件不存在，需要从数据库读取当前状态
            logger.info("状态文件不存在，从数据库读取当前状态")
            current_status = {
                "stock_data": self.get_stock_data_status(use_cache=False),
                "details_data": self.get_details_data_status(use_cache=False)
            }
            return {
                "status_file_exists": False,
                "current_status": current_status,
                "differences": {},
                "recommendation": "生成新的状态文件"
            }
        
        # 如果使用缓存，先尝试从状态文件获取当前状态（避免不必要的数据库查询）
        if use_cache:
            # 先使用缓存的状态进行比较
            cached_current_status = {
                "stock_data": self.get_stock_data_status(use_cache=True),
                "details_data": self.get_details_data_status(use_cache=True)
            }
            
            # 比较差异
            differences = self._compare_status(saved_status, cached_current_status)
            
            # 如果没有差异，直接返回缓存的结果
            if not differences:
                logger.info("数据库状态与状态文件一致（使用缓存）")
                return {
                    "status_file_exists": True,
                    "status_file_generated_at": saved_status.get("generated_at"),
                    "current_status": cached_current_status,
                    "saved_status": {
                        "stock_data": saved_status.get("stock_data", {}),
                        "details_data": saved_status.get("details_data", {})
                    },
                    "differences": differences,
                    "has_changes": False,
                    "recommendation": "状态文件是最新的"
                }
            
            # 有差异，需要从数据库重新读取以确认
            logger.info(f"发现潜在差异，从数据库重新读取以确认")
        
        # 从数据库读取当前状态（用于比较或确认差异）
        current_status = {
            "stock_data": self.get_stock_data_status(use_cache=False),
            "details_data": self.get_details_data_status(use_cache=False)
        }
        
        # 比较差异
        differences = self._compare_status(saved_status, current_status)
        
        result = {
            "status_file_exists": True,
            "status_file_generated_at": saved_status.get("generated_at"),
            "current_status": current_status,
            "saved_status": {
                "stock_data": saved_status.get("stock_data", {}),
                "details_data": saved_status.get("details_data", {})
            },
            "differences": differences,
            "has_changes": len(differences) > 0
        }
        
        if differences:
            logger.warning(f"发现 {len(differences)} 处差异")
            result["recommendation"] = "建议更新状态文件"
        else:
            logger.info("数据库状态与状态文件一致")
            result["recommendation"] = "状态文件是最新的"
        
        return result
    
    def _compare_status(self, saved: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较已保存的状态和当前状态
        
        Args:
            saved: 已保存的状态
            current: 当前状态
            
        Returns:
            包含差异的字典
        """
        differences = {}
        
        # 比较股票数据
        saved_stock = saved.get("stock_data", {})
        current_stock = current.get("stock_data", {})
        
        stock_diff = {}
        
        # 检查数据库是否存在
        if saved_stock.get("database_exists") != current_stock.get("database_exists"):
            stock_diff["database_exists"] = {
                "saved": saved_stock.get("database_exists"),
                "current": current_stock.get("database_exists")
            }
        
        # 比较总记录数
        if saved_stock.get("total_records") != current_stock.get("total_records"):
            stock_diff["total_records"] = {
                "saved": saved_stock.get("total_records"),
                "current": current_stock.get("total_records")
            }
        
        # 比较总股票数
        if saved_stock.get("total_stocks") != current_stock.get("total_stocks"):
            stock_diff["total_stocks"] = {
                "saved": saved_stock.get("total_stocks"),
                "current": current_stock.get("total_stocks")
            }
        
        # 比较每个复权类型
        saved_adj_types = saved_stock.get("adj_types", {})
        current_adj_types = current_stock.get("adj_types", {})
        
        all_adj_types = set(saved_adj_types.keys()) | set(current_adj_types.keys())
        
        for adj_type in all_adj_types:
            saved_adj = saved_adj_types.get(adj_type, {})
            current_adj = current_adj_types.get(adj_type, {})
            
            adj_diff = {}
            
            # 比较日期范围
            if saved_adj.get("min_date") != current_adj.get("min_date"):
                adj_diff["min_date"] = {
                    "saved": saved_adj.get("min_date"),
                    "current": current_adj.get("min_date")
                }
            
            if saved_adj.get("max_date") != current_adj.get("max_date"):
                adj_diff["max_date"] = {
                    "saved": saved_adj.get("max_date"),
                    "current": current_adj.get("max_date")
                }
            
            # 比较记录数
            if saved_adj.get("total_records") != current_adj.get("total_records"):
                adj_diff["total_records"] = {
                    "saved": saved_adj.get("total_records"),
                    "current": current_adj.get("total_records")
                }
            
            # 比较股票数
            if saved_adj.get("stock_count") != current_adj.get("stock_count"):
                adj_diff["stock_count"] = {
                    "saved": saved_adj.get("stock_count"),
                    "current": current_adj.get("stock_count")
                }
            
            if adj_diff:
                stock_diff[f"adj_type_{adj_type}"] = adj_diff
        
        if stock_diff:
            differences["stock_data"] = stock_diff
        
        # 比较细节数据
        saved_details = saved.get("details_data", {})
        current_details = current.get("details_data", {})
        
        details_diff = {}
        
        # 检查数据库是否存在
        if saved_details.get("database_exists") != current_details.get("database_exists"):
            details_diff["database_exists"] = {
                "saved": saved_details.get("database_exists"),
                "current": current_details.get("database_exists")
            }
        
        # 比较日期范围
        if saved_details.get("min_date") != current_details.get("min_date"):
            details_diff["min_date"] = {
                "saved": saved_details.get("min_date"),
                "current": current_details.get("min_date")
            }
        
        if saved_details.get("max_date") != current_details.get("max_date"):
            details_diff["max_date"] = {
                "saved": saved_details.get("max_date"),
                "current": current_details.get("max_date")
            }
        
        # 比较记录数
        if saved_details.get("total_records") != current_details.get("total_records"):
            details_diff["total_records"] = {
                "saved": saved_details.get("total_records"),
                "current": current_details.get("total_records")
            }
        
        # 比较股票数
        if saved_details.get("stock_count") != current_details.get("stock_count"):
            details_diff["stock_count"] = {
                "saved": saved_details.get("stock_count"),
                "current": current_details.get("stock_count")
            }
        
        if details_diff:
            differences["details_data"] = details_diff
        
        return differences
    
    def update_status_file(self) -> Dict[str, Any]:
        """
        更新状态文件
        
        Returns:
            更新后的状态字典
        """
        logger.info("更新数据库状态文件...")
        return self.generate_status_file()

# 全局数据库管理器实例
_database_manager = None
_database_manager_lock = threading.Lock()

def get_database_manager() -> DatabaseManager:
    """获取数据库管理器实例
    
    在多进程环境下，每个进程会创建自己的 DatabaseManager 实例。
    这是正常行为，因为进程间不能共享线程锁和队列等资源。
    
    每个进程的 DatabaseManager 会：
    1. 自动检测是否在多进程环境下
    2. 根据进程数调整连接池大小
    3. 使用持久化的 DuckDB 配置确保跨进程一致性
    """
    global _database_manager
    
    if _database_manager is None:
        with _database_manager_lock:
            if _database_manager is None:
                _database_manager = DatabaseManager()
                # 在多进程环境下，确保子进程能正确初始化
                if _database_manager._is_multiprocess:
                    logger.info(f"子进程 {_database_manager._process_name} (PID: {_database_manager._process_id}) 的 DatabaseManager 初始化完成")
    
    return _database_manager

# 便捷函数
def execute_query(db_path: str, sql: str, params: Optional[List[Any]] = None, 
                 timeout: float = 30.0) -> pd.DataFrame:
    """执行查询的便捷函数"""
    manager = get_database_manager()
    return manager.execute_sync_query(db_path, sql, params, timeout)


def execute_write(db_path: str, sql: str, params: Optional[List[Any]] = None,
                 timeout: float = 30.0) -> bool:
    """执行写入的便捷函数"""
    manager = get_database_manager()
    return manager.execute_sync_write(db_path, sql, params, timeout)

@contextmanager
def get_connection(db_path: str, read_only: bool = True, timeout: float = 30.0):
    """获取数据库连接的便捷函数 - 严格统一连接打开姿势"""
    manager = get_database_manager()
    # 严格使用统一的连接管理器，避免直接使用duckdb.connect
    with manager.get_connection(db_path, read_only, timeout) as conn:
        yield conn


def close_all_connections():
    """关闭所有连接的便捷函数"""
    manager = get_database_manager()
    manager.close_all()


def clear_connections_only():
    """仅清理连接池，不关闭工作线程 - 用于评分线程中的轻量级清理"""
    manager = get_database_manager()
    manager.clear_connections_only()

# ================= 股票数据特化便捷函数 =================

def get_stock_list_from_cache() -> List[str]:
    """从缓存获取股票列表的便捷函数"""
    manager = get_database_manager()
    return manager.get_stock_list_from_cache()


def get_trade_dates(db_path: str = None) -> List[str]:
    """获取交易日列表的便捷函数"""
    manager = get_database_manager()
    return manager.get_trade_dates(db_path)


def get_trade_dates_from_db(db_path: str = None, *, table: str = "stock_data") -> List[str]:
    """获取数据库中已有交易日的便捷函数（按日期升序）。"""
    manager = get_database_manager()
    return manager.get_trade_dates_from_db(db_path, table=table)


def get_trade_calendar_cached(start_date: Optional[str] = None, end_date: Optional[str] = None, refresh_if_insufficient: bool = True) -> List[str]:
    """获取交易日历（带缓存）的便捷函数。"""
    manager = get_database_manager()
    return manager.get_trade_calendar_cached(start_date, end_date, refresh_if_insufficient)


def get_stock_list(db_path: str = None, adj_type: str = "raw") -> List[str]:
    """获取股票代码列表的便捷函数"""
    manager = get_database_manager()
    return manager.get_stock_list(db_path, adj_type)


def query_stock_data(db_path: str = None, ts_code: str = None, start_date: str = None, 
                    end_date: str = None, columns: List[str] = None, adj_type: str = "qfq", limit: Optional[int] = None,
                    order: str = "asc") -> pd.DataFrame:
    """查询股票数据的便捷函数"""
    manager = get_database_manager()
    return manager.query_stock_data(db_path, ts_code, start_date, end_date, columns, adj_type, limit, order)


def register_preload_cache(cache_name: str, data: pd.DataFrame, ref_date: str, 
                           start_date: str, columns: List[str]) -> None:
    """注册预加载数据缓存的便捷函数"""
    manager = get_database_manager()
    return manager.register_preload_cache(cache_name, data, ref_date, start_date, columns)


def clear_preload_cache(cache_name: str = None) -> None:
    """清除预加载数据缓存的便捷函数"""
    manager = get_database_manager()
    return manager.clear_preload_cache(cache_name)


def count_stock_data(db_path: str = None, ts_code: str = None, start_date: str = None,
                    end_date: str = None, adj_type: str = "qfq") -> int:
    """统计股票数据总行数的便捷函数"""
    manager = get_database_manager()
    return manager.count_stock_data(db_path, ts_code, start_date, end_date, adj_type)


def batch_query_stock_data(db_path: str = None, ts_codes: List[str] = None, 
                          start_date: str = None, end_date: str = None, 
                          columns: List[str] = None, adj_type: str = "qfq") -> pd.DataFrame:
    """批量查询多只股票数据的便捷函数"""
    manager = get_database_manager()
    return manager.batch_query_stock_data(db_path, ts_codes, start_date, end_date, columns, adj_type)


def is_using_unified_db(db_path: str = None) -> bool:
    """检查是否使用统一数据库的便捷函数"""
    manager = get_database_manager()
    return manager.is_using_unified_db(db_path)


def get_database_info(db_path: str = None, use_cache: bool = True) -> Dict[str, Any]:
    """获取数据库信息的便捷函数（优先使用状态文件缓存）"""
    manager = get_database_manager()
    return manager.get_database_info(db_path, use_cache=use_cache)


def get_data_source_status(db_path: str = None) -> Dict[str, Any]:
    """获取数据源状态的便捷函数"""
    manager = get_database_manager()
    return manager.get_data_source_status(db_path)


def get_latest_trade_date(db_path: str = None) -> Optional[str]:
    """获取最新交易日的便捷函数"""
    manager = get_database_manager()
    return manager.get_latest_trade_date(db_path)


def get_smart_end_date(end_date_config: str) -> str:
    """获取智能结束日期的便捷函数"""
    manager = get_database_manager()
    return manager.get_smart_end_date(end_date_config)

# ================= 数据库表结构初始化功能 =================
class DatabaseSchemaManager:
    """数据库表结构管理器"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager
    
    def _get_indicator_columns(self) -> List[str]:
        """获取所有指标列名"""
        try:
            from indicators import REGISTRY
            # 从REGISTRY中获取所有指标的输出列
            all_columns = []
            for meta in REGISTRY.values():
                all_columns.extend(meta.out.keys())
            # 额外补充pro_bar因子返回但非指标的字段（如换手率）
            extra_factor_cols = ['tor']
            for col in extra_factor_cols:
                if col not in all_columns:
                    all_columns.append(col)
            return all_columns
        except ImportError:
            # 如果无法导入indicators模块，返回默认的指标列
            return [
                'ma5', 'ma10', 'ma20', 'ma60',
                'ema5', 'ema10', 'ema20',
                'k', 'd', 'j',
                'rsi6', 'rsi12', 'rsi24',
            'macd', 'macd_signal', 'macd_hist',
            'boll_upper', 'boll_mid', 'boll_lower',
            'atr', 'cci', 'williams_r', 'obv',
            # pro_bar换手率因子（短名）已作为基础列，避免重复
        ]
    
    def _build_indicator_columns_sql(self) -> List[str]:
        """构建指标列定义SQL"""
        indicator_columns = self._get_indicator_columns()

        # 移除已经作为基础行情列存在的字段，避免重复定义
        base_cols = {
            "ts_code", "trade_date", "adj_type",
            "open", "high", "low", "close",
            "pre_close", "change", "pct_chg",
            "vol", "amount", "tor",
        }
        indicator_columns = [c for c in indicator_columns if c not in base_cols]
        indicator_cols_sql = []
        
        # SQL保留字列表
        sql_reserved_words = [
            'DIFF', 'ORDER', 'GROUP', 'SELECT', 'FROM', 'WHERE', 'HAVING', 'UNION', 'INTERSECT', 'EXCEPT',
            'Z_SCORE', 'SCORE', 'RANK', 'INDEX', 'KEY', 'VALUE', 'TABLE', 'COLUMN', 'ROW', 'COUNT',
            'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'LIMIT', 'OFFSET', 'JOIN', 'INNER', 'LEFT', 'RIGHT',
            'OUTER', 'ON', 'AS', 'IS', 'NULL', 'NOT', 'AND', 'OR', 'IN', 'BETWEEN', 'LIKE', 'EXISTS'
        ]
        
        for col in indicator_columns:
            if col.upper() in sql_reserved_words:
                indicator_cols_sql.append(f'            "{col}" DECIMAL(15,4),')
            else:
                indicator_cols_sql.append(f"            {col} DECIMAL(15,4),")
        
        return indicator_cols_sql
    
    def init_stock_data_tables(self, db_path: str, db_type: str = "duckdb"):
        """初始化股票数据表结构"""
        # 在多进程环境下，如果是子进程，跳过初始化（表结构应该已经在主进程中创建）
        if _MULTIPROCESSING_AVAILABLE:
            current_process = _get_current_process()
            if current_process.name != "MainProcess":
                logger.debug(f"子进程 {current_process.name} 跳过表结构初始化（表结构应在主进程中创建）")
                return
        
        try:
            # 确保数据库目录存在
            db_dir = os.path.dirname(db_path)
            if db_dir:  # 只有当目录路径不为空时才创建
                os.makedirs(db_dir, exist_ok=True)
            
            if db_type.lower() == "duckdb":
                self._init_duckdb_stock_tables(db_path)
            elif db_type.lower() == "sqlite":
                self._init_sqlite_stock_tables(db_path)
            else:
                raise ValueError(f"不支持的数据库类型: {db_type}")
                
            logger.info(f"数据库表结构初始化完成: {db_path} ({db_type})")
            
        except Exception as e:
            logger.error(f"初始化数据库表结构失败: {e}")
            raise
    
    def _init_duckdb_stock_tables(self, db_path: str):
        """初始化DuckDB股票数据表结构"""
        # 在多进程环境下，如果是子进程，跳过初始化（表结构应该已经在主进程中创建）
        if _MULTIPROCESSING_AVAILABLE:
            current_process = _get_current_process()
            if current_process.name != "MainProcess":
                logger.debug(f"子进程 {current_process.name} 跳过表结构初始化（表结构应在主进程中创建）")
                return
        
        # 通过DatabaseManager的连接池创建连接，避免配置冲突
        logger.info(f"[数据库初始化] 开始初始化stock_data数据库表结构: {db_path}")
        with self.db_manager.get_connection(db_path, read_only=False) as conn:
            # 构建指标列定义
            indicator_cols_sql = self._build_indicator_columns_sql()
            
            # 创建主表
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS stock_data (
                ts_code VARCHAR(20) NOT NULL,
                trade_date VARCHAR(8) NOT NULL,
                adj_type VARCHAR(10) NOT NULL DEFAULT 'raw',
                open DECIMAL(10,2),
                high DECIMAL(10,2),
                low DECIMAL(10,2),
                close DECIMAL(10,2),
                pre_close DECIMAL(10,2),
                change DECIMAL(10,2),
                pct_chg DECIMAL(10,4),
                vol DECIMAL(15,2),
                amount DECIMAL(20,2),
                tor DECIMAL(10,4),
                -- 技术指标列（动态添加）
{chr(10).join(indicator_cols_sql)[:-1]},
                PRIMARY KEY (ts_code, trade_date, adj_type)
            )
            """
            conn.execute(create_table_sql)
            
            # 动态添加缺失的指标列
            self._add_missing_indicator_columns(conn, "duckdb")
            # 仅补充必需的基础行情列（pre_close/change/pct_chg）
            self._ensure_stock_base_columns(conn, "duckdb")
            
            # 创建优化索引 - 简化索引配置
            # 主键 (ts_code, trade_date, adj_type) 已经覆盖了大部分查询场景
            # 只保留按日期和复权类型查询的索引
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_date_adj ON stock_data (trade_date, adj_type)"
            ]
            
            for idx_sql in indexes:
                try:
                    conn.execute(idx_sql)
                except Exception as e:
                    logger.warning(f"创建索引失败: {e}")
            
            # DuckDB 优化参数已在连接配置中设置
    
    def _init_sqlite_stock_tables(self, db_path: str):
        """初始化SQLite股票数据表结构"""
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        try:
            # 构建指标列定义
            indicator_cols_sql = self._build_indicator_columns_sql()
            
            # 创建主表
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS stock_data (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                adj_type TEXT NOT NULL DEFAULT 'raw',
                open DECIMAL(10,2),
                high DECIMAL(10,2),
                low DECIMAL(10,2),
                close DECIMAL(10,2),
                pre_close DECIMAL(10,2),
                change DECIMAL(10,2),
                pct_chg DECIMAL(10,4),
                vol DECIMAL(15,2),
                amount DECIMAL(20,2),
                tor DECIMAL(10,4),
                -- 技术指标列（动态添加）
{chr(10).join(indicator_cols_sql)[:-1]},
                PRIMARY KEY (ts_code, trade_date, adj_type)
            )
            """
            conn.execute(create_table_sql)
            
            # 动态添加缺失的指标列
            self._add_missing_indicator_columns(conn, "sqlite")
            # 仅补充必需的基础行情列（pre_close/change/pct_chg）
            self._ensure_stock_base_columns(conn, "sqlite")
            
            # 创建索引 - 优化后只保留必要的复合索引
            # 主键 (ts_code, trade_date, adj_type) 已经覆盖了大部分查询场景
            # 只保留按日期和复权类型查询的索引
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_date_adj ON stock_data(trade_date, adj_type)"
            ]
            
            for idx_sql in indexes:
                try:
                    conn.execute(idx_sql)
                except Exception as e:
                    logger.warning(f"创建索引失败: {e}")
            
            conn.commit()
            
        finally:
            conn.close()
    
    def _add_missing_indicator_columns(self, conn, db_type: str):
        """动态添加缺失的指标列"""
        try:
            # 获取现有列
            if db_type == "duckdb":
                result = conn.execute("DESCRIBE stock_data").fetchall()
                existing_cols = [row[0] for row in result]
            else:  # sqlite
                cursor = conn.execute("PRAGMA table_info(stock_data)")
                existing_cols = [row[1] for row in cursor.fetchall()]
            
            # 添加缺失的指标列
            indicator_columns = self._get_indicator_columns()
            for col in indicator_columns:
                if col not in existing_cols:
                    try:
                        if db_type == "duckdb":
                            alter_sql = f'ALTER TABLE stock_data ADD COLUMN "{col}" DECIMAL(15,4)'
                        else:
                            alter_sql = f'ALTER TABLE stock_data ADD COLUMN "{col}" DECIMAL(15,4)'
                        conn.execute(alter_sql)
                        logger.debug(f"添加指标列: {col}")
                    except Exception as e:
                        logger.warning(f"添加指标列失败 {col}: {e}")
        except Exception as e:
            logger.warning(f"动态添加指标列失败: {e}")

    def _ensure_stock_base_columns(self, conn, db_type: str):
        """为 stock_data 表补充必需的基础行情列（不含市值/股本）。"""
        base_columns = {
            "pre_close": "DECIMAL(10,2)",
            "change": "DECIMAL(10,2)",
            "pct_chg": "DECIMAL(10,4)",
            "tor": "DECIMAL(10,4)",
        }
        try:
            if db_type == "duckdb":
                result = conn.execute("DESCRIBE stock_data").fetchall()
                existing_cols = [row[0] for row in result]
            else:
                cursor = conn.execute("PRAGMA table_info(stock_data)")
                existing_cols = [row[1] for row in cursor.fetchall()]

            for col, type_sql in base_columns.items():
                if col not in existing_cols:
                    try:
                        alter_sql = f'ALTER TABLE stock_data ADD COLUMN "{col}" {type_sql}'
                        conn.execute(alter_sql)
                        logger.info(f"补充 stock_data 基础列: {col}")
                    except Exception as e:
                        logger.warning(f"添加基础列失败 {col}: {e}")
        except Exception as e:
            logger.warning(f"检查/补充基础列失败: {e}")
    
    def init_stock_details_tables(self, db_path: str, db_type: str = "duckdb"):
        """初始化股票详情表结构"""
        # 在多进程环境下，如果是子进程，跳过初始化（表结构应该已经在主进程中创建）
        if _MULTIPROCESSING_AVAILABLE:
            current_process = _get_current_process()
            if current_process.name != "MainProcess":
                logger.debug(f"子进程 {current_process.name} 跳过表结构初始化（表结构应在主进程中创建）")
                return
        
        try:
            if db_type.lower() == "duckdb":
                self._init_duckdb_details_tables(db_path)
            elif db_type.lower() == "sqlite":
                self._init_sqlite_details_tables(db_path)
            else:
                raise ValueError(f"不支持的数据库类型: {db_type}")
                
            logger.info(f"股票详情表结构初始化完成: {db_path} ({db_type})")
            
        except Exception as e:
            logger.error(f"初始化股票详情表结构失败: {e}")
            raise
    
    def _init_duckdb_details_tables(self, db_path: str):
        """初始化DuckDB股票详情表结构"""
        # 在多进程环境下，如果是子进程，跳过初始化（表结构应该已经在主进程中创建）
        if _MULTIPROCESSING_AVAILABLE:
            current_process = _get_current_process()
            if current_process.name != "MainProcess":
                logger.debug(f"子进程 {current_process.name} 跳过表结构初始化（表结构应在主进程中创建）")
                return
        
        # 通过DatabaseManager的连接池创建连接，避免配置冲突
        logger.info(f"[数据库初始化] 开始初始化details数据库表结构: {db_path}")
        # 通过连接管理器创建连接，不使用直连避免配置冲突
        with self.db_manager.get_connection(db_path, read_only=False) as conn:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS stock_details (
                ts_code VARCHAR,
                ref_date VARCHAR,
                score DOUBLE,
                tiebreak DOUBLE,
                rank INTEGER,
                total INTEGER,
                rules JSON,
                PRIMARY KEY (ts_code, ref_date)
            )
            """
            conn.execute(create_table_sql)
            logger.info(f"[数据库初始化] stock_details表结构初始化完成: {db_path}")
    
    def _init_sqlite_details_tables(self, db_path: str):
        """初始化SQLite股票详情表结构"""
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS stock_details (
                ts_code TEXT NOT NULL,
                ref_date TEXT NOT NULL,
                score REAL,
                tiebreak REAL,
                rank INTEGER,
                total INTEGER,
                rules TEXT,
                PRIMARY KEY (ts_code, ref_date)
            )
            """
            conn.execute(create_table_sql)
            
            # 创建索引
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_ts_code ON stock_details(ts_code)",
                "CREATE INDEX IF NOT EXISTS idx_ref_date ON stock_details(ref_date)",
                "CREATE INDEX IF NOT EXISTS idx_score ON stock_details(score)",
                "CREATE INDEX IF NOT EXISTS idx_rank ON stock_details(rank)"
            ]
            
            for idx_sql in indexes:
                try:
                    conn.execute(idx_sql)
                except Exception as e:
                    logger.warning(f"创建索引失败: {e}")
            
            conn.commit()
            
        finally:
            conn.close()
    
# 扩展DatabaseManager类，添加表结构初始化功能
def _add_schema_manager_to_database_manager():
    """为DatabaseManager添加表结构管理器"""
    if not hasattr(DatabaseManager, '_schema_manager'):
        DatabaseManager._schema_manager = None
        DatabaseManager._schema_manager_lock = threading.Lock()
    
    def get_schema_manager(self) -> DatabaseSchemaManager:
        """获取表结构管理器"""
        if self._schema_manager is None:
            with self._schema_manager_lock:
                if self._schema_manager is None:
                    self._schema_manager = DatabaseSchemaManager(self)
        return self._schema_manager
    
    def init_stock_data_tables(self, db_path: str, db_type: str = "duckdb"):
        """初始化股票数据表结构"""
        schema_manager = self.get_schema_manager()
        return schema_manager.init_stock_data_tables(db_path, db_type)
    
    def init_stock_details_tables(self, db_path: str, db_type: str = "duckdb"):
        """初始化股票详情表结构"""
        schema_manager = self.get_schema_manager()
        return schema_manager.init_stock_details_tables(db_path, db_type)
    
    # 将方法添加到DatabaseManager类
    DatabaseManager.get_schema_manager = get_schema_manager
    DatabaseManager.init_stock_data_tables = init_stock_data_tables
    DatabaseManager.init_stock_details_tables = init_stock_details_tables

# 初始化表结构管理功能
_add_schema_manager_to_database_manager()

# 便捷函数
def init_stock_data_tables(db_path: str, db_type: str = "duckdb"):
    """初始化股票数据表结构的便捷函数"""
    manager = get_database_manager()
    return manager.init_stock_data_tables(db_path, db_type)


def init_stock_details_tables(db_path: str, db_type: str = "duckdb"):
    """初始化股票详情表结构的便捷函数"""
    manager = get_database_manager()
    return manager.init_stock_details_tables(db_path, db_type)

# ================= 数据接收和导入功能 =================
@dataclass
class DataImportRequest:
    """数据导入请求"""
    request_id: str
    source_module: str
    data_type: str  # 'stock_data', 'indicator_data', 'score_data', 'custom'
    data: Union[pd.DataFrame, List[Dict], str]  # 支持DataFrame、字典列表、JSON字符串
    table_name: str
    mode: str = "append"  # 'append', 'replace', 'upsert'
    validation_rules: Optional[Dict[str, Any]] = None
    callback: Optional[Callable] = None
    timeout: float = 60.0
    priority: int = 0
    db_path: Optional[str] = None  # 数据库路径
    
    def __lt__(self, other):
        """支持优先级队列排序"""
        if not isinstance(other, DataImportRequest):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.request_id < other.request_id

@dataclass
class DataImportResponse:
    """数据导入响应"""
    request_id: str
    success: bool
    rows_imported: int = 0
    rows_skipped: int = 0
    error: Optional[str] = None
    execution_time: float = 0.0
    validation_errors: List[str] = None


class DataReceiver:
    """数据接收器 - 负责接收和验证外部模块的数据"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager
        self._validation_rules = self._load_default_validation_rules()
        self._import_queue = queue.PriorityQueue()
        self._worker_threads = []
        self._shutdown = False
        self._start_import_workers()
        
        logger.info("数据接收器初始化完成")
    
    def _load_default_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """加载默认验证规则"""
        return {
            "stock_data": {
                "required_columns": ["ts_code", "trade_date", "open", "high", "low", "close", "vol"],
                "data_types": {
                    "ts_code": str,
                    "trade_date": str,
                    "open": (int, float),
                    "high": (int, float),
                    "low": (int, float),
                    "close": (int, float),
                    "vol": (int, float)
                },
                "constraints": {
                    "ts_code": r"^\d{6}\.(SH|SZ|BJ)$",
                    "trade_date": r"^\d{8}$",
                    "open": lambda x: x > 0,
                    "high": lambda x: x > 0,
                    "low": lambda x: x > 0,
                    "close": lambda x: x > 0,
                    "vol": lambda x: x >= 0
                }
            },
            "indicator_data": {
                "required_columns": ["ts_code", "trade_date", "indicator_name", "value"],
                "data_types": {
                    "ts_code": str,
                    "trade_date": str,
                    "indicator_name": str,
                    "value": (int, float)
                }
            },
            "score_data": {
                "required_columns": ["ts_code", "trade_date", "score", "rank"],
                "data_types": {
                    "ts_code": str,
                    "trade_date": str,
                    "score": (int, float),
                    "rank": int
                }
            }
        }
    
    def _start_import_workers(self):
        """启动数据导入工作线程"""
        worker_count = min(os.cpu_count() or 2, 4)
        for i in range(worker_count):
            worker = threading.Thread(target=self._import_worker_loop, daemon=True)
            worker.start()
            self._worker_threads.append(worker)
    
    def _import_worker_loop(self):
        """数据导入工作线程循环"""
        while not self._shutdown:
            try:
                request = self._import_queue.get(timeout=0.5)
                self._process_import_request(request)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"数据导入工作线程出错: {e}")
    
    def _process_import_request(self, request: DataImportRequest):
        """处理数据导入请求"""
        start_time = time.time()
        response = None
        
        try:
            logger.info(f"[数据导入] 开始处理导入请求 {request.request_id} (来源: {request.source_module}, 表: {request.table_name}, 模式: {request.mode})")
            
            # 验证数据
            validation_result = self._validate_data(request)
            if not validation_result["valid"]:
                logger.error(f"[数据导入] 数据验证失败 {request.request_id}: {', '.join(validation_result['errors'])}")
                response = DataImportResponse(
                    request_id=request.request_id,
                    success=False,
                    error=f"数据验证失败: {', '.join(validation_result['errors'])}",
                    validation_errors=validation_result["errors"],
                    execution_time=time.time() - start_time
                )
            else:
                # 转换数据格式
                df = self._convert_to_dataframe(request.data)
                # 统一移除不需要写入 stock_data 的字段
                if request.data_type == "stock_data":
                    df = df.drop(columns=["total_share"], errors="ignore")
                logger.info(f"[数据导入] 数据验证通过，开始导入 {len(df)} 条记录到表 {request.table_name}...")
                
                # 执行导入
                db_path = request.db_path
                if db_path is None:
                    from config import DATA_ROOT, UNIFIED_DB_PATH
                    db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                
                import_result = self._import_dataframe(
                    df, request.table_name, request.mode, db_path
                )
                
                execution_time = time.time() - start_time
                logger.info(f"[数据导入] 导入完成 {request.request_id}: 导入 {import_result['imported']} 条记录，跳过 {import_result['skipped']} 条，耗时 {execution_time:.2f} 秒")
                
                response = DataImportResponse(
                    request_id=request.request_id,
                    success=True,
                    rows_imported=import_result["imported"],
                    rows_skipped=import_result["skipped"],
                    execution_time=execution_time
                )
                
        except Exception as e:
            response = DataImportResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
            logger.error(f"数据导入失败 {request.request_id}: {e}")
        
        # 调用回调函数
        if request.callback and response:
            try:
                request.callback(response)
            except Exception as e:
                logger.error(f"导入回调函数执行失败 {request.request_id}: {e}")
        
        # 返回响应
        return response
    
    def _validate_data(self, request: DataImportRequest) -> Dict[str, Any]:
        """验证数据"""
        try:
            # 获取验证规则
            rules = request.validation_rules or self._validation_rules.get(request.data_type, {})
            if not rules:
                return {"valid": True, "errors": []}
            
            # 转换数据为DataFrame进行验证
            df = self._convert_to_dataframe(request.data)
            errors = []
            
            # 检查必需列
            required_columns = rules.get("required_columns", [])
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"缺少必需列: {missing_columns}")
            
            # 检查数据类型
            data_types = rules.get("data_types", {})
            for col, expected_type in data_types.items():
                if col in df.columns:
                    if not df[col].dtype == expected_type and not isinstance(df[col].iloc[0], expected_type):
                        errors.append(f"列 {col} 数据类型不匹配，期望 {expected_type}")
            
            # 检查约束条件
            constraints = rules.get("constraints", {})
            for col, constraint in constraints.items():
                if col in df.columns:
                    if isinstance(constraint, str):  # 正则表达式
                        import re
                        invalid_rows = df[~df[col].astype(str).str.match(constraint)]
                        if not invalid_rows.empty:
                            errors.append(f"列 {col} 格式不符合要求: {constraint}")
                    elif callable(constraint):  # 函数约束
                        invalid_rows = df[~df[col].apply(constraint)]
                        if not invalid_rows.empty:
                            errors.append(f"列 {col} 值不符合约束条件")
            
            return {"valid": len(errors) == 0, "errors": errors}
            
        except Exception as e:
            return {"valid": False, "errors": [f"验证过程出错: {str(e)}"]}
    
    def _convert_to_dataframe(self, data: Union[pd.DataFrame, List[Dict], str]) -> pd.DataFrame:
        """将各种数据格式转换为DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, str):
            try:
                # 尝试解析JSON
                json_data = json.loads(data)
                if isinstance(json_data, list):
                    return pd.DataFrame(json_data)
                else:
                    return pd.DataFrame([json_data])
            except json.JSONDecodeError:
                # 尝试解析CSV
                from io import StringIO
                return pd.read_csv(StringIO(data))
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
    
    def _import_dataframe(self, df: pd.DataFrame, table_name: str, mode: str, db_path: str) -> Dict[str, int]:
        """将DataFrame导入到数据库"""
        import tempfile
        import os
        import uuid
        import atexit
        
        # 使用上下文管理器确保临时文件被正确清理
        class TempFileManager:
            def __init__(self, suffix='.csv'):
                self.temp_file = None
                self.temp_file_path = None
                self.suffix = suffix
                
            def __enter__(self):
                self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=self.suffix, delete=False)
                self.temp_file_path = self.temp_file.name
                return self.temp_file
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # 关闭文件句柄
                if self.temp_file:
                    self.temp_file.close()
                
                # 确保临时文件被删除
                if self.temp_file_path and os.path.exists(self.temp_file_path):
                    try:
                        os.unlink(self.temp_file_path)
                        logger.debug(f"临时文件已清理: {self.temp_file_path}")
                    except Exception as e:
                        logger.warning(f"清理临时文件失败 {self.temp_file_path}: {e}")
                        # 注册退出时清理
                        atexit.register(lambda: self._cleanup_on_exit())
                
            def _cleanup_on_exit(self):
                """程序退出时清理临时文件"""
                if self.temp_file_path and os.path.exists(self.temp_file_path):
                    try:
                        os.unlink(self.temp_file_path)
                    except:
                        pass
        
        try:
            # 使用统一的数据库管理器进行导入，避免连接冲突
            with self.db_manager.get_connection(db_path, read_only=False) as conn:
                if mode == "replace":
                    # 替换模式：先删除表，再插入
                    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    
                    with TempFileManager() as tmp_file:
                        df.to_csv(tmp_file.name, index=False)
                        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{tmp_file.name}')")
                    
                    return {"imported": len(df), "skipped": 0}
                
                elif mode == "append":
                    # 追加模式：将DataFrame写入临时文件，然后导入
                    with TempFileManager() as tmp_file:
                        df.to_csv(tmp_file.name, index=False)
                        
                        # 获取目标表的所有列
                        table_columns = conn.execute(f"DESCRIBE {table_name}").df()['column_name'].tolist()
                        
                        # 创建临时表
                        temp_table_name = f"temp_append_{uuid.uuid4().hex[:8]}"
                        conn.execute(f"CREATE TEMP TABLE {temp_table_name} AS SELECT * FROM read_csv_auto('{tmp_file.name}')")
                        temp_columns = conn.execute(f"DESCRIBE {temp_table_name}").df()['column_name'].tolist()
                        
                        # 只选择存在的列进行插入
                        common_columns = [col for col in temp_columns if col in table_columns]
                        if not common_columns:
                            raise ValueError("没有匹配的列可以插入")
                        
                        # 构建列名列表
                        columns_str = ', '.join(common_columns)
                        values_str = ', '.join([f"{col}" for col in common_columns])
                        
                        conn.execute(f"INSERT INTO {table_name} ({columns_str}) SELECT {values_str} FROM {temp_table_name}")
                        conn.execute(f"DROP TABLE {temp_table_name}")
                    
                    return {"imported": len(df), "skipped": 0}
                
                elif mode == "upsert":
                    # 更新插入模式：使用ON CONFLICT DO UPDATE，只更新传入的字段，保留其他字段的值
                    with TempFileManager() as tmp_file:
                        df.to_csv(tmp_file.name, index=False)
                        
                        # 使用唯一的临时表名避免冲突
                        temp_table_name = f"temp_import_{uuid.uuid4().hex[:8]}"
                        conn.execute(f"CREATE TEMP TABLE {temp_table_name} AS SELECT * FROM read_csv_auto('{tmp_file.name}')")
                        
                        # 获取目标表的所有列
                        table_columns = conn.execute(f"DESCRIBE {table_name}").df()['column_name'].tolist()
                        temp_columns = conn.execute(f"DESCRIBE {temp_table_name}").df()['column_name'].tolist()
                        
                        # 只选择存在的列进行插入
                        common_columns = [col for col in temp_columns if col in table_columns]
                        if not common_columns:
                            raise ValueError("没有匹配的列可以插入")
                        
                        # 尝试获取主键信息
                        primary_key_columns = []
                        try:
                            # 查询表的主键约束
                            # 对于DuckDB，可以通过PRAGMA或查询系统表获取主键
                            # 简化处理：如果是stock_details表，主键是(ts_code, ref_date)
                            if table_name == "stock_details":
                                primary_key_columns = ["ts_code", "ref_date"]
                            elif table_name == "stock_data":
                                primary_key_columns = ["ts_code", "trade_date", "adj_type"]
                            else:
                                # 对于其他表，尝试从表结构推断主键（NOT NULL且唯一）
                                # 如果无法确定，使用所有匹配的列作为主键
                                table_info = conn.execute(f"DESCRIBE {table_name}").df()
                                # 检查是否有明确的PRIMARY KEY约束信息
                                # 如果没有，回退到使用所有common_columns
                                primary_key_columns = common_columns
                        except Exception as e:
                            logger.warning(f"无法获取表 {table_name} 的主键信息: {e}，使用所有列作为键")
                            primary_key_columns = common_columns
                        
                        # 如果没有明确的主键，使用所有列作为键（但这可能导致问题）
                        if not primary_key_columns:
                            primary_key_columns = common_columns
                        
                        # 确定需要更新的列（排除主键列，只更新非主键列）
                        update_columns = [col for col in common_columns if col not in primary_key_columns]
                        
                        # 构建列名列表
                        columns_str = ', '.join(common_columns)
                        values_str = ', '.join([f"{col}" for col in common_columns])
                        primary_key_str = ', '.join(primary_key_columns)
                        
                        # 使用UPDATE + INSERT方式实现upsert
                        # 这样可以只更新传入的字段，保留其他字段的值
                        if update_columns:
                            # 先更新已存在的记录（只更新非主键字段）
                            # 使用子查询方式，兼容DuckDB语法
                            update_set_clause = ', '.join([f"{col} = (SELECT {col} FROM {temp_table_name} WHERE {' AND '.join([f'{temp_table_name}.{pk} = {table_name}.{pk}' for pk in primary_key_columns])})" for col in update_columns])
                            where_clause = f"EXISTS (SELECT 1 FROM {temp_table_name} WHERE {' AND '.join([f'{temp_table_name}.{pk} = {table_name}.{pk}' for pk in primary_key_columns])})"
                            
                            # 使用子查询方式进行UPDATE
                            update_sql = f"""
                                UPDATE {table_name} 
                                SET {update_set_clause}
                                WHERE {where_clause}
                            """
                            conn.execute(update_sql)
                            
                            # 然后插入不存在的记录
                            # 找出temp_table中存在但目标表中不存在的记录
                            join_condition = ' AND '.join([f"temp.{pk} = target.{pk}" for pk in primary_key_columns])
                            # 明确指定使用temp表的列，避免列名歧义
                            values_str_with_alias = ', '.join([f"temp.{col}" for col in common_columns])
                            insert_sql = f"""
                                INSERT INTO {table_name} ({columns_str})
                                SELECT {values_str_with_alias} FROM {temp_table_name} AS temp
                                LEFT JOIN {table_name} AS target ON {join_condition}
                                WHERE target.{primary_key_columns[0]} IS NULL
                            """
                            conn.execute(insert_sql)
                        else:
                            # 只有主键列，使用INSERT OR IGNORE（不更新）
                            insert_sql = f"""
                                INSERT OR IGNORE INTO {table_name} ({columns_str}) 
                                SELECT {values_str} FROM {temp_table_name}
                            """
                            conn.execute(insert_sql)
                        conn.execute(f"DROP TABLE {temp_table_name}")
                    
                    return {"imported": len(df), "skipped": 0}
                
                else:
                    raise ValueError(f"不支持的导入模式: {mode}")
                    
        except Exception as e:
            logger.error(f"DataFrame导入失败: {e}")
            logger.error(f"导入参数: table_name={table_name}, mode={mode}, df_shape={df.shape}, df_columns={list(df.columns)}")
            raise
    
    def receive_data(self, source_module: str, data_type: str, data: Union[pd.DataFrame, List[Dict], str],
                    table_name: str, mode: str = "append", validation_rules: Optional[Dict] = None,
                    callback: Optional[Callable] = None, timeout: float = 60.0, 
                    priority: int = 0, db_path: str = None) -> str:
        """接收外部模块的数据"""
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            request_id = f"import_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            
            request = DataImportRequest(
                request_id=request_id,
                source_module=source_module,
                data_type=data_type,
                data=data,
                table_name=table_name,
                mode=mode,
                validation_rules=validation_rules,
                callback=callback,
                timeout=timeout,
                priority=priority,
                db_path=db_path
            )
            
            self._import_queue.put(request)
            logger.debug(f"数据导入请求已提交: {request_id} (来源: {source_module}, 类型: {data_type})")
            return request_id
            
        except Exception as e:
            logger.error(f"接收数据失败: {e}")
            raise
    
    def receive_stock_data(self, source_module: str, data: Union[pd.DataFrame, List[Dict], str],
                          mode: str = "append", callback: Optional[Callable] = None,
                          db_path: str = None) -> str:
        """接收股票数据"""
        return self.receive_data(
            source_module=source_module,
            data_type="stock_data",
            data=data,
            table_name="stock_data",
            mode=mode,
            callback=callback,
            db_path=db_path
        )
    
    def receive_indicator_data(self, source_module: str, data: Union[pd.DataFrame, List[Dict], str],
                              table_name: str = "indicator_data", mode: str = "append",
                              callback: Optional[Callable] = None, db_path: str = None) -> str:
        """接收指标数据"""
        return self.receive_data(
            source_module=source_module,
            data_type="indicator_data",
            data=data,
            table_name=table_name,
            mode=mode,
            callback=callback,
            db_path=db_path
        )
    
    def receive_score_data(self, source_module: str, data: Union[pd.DataFrame, List[Dict], str],
                          table_name: str = "score_data", mode: str = "append",
                          callback: Optional[Callable] = None, db_path: str = None) -> str:
        """接收评分数据"""
        return self.receive_data(
            source_module=source_module,
            data_type="score_data",
            data=data,
            table_name=table_name,
            mode=mode,
            callback=callback,
            db_path=db_path
        )
    
    def get_import_stats(self) -> Dict[str, Any]:
        """获取导入统计信息"""
        return {
            "queue_size": self._import_queue.qsize(),
            "worker_count": len(self._worker_threads),
            "shutdown": self._shutdown
        }
    
    def import_data_sync(self, source_module: str, data_type: str, data: Union[pd.DataFrame, List[Dict], str],
                        table_name: str, mode: str = "append", validation_rules: Optional[Dict] = None,
                        db_path: str = None) -> DataImportResponse:
        """同步导入数据（用于测试和调试）"""
        request_id = f"sync_import_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        try:
            if db_path is None:
                from config import DATA_ROOT, UNIFIED_DB_PATH
                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            request = DataImportRequest(
                request_id=request_id,
                source_module=source_module,
                data_type=data_type,
                data=data,
                table_name=table_name,
                mode=mode,
                validation_rules=validation_rules,
                db_path=db_path
            )
            
            # 直接处理请求
            result = self._process_import_request(request)
            logger.debug(f"同步导入完成: {request_id}, 结果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"同步导入失败: {e}")
            return DataImportResponse(
                request_id=request_id,
                success=False,
                error=str(e),
                execution_time=0.0
            )
    
    def shutdown(self):
        """关闭数据接收器"""
        self._shutdown = True
        for worker in self._worker_threads:
            worker.join(timeout=5.0)
        logger.info("数据接收器已关闭")

# 扩展DatabaseManager类，添加数据接收功能
def _add_data_receiver_to_manager():
    """为DatabaseManager添加数据接收器"""
    if not hasattr(DatabaseManager, '_data_receiver'):
        DatabaseManager._data_receiver = None
        DatabaseManager._receiver_lock = threading.Lock()
    
    def get_data_receiver(self) -> DataReceiver:
        """获取数据接收器"""
        if self._data_receiver is None:
            with self._receiver_lock:
                if self._data_receiver is None:
                    self._data_receiver = DataReceiver(self)
        return self._data_receiver
    
    def receive_data(self, source_module: str, data_type: str, data: Union[pd.DataFrame, List[Dict], str],
                    table_name: str, mode: str = "append", validation_rules: Optional[Dict] = None,
                    callback: Optional[Callable] = None, timeout: float = 60.0, 
                    priority: int = 0, db_path: str = None) -> str:
        """接收外部模块的数据"""
        receiver = self.get_data_receiver()
        return receiver.receive_data(source_module, data_type, data, table_name, mode, 
                                   validation_rules, callback, timeout, priority, db_path)
    
    def receive_stock_data(self, source_module: str, data: Union[pd.DataFrame, List[Dict], str],
                          mode: str = "append", callback: Optional[Callable] = None,
                          db_path: str = None) -> str:
        """接收股票数据"""
        receiver = self.get_data_receiver()
        return receiver.receive_stock_data(source_module, data, mode, callback, db_path)
    
    def receive_indicator_data(self, source_module: str, data: Union[pd.DataFrame, List[Dict], str],
                              table_name: str = "indicator_data", mode: str = "append",
                              callback: Optional[Callable] = None, db_path: str = None) -> str:
        """接收指标数据"""
        receiver = self.get_data_receiver()
        return receiver.receive_indicator_data(source_module, data, table_name, mode, callback, db_path)
    
    def receive_score_data(self, source_module: str, data: Union[pd.DataFrame, List[Dict], str],
                          table_name: str = "score_data", mode: str = "append",
                          callback: Optional[Callable] = None, db_path: str = None) -> str:
        """接收评分数据"""
        receiver = self.get_data_receiver()
        return receiver.receive_score_data(source_module, data, table_name, mode, callback, db_path)
    
    def import_stock_data_sync(self, source_module: str, data: Union[pd.DataFrame, List[Dict], str],
                              mode: str = "append", db_path: str = None) -> DataImportResponse:
        """同步导入股票数据（用于测试和调试）"""
        receiver = self.get_data_receiver()
        return receiver.import_data_sync(source_module, "stock_data", data, "stock_data", mode, None, db_path)
    
    # 将方法添加到DatabaseManager类
    DatabaseManager.get_data_receiver = get_data_receiver
    DatabaseManager.receive_data = receive_data
    DatabaseManager.receive_stock_data = receive_stock_data
    DatabaseManager.receive_indicator_data = receive_indicator_data
    DatabaseManager.receive_score_data = receive_score_data
    DatabaseManager.import_stock_data_sync = import_stock_data_sync

# 初始化数据接收功能
_add_data_receiver_to_manager()

# 便捷函数
def receive_data(source_module: str, data_type: str, data: Union[pd.DataFrame, List[Dict], str],
                table_name: str, mode: str = "append", validation_rules: Optional[Dict] = None,
                callback: Optional[Callable] = None, timeout: float = 60.0, 
                priority: int = 0, db_path: str = None) -> str:
    """接收外部模块数据的便捷函数"""
    manager = get_database_manager()
    return manager.receive_data(source_module, data_type, data, table_name, mode,
                              validation_rules, callback, timeout, priority, db_path)


def receive_stock_data(source_module: str, data: Union[pd.DataFrame, List[Dict], str],
                      mode: str = "append", callback: Optional[Callable] = None,
                      db_path: str = None) -> str:
    """接收股票数据的便捷函数"""
    manager = get_database_manager()
    return manager.receive_stock_data(source_module, data, mode, callback, db_path)


def receive_indicator_data(source_module: str, data: Union[pd.DataFrame, List[Dict], str],
                          table_name: str = "indicator_data", mode: str = "append",
                          callback: Optional[Callable] = None, db_path: str = None) -> str:
    """接收指标数据的便捷函数"""
    manager = get_database_manager()
    return manager.receive_indicator_data(source_module, data, table_name, mode, callback, db_path)


def receive_score_data(source_module: str, data: Union[pd.DataFrame, List[Dict], str],
                      table_name: str = "score_data", mode: str = "append",
                      callback: Optional[Callable] = None, db_path: str = None) -> str:
    """接收评分数据的便捷函数"""
    manager = get_database_manager()
    return manager.receive_score_data(source_module, data, table_name, mode, callback, db_path)

# ==================== Details数据库读取功能 ====================
def is_details_db_reading_enabled() -> bool:
    """
    检查details数据库读取是否启用
    
    优先从streamlit session_state读取，如果没有streamlit环境则返回False
    
    Returns:
        bool: 如果数据库读取已启用返回True，否则返回False
    """
    try:
        # 尝试从streamlit session_state读取
        import streamlit as st
        return st.session_state.get("details_db_reading_enabled", False)
    except (ImportError, AttributeError, RuntimeError):
        # 如果没有streamlit环境或不在streamlit上下文中，返回False
        return False


def get_details_db_path_with_fallback(db_path: Optional[str] = None, fallback_to_unified: bool = True) -> Optional[str]:
    """
    获取Details数据库路径，包含回退到统一数据库的逻辑
    
    Args:
        db_path: 指定数据库路径，如果为None则使用配置文件中的路径
        fallback_to_unified: 如果details数据库不存在，是否回退到统一数据库（默认True）
        
    Returns:
        数据库文件路径，如果不存在且不启用回退则返回None
    """
    # 首先尝试使用指定的路径或默认路径
    if db_path:
        details_db_path = _abs_norm(db_path)
    else:
        details_db_path = get_details_db_path()
    
    # 检查数据库文件是否存在
    if os.path.exists(details_db_path):
        return details_db_path
    
    # 如果不存在且启用回退，尝试从统一数据库读取（兼容性）
    if fallback_to_unified:
        try:
            from config import DATA_ROOT, UNIFIED_DB_PATH
            unified_db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            if os.path.exists(unified_db_path):
                logger.debug(f"Details数据库不存在，使用统一数据库: {unified_db_path}")
                return _abs_norm(unified_db_path)
        except (ImportError, AttributeError):
            pass
    
    # 如果都不存在，返回None
    logger.debug(f"Details数据库文件不存在: {details_db_path}")
    return None


def is_details_db_available(check_file_exists: bool = True) -> bool:
    """
    检查details数据库是否可用（配置支持且文件存在）
    
    Args:
        check_file_exists: 是否检查文件存在（默认True）
        
    Returns:
        bool: 如果数据库可用返回True，否则返回False
    """
    try:
        from config import SC_USE_DB_STORAGE, SC_DETAIL_STORAGE
        
        # 检查配置是否支持数据库存储
        if not SC_USE_DB_STORAGE:
            return False
        
        if SC_DETAIL_STORAGE not in ["database", "both", "db"]:
            return False
        
        # 如果不需要检查文件存在，仅检查配置即可
        if not check_file_exists:
            return True
        
        # 检查数据库文件是否存在
        db_path = get_details_db_path_with_fallback(fallback_to_unified=True)
        return db_path is not None and os.path.exists(db_path)
        
    except (ImportError, AttributeError):
        return False


def _handle_details_db_error(e: Exception, db_path: Optional[str] = None, operation: str = "操作") -> None:
    """
    统一的Details数据库错误处理函数
    
    Args:
        e: 异常对象
        db_path: 数据库路径（用于日志）
        operation: 操作名称（用于日志）
    """
    error_msg = str(e).lower()
    if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
        logger.debug(f"Details数据库表不存在: {db_path if db_path else 'unknown'}")
    elif isinstance(e, (FileNotFoundError, RuntimeError, AttributeError, ImportError)):
        logger.debug(f"{operation}失败: {e}")
    else:
        logger.error(f"{operation}失败: {e}")


def get_details_db_path(db_path: Optional[str] = None) -> str:
    """
    获取Details数据库路径
    
    Args:
        db_path: 指定数据库路径，如果为None则使用配置文件中的路径
        
    Returns:
        数据库文件路径
    """
    if db_path:
        return _abs_norm(db_path)
    
    try:
        from config import SC_OUTPUT_DIR, SC_DETAIL_DB_TYPE, SC_DETAIL_DB_PATH
        if SC_DETAIL_DB_TYPE == "duckdb":
            return _abs_norm(os.path.join(SC_OUTPUT_DIR, 'details', 'details.db'))
        else:
            return _abs_norm(os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH))
    except ImportError:
        # 如果无法导入配置，使用默认路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return _abs_norm(os.path.join(base_dir, 'output', 'score', 'details', 'details.db'))


def get_details_table_info(db_path: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    获取Details数据库表信息（优先使用状态文件缓存）
    
    Args:
        db_path: 数据库文件路径，如果为None则使用配置文件中的路径
        use_cache: 是否优先使用状态文件缓存，默认为True
        
    Returns:
        包含表结构、记录数、日期范围等信息的字典
    """
    try:
        db_path = get_details_db_path(db_path)
        
        if not os.path.exists(db_path):
            return {'error': f'数据库文件不存在: {db_path}'}
        
        manager = get_database_manager()
        
        # 优先从状态文件读取统计信息
        result = {
            'table_structure': [],
            'total_records': 0,
            'date_range': {},
            'stock_count': 0,
            'database_path': db_path
        }
        
        if use_cache:
            try:
                saved_status = manager.load_status_file()
                
                if saved_status and "details_data" in saved_status:
                    details_status = saved_status["details_data"]
                    # 验证数据库路径是否匹配
                    if details_status.get("database_path") == db_path:
                        # 从状态文件获取统计信息
                        result['total_records'] = details_status.get("total_records", 0)
                        result['stock_count'] = details_status.get("stock_count", 0)
                        
                        # 获取日期范围
                        min_date = details_status.get("min_date")
                        max_date = details_status.get("max_date")
                        if min_date or max_date:
                            date_range = {}
                            if min_date:
                                date_range['min_date'] = min_date
                            if max_date:
                                date_range['max_date'] = max_date
                            result['date_range'] = date_range
                        
                        logger.debug("从状态文件缓存获取Details表统计信息")
            except Exception as cache_error:
                logger.debug(f"从状态文件读取失败，回退到数据库查询: {cache_error}")
        
        # 表结构需要从数据库读取（状态文件不包含表结构信息）
        try:
            sql = "DESCRIBE stock_details"
            df = manager.execute_sync_query(db_path, sql, timeout=30.0)
            result['table_structure'] = df.to_dict('records') if not df.empty else []
        except Exception as e:
            _handle_details_db_error(e, db_path, "获取表结构")
            if 'table' not in str(e).lower() or 'does not exist' not in str(e).lower():
                logger.warning(f"获取表结构失败: {e}")
        
        # 如果缓存中没有统计信息，从数据库读取
        if result['total_records'] == 0 and result['stock_count'] == 0 and not result.get('date_range'):
            logger.debug("从数据库读取Details表统计信息")
            try:
                # 获取记录数
                count_sql = "SELECT COUNT(*) as total FROM stock_details"
                count_df = manager.execute_sync_query(db_path, count_sql, timeout=30.0)
                result['total_records'] = count_df.iloc[0]['total'] if not count_df.empty else 0
                
                # 获取日期范围
                date_sql = """
                SELECT 
                    MIN(ref_date) as min_date,
                    MAX(ref_date) as max_date,
                    COUNT(DISTINCT ref_date) as date_count
                FROM stock_details
                """
                date_df = manager.execute_sync_query(db_path, date_sql, timeout=30.0)
                if not date_df.empty:
                    result['date_range'] = date_df.to_dict('records')[0]
                
                # 获取股票数量
                stock_sql = "SELECT COUNT(DISTINCT ts_code) as stock_count FROM stock_details"
                stock_df = manager.execute_sync_query(db_path, stock_sql, timeout=30.0)
                result['stock_count'] = stock_df.iloc[0]['stock_count'] if not stock_df.empty else 0
            except Exception as e:
                _handle_details_db_error(e, db_path, "从数据库读取统计信息")
        
        return result
    except Exception as e:
        logger.error(f"get_details_table_info 失败: {e}")
        return {'error': str(e)}


def query_details_by_stock(ts_code: str, limit: int = 10, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    根据股票代码查询Details数据
    
    Args:
        ts_code: 股票代码
        limit: 返回记录数限制
        db_path: 数据库文件路径，如果为None则使用配置文件中的路径
        
    Returns:
        包含股票详情的DataFrame
    """
    try:
        db_path = get_details_db_path(db_path)
        
        # 检查数据库文件是否存在
        if not os.path.exists(db_path):
            logger.debug(f"Details数据库文件不存在: {db_path}")
            return pd.DataFrame()
        
        manager = get_database_manager()
        if not manager:
            logger.debug("无法获取数据库管理器")
            return pd.DataFrame()
        
        sql = """
        SELECT * FROM stock_details 
        WHERE ts_code = ? 
        ORDER BY ref_date DESC 
        LIMIT ?
        """
        return manager.execute_sync_query(db_path, sql, [ts_code, limit], timeout=30.0)
    except Exception as e:
        _handle_details_db_error(e, db_path, "query_details_by_stock")
        return pd.DataFrame()


def query_details_by_date(ref_date: str, limit: int = 100, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    根据日期查询Details数据
    
    Args:
        ref_date: 参考日期 (YYYYMMDD)
        limit: 返回记录数限制，-1表示返回全部记录
        db_path: 数据库文件路径，如果为None则使用配置文件中的路径
        
    Returns:
        包含该日期所有股票详情的DataFrame
    """
    try:
        db_path = get_details_db_path(db_path)
        
        # 检查数据库文件是否存在
        if not os.path.exists(db_path):
            logger.debug(f"Details数据库文件不存在: {db_path}")
            return pd.DataFrame()
        
        manager = get_database_manager()
        if not manager:
            logger.debug("无法获取数据库管理器")
            return pd.DataFrame()
        
        if limit == -1:
            # -1表示返回全部记录，不使用LIMIT
            sql = """
            SELECT * FROM stock_details 
            WHERE ref_date = ? 
            ORDER BY score DESC, rank ASC
            """
            return manager.execute_sync_query(db_path, sql, [ref_date], timeout=30.0)
        else:
            sql = """
            SELECT * FROM stock_details 
            WHERE ref_date = ? 
            ORDER BY score DESC, rank ASC
            LIMIT ?
            """
            return manager.execute_sync_query(db_path, sql, [ref_date, limit], timeout=30.0)
    except Exception as e:
        _handle_details_db_error(e, db_path, "query_details_by_date")
        return pd.DataFrame()


def query_details_top_stocks(ref_date: str, top_k: int = 50, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    查询指定日期的Top-K股票
    
    Args:
        ref_date: 参考日期 (YYYYMMDD)
        top_k: 返回前K名股票
        db_path: 数据库文件路径，如果为None则使用配置文件中的路径
        
    Returns:
        包含Top-K股票详情的DataFrame
    """
    try:
        db_path = get_details_db_path(db_path)
        
        # 检查数据库文件是否存在
        if not os.path.exists(db_path):
            logger.debug(f"Details数据库文件不存在: {db_path}")
            return pd.DataFrame()
        
        manager = get_database_manager()
        if not manager:
            logger.debug("无法获取数据库管理器")
            return pd.DataFrame()
        
        sql = """
        SELECT * FROM stock_details 
        WHERE ref_date = ? 
        ORDER BY score DESC, rank ASC
        LIMIT ?
        """
        return manager.execute_sync_query(db_path, sql, [ref_date, top_k], timeout=30.0)
    except (FileNotFoundError, RuntimeError, AttributeError, ImportError) as e:
        logger.debug(f"query_details_top_stocks 失败: {e}")
        return pd.DataFrame()
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['table', 'does not exist', 'no such table', 'catalog', 'relation']):
            logger.debug(f"Details数据库表不存在: {db_path}")
        else:
            logger.error(f"query_details_top_stocks 失败: {e}")
        return pd.DataFrame()


def query_details_score_range(ref_date: str, min_score: float, max_score: float, 
                              db_path: Optional[str] = None) -> pd.DataFrame:
    """
    查询指定分数范围的股票
    
    Args:
        ref_date: 参考日期 (YYYYMMDD)
        min_score: 最低分数
        max_score: 最高分数
        db_path: 数据库文件路径，如果为None则使用配置文件中的路径
        
    Returns:
        包含分数范围内股票详情的DataFrame
    """
    try:
        db_path = get_details_db_path(db_path)
        
        # 检查数据库文件是否存在
        if not os.path.exists(db_path):
            logger.debug(f"Details数据库文件不存在: {db_path}")
            return pd.DataFrame()
        
        manager = get_database_manager()
        if not manager:
            logger.debug("无法获取数据库管理器")
            return pd.DataFrame()
        
        sql = """
        SELECT * FROM stock_details 
        WHERE ref_date = ? AND score >= ? AND score <= ?
        ORDER BY score DESC, rank ASC
        """
        return manager.execute_sync_query(db_path, sql, [ref_date, min_score, max_score], timeout=30.0)
    except Exception as e:
        _handle_details_db_error(e, db_path, "query_details_score_range")
        return pd.DataFrame()


def query_details_recent_dates(days: int = 7, db_path: Optional[str] = None) -> List[str]:
    """
    查询最近的N个交易日
    
    Args:
        days: 查询最近几天
        db_path: 数据库文件路径，如果为None则使用配置文件中的路径
        
    Returns:
        最近N个交易日的日期列表
    """
    try:
        db_path = get_details_db_path(db_path)
        
        # 检查数据库文件是否存在
        if not os.path.exists(db_path):
            logger.debug(f"Details数据库文件不存在: {db_path}")
            return []
        
        logger.debug(f"[数据库连接] 开始获取数据库管理器实例 (查询最近{days}个交易日的details数据)")
        manager = get_database_manager()
        if not manager:
            logger.debug("无法获取数据库管理器")
            return []
        
        sql = """
        SELECT DISTINCT ref_date 
        FROM stock_details 
        ORDER BY ref_date DESC 
        LIMIT ?
        """
        df = manager.execute_sync_query(db_path, sql, [days], timeout=30.0)
        return df['ref_date'].tolist() if not df.empty else []
    except Exception as e:
        _handle_details_db_error(e, db_path, "query_details_recent_dates")
        return []


def get_details_stock_summary(ts_code: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取股票的历史评分摘要
    
    Args:
        ts_code: 股票代码
        db_path: 数据库文件路径，如果为None则使用配置文件中的路径
        
    Returns:
        包含股票历史评分摘要的字典
    """
    try:
        db_path = get_details_db_path(db_path)
        
        # 检查数据库文件是否存在
        if not os.path.exists(db_path):
            logger.debug(f"Details数据库文件不存在: {db_path}")
            return {}
        
        manager = get_database_manager()
        if not manager:
            logger.debug("无法获取数据库管理器")
            return {}
        
        sql = """
        SELECT 
            ts_code,
            COUNT(*) as total_days,
            MIN(ref_date) as first_date,
            MAX(ref_date) as last_date,
            AVG(score) as avg_score,
            MIN(score) as min_score,
            MAX(score) as max_score,
            AVG(rank) as avg_rank,
            MIN(rank) as best_rank,
            MAX(rank) as worst_rank
        FROM stock_details 
        WHERE ts_code = ?
        GROUP BY ts_code
        """
        df = manager.execute_sync_query(db_path, sql, [ts_code], timeout=30.0)
        return df.to_dict('records')[0] if not df.empty else {}
    except Exception as e:
        _handle_details_db_error(e, db_path, "get_details_stock_summary")
        return {}

# ========== 状态管理便捷函数（保持向后兼容，合并自 database_status） ==========
def generate_database_status(status_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    生成数据库状态文件（便捷函数）
    
    Args:
        status_file_path: 状态文件路径，如果为None则使用默认路径
        
    Returns:
        包含完整状态的字典
    """
    manager = get_database_manager()
    if status_file_path:
        manager.set_status_file_path(status_file_path)
    return manager.generate_status_file()


def check_database_status(status_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    检查数据库状态（便捷函数）
    
    Args:
        status_file_path: 状态文件路径，如果为None则使用默认路径
        
    Returns:
        包含检查结果的字典
    """
    manager = get_database_manager()
    if status_file_path:
        manager.set_status_file_path(status_file_path)
    return manager.check_status()


def update_database_status(status_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    更新数据库状态文件（便捷函数）
    
    Args:
        status_file_path: 状态文件路径，如果为None则使用默认路径
        
    Returns:
        更新后的状态字典
    """
    manager = get_database_manager()
    if status_file_path:
        manager.set_status_file_path(status_file_path)
    return manager.update_status_file()


def update_stock_data_status(status_file_path: Optional[str] = None) -> None:
    """
    更新股票数据状态（仅更新股票数据部分）
    
    Args:
        status_file_path: 状态文件路径，如果为None则使用默认路径
    """
    try:
        manager = get_database_manager()
        if status_file_path:
            manager.set_status_file_path(status_file_path)
        # 生成完整状态（包含股票数据和细节数据）
        status = manager.generate_status_file()
        logger.info("股票数据状态已更新")
    except Exception as e:
        logger.error(f"更新股票数据状态失败: {e}")


def update_details_data_status(status_file_path: Optional[str] = None) -> None:
    """
    更新细节数据状态（仅更新细节数据部分）
    
    Args:
        status_file_path: 状态文件路径，如果为None则使用默认路径
    """
    try:
        manager = get_database_manager()
        if status_file_path:
            manager.set_status_file_path(status_file_path)
        # 生成完整状态（包含股票数据和细节数据）
        status = manager.generate_status_file()
        logger.info("细节数据状态已更新")
    except Exception as e:
        logger.error(f"更新细节数据状态失败: {e}")


# 清理函数
import atexit
atexit.register(close_all_connections)
