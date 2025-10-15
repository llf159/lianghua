#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个股详情数据库存储模块
支持SQLite和DuckDB两种存储方式
"""

import sqlite3
import json
import os
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)

class DetailDB:
    """个股详情数据库存储类"""
    
    def __init__(self, db_path: str = "details.db", db_type: str = "sqlite"):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径
            db_type: 数据库类型 ("sqlite" 或 "duckdb")
        """
        self.db_path = db_path
        self.db_type = db_type.lower()
        self._local = threading.local()  # 线程本地存储
        self._init_tables()
    
    def _get_connection(self):
        """获取线程本地的数据库连接"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            try:
                if self.db_type == "sqlite":
                    self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    self._local.conn.row_factory = sqlite3.Row  # 支持字典式访问
                elif self.db_type == "duckdb":
                    import duckdb
                    self._local.conn = duckdb.connect(self.db_path)
                else:
                    raise ValueError(f"不支持的数据库类型: {self.db_type}")
                
                LOGGER.debug(f"数据库连接成功: {self.db_path} ({self.db_type})")
            except Exception as e:
                LOGGER.error(f"数据库连接失败: {e}")
                raise
        
        return self._local.conn
    
    def _init_tables(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        if self.db_type == "sqlite":
            self._init_sqlite_tables(conn)
        elif self.db_type == "duckdb":
            self._init_duckdb_tables(conn)
    
    def _init_sqlite_tables(self, conn):
        """初始化SQLite表结构"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_details (
            ts_code TEXT NOT NULL,
            ref_date TEXT NOT NULL,
            score REAL,
            tiebreak REAL,
            highlights TEXT,
            drawbacks TEXT,
            opportunities TEXT,
            rank INTEGER,
            total INTEGER,
            rules TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ts_code, ref_date)
        )
        """
        
        create_index_sql = [
            "CREATE INDEX IF NOT EXISTS idx_ts_code ON stock_details(ts_code)",
            "CREATE INDEX IF NOT EXISTS idx_ref_date ON stock_details(ref_date)",
            "CREATE INDEX IF NOT EXISTS idx_score ON stock_details(score)",
            "CREATE INDEX IF NOT EXISTS idx_rank ON stock_details(rank)"
        ]
        
        try:
            conn.execute(create_table_sql)
            for idx_sql in create_index_sql:
                conn.execute(idx_sql)
            conn.commit()
            LOGGER.debug("SQLite表结构初始化完成")
        except Exception as e:
            LOGGER.error(f"SQLite表初始化失败: {e}")
            raise
    
    def _init_duckdb_tables(self, conn):
        """初始化DuckDB表结构"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_details (
            ts_code VARCHAR,
            ref_date VARCHAR,
            score DOUBLE,
            tiebreak DOUBLE,
            highlights VARCHAR,
            drawbacks VARCHAR,
            opportunities VARCHAR,
            rank INTEGER,
            total INTEGER,
            rules VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ts_code, ref_date)
        )
        """
        
        try:
            conn.execute(create_table_sql)
            LOGGER.debug("DuckDB表结构初始化完成")
        except Exception as e:
            LOGGER.error(f"DuckDB表初始化失败: {e}")
            raise
    
    def save_detail(self, ts_code: str, ref_date: str, summary: Dict, rules: List[Dict]) -> bool:
        """
        保存个股详情到数据库
        
        Args:
            ts_code: 股票代码
            ref_date: 参考日期
            summary: 摘要信息
            rules: 规则详情列表
            
        Returns:
            bool: 保存是否成功
        """
        try:
            conn = self._get_connection()
            
            # 提取summary字段
            score = summary.get("score", 0.0)
            tiebreak = summary.get("tiebreak")
            highlights = json.dumps(summary.get("highlights", []), ensure_ascii=False)
            drawbacks = json.dumps(summary.get("drawbacks", []), ensure_ascii=False)
            opportunities = json.dumps(summary.get("opportunities", []), ensure_ascii=False)
            rank = summary.get("rank")
            total = summary.get("total")
            
            # 序列化rules
            rules_json = json.dumps(rules, ensure_ascii=False)
            
            if self.db_type == "sqlite":
                sql = """
                INSERT OR REPLACE INTO stock_details 
                (ts_code, ref_date, score, tiebreak, highlights, drawbacks, opportunities, 
                 rank, total, rules, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """
                conn.execute(sql, (
                    ts_code, ref_date, score, tiebreak, highlights, drawbacks, 
                    opportunities, rank, total, rules_json
                ))
            else:  # duckdb
                sql = """
                INSERT OR REPLACE INTO stock_details 
                (ts_code, ref_date, score, tiebreak, highlights, drawbacks, opportunities, 
                 rank, total, rules, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """
                conn.execute(sql, [
                    ts_code, ref_date, score, tiebreak, highlights, drawbacks, 
                    opportunities, rank, total, rules_json
                ])
            
            conn.commit()
            LOGGER.debug(f"保存详情成功: {ts_code}_{ref_date}")
            return True
            
        except Exception as e:
            LOGGER.error(f"保存详情失败 {ts_code}_{ref_date}: {e}")
            return False
    
    def load_detail(self, ts_code: str, ref_date: str) -> Optional[Dict]:
        """
        从数据库加载个股详情
        
        Args:
            ts_code: 股票代码
            ref_date: 参考日期
            
        Returns:
            Dict: 详情数据，格式与JSON文件相同
        """
        try:
            conn = self._get_connection()
            sql = "SELECT * FROM stock_details WHERE ts_code = ? AND ref_date = ?"
            
            if self.db_type == "sqlite":
                cursor = conn.execute(sql, (ts_code, ref_date))
                row = cursor.fetchone()
                if not row:
                    return None
                
                # 转换为字典
                data = dict(row)
            else:  # duckdb
                result = conn.execute(sql, [ts_code, ref_date]).fetchone()
                if not result:
                    return None
                
                # 获取列名
                columns = [desc[0] for desc in conn.description]
                data = dict(zip(columns, result))
            
            # 重构为原始JSON格式
            summary = {
                "score": data.get("score", 0.0),
                "tiebreak": data.get("tiebreak"),
                "highlights": json.loads(data.get("highlights", "[]")),
                "drawbacks": json.loads(data.get("drawbacks", "[]")),
                "opportunities": json.loads(data.get("opportunities", "[]")),
                "rank": data.get("rank"),
                "total": data.get("total")
            }
            
            rules = json.loads(data.get("rules", "[]"))
            
            return {
                "ts_code": data["ts_code"],
                "ref_date": data["ref_date"],
                "summary": summary,
                "rules": rules
            }
            
        except Exception as e:
            LOGGER.error(f"加载详情失败 {ts_code}_{ref_date}: {e}")
            return None
    
    def batch_update_ranks(self, ref_date: str, scored_sorted: List[Any]) -> bool:
        """
        批量更新排名信息
        
        Args:
            ref_date: 参考日期
            scored_sorted: 已排序的股票列表，包含ts_code属性
            
        Returns:
            bool: 更新是否成功
        """
        try:
            conn = self._get_connection()
            total_n = len(scored_sorted)
            
            if self.db_type == "sqlite":
                # 使用事务批量更新
                conn.execute("BEGIN TRANSACTION")
                for i, item in enumerate(scored_sorted):
                    ts_code = getattr(item, 'ts_code', str(item))
                    rank = i + 1
                    sql = """
                    UPDATE stock_details 
                    SET rank = ?, total = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE ts_code = ? AND ref_date = ?
                    """
                    conn.execute(sql, (rank, total_n, ts_code, ref_date))
                conn.execute("COMMIT")
            else:  # duckdb
                # DuckDB批量更新
                for i, item in enumerate(scored_sorted):
                    ts_code = getattr(item, 'ts_code', str(item))
                    rank = i + 1
                    sql = """
                    UPDATE stock_details 
                    SET rank = ?, total = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE ts_code = ? AND ref_date = ?
                    """
                    conn.execute(sql, [rank, total_n, ts_code, ref_date])
            
            LOGGER.debug(f"批量更新排名完成: {ref_date}, 共{total_n}只股票")
            return True
            
        except Exception as e:
            LOGGER.error(f"批量更新排名失败: {e}")
            if self.db_type == "sqlite":
                conn.execute("ROLLBACK")
            return False
    
    def query_by_date(self, ref_date: str) -> pd.DataFrame:
        """
        查询指定日期的所有股票详情
        
        Args:
            ref_date: 参考日期
            
        Returns:
            pd.DataFrame: 详情数据框
        """
        try:
            conn = self._get_connection()
            sql = "SELECT * FROM stock_details WHERE ref_date = ? ORDER BY score DESC"
            
            if self.db_type == "sqlite":
                df = pd.read_sql_query(sql, conn, params=(ref_date,))
            else:  # duckdb
                result = conn.execute(sql, [ref_date]).fetchall()
                columns = [desc[0] for desc in conn.description]
                df = pd.DataFrame(result, columns=columns)
            
            return df
            
        except Exception as e:
            LOGGER.error(f"查询日期数据失败 {ref_date}: {e}")
            return pd.DataFrame()
    
    def query_by_codes(self, ts_codes: List[str], ref_date: str) -> pd.DataFrame:
        """
        查询指定股票代码列表的详情
        
        Args:
            ts_codes: 股票代码列表
            ref_date: 参考日期
            
        Returns:
            pd.DataFrame: 详情数据框
        """
        try:
            if not ts_codes:
                return pd.DataFrame()
            
            conn = self._get_connection()
            placeholders = ",".join(["?" for _ in ts_codes])
            sql = f"""
            SELECT * FROM stock_details 
            WHERE ref_date = ? AND ts_code IN ({placeholders})
            ORDER BY score DESC
            """
            
            params = [ref_date] + ts_codes
            
            if self.db_type == "sqlite":
                df = pd.read_sql_query(sql, conn, params=params)
            else:  # duckdb
                result = conn.execute(sql, params).fetchall()
                columns = [desc[0] for desc in conn.description]
                df = pd.DataFrame(result, columns=columns)
            
            return df
            
        except Exception as e:
            LOGGER.error(f"查询股票数据失败: {e}")
            return pd.DataFrame()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            conn = self._get_connection()
            stats = {}
            
            # 总记录数
            if self.db_type == "sqlite":
                cursor = conn.execute("SELECT COUNT(*) FROM stock_details")
                stats["total_records"] = cursor.fetchone()[0]
                
                # 按日期统计
                cursor = conn.execute("""
                    SELECT ref_date, COUNT(*) as count 
                    FROM stock_details 
                    GROUP BY ref_date 
                    ORDER BY ref_date DESC
                """)
                stats["by_date"] = dict(cursor.fetchall())
                
            else:  # duckdb
                result = conn.execute("SELECT COUNT(*) FROM stock_details").fetchone()
                stats["total_records"] = result[0]
                
                result = conn.execute("""
                    SELECT ref_date, COUNT(*) as count 
                    FROM stock_details 
                    GROUP BY ref_date 
                    ORDER BY ref_date DESC
                """).fetchall()
                stats["by_date"] = dict(result)
            
            return stats
            
        except Exception as e:
            LOGGER.error(f"获取统计信息失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
            LOGGER.debug("数据库连接已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 全局数据库实例
_detail_db = None

def get_detail_db() -> DetailDB:
    """获取全局数据库实例"""
    global _detail_db
    if _detail_db is None:
        from config import SC_OUTPUT_DIR
        db_path = os.path.join(SC_OUTPUT_DIR, "details.db")
        _detail_db = DetailDB(db_path)
    return _detail_db

def close_detail_db():
    """关闭全局数据库连接"""
    global _detail_db
    if _detail_db:
        _detail_db.close()
        _detail_db = None
