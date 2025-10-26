#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Details 数据库读取工具
用于读取和查询股票评分详情数据库
"""

import os
import sys
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SC_OUTPUT_DIR, SC_DETAIL_DB_PATH, SC_DETAIL_DB_TYPE
from database_manager import get_database_manager

class DetailsDBReader:
    """Details 数据库读取器"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据库读取器
        
        Args:
            db_path: 数据库文件路径，如果为None则使用配置文件中的路径
        """
        if db_path is None:
            # 使用配置文件中的路径
            if SC_DETAIL_DB_TYPE == "duckdb":
                self.db_path = os.path.join(SC_OUTPUT_DIR, 'details', 'details.db')
            else:
                self.db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
        else:
            self.db_path = db_path
            
        self.manager = get_database_manager()
        
        # 检查数据库文件是否存在
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
    
    def get_table_info(self) -> Dict[str, Any]:
        """获取数据库表信息"""
        try:
            # 获取表结构
            sql = "DESCRIBE stock_details"
            df = self.manager.execute_sync_query(self.db_path, sql, timeout=30.0)
            
            # 获取记录数
            count_sql = "SELECT COUNT(*) as total FROM stock_details"
            count_df = self.manager.execute_sync_query(self.db_path, count_sql, timeout=30.0)
            total_count = count_df.iloc[0]['total'] if not count_df.empty else 0
            
            # 获取日期范围
            date_sql = """
            SELECT 
                MIN(ref_date) as min_date,
                MAX(ref_date) as max_date,
                COUNT(DISTINCT ref_date) as date_count
            FROM stock_details
            """
            date_df = self.manager.execute_sync_query(self.db_path, date_sql, timeout=30.0)
            
            # 获取股票数量
            stock_sql = "SELECT COUNT(DISTINCT ts_code) as stock_count FROM stock_details"
            stock_df = self.manager.execute_sync_query(self.db_path, stock_sql, timeout=30.0)
            
            return {
                'table_structure': df.to_dict('records') if not df.empty else [],
                'total_records': total_count,
                'date_range': date_df.to_dict('records')[0] if not date_df.empty else {},
                'stock_count': stock_df.iloc[0]['stock_count'] if not stock_df.empty else 0,
                'database_path': self.db_path
            }
        except Exception as e:
            return {'error': str(e)}
    
    def query_by_stock(self, ts_code: str, limit: int = 10) -> pd.DataFrame:
        """
        根据股票代码查询详情
        
        Args:
            ts_code: 股票代码
            limit: 返回记录数限制
            
        Returns:
            包含股票详情的DataFrame
        """
        sql = """
        SELECT * FROM stock_details 
        WHERE ts_code = ? 
        ORDER BY ref_date DESC 
        LIMIT ?
        """
        return self.manager.execute_sync_query(self.db_path, sql, [ts_code, limit], timeout=30.0)
    
    def query_by_date(self, ref_date: str, limit: int = 100) -> pd.DataFrame:
        """
        根据日期查询详情
        
        Args:
            ref_date: 参考日期 (YYYYMMDD)
            limit: 返回记录数限制
            
        Returns:
            包含该日期所有股票详情的DataFrame
        """
        sql = """
        SELECT * FROM stock_details 
        WHERE ref_date = ? 
        ORDER BY score DESC, rank ASC
        LIMIT ?
        """
        return self.manager.execute_sync_query(self.db_path, sql, [ref_date, limit], timeout=30.0)
    
    def query_top_stocks(self, ref_date: str, top_k: int = 50) -> pd.DataFrame:
        """
        查询指定日期的Top-K股票
        
        Args:
            ref_date: 参考日期 (YYYYMMDD)
            top_k: 返回前K名股票
            
        Returns:
            包含Top-K股票详情的DataFrame
        """
        sql = """
        SELECT * FROM stock_details 
        WHERE ref_date = ? 
        ORDER BY score DESC, rank ASC
        LIMIT ?
        """
        return self.manager.execute_sync_query(self.db_path, sql, [ref_date, top_k], timeout=30.0)
    
    def query_score_range(self, ref_date: str, min_score: float, max_score: float) -> pd.DataFrame:
        """
        查询指定分数范围的股票
        
        Args:
            ref_date: 参考日期 (YYYYMMDD)
            min_score: 最低分数
            max_score: 最高分数
            
        Returns:
            包含分数范围内股票详情的DataFrame
        """
        sql = """
        SELECT * FROM stock_details 
        WHERE ref_date = ? AND score >= ? AND score <= ?
        ORDER BY score DESC, rank ASC
        """
        return self.manager.execute_sync_query(self.db_path, sql, [ref_date, min_score, max_score], timeout=30.0)
    
    def query_recent_dates(self, days: int = 7) -> List[str]:
        """
        查询最近的N个交易日
        
        Args:
            days: 查询最近几天
            
        Returns:
            最近N个交易日的日期列表
        """
        sql = """
        SELECT DISTINCT ref_date 
        FROM stock_details 
        ORDER BY ref_date DESC 
        LIMIT ?
        """
        df = self.manager.execute_sync_query(self.db_path, sql, [days], timeout=30.0)
        return df['ref_date'].tolist() if not df.empty else []
    
    def get_stock_summary(self, ts_code: str) -> Dict[str, Any]:
        """
        获取股票的历史评分摘要
        
        Args:
            ts_code: 股票代码
            
        Returns:
            包含股票历史评分摘要的字典
        """
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
        df = self.manager.execute_sync_query(self.db_path, sql, [ts_code], timeout=30.0)
        return df.to_dict('records')[0] if not df.empty else {}
    
    def export_to_csv(self, output_file: str, ref_date: str = None, limit: int = None):
        """
        导出数据到CSV文件
        
        Args:
            output_file: 输出文件路径
            ref_date: 指定日期，如果为None则导出所有数据
            limit: 限制记录数，如果为None则导出所有记录
        """
        if ref_date:
            sql = "SELECT * FROM stock_details WHERE ref_date = ? ORDER BY score DESC, rank ASC"
            params = [ref_date]
        else:
            sql = "SELECT * FROM stock_details ORDER BY ref_date DESC, score DESC, rank ASC"
            params = []
        
        if limit:
            sql += f" LIMIT {limit}"
        
        df = self.manager.execute_sync_query(self.db_path, sql, params, timeout=60.0)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"数据已导出到: {output_file}")
        print(f"共导出 {len(df)} 条记录")

def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='Details 数据库读取工具')
    parser.add_argument('--db-path', help='数据库文件路径')
    parser.add_argument('--info', action='store_true', help='显示数据库信息')
    parser.add_argument('--stock', help='查询指定股票代码')
    parser.add_argument('--date', help='查询指定日期 (YYYYMMDD)')
    parser.add_argument('--top', type=int, help='查询Top-K股票')
    parser.add_argument('--score-min', type=float, help='最低分数')
    parser.add_argument('--score-max', type=float, help='最高分数')
    parser.add_argument('--recent', type=int, help='查询最近N个交易日')
    parser.add_argument('--summary', help='获取股票历史摘要')
    parser.add_argument('--export', help='导出到CSV文件')
    parser.add_argument('--limit', type=int, default=20, help='限制返回记录数')
    
    args = parser.parse_args()
    
    try:
        # 初始化读取器
        reader = DetailsDBReader(args.db_path)
        
        # 显示数据库信息
        if args.info:
            info = reader.get_table_info()
            print("=== 数据库信息 ===")
            print(f"数据库路径: {info.get('database_path', 'N/A')}")
            print(f"总记录数: {info.get('total_records', 'N/A')}")
            print(f"股票数量: {info.get('stock_count', 'N/A')}")
            if 'date_range' in info:
                date_info = info['date_range']
                print(f"日期范围: {date_info.get('min_date', 'N/A')} ~ {date_info.get('max_date', 'N/A')}")
                print(f"交易日数: {date_info.get('date_count', 'N/A')}")
            
            print("\n=== 表结构 ===")
            for col in info.get('table_structure', []):
                print(f"{col.get('column_name', 'N/A')}: {col.get('column_type', 'N/A')}")
            return
        
        # 查询最近交易日
        if args.recent:
            dates = reader.query_recent_dates(args.recent)
            print(f"最近 {args.recent} 个交易日:")
            for date in dates:
                print(f"  {date}")
            return
        
        # 查询指定股票
        if args.stock:
            df = reader.query_by_stock(args.stock, args.limit)
            if not df.empty:
                print(f"=== 股票 {args.stock} 的评分详情 ===")
                print(df[['ref_date', 'score', 'rank', 'total', 'tiebreak']].to_string(index=False))
            else:
                print(f"未找到股票 {args.stock} 的数据")
            return
        
        # 查询指定日期
        if args.date:
            if args.top:
                df = reader.query_top_stocks(args.date, args.top)
                print(f"=== {args.date} Top-{args.top} 股票 ===")
            elif args.score_min is not None and args.score_max is not None:
                df = reader.query_score_range(args.date, args.score_min, args.score_max)
                print(f"=== {args.date} 分数范围 {args.score_min}-{args.score_max} 的股票 ===")
            else:
                df = reader.query_by_date(args.date, args.limit)
                print(f"=== {args.date} 所有股票评分 ===")
            
            if not df.empty:
                print(df[['ts_code', 'score', 'rank', 'total', 'tiebreak']].to_string(index=False))
            else:
                print(f"未找到 {args.date} 的数据")
            return
        
        # 获取股票摘要
        if args.summary:
            summary = reader.get_stock_summary(args.summary)
            if summary:
                print(f"=== 股票 {args.summary} 历史摘要 ===")
                for key, value in summary.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
            else:
                print(f"未找到股票 {args.summary} 的数据")
            return
        
        # 导出数据
        if args.export:
            reader.export_to_csv(args.export, args.date, args.limit)
            return
        
        # 默认显示帮助
        parser.print_help()
        
    except Exception as e:
        print(f"错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
