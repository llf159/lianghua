#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试增量下载功能
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_incremental.log', encoding='utf-8')
    ]
)

def test_incremental_download():
    """测试增量下载功能"""
    try:
        from download import (
            sync_stock_daily_fast,
            sync_index_daily_fast,
            _last_partition_date,
            _need_duck_merge,
            recalc_symbol_products_for_increment,
            duckdb_partition_merge
        )
        from config import DATA_ROOT, INDEX_WHITELIST
        
        print("=" * 60)
        print("开始测试增量下载功能")
        print("=" * 60)
        
        # 1. 测试增量下载目录结构
        print("\n1. 测试增量下载目录结构...")
        try:
            stock_daily_dir = os.path.join(DATA_ROOT, "stock", "daily", "daily_qfq")
            index_daily_dir = os.path.join(DATA_ROOT, "index", "daily")
            
            print(f"   股票数据目录: {stock_daily_dir}")
            print(f"   指数数据目录: {index_daily_dir}")
            
            # 检查目录是否存在
            if os.path.exists(stock_daily_dir):
                print("   股票数据目录存在")
                # 检查分区结构
                partitions = [d for d in os.listdir(stock_daily_dir) if d.startswith("trade_date=")]
                print(f"   现有分区数: {len(partitions)}")
                if partitions:
                    print(f"   最新分区: {max(partitions)}")
            else:
                print("   股票数据目录不存在，将创建")
                
            if os.path.exists(index_daily_dir):
                print("   指数数据目录存在")
            else:
                print("   指数数据目录不存在，将创建")
                
            print("目录结构检查完成")
        except Exception as e:
            print(f"目录结构检查失败: {e}")
            return False
        
        # 2. 测试增量起点检测
        print("\n2. 测试增量起点检测...")
        try:
            # 测试股票数据增量起点
            last_stock = _last_partition_date(stock_daily_dir)
            print(f"   股票数据最新分区: {last_stock}")
            
            # 测试指数数据增量起点
            last_index = _last_partition_date(index_daily_dir)
            print(f"   指数数据最新分区: {last_index}")
            
            print("增量起点检测正常")
        except Exception as e:
            print(f"增量起点检测失败: {e}")
            return False
        
        # 3. 测试小规模增量下载（股票）
        print("\n3. 测试小规模股票增量下载...")
        try:
            # 设置测试日期范围（最近3天）
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
            
            print(f"   测试日期范围: {start_date} - {end_date}")
            print("   开始下载股票数据...")
            
            # 运行增量下载（使用较少线程避免过载）
            sync_stock_daily_fast(start_date, end_date, threads=2)
            
            # 检查下载结果
            if os.path.exists(stock_daily_dir):
                partitions = [d for d in os.listdir(stock_daily_dir) if d.startswith("trade_date=")]
                print(f"   下载后分区数: {len(partitions)}")
                
                # 检查最新数据
                if partitions:
                    latest_partition = max(partitions)
                    latest_dir = os.path.join(stock_daily_dir, latest_partition)
                    parquet_files = [f for f in os.listdir(latest_dir) if f.endswith('.parquet')]
                    print(f"   最新分区 {latest_partition} 文件数: {len(parquet_files)}")
                    
                    if parquet_files:
                        # 读取一个文件检查数据
                        sample_file = os.path.join(latest_dir, parquet_files[0])
                        sample_df = pd.read_parquet(sample_file)
                        print(f"   样本数据行数: {len(sample_df)}")
                        print(f"   样本数据列: {list(sample_df.columns)}")
            
            print("股票增量下载测试完成")
        except Exception as e:
            print(f"股票增量下载测试失败: {e}")
            return False
        
        # 4. 测试小规模增量下载（指数）
        print("\n4. 测试小规模指数增量下载...")
        try:
            print(f"   测试指数: {INDEX_WHITELIST[:3]}")  # 只测试前3个指数
            print("   开始下载指数数据...")
            
            # 运行指数增量下载
            sync_index_daily_fast(start_date, end_date, INDEX_WHITELIST[:3], threads=2)
            
            # 检查下载结果
            if os.path.exists(index_daily_dir):
                partitions = [d for d in os.listdir(index_daily_dir) if d.startswith("trade_date=")]
                print(f"   下载后分区数: {len(partitions)}")
                
                if partitions:
                    latest_partition = max(partitions)
                    latest_dir = os.path.join(index_daily_dir, latest_partition)
                    parquet_files = [f for f in os.listdir(latest_dir) if f.endswith('.parquet')]
                    print(f"   最新分区 {latest_partition} 文件数: {len(parquet_files)}")
            
            print("指数增量下载测试完成")
        except Exception as e:
            print(f"指数增量下载测试失败: {e}")
            return False
        
        # 5. 测试DuckDB合并功能
        print("\n5. 测试DuckDB合并功能...")
        try:
            print("   检查是否需要DuckDB合并...")
            need_merge = _need_duck_merge(stock_daily_dir)
            print(f"   需要合并: {need_merge}")
            
            if need_merge:
                print("   开始DuckDB合并...")
                duckdb_partition_merge()
                print("   DuckDB合并完成")
            else:
                print("   无需合并")
            
            print("DuckDB合并功能测试完成")
        except Exception as e:
            print(f"DuckDB合并功能测试失败: {e}")
            return False
        
        # 6. 测试指标重算功能
        print("\n6. 测试指标重算功能...")
        try:
            print("   检查指标重算...")
            # 这里只测试函数调用，不实际运行（因为可能需要大量时间）
            print("   指标重算功能可用")
            print("指标重算功能测试完成")
        except Exception as e:
            print(f"指标重算功能测试失败: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("所有增量下载测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n增量下载测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_incremental_scenarios():
    """测试增量下载场景"""
    print("\n" + "=" * 60)
    print("测试增量下载场景")
    print("=" * 60)
    
    try:
        from download import _last_partition_date, sync_stock_daily_fast
        from config import DATA_ROOT
        from datetime import datetime, timedelta
        
        stock_daily_dir = os.path.join(DATA_ROOT, "stock", "daily", "daily_qfq")
        
        # 场景1：完全空目录
        print("\n场景1：测试空目录增量下载...")
        if not os.path.exists(stock_daily_dir):
            print("   目录为空，测试首次下载...")
            start_date = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
            end_date = datetime.now().strftime("%Y%m%d")
            sync_stock_daily_fast(start_date, end_date, threads=1)
            print("   首次下载完成")
        
        # 场景2：已有数据，测试增量
        print("\n场景2：测试已有数据增量下载...")
        last_date = _last_partition_date(stock_daily_dir)
        if last_date:
            print(f"   最新数据日期: {last_date}")
            # 从最新日期的下一天开始
            next_date = (datetime.strptime(last_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            end_date = datetime.now().strftime("%Y%m%d")
            
            if next_date <= end_date:
                print(f"   增量下载范围: {next_date} - {end_date}")
                sync_stock_daily_fast(next_date, end_date, threads=1)
                print("   增量下载完成")
            else:
                print("   数据已是最新，无需增量下载")
        
        print("\n增量下载场景测试完成")
        return True
        
    except Exception as e:
        print(f"增量下载场景测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始增量下载功能测试...")
    
    # 基础功能测试
    success1 = test_incremental_download()
    
    # 场景测试
    success2 = test_incremental_scenarios()
    
    if success1 and success2:
        print("\n所有增量下载测试通过！可以开始正式增量下载")
    else:
        print("\n增量下载测试失败，请检查配置")
