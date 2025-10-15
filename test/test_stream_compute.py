#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试边下载边计算指标功能
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
        logging.FileHandler('test_stream_compute.log', encoding='utf-8')
    ]
)

def test_stream_compute_config():
    """测试边下载边计算配置"""
    print("=" * 60)
    print("测试边下载边计算配置")
    print("=" * 60)
    
    try:
        from config import (
            INC_STREAM_COMPUTE_INDICATORS,
            INC_STREAM_UPDATE_FAST_CACHE,
            INC_STREAM_MERGE_IND_SUBSET,
            WRITE_SYMBOL_INDICATORS,
            API_ADJ
        )
        
        print(f"边下载边计算指标: {INC_STREAM_COMPUTE_INDICATORS}")
        print(f"边下载边更新缓存: {INC_STREAM_UPDATE_FAST_CACHE}")
        print(f"边下载边合并分区: {INC_STREAM_MERGE_IND_SUBSET}")
        print(f"写入指标数据: {WRITE_SYMBOL_INDICATORS}")
        print(f"API复权类型: {API_ADJ}")
        
        if INC_STREAM_COMPUTE_INDICATORS and WRITE_SYMBOL_INDICATORS:
            print("边下载边计算功能已启用")
            return True
        else:
            print("边下载边计算功能未启用")
            return False
            
    except Exception as e:
        print(f"配置检查失败: {e}")
        return False

def test_stream_compute_functions():
    """测试边下载边计算相关函数"""
    print("\n" + "=" * 60)
    print("测试边下载边计算相关函数")
    print("=" * 60)
    
    try:
        from download import (
            _WRITE_SYMBOL_INDICATORS,
            _update_fast_init_cache,
            _decide_symbol_adj_for_fast_init,
            duckdb_merge_symbol_products_to_daily_subset,
            _with_api_adj
        )
        
        print("所有边下载边计算相关函数导入成功")
        
        # 测试复权类型决策
        adj_type = _decide_symbol_adj_for_fast_init()
        print(f"   复权类型决策: {adj_type}")
        
        return True
        
    except Exception as e:
        print(f"函数导入失败: {e}")
        return False

def test_stream_compute_with_real_data():
    """使用真实数据测试边下载边计算"""
    print("\n" + "=" * 60)
    print("使用真实数据测试边下载边计算")
    print("=" * 60)
    
    try:
        from download import (
            sync_stock_daily_fast,
            _fetch_stock_list,
            _clean_and_validate_dataframe
        )
        from config import DATA_ROOT
        
        # 1. 获取测试股票
        print("1. 获取测试股票...")
        stocks = _fetch_stock_list()
        test_codes = stocks.head(2)['ts_code'].tolist()  # 只测试2只股票
        print(f"   测试股票: {test_codes}")
        
        # 2. 设置测试日期（最近2天）
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
        print(f"   测试日期: {start_date} - {end_date}")
        
        # 3. 检查指标输出目录
        print("2. 检查指标输出目录...")
        adj_type = "qfq"  # 从配置中获取
        single_ind_dir = os.path.join(DATA_ROOT, "stock", "single", f"single_{adj_type}_indicators")
        daily_ind_dir = os.path.join(DATA_ROOT, "stock", "daily", f"daily_{adj_type}_indicators")
        
        print(f"   单股指标目录: {single_ind_dir}")
        print(f"   分区指标目录: {daily_ind_dir}")
        
        # 4. 运行边下载边计算
        print("3. 运行边下载边计算...")
        print("   开始下载（包含边下载边计算）...")
        
        # 使用较少线程避免过载
        sync_stock_daily_fast(start_date, end_date, threads=1)
        
        print("   下载完成")
        
        # 5. 检查结果
        print("4. 检查计算结果...")
        
        # 检查单股指标文件
        if os.path.exists(single_ind_dir):
            ind_files = [f for f in os.listdir(single_ind_dir) if f.endswith('.parquet')]
            print(f"   单股指标文件数: {len(ind_files)}")
            
            # 检查测试股票是否有指标文件
            for code in test_codes:
                ind_file = os.path.join(single_ind_dir, f"{code}.parquet")
                if os.path.exists(ind_file):
                    print(f"   {code} 指标文件存在")
                    # 读取并检查指标数据
                    try:
                        df_ind = pd.read_parquet(ind_file)
                        print(f"      指标数据行数: {len(df_ind)}")
                        print(f"      指标数据列: {list(df_ind.columns)}")
                        
                        # 检查是否有指标列（非基础OHLCV列）
                        base_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
                        indicator_cols = [col for col in df_ind.columns if col not in base_cols]
                        print(f"      指标列数: {len(indicator_cols)}")
                        if indicator_cols:
                            print(f"      指标列示例: {indicator_cols[:5]}")
                    except Exception as e:
                        print(f"      读取指标文件失败: {e}")
                else:
                    print(f"   {code} 指标文件不存在")
        else:
            print("   单股指标目录不存在")
        
        # 检查分区指标文件
        if os.path.exists(daily_ind_dir):
            partitions = [d for d in os.listdir(daily_ind_dir) if d.startswith("trade_date=")]
            print(f"   分区指标目录数: {len(partitions)}")
            if partitions:
                latest_partition = max(partitions)
                latest_dir = os.path.join(daily_ind_dir, latest_partition)
                ind_files = [f for f in os.listdir(latest_dir) if f.endswith('.parquet')]
                print(f"   最新分区 {latest_partition} 指标文件数: {len(ind_files)}")
        else:
            print("   分区指标目录不存在")
        
        print("边下载边计算测试完成")
        return True
        
    except Exception as e:
        print(f"边下载边计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_computation():
    """测试指标计算功能"""
    print("\n" + "=" * 60)
    print("测试指标计算功能")
    print("=" * 60)
    
    try:
        from download import _WRITE_SYMBOL_INDICATORS
        from config import SYMBOL_PRODUCT_INDICATORS
        import pandas as pd
        import numpy as np
        
        print(f"配置的指标: {SYMBOL_PRODUCT_INDICATORS}")
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 10,
            'trade_date': [f'2024010{i}' for i in range(1, 11)],
            'open': np.random.uniform(10, 12, 10),
            'high': np.random.uniform(11, 13, 10),
            'low': np.random.uniform(9, 11, 10),
            'close': np.random.uniform(10, 12, 10),
            'vol': np.random.uniform(1000, 2000, 10),
            'amount': np.random.uniform(10000, 20000, 10)
        })
        
        print(f"   测试数据: {len(test_data)} 行")
        
        # 测试指标计算
        print("   开始计算指标...")
        _WRITE_SYMBOL_INDICATORS("000001.SZ", test_data, "20240110")
        print("   指标计算完成")
        
        return True
        
    except Exception as e:
        print(f"指标计算测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始边下载边计算功能测试...")
    
    # 1. 配置检查
    config_ok = test_stream_compute_config()
    
    # 2. 函数检查
    func_ok = test_stream_compute_functions()
    
    # 3. 指标计算测试
    indicator_ok = test_indicator_computation()
    
    # 4. 真实数据测试
    if config_ok and func_ok:
        real_data_ok = test_stream_compute_with_real_data()
    else:
        real_data_ok = False
    
    print("\n" + "=" * 60)
    print("边下载边计算功能测试结果")
    print("=" * 60)
    print(f"配置检查: {'通过' if config_ok else '失败'}")
    print(f"函数检查: {'通过' if func_ok else '失败'}")
    print(f"指标计算: {'通过' if indicator_ok else '失败'}")
    print(f"真实数据: {'通过' if real_data_ok else '失败'}")
    
    if all([config_ok, func_ok, indicator_ok, real_data_ok]):
        print("\n所有边下载边计算功能测试通过！")
    else:
        print("\n部分测试失败，请检查配置和实现")
