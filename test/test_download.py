#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试数据下载功能
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_download.log', encoding='utf-8')
    ]
)

def test_download():
    """测试下载功能"""
    try:
        # 导入下载模块
        from download import (
            _require_pro, 
            _clean_and_validate_dataframe,
            _safe_write_parquet,
            _fetch_stock_list,
            _rate_limit,
            _retry
        )
        
        print("=" * 60)
        print("开始测试数据下载功能")
        print("=" * 60)
        
        # 1. 测试Token初始化
        print("\n1. 测试Token初始化...")
        try:
            pro = _require_pro()
            print("Token初始化成功")
        except Exception as e:
            print(f"Token初始化失败: {e}")
            return False
        
        # 2. 测试限频机制
        print("\n2. 测试限频机制...")
        try:
            _rate_limit()
            print("限频机制正常")
        except Exception as e:
            print(f"限频机制异常: {e}")
            return False
        
        # 3. 测试股票列表获取
        print("\n3. 测试股票列表获取...")
        try:
            stocks = _fetch_stock_list()
            print(f"获取股票列表成功，共 {len(stocks)} 只股票")
            print(f"   示例股票: {stocks.head(3)['ts_code'].tolist()}")
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return False
        
        # 4. 测试数据清理功能
        print("\n4. 测试数据清理功能...")
        try:
            import pandas as pd
            import numpy as np
            
            # 创建测试数据（包含异常值）
            test_data = pd.DataFrame({
                'ts_code': ['000001.SZ'] * 5,
                'trade_date': ['20240101', '20240102', '20240103', '20240104', '20240105'],
                'open': [10.0, 0.0, 11.0, np.nan, 12.0],  # 包含0和NaN
                'high': [10.5, 0.0, 11.5, 12.0, 12.5],
                'low': [9.5, 0.0, 10.5, 11.0, 11.5],
                'close': [10.2, 0.0, 11.2, 11.8, 12.2],
                'vol': [1000, -100, 1100, 1200, 1300]  # 包含负数
            })
            
            print(f"   原始数据: {len(test_data)} 行")
            cleaned_data = _clean_and_validate_dataframe(test_data, "000001.SZ")
            print(f"   清理后数据: {len(cleaned_data)} 行")
            print("数据清理功能正常")
        except Exception as e:
            print(f"数据清理功能异常: {e}")
            return False
        
        # 5. 测试小规模下载（只下载几只股票）
        print("\n5. 测试小规模下载...")
        try:
            # 选择前3只股票进行测试
            test_codes = stocks.head(3)['ts_code'].tolist()
            print(f"   测试股票: {test_codes}")
            
            # 设置测试日期范围（最近7天）
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
            
            print(f"   测试日期范围: {start_date} - {end_date}")
            
            # 测试单只股票下载
            from download import ts
            test_code = test_codes[0]
            print(f"   测试下载股票: {test_code}")
            
            def test_call():
                return ts.pro_bar(
                    ts_code=test_code,
                    start_date=start_date,
                    end_date=end_date,
                    adj=None,
                    freq='D',
                    asset='E'
                )
            
            df = _retry(test_call, f"test_{test_code}")
            
            if df is not None and not df.empty:
                print(f"下载成功，获取 {len(df)} 行数据")
                print(f"   数据列: {list(df.columns)}")
                print(f"   日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
                
                # 测试数据清理
                cleaned_df = _clean_and_validate_dataframe(df, test_code)
                print(f"   清理后数据: {len(cleaned_df)} 行")
            else:
                print("下载数据为空（可能是非交易日）")
            
        except Exception as e:
            print(f"小规模下载测试失败: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("所有测试通过！下载功能正常")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_download()
    if success:
        print("\n测试完成，可以开始正式下载")
    else:
        print("\n测试失败，请检查配置和网络连接")
