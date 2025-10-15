#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个股详情数据库功能测试脚本
"""

import os
import json
import tempfile
import logging
from pathlib import Path

from detail_db import DetailDB
from config import SC_OUTPUT_DIR

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def test_basic_operations():
    """测试基本数据库操作"""
    LOGGER.info("测试基本数据库操作...")
    
    # 使用临时数据库
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # 初始化数据库
        db = DetailDB(db_path, "sqlite")
        
        # 测试数据
        test_data = {
            "ts_code": "000001.SZ",
            "ref_date": "20251014",
            "summary": {
                "score": 135.0,
                "tiebreak": -3.54,
                "highlights": ["过去60日涨幅过大"],
                "drawbacks": [],
                "opportunities": ["b1", "长期成本线附近"],
                "rank": 1,
                "total": 100
            },
            "rules": [
                {
                    "name": "高于趋势",
                    "scope": "EACH",
                    "timeframe": "D",
                    "window": 20,
                    "points": 2.0,
                    "ok": True,
                    "cnt": 14,
                    "add": 28.0,
                    "explain": "高于主力成本"
                }
            ]
        }
        
        # 测试保存
        LOGGER.info("测试保存数据...")
        success = db.save_detail(
            test_data["ts_code"],
            test_data["ref_date"],
            test_data["summary"],
            test_data["rules"]
        )
        assert success, "保存数据失败"
        LOGGER.info("✓ 保存数据成功")
        
        # 测试读取
        LOGGER.info("测试读取数据...")
        loaded = db.load_detail(test_data["ts_code"], test_data["ref_date"])
        assert loaded is not None, "读取数据失败"
        assert loaded["ts_code"] == test_data["ts_code"], "股票代码不匹配"
        assert loaded["ref_date"] == test_data["ref_date"], "参考日期不匹配"
        assert loaded["summary"]["score"] == test_data["summary"]["score"], "评分不匹配"
        LOGGER.info("✓ 读取数据成功")
        
        # 测试查询
        LOGGER.info("测试按日期查询...")
        df = db.query_by_date(test_data["ref_date"])
        assert not df.empty, "按日期查询失败"
        assert len(df) == 1, "查询结果数量不正确"
        LOGGER.info("✓ 按日期查询成功")
        
        # 测试按代码查询
        LOGGER.info("测试按代码查询...")
        df = db.query_by_codes([test_data["ts_code"]], test_data["ref_date"])
        assert not df.empty, "按代码查询失败"
        assert len(df) == 1, "查询结果数量不正确"
        LOGGER.info("✓ 按代码查询成功")
        
        # 测试统计信息
        LOGGER.info("测试统计信息...")
        stats = db.get_stats()
        assert stats["total_records"] == 1, "记录数不正确"
        LOGGER.info("✓ 统计信息正确")
        
        LOGGER.info("✓ 所有基本操作测试通过!")
        
    finally:
        # 清理临时文件
        db.close()
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_batch_operations():
    """测试批量操作"""
    LOGGER.info("测试批量操作...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        db = DetailDB(db_path, "sqlite")
        
        # 准备测试数据
        test_stocks = [
            {"ts_code": "000001.SZ", "score": 135.0, "rank": 1},
            {"ts_code": "000002.SZ", "score": 120.0, "rank": 2},
            {"ts_code": "000003.SZ", "score": 110.0, "rank": 3},
        ]
        
        ref_date = "20251014"
        
        # 批量保存
        LOGGER.info("测试批量保存...")
        for stock in test_stocks:
            summary = {
                "score": stock["score"],
                "tiebreak": None,
                "highlights": [],
                "drawbacks": [],
                "opportunities": []
            }
            rules = [{"name": "测试规则", "ok": True, "add": stock["score"]}]
            
            success = db.save_detail(stock["ts_code"], ref_date, summary, rules)
            assert success, f"保存 {stock['ts_code']} 失败"
        
        LOGGER.info("✓ 批量保存成功")
        
        # 测试批量更新排名
        LOGGER.info("测试批量更新排名...")
        class MockStock:
            def __init__(self, ts_code):
                self.ts_code = ts_code
        
        scored_sorted = [MockStock(stock["ts_code"]) for stock in test_stocks]
        success = db.batch_update_ranks(ref_date, scored_sorted)
        assert success, "批量更新排名失败"
        LOGGER.info("✓ 批量更新排名成功")
        
        # 验证排名更新
        df = db.query_by_date(ref_date)
        for _, row in df.iterrows():
            expected_rank = next(s["rank"] for s in test_stocks if s["ts_code"] == row["ts_code"])
            assert row["rank"] == expected_rank, f"排名更新失败: {row['ts_code']}"
        
        LOGGER.info("✓ 批量操作测试通过!")
        
    finally:
        db.close()
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_migration_compatibility():
    """测试与现有JSON文件的兼容性"""
    LOGGER.info("测试与现有JSON文件的兼容性...")
    
    # 查找现有的JSON文件
    details_dir = Path(SC_OUTPUT_DIR) / "details"
    if not details_dir.exists():
        LOGGER.warning("未找到details目录，跳过兼容性测试")
        return
    
    json_files = []
    for date_dir in details_dir.iterdir():
        if date_dir.is_dir():
            for json_file in date_dir.glob("*.json"):
                json_files.append(json_file)
                if len(json_files) >= 3:  # 只测试前3个文件
                    break
        if len(json_files) >= 3:
            break
    
    if not json_files:
        LOGGER.warning("未找到JSON文件，跳过兼容性测试")
        return
    
    LOGGER.info(f"找到 {len(json_files)} 个JSON文件进行测试")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        db = DetailDB(db_path, "sqlite")
        
        # 测试迁移JSON文件
        for json_file in json_files:
            LOGGER.info(f"测试迁移文件: {json_file.name}")
            
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            ts_code = data.get("ts_code")
            ref_date = data.get("ref_date")
            summary = data.get("summary", {})
            rules = data.get("rules", [])
            
            if not ts_code or not ref_date:
                LOGGER.warning(f"跳过无效文件: {json_file}")
                continue
            
            # 保存到数据库
            success = db.save_detail(ts_code, ref_date, summary, rules)
            assert success, f"迁移文件失败: {json_file}"
            
            # 验证数据完整性
            loaded = db.load_detail(ts_code, ref_date)
            assert loaded is not None, f"读取迁移数据失败: {ts_code}"
            assert loaded["ts_code"] == ts_code, "股票代码不匹配"
            assert loaded["ref_date"] == ref_date, "参考日期不匹配"
        
        LOGGER.info("✓ 兼容性测试通过!")
        
    finally:
        db.close()
        if os.path.exists(db_path):
            os.unlink(db_path)

def main():
    """运行所有测试"""
    LOGGER.info("开始个股详情数据库功能测试")
    LOGGER.info("=" * 50)
    
    try:
        test_basic_operations()
        test_batch_operations()
        test_migration_compatibility()
        
        LOGGER.info("=" * 50)
        LOGGER.info("✓ 所有测试通过!")
        
    except Exception as e:
        LOGGER.error(f"测试失败: {e}")
        raise

if __name__ == "__main__":
    main()
