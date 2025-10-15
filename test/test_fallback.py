#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据库回退机制
"""

import os
import tempfile
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def test_database_fallback():
    """测试数据库失败时的回退机制"""
    LOGGER.info("测试数据库回退机制...")
    
    # 临时修改配置，模拟数据库失败
    original_config = {}
    try:
        # 导入配置
        import config
        original_config = {
            'SC_DETAIL_STORAGE': config.SC_DETAIL_STORAGE,
            'SC_USE_DB_STORAGE': config.SC_USE_DB_STORAGE,
            'SC_DB_FALLBACK_TO_JSON': getattr(config, 'SC_DB_FALLBACK_TO_JSON', True)
        }
        
        # 设置测试配置
        config.SC_DETAIL_STORAGE = "database"
        config.SC_USE_DB_STORAGE = True
        config.SC_DB_FALLBACK_TO_JSON = True
        
        # 测试数据
        test_data = {
            "ts_code": "000001.SZ",
            "ref_date": "20251014",
            "summary": {
                "score": 135.0,
                "tiebreak": -3.54,
                "highlights": ["测试数据"],
                "drawbacks": [],
                "opportunities": ["测试机会"]
            },
            "rules": [
                {
                    "name": "测试规则",
                    "ok": True,
                    "add": 10.0,
                    "explain": "测试规则说明"
                }
            ]
        }
        
        # 1. 测试正常情况（数据库可用）
        LOGGER.info("测试正常数据库操作...")
        from scoring_core import _write_detail_json
        from score_ui import _load_detail_json
        
        # 写入数据
        _write_detail_json(
            test_data["ts_code"],
            test_data["ref_date"],
            test_data["summary"],
            test_data["rules"]
        )
        
        # 读取数据
        loaded = _load_detail_json(test_data["ref_date"], test_data["ts_code"])
        if loaded:
            LOGGER.info("✓ 正常数据库操作成功")
        else:
            LOGGER.warning("⚠ 正常数据库操作失败，将使用JSON回退")
        
        # 2. 测试数据库不可用的情况
        LOGGER.info("测试数据库不可用时的回退...")
        
        # 模拟数据库连接失败
        config.SC_USE_DB_STORAGE = False
        
        # 写入数据（应该回退到JSON）
        _write_detail_json(
            test_data["ts_code"] + "_fallback",
            test_data["ref_date"],
            test_data["summary"],
            test_data["rules"]
        )
        
        # 读取数据（应该从JSON读取）
        loaded = _load_detail_json(test_data["ref_date"], test_data["ts_code"] + "_fallback")
        if loaded:
            LOGGER.info("✓ JSON回退机制工作正常")
        else:
            LOGGER.error("✗ JSON回退机制失败")
        
        # 3. 验证JSON文件是否存在
        json_path = Path(config.SC_OUTPUT_DIR) / "details" / test_data["ref_date"] / f"{test_data['ts_code']}_fallback_{test_data['ref_date']}.json"
        if json_path.exists():
            LOGGER.info("✓ JSON文件已创建")
        else:
            LOGGER.error("✗ JSON文件未创建")
        
        LOGGER.info("回退机制测试完成")
        
    except Exception as e:
        LOGGER.error(f"测试失败: {e}")
        raise
    finally:
        # 恢复原始配置
        try:
            for key, value in original_config.items():
                setattr(config, key, value)
        except:
            pass

def test_config_validation():
    """测试配置验证"""
    LOGGER.info("测试配置验证...")
    
    try:
        import config
        
        # 检查必要配置
        required_configs = [
            'SC_DETAIL_STORAGE',
            'SC_DETAIL_DB_TYPE', 
            'SC_DETAIL_DB_PATH',
            'SC_USE_DB_STORAGE'
        ]
        
        for cfg in required_configs:
            if not hasattr(config, cfg):
                LOGGER.error(f"缺少配置项: {cfg}")
                return False
        
        # 检查回退配置
        if not hasattr(config, 'SC_DB_FALLBACK_TO_JSON'):
            LOGGER.warning("缺少回退配置 SC_DB_FALLBACK_TO_JSON，将使用默认值 True")
            config.SC_DB_FALLBACK_TO_JSON = True
        
        LOGGER.info("✓ 配置验证通过")
        LOGGER.info(f"存储方式: {config.SC_DETAIL_STORAGE}")
        LOGGER.info(f"数据库类型: {config.SC_DETAIL_DB_TYPE}")
        LOGGER.info(f"启用数据库: {config.SC_USE_DB_STORAGE}")
        LOGGER.info(f"回退到JSON: {getattr(config, 'SC_DB_FALLBACK_TO_JSON', True)}")
        
        return True
        
    except Exception as e:
        LOGGER.error(f"配置验证失败: {e}")
        return False

def main():
    """主函数"""
    LOGGER.info("开始测试数据库回退机制")
    LOGGER.info("=" * 50)
    
    try:
        # 1. 配置验证
        if not test_config_validation():
            LOGGER.error("配置验证失败，退出测试")
            return
        
        # 2. 回退机制测试
        test_database_fallback()
        
        LOGGER.info("=" * 50)
        LOGGER.info("✓ 所有测试完成")
        
    except Exception as e:
        LOGGER.error(f"测试失败: {e}")
        raise

if __name__ == "__main__":
    main()
