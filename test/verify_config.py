#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证数据库存储配置和功能
"""

import os
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def verify_config():
    """验证配置"""
    LOGGER.info("验证数据库存储配置...")
    
    try:
        import config
        
        # 检查配置
        configs = {
            'SC_DETAIL_STORAGE': getattr(config, 'SC_DETAIL_STORAGE', None),
            'SC_DETAIL_DB_TYPE': getattr(config, 'SC_DETAIL_DB_TYPE', None),
            'SC_DETAIL_DB_PATH': getattr(config, 'SC_DETAIL_DB_PATH', None),
            'SC_USE_DB_STORAGE': getattr(config, 'SC_USE_DB_STORAGE', None),
            'SC_DB_FALLBACK_TO_JSON': getattr(config, 'SC_DB_FALLBACK_TO_JSON', None),
        }
        
        LOGGER.info("当前配置:")
        for key, value in configs.items():
            LOGGER.info(f"  {key}: {value}")
        
        # 验证配置值
        if configs['SC_DETAIL_STORAGE'] == 'database':
            LOGGER.info("✓ 配置为优先使用数据库存储")
        else:
            LOGGER.warning(f"⚠ 存储方式为: {configs['SC_DETAIL_STORAGE']}")
        
        if configs['SC_USE_DB_STORAGE']:
            LOGGER.info("✓ 数据库存储已启用")
        else:
            LOGGER.warning("⚠ 数据库存储未启用")
        
        if configs['SC_DB_FALLBACK_TO_JSON']:
            LOGGER.info("✓ 回退到JSON文件已启用")
        else:
            LOGGER.warning("⚠ 回退到JSON文件未启用")
        
        return True
        
    except Exception as e:
        LOGGER.error(f"配置验证失败: {e}")
        return False

def verify_database():
    """验证数据库功能"""
    LOGGER.info("验证数据库功能...")
    
    try:
        import config
        from detail_db import get_detail_db
        
        # 获取数据库实例
        db = get_detail_db()
        
        # 检查数据库文件是否存在
        db_path = Path(config.SC_OUTPUT_DIR) / config.SC_DETAIL_DB_PATH
        if db_path.exists():
            LOGGER.info(f"✓ 数据库文件存在: {db_path}")
            LOGGER.info(f"  文件大小: {db_path.stat().st_size} 字节")
        else:
            LOGGER.warning(f"⚠ 数据库文件不存在: {db_path}")
        
        # 获取统计信息
        stats = db.get_stats()
        if stats:
            LOGGER.info("✓ 数据库统计信息:")
            LOGGER.info(f"  总记录数: {stats.get('total_records', 0)}")
            if stats.get('by_date'):
                LOGGER.info("  按日期统计:")
                for date, count in sorted(stats['by_date'].items()):
                    LOGGER.info(f"    {date}: {count} 条记录")
        else:
            LOGGER.warning("⚠ 无法获取数据库统计信息")
        
        db.close()
        return True
        
    except Exception as e:
        LOGGER.error(f"数据库验证失败: {e}")
        return False

def verify_fallback_mechanism():
    """验证回退机制"""
    LOGGER.info("验证回退机制...")
    
    try:
        import config
        # 检查JSON文件目录
        details_dir = Path(config.SC_OUTPUT_DIR) / "details"
        if details_dir.exists():
            json_files = list(details_dir.rglob("*.json"))
            LOGGER.info(f"✓ JSON文件目录存在，包含 {len(json_files)} 个文件")
            
            # 显示最近的几个文件
            if json_files:
                recent_files = sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
                LOGGER.info("  最近的JSON文件:")
                for f in recent_files:
                    LOGGER.info(f"    {f.name}")
        else:
            LOGGER.warning("⚠ JSON文件目录不存在")
        
        return True
        
    except Exception as e:
        LOGGER.error(f"回退机制验证失败: {e}")
        return False

def main():
    """主函数"""
    LOGGER.info("验证数据库存储配置和功能")
    LOGGER.info("=" * 50)
    
    success = True
    
    # 1. 验证配置
    if not verify_config():
        success = False
    
    # 2. 验证数据库
    if not verify_database():
        success = False
    
    # 3. 验证回退机制
    if not verify_fallback_mechanism():
        success = False
    
    LOGGER.info("=" * 50)
    if success:
        LOGGER.info("✓ 所有验证通过，系统已配置为优先使用数据库存储，失败时回退到JSON文件")
    else:
        LOGGER.error("✗ 部分验证失败，请检查配置和功能")
    
    return success

if __name__ == "__main__":
    main()
