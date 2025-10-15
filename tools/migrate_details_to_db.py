#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个股详情数据迁移脚本
将现有的JSON文件迁移到数据库存储
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

from config import SC_OUTPUT_DIR, SC_DETAIL_DB_TYPE, SC_DETAIL_DB_PATH
from detail_db import DetailDB

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def find_json_files(details_dir: Path) -> List[Path]:
    """查找所有JSON文件"""
    json_files = []
    if details_dir.exists():
        for date_dir in details_dir.iterdir():
            if date_dir.is_dir():
                for json_file in date_dir.glob("*.json"):
                    json_files.append(json_file)
    return sorted(json_files)

def load_json_file(json_path: Path) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        LOGGER.error(f"加载JSON文件失败 {json_path}: {e}")
        return None

def migrate_single_file(db: DetailDB, json_path: Path) -> bool:
    """迁移单个JSON文件到数据库"""
    data = load_json_file(json_path)
    if not data:
        return False
    
    ts_code = data.get("ts_code")
    ref_date = data.get("ref_date")
    summary = data.get("summary", {})
    rules = data.get("rules", [])
    
    if not ts_code or not ref_date:
        LOGGER.warning(f"JSON文件缺少必要字段: {json_path}")
        return False
    
    return db.save_detail(ts_code, ref_date, summary, rules)

def migrate_all_files(details_dir: Path, db_path: str, db_type: str = "sqlite") -> Dict[str, Any]:
    """迁移所有JSON文件到数据库"""
    LOGGER.info(f"开始迁移数据到数据库: {db_path} ({db_type})")
    
    # 查找所有JSON文件
    json_files = find_json_files(details_dir)
    if not json_files:
        LOGGER.warning("未找到任何JSON文件")
        return {"success": False, "message": "未找到JSON文件"}
    
    LOGGER.info(f"找到 {len(json_files)} 个JSON文件")
    
    # 初始化数据库
    try:
        db = DetailDB(db_path, db_type)
    except Exception as e:
        LOGGER.error(f"数据库初始化失败: {e}")
        return {"success": False, "message": f"数据库初始化失败: {e}"}
    
    # 统计信息
    stats = {
        "total_files": len(json_files),
        "success_count": 0,
        "failed_count": 0,
        "failed_files": [],
        "by_date": {}
    }
    
    # 迁移文件
    with tqdm(json_files, desc="迁移进度") as pbar:
        for json_path in pbar:
            try:
                if migrate_single_file(db, json_path):
                    stats["success_count"] += 1
                    
                    # 按日期统计
                    ref_date = json_path.parent.name
                    if ref_date not in stats["by_date"]:
                        stats["by_date"][ref_date] = 0
                    stats["by_date"][ref_date] += 1
                else:
                    stats["failed_count"] += 1
                    stats["failed_files"].append(str(json_path))
                
                pbar.set_postfix({
                    "成功": stats["success_count"],
                    "失败": stats["failed_count"]
                })
                
            except Exception as e:
                LOGGER.error(f"迁移文件失败 {json_path}: {e}")
                stats["failed_count"] += 1
                stats["failed_files"].append(str(json_path))
    
    # 关闭数据库连接
    db.close()
    
    # 输出统计信息
    LOGGER.info("=" * 50)
    LOGGER.info("迁移完成统计:")
    LOGGER.info(f"总文件数: {stats['total_files']}")
    LOGGER.info(f"成功迁移: {stats['success_count']}")
    LOGGER.info(f"迁移失败: {stats['failed_count']}")
    LOGGER.info(f"成功率: {stats['success_count']/stats['total_files']*100:.1f}%")
    
    if stats["by_date"]:
        LOGGER.info("按日期统计:")
        for date, count in sorted(stats["by_date"].items()):
            LOGGER.info(f"  {date}: {count} 个文件")
    
    if stats["failed_files"]:
        LOGGER.warning(f"失败文件列表 (前10个):")
        for f in stats["failed_files"][:10]:
            LOGGER.warning(f"  {f}")
        if len(stats["failed_files"]) > 10:
            LOGGER.warning(f"  ... 还有 {len(stats['failed_files']) - 10} 个失败文件")
    
    stats["success"] = stats["failed_count"] == 0
    return stats

def verify_migration(db_path: str, db_type: str = "sqlite") -> Dict[str, Any]:
    """验证迁移结果"""
    LOGGER.info("验证迁移结果...")
    
    try:
        db = DetailDB(db_path, db_type)
        stats = db.get_stats()
        db.close()
        
        LOGGER.info("数据库统计信息:")
        LOGGER.info(f"总记录数: {stats.get('total_records', 0)}")
        
        if stats.get("by_date"):
            LOGGER.info("按日期统计:")
            for date, count in sorted(stats["by_date"].items()):
                LOGGER.info(f"  {date}: {count} 条记录")
        
        return {"success": True, "stats": stats}
        
    except Exception as e:
        LOGGER.error(f"验证失败: {e}")
        return {"success": False, "message": str(e)}

def main():
    """主函数"""
    # 配置路径
    details_dir = Path(SC_OUTPUT_DIR) / "details"
    db_path = os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH)
    
    LOGGER.info("个股详情数据迁移工具")
    LOGGER.info(f"源目录: {details_dir}")
    LOGGER.info(f"目标数据库: {db_path} ({SC_DETAIL_DB_TYPE})")
    
    if not details_dir.exists():
        LOGGER.error(f"源目录不存在: {details_dir}")
        return
    
    # 检查是否已有数据库文件
    if os.path.exists(db_path):
        response = input(f"数据库文件已存在: {db_path}\n是否覆盖? (y/N): ")
        if response.lower() != 'y':
            LOGGER.info("取消迁移")
            return
        else:
            # 备份现有数据库
            backup_path = f"{db_path}.backup"
            try:
                import shutil
                shutil.move(db_path, backup_path)
                LOGGER.info(f"已备份现有数据库到: {backup_path}")
            except Exception as e:
                LOGGER.warning(f"备份失败: {e}")
    
    # 执行迁移
    result = migrate_all_files(details_dir, db_path, SC_DETAIL_DB_TYPE)
    
    if result["success"]:
        LOGGER.info("迁移成功完成!")
        
        # 验证迁移结果
        verify_result = verify_migration(db_path, SC_DETAIL_DB_TYPE)
        if verify_result["success"]:
            LOGGER.info("验证通过!")
        else:
            LOGGER.warning(f"验证失败: {verify_result.get('message', '未知错误')}")
    else:
        LOGGER.error("迁移失败!")
        if result.get("failed_files"):
            LOGGER.error(f"失败文件数: {len(result['failed_files'])}")

if __name__ == "__main__":
    main()
