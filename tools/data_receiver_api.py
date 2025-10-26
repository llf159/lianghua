#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_receiver_api.py - 数据接收API接口
提供HTTP API接口，供其他计算模块通过HTTP请求发送数据
"""

import json
import logging
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from database_manager import get_database_manager, receive_data, receive_stock_data, receive_indicator_data, receive_score_data

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 获取数据库管理器
db_manager = get_database_manager()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        stats = db_manager.get_stats()
        return jsonify({
            "status": "healthy",
            "database_manager": {
                "queue_size": stats['queue_size'],
                "paused": stats['paused']
            }
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/api/v1/data/stock', methods=['POST'])
def receive_stock_data_api():
    """接收股票数据的API接口"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据不能为空"}), 400
        
        # 获取参数
        source_module = request.headers.get('X-Source-Module', 'unknown')
        mode = data.get('mode', 'append')
        callback_url = data.get('callback_url')
        
        # 处理回调函数
        callback = None
        if callback_url:
            def http_callback(response):
                # 这里可以实现HTTP回调逻辑
                logger.info(f"数据导入完成，回调URL: {callback_url}")
            callback = http_callback
        
        # 接收数据
        request_id = receive_stock_data(
            source_module=source_module,
            data=data.get('data'),
            mode=mode,
            callback=callback
        )
        
        return jsonify({
            "success": True,
            "request_id": request_id,
            "message": "股票数据接收成功"
        })
        
    except Exception as e:
        logger.error(f"接收股票数据失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/data/indicator', methods=['POST'])
def receive_indicator_data_api():
    """接收指标数据的API接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据不能为空"}), 400
        
        source_module = request.headers.get('X-Source-Module', 'unknown')
        table_name = data.get('table_name', 'indicator_data')
        mode = data.get('mode', 'append')
        
        request_id = receive_indicator_data(
            source_module=source_module,
            data=data.get('data'),
            table_name=table_name,
            mode=mode
        )
        
        return jsonify({
            "success": True,
            "request_id": request_id,
            "message": "指标数据接收成功"
        })
        
    except Exception as e:
        logger.error(f"接收指标数据失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/data/score', methods=['POST'])
def receive_score_data_api():
    """接收评分数据的API接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据不能为空"}), 400
        
        source_module = request.headers.get('X-Source-Module', 'unknown')
        table_name = data.get('table_name', 'score_data')
        mode = data.get('mode', 'append')
        
        request_id = receive_score_data(
            source_module=source_module,
            data=data.get('data'),
            table_name=table_name,
            mode=mode
        )
        
        return jsonify({
            "success": True,
            "request_id": request_id,
            "message": "评分数据接收成功"
        })
        
    except Exception as e:
        logger.error(f"接收评分数据失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/data/custom', methods=['POST'])
def receive_custom_data_api():
    """接收自定义数据的API接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据不能为空"}), 400
        
        source_module = request.headers.get('X-Source-Module', 'unknown')
        data_type = data.get('data_type', 'custom')
        table_name = data.get('table_name')
        mode = data.get('mode', 'append')
        validation_rules = data.get('validation_rules')
        
        if not table_name:
            return jsonify({"error": "table_name参数不能为空"}), 400
        
        request_id = receive_data(
            source_module=source_module,
            data_type=data_type,
            data=data.get('data'),
            table_name=table_name,
            mode=mode,
            validation_rules=validation_rules
        )
        
        return jsonify({
            "success": True,
            "request_id": request_id,
            "message": "自定义数据接收成功"
        })
        
    except Exception as e:
        logger.error(f"接收自定义数据失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/stats', methods=['GET'])
def get_stats_api():
    """获取统计信息的API接口"""
    try:
        # 数据库管理器统计
        db_stats = db_manager.get_stats()
        
        # 数据接收器统计
        receiver = db_manager.get_data_receiver()
        import_stats = receiver.get_import_stats()
        
        return jsonify({
            "database_manager": db_stats,
            "data_receiver": import_stats
        })
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        "error": "接口不存在",
        "message": "请检查请求路径是否正确"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        "error": "服务器内部错误",
        "message": "请联系系统管理员"
    }), 500

def create_app():
    """创建Flask应用"""
    return app

if __name__ == '__main__':
    # 启动API服务器
    print("启动数据接收API服务器...")
    print("API接口:")
    print("  GET  /health - 健康检查")
    print("  POST /api/v1/data/stock - 接收股票数据")
    print("  POST /api/v1/data/indicator - 接收指标数据")
    print("  POST /api/v1/data/score - 接收评分数据")
    print("  POST /api/v1/data/custom - 接收自定义数据")
    print("  GET  /api/v1/stats - 获取统计信息")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
