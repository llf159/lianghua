# -*- coding: utf-8 -*-
"""
strategies_repo.py — Python 版策略仓库
策略编辑请参考文档
"""
RANKING_TITLE = "排名策略（由 config.SC_RULES 抽取）"
FILTER_TITLE = "筛选策略（hard_penalty=True）"
PREDICTION_TITLE = "模拟策略（预留，当前为空）"
POSITION_TITLE = "持仓检查策略（个股）"
OPPORTUNITY_TITLE = "买点策略（个股）"

# 排名策略
RANKING_RULES = [
]

# 初选策略（硬淘汰）
FILTER_RULES = [
]

# 模拟策略
PREDICTION_RULES = [
    # {
    #     "name": "反推KDJ-J买点",
    #     "check": "j <= 13",
    #     "scenario": {
    #         "mode": "reverse_indicator",    # 反推模式
    #         "reverse_indicator": "j",       # 指标名称
    #         "reverse_target_value": 10.0,   # 目标J值
    #         "reverse_method": "optimize",   # 求解方法
    #         "reverse_tolerance": 1e-6,      # 求解精度
    #         "reverse_max_iterations": 1000, # 最大迭代次数
    #         "hl_mode": "follow",            # 高低点跟随
    #         "vol_mode": "same",             # 成交量保持不变
    #         "lock_higher_than_open": True   # 收盘价不低于开盘价
    #     }
    # },
    # {
    #     "name": "反推RSI超卖",
    #     "check": "rsi <= 30",
    #     "scenario": {
    #         "mode": "reverse_indicator",
    #         "reverse_indicator": "rsi",     # 指标名称
    #         "reverse_target_value": 25.0,   # 目标RSI值
    #         "reverse_method": "binary_search",
    #         "reverse_tolerance": 1e-4,
    #         "reverse_max_iterations": 100,
    #         "hl_mode": "range_pct",
    #         "range_pct": 2.0,
    #         "vol_mode": "mult",
    #         "vol_arg": 1.5
    #     }
    # },
]

# 个股持仓检查策略（仅用于个股页签的触发表展示，不生成动作）
POSITION_POLICIES = [
]

# 个股买点策略（用于给出买点价格来源；可选，不影响持仓检查）
OPPORTUNITY_POLICIES = [
    # {
    #     'name': 'KDJ_J最低点买点',
    #     'when': 'GET_LAST_CONDITION_PRICE("j < 13", 100) > 0',
    #     'explain': '获取上一次J值低于13的收盘价作为买点价格'
    # },
    # {
    #     'name': 'KDJ_J最低点买点(严格)',
    #     'when': 'GET_LAST_CONDITION_PRICE("j < 10", 50) > 0',
    #     'explain': '获取上一次J值低于10的收盘价作为买点价格'
    # },
    # {
    #     'name': '价格突破20日均线买点',
    #     'when': 'GET_LAST_CONDITION_PRICE("C > MA(C, 20)", 100) > 0',
    #     'explain': '获取上一次价格突破20日均线的收盘价作为买点价格'
    # },
    # {
    #     'name': '成交量放大买点',
    #     'when': 'GET_LAST_CONDITION_PRICE("V > MA(V, 5) * 1.5", 100) > 0',
    #     'explain': '获取上一次成交量放大的收盘价作为买点价格'
    # }
]


# -----------------------------------------------------------------------------
# 下面是来自 config.py 的【已注释样例 - 排名策略】（保留原格式，便于日后启用）
# {
#     "name": "相对强于深证",
#     "timeframe": "D",
#     "window": 20,
#     "when": "RS_399001_SZ_3 > 1.02",   # 20日RS>1.02（强于基准≈2%）
#     "scope": "ANY",
#     "points": +4,
#     "explain": "20日跑赢深证"
# },
# {
#     "name": "当日振幅≥5%",
#     "timeframe": "D",
#     "window": 10,
#     "when": "SAFE_DIV(H - L, REF(C,1)) >= 0.05 AND SAFE_DIV(ABS(C - REF(C,1)), REF(C,1)) <= 0.02",
#     "scope": "EACH",
#     "points": -5,
#     "explain": "大波动"
# },
# {
#     "name": "健康缩量",
#     "timeframe": "D",
#     "window": 60,
#     "when": "(COUNT( (CROSS(C, HHV(H, 60)) AND V <= 1.5 * MA(V, 20)), 5 ) >= 1) AND (TS_PCT(V, 20) <= 0.35)",
#     "scope": "ANY",
#     "points": +5,
#     "explain": "健康缩量",
#     "show_reason": False
# },
# {
#     "name": "3/4 阴量线",
#     "timeframe": "D",
#     "window": 20,
#     "when": "REF(TS_PCT(C,20),1) > 0.9 AND (C < O) AND (C < REF(C, 1)) AND (SAFE_DIV(V, REF(V, 1)) >= 0.6) AND (SAFE_DIV(V, REF(V, 1)) <= 0.8)",
#     "scope": "ANY",
#     "points": -15,
#     "explain": "3/4 阴量线",
# },
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 下面是来自 config.py 的【已注释样例 - 筛选策略（硬淘汰）】（保留原格式，便于日后启用）
# # a) 周线下行并放量：12周内至少3次
# {
#     "name": "W_下行放量_硬淘汰",
#     "timeframe": "W",
#     "window": 12,
#     "when": " (C<REF(C,1)) AND (V>1.5*MA(V,10)) ",
#     "scope": "COUNT>=3",
#     "hard_penalty": True,
#     "reason": "周线下行并放量(12周内≥3次)"
# },
# # b) 月线破位（跌破半年均线）
# {
#     "name": "M_跌破半年均线_硬淘汰",
#     "timeframe": "M",
#     "window": 12,
#     "when": " C<MA(C,6) ",
#     "scope": "LAST",
#     "hard_penalty": True,
#     "reason": "月线跌破半年均线"
# },
# -----------------------------------------------------------------------------

# =============================================================================
# 策略验证器功能（从 strategy_validator.py 合并）
# =============================================================================

import json
import re
import ast
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import pandas as pd
import numpy as np

# 导入现有的模块
try:
    from tdx_compat import evaluate_bool, evaluate
    from indicators import names_in_expr, REGISTRY
    from scoring_core import diagnose_expr, _pick_ref_date, _list_codes_for_day, _read_stock_df, _start_for_tf_window, _resample, _window_slice, _ctx_df
except ImportError:
    # 兜底处理
    def evaluate_bool(*args, **kwargs): return [False]
    def evaluate(*args, **kwargs): return {"sig": [False]}
    def names_in_expr(expr): return []
    REGISTRY = {}
    def diagnose_expr(expr): return {"ok": True, "error": None, "missing": [], "need_cols": []}
    def _pick_ref_date(*args, **kwargs): return "20241201"
    def _list_codes_for_day(*args, **kwargs): return []
    def _read_stock_df(*args, **kwargs): return pd.DataFrame()
    def _start_for_tf_window(*args, **kwargs): return 0
    def _resample(*args, **kwargs): return pd.DataFrame()
    def _window_slice(*args, **kwargs): return pd.DataFrame()
    def _ctx_df(*args, **kwargs): return pd.DataFrame()


class StrategyValidationResult:
    """策略验证结果"""
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.suggestions = []
        self.missing_columns = []
        self.missing_indicators = []
        self.syntax_issues = []
        self.required_fields = []
        self.optional_fields = []
        
    def add_error(self, message: str, field: str = None):
        self.is_valid = False
        self.errors.append({
            "message": message,
            "field": field,
            "type": "error"
        })
    
    def add_warning(self, message: str, field: str = None):
        self.warnings.append({
            "message": message,
            "field": field,
            "type": "warning"
        })
    
    def add_suggestion(self, message: str, field: str = None):
        self.suggestions.append({
            "message": message,
            "field": field,
            "type": "suggestion"
        })


class StrategyValidator:
    """策略验证器"""
    
    # 必填字段
    REQUIRED_FIELDS = {
        "ranking": ["when"],  # 排名策略
        "filter": ["when"],   # 筛选策略
        "prediction": ["check"],  # 模拟策略
        "position": ["when"],     # 持仓策略
        "opportunity": ["when"]   # 买点策略
    }
    
    # 可选字段
    OPTIONAL_FIELDS = {
        "ranking": ["name", "timeframe", "window", "score_windows", "scope", "points", "explain", "show_reason", "as", "gate", "clauses", "dist_points"],
        "filter": ["name", "timeframe", "window", "score_windows", "scope", "reason", "hard_penalty", "gate", "clauses"],
        "prediction": ["name", "scenario"],
        "position": ["name", "explain"],
        "opportunity": ["name", "explain"]
    }
    
    # 支持的timeframe
    SUPPORTED_TIMEFRAMES = {"D", "W", "M", "60MIN"}
    
    # 支持的scope
    SUPPORTED_SCOPES = {"LAST", "ANY", "ALL", "EACH", "RECENT", "DIST", "NEAR", "CONSEC", "COUNT"}
    
    def __init__(self):
        self.available_columns = set()
        self.available_indicators = set()
        self._load_available_resources()
    
    def _load_available_resources(self):
        """加载可用的列和指标"""
        # 基础列
        self.available_columns = {
            "open", "high", "low", "close", "vol", "amount", "o", "h", "l", "c", "v",
            "trade_date", "ts_code", "adj_factor"
        }
        
        # 从指标注册表获取可用指标
        for indicator_name, meta in REGISTRY.items():
            if hasattr(meta, 'out') and meta.out:
                for col in meta.out.keys():
                    self.available_indicators.add(col.lower())
                    self.available_columns.add(col.lower())
    
    def validate_strategy_file(self, file_path: str) -> StrategyValidationResult:
        """验证策略文件"""
        result = StrategyValidationResult()
        
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 尝试解析为Python模块
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("strategy_module", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 检查各个策略列表
                strategy_lists = [
                    ("RANKING_RULES", "ranking"),
                    ("FILTER_RULES", "filter"), 
                    ("PREDICTION_RULES", "prediction"),
                    ("POSITION_POLICIES", "position"),
                    ("OPPORTUNITY_POLICIES", "opportunity")
                ]
                
                for list_name, category in strategy_lists:
                    if hasattr(module, list_name):
                        rules = getattr(module, list_name)
                        if isinstance(rules, list):
                            for i, rule in enumerate(rules):
                                rule_result = self.validate_rule(rule, category)
                                if not rule_result.is_valid:
                                    result.is_valid = False
                                    for error in rule_result.errors:
                                        result.add_error(f"[{list_name}[{i}]] {error['message']}", error.get('field'))
                                    for warning in rule_result.warnings:
                                        result.add_warning(f"[{list_name}[{i}]] {warning['message']}", warning.get('field'))
                                    for suggestion in rule_result.suggestions:
                                        result.add_suggestion(f"[{list_name}[{i}]] {suggestion['message']}", suggestion.get('field'))
                
            except Exception as e:
                result.add_error(f"无法解析Python文件: {e}")
                
        except Exception as e:
            result.add_error(f"无法读取文件: {e}")
        
        return result
    
    def validate_rule(self, rule: Dict[str, Any], category: str = "ranking") -> StrategyValidationResult:
        """验证单个策略规则"""
        result = StrategyValidationResult()
        
        if not isinstance(rule, dict):
            result.add_error("规则必须是字典格式")
            return result
        
        # 检查必填字段
        required_fields = self.REQUIRED_FIELDS.get(category, [])
        for field in required_fields:
            if field not in rule or not rule[field]:
                result.add_error(f"缺少必填字段: {field}", field)
                result.required_fields.append(field)
        
        # 检查字段类型和值
        self._validate_field_types(rule, category, result)
        
        # 检查表达式语法
        if "when" in rule and rule["when"]:
            expr_result = self._validate_expression(rule["when"], "when")
            if not expr_result["valid"]:
                result.add_error(f"when表达式错误: {expr_result['error']}", "when")
                result.syntax_issues.extend(expr_result.get("issues", []))
            else:
                result.missing_columns.extend(expr_result.get("missing_columns", []))
                result.missing_indicators.extend(expr_result.get("missing_indicators", []))
        
        if "check" in rule and rule["check"]:
            expr_result = self._validate_expression(rule["check"], "check")
            if not expr_result["valid"]:
                result.add_error(f"check表达式错误: {expr_result['error']}", "check")
                result.syntax_issues.extend(expr_result.get("issues", []))
            else:
                result.missing_columns.extend(expr_result.get("missing_columns", []))
                result.missing_indicators.extend(expr_result.get("missing_indicators", []))
        
        # 检查gate表达式
        if "gate" in rule and rule["gate"]:
            if isinstance(rule["gate"], str):
                expr_result = self._validate_expression(rule["gate"], "gate")
                if not expr_result["valid"]:
                    result.add_warning(f"gate表达式错误: {expr_result['error']}", "gate")
        
        # 检查clauses
        if "clauses" in rule and rule["clauses"]:
            if not isinstance(rule["clauses"], list):
                result.add_error("clauses必须是列表", "clauses")
            else:
                for i, clause in enumerate(rule["clauses"]):
                    if not isinstance(clause, dict):
                        result.add_error(f"clauses[{i}]必须是字典", f"clauses[{i}]")
                    else:
                        clause_result = self.validate_rule(clause, category)
                        if not clause_result.is_valid:
                            for error in clause_result.errors:
                                result.add_error(f"clauses[{i}]: {error['message']}", f"clauses[{i}].{error.get('field', '')}")
        
        # 提供建议
        self._provide_suggestions(rule, category, result)
        
        return result
    
    def _validate_field_types(self, rule: Dict[str, Any], category: str, result: StrategyValidationResult):
        """验证字段类型"""
        # 检查timeframe
        if "timeframe" in rule:
            tf = rule["timeframe"]
            if not isinstance(tf, str) or tf.upper() not in self.SUPPORTED_TIMEFRAMES:
                result.add_error(f"不支持的timeframe: {tf}，支持: {', '.join(self.SUPPORTED_TIMEFRAMES)}", "timeframe")
        
        # 检查window
        if "window" in rule:
            window = rule["window"]
            if not isinstance(window, (int, float)) or window <= 0:
                result.add_error(f"window必须是正整数: {window}", "window")
        
        # 检查score_windows（可选）
        if "score_windows" in rule:
            score_windows = rule["score_windows"]
            if score_windows is not None and (not isinstance(score_windows, (int, float)) or score_windows <= 0):
                result.add_error(f"score_windows必须是正整数: {score_windows}", "score_windows")
        
        # 检查scope
        if "scope" in rule:
            scope = rule["scope"]
            if not isinstance(scope, str):
                result.add_error(f"scope必须是字符串: {scope}", "scope")
            else:
                # 检查scope格式
                scope_upper = scope.upper()
                if not any(scope_upper.startswith(s) for s in self.SUPPORTED_SCOPES):
                    result.add_warning(f"scope格式可能不正确: {scope}", "scope")
        
        # 检查points
        if "points" in rule:
            points = rule["points"]
            if not isinstance(points, (int, float)):
                result.add_error(f"points必须是数字: {points}", "points")
        
        # 检查dist_points
        if "dist_points" in rule:
            dist_points = rule["dist_points"]
            if not isinstance(dist_points, list):
                result.add_error(f"dist_points必须是列表: {dist_points}", "dist_points")
            else:
                for i, dp in enumerate(dist_points):
                    if isinstance(dp, list) and len(dp) == 3:
                        if not all(isinstance(x, (int, float)) for x in dp):
                            result.add_error(f"dist_points[{i}]格式错误，应为[min, max, points]", f"dist_points[{i}]")
                    elif isinstance(dp, dict):
                        required_keys = {"min", "max", "points"}
                        if not all(k in dp for k in required_keys):
                            result.add_error(f"dist_points[{i}]缺少必要字段: {required_keys - set(dp.keys())}", f"dist_points[{i}]")
                    else:
                        result.add_error(f"dist_points[{i}]格式错误", f"dist_points[{i}]")
    
    def _validate_expression(self, expr: str, field: str) -> Dict[str, Any]:
        """验证TDX表达式"""
        result = {
            "valid": False,
            "error": None,
            "missing_columns": [],
            "missing_indicators": [],
            "issues": []
        }
        
        if not expr or not expr.strip():
            result["error"] = "表达式为空"
            return result
        
        try:
            # 使用现有的诊断功能
            diag_result = diagnose_expr(expr)
            
            if not diag_result["ok"]:
                result["error"] = diag_result["error"]
                result["missing_columns"] = diag_result.get("missing", [])
                return result
            
            # 检查语法
            syntax_issues = self._check_expression_syntax(expr)
            result["issues"] = syntax_issues
            
            # 检查缺失的列和指标
            need_cols = diag_result.get("need_cols", [])
            missing = diag_result.get("missing", [])
            result["missing_columns"] = missing
            
            # 检查指标依赖
            indicators = names_in_expr(expr)
            missing_indicators = []
            for ind in indicators:
                if ind not in self.available_indicators:
                    missing_indicators.append(ind)
            result["missing_indicators"] = missing_indicators
            
            result["valid"] = True
            
        except Exception as e:
            result["error"] = f"表达式验证异常: {e}"
        
        return result
    
    def _check_expression_syntax(self, expr: str) -> List[str]:
        """检查表达式语法问题"""
        issues = []
        
        # 检查括号配对
        if expr.count('(') != expr.count(')'):
            issues.append("括号不配对")
        
        # 检查语法错误
        if re.search(r'[=]{2,}', expr):  # 连续等号
            issues.append("发现连续等号，可能应为单个等号")
        
        if re.search(r'[&]{2,}', expr):  # 连续&
            issues.append("发现连续&符号，可能应为单个&")
        
        if re.search(r'[|]{2,}', expr):  # 连续|
            issues.append("发现连续|符号，可能应为单个|")
        
        # 检查函数调用格式
        if re.search(r'[A-Z_][A-Z0-9_]*\s*\([^)]*\)', expr):
            # 检查函数名后是否有空格
            if re.search(r'[A-Z_][A-Z0-9_]*\s+\(', expr):
                issues.append("函数名后不应有空格")
        
        # 检查比较运算符
        if re.search(r'[<>=!]=', expr):
            # 检查是否有无效的比较运算符组合
            if re.search(r'[<>=!]{3,}', expr):
                issues.append("发现无效的比较运算符组合")
        
        return issues
    
    def _provide_suggestions(self, rule: Dict[str, Any], category: str, result: StrategyValidationResult):
        """提供改进建议"""
        # 检查是否有name字段
        if "name" not in rule or not rule["name"]:
            result.add_suggestion("建议添加name字段以便识别规则", "name")
        
        # 检查是否有explain字段
        if "explain" not in rule or not rule["explain"]:
            result.add_suggestion("建议添加explain字段说明规则用途", "explain")
        
        # 检查scope设置
        if "scope" in rule and rule["scope"] == "ANY":
            result.add_suggestion("scope为ANY时建议考虑使用LAST或EACH", "scope")
        
        # 检查window设置
        if "window" in rule:
            window = rule["window"]
            if window > 100:
                result.add_warning(f"window值较大({window})，可能影响性能", "window")
            elif window < 5:
                result.add_warning(f"window值较小({window})，可能数据不足", "window")
        
        # 检查表达式复杂度
        if "when" in rule and rule["when"]:
            expr = rule["when"]
            if len(expr) > 200:
                result.add_suggestion("表达式较长，建议拆分为多个简单规则", "when")
            
            # 检查是否使用了安全除法
            if "/" in expr and "SAFE_DIV" not in expr:
                result.add_suggestion("建议使用SAFE_DIV避免除零错误", "when")
        
        # 检查points设置
        if "points" in rule:
            points = rule["points"]
            if isinstance(points, (int, float)) and abs(points) > 50:
                result.add_warning(f"points值较大({points})，可能影响评分平衡", "points")
    
    def validate_json_strategy(self, json_str: str) -> StrategyValidationResult:
        """验证JSON格式的策略"""
        result = StrategyValidationResult()
        
        try:
            data = json.loads(json_str)
            
            if isinstance(data, dict):
                # 检查是否有规则列表
                rule_lists = ["prescreen", "rules", "PREDICTION_RULES", "POSITION_POLICIES", "OPPORTUNITY_POLICIES"]
                found_rules = False
                
                for list_name in rule_lists:
                    if list_name in data and isinstance(data[list_name], list):
                        found_rules = True
                        category = "ranking" if list_name == "rules" else "filter" if list_name == "prescreen" else list_name.lower()
                        
                        for i, rule in enumerate(data[list_name]):
                            if isinstance(rule, dict):
                                rule_result = self.validate_rule(rule, category)
                                if not rule_result.is_valid:
                                    result.is_valid = False
                                    for error in rule_result.errors:
                                        result.add_error(f"[{list_name}[{i}]] {error['message']}", error.get('field'))
                                for warning in rule_result.warnings:
                                    result.add_warning(f"[{list_name}[{i}]] {warning['message']}", warning.get('field'))
                                for suggestion in rule_result.suggestions:
                                    result.add_suggestion(f"[{list_name}[{i}]] {suggestion['message']}", suggestion.get('field'))
                
                if not found_rules:
                    result.add_error("未找到有效的规则列表")
            
            elif isinstance(data, list):
                # 直接是规则列表
                for i, rule in enumerate(data):
                    if isinstance(rule, dict):
                        rule_result = self.validate_rule(rule)
                        if not rule_result.is_valid:
                            result.is_valid = False
                            for error in rule_result.errors:
                                result.add_error(f"[{i}] {error['message']}", error.get('field'))
                        for warning in rule_result.warnings:
                            result.add_warning(f"[{i}] {warning['message']}", warning.get('field'))
                        for suggestion in rule_result.suggestions:
                            result.add_suggestion(f"[{i}] {suggestion['message']}", suggestion.get('field'))
            
            else:
                result.add_error("JSON格式不正确，应为对象或数组")
                
        except json.JSONDecodeError as e:
            result.add_error(f"JSON解析错误: {e}")
        except Exception as e:
            result.add_error(f"验证过程出错: {e}")
        
        return result


# 便捷函数
def validate_strategy_file(file_path: str) -> StrategyValidationResult:
    """验证策略文件的便捷函数"""
    validator = StrategyValidator()
    return validator.validate_strategy_file(file_path)


def validate_json_strategy(json_str: str) -> StrategyValidationResult:
    """验证JSON策略的便捷函数"""
    validator = StrategyValidator()
    return validator.validate_json_strategy(json_str)


def validate_rule(rule: Dict[str, Any], category: str = "ranking") -> StrategyValidationResult:
    """验证单个规则的便捷函数"""
    validator = StrategyValidator()
    return validator.validate_rule(rule, category)


# 验证当前策略文件
def validate_current_strategies() -> StrategyValidationResult:
    """验证当前策略文件中的所有规则"""
    validator = StrategyValidator()
    result = StrategyValidationResult()
    
    # 验证各个策略列表
    strategy_lists = [
        (RANKING_RULES, "RANKING_RULES", "ranking"),
        (FILTER_RULES, "FILTER_RULES", "filter"),
        (PREDICTION_RULES, "PREDICTION_RULES", "prediction"),
        (POSITION_POLICIES, "POSITION_POLICIES", "position"),
        (OPPORTUNITY_POLICIES, "OPPORTUNITY_POLICIES", "opportunity")
    ]
    
    for rules, list_name, category in strategy_lists:
        if rules:
            for i, rule in enumerate(rules):
                rule_result = validator.validate_rule(rule, category)
                if not rule_result.is_valid:
                    result.is_valid = False
                    for error in rule_result.errors:
                        result.add_error(f"[{list_name}[{i}]] {error['message']}", error.get('field'))
                for warning in rule_result.warnings:
                    result.add_warning(f"[{list_name}[{i}]] {warning['message']}", warning.get('field'))
                for suggestion in rule_result.suggestions:
                    result.add_suggestion(f"[{list_name}[{i}]] {suggestion['message']}", suggestion.get('field'))
    
    return result


# -----------------------------------------------------------------------------
