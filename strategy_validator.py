# -*- coding: utf-8 -*-
"""
策略验证器 - 用于检查策略文件的语法和可用性
"""
import json
import re
import ast
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import pandas as pd
import numpy as np

# 导入现有的模块
from tdx_compat import evaluate_bool, evaluate
from indicators import names_in_expr, REGISTRY
from scoring_core import diagnose_expr, _pick_ref_date, _list_codes_for_day, _read_stock_df, _start_for_tf_window, _resample, _window_slice, _ctx_df


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
        "ranking": ["name", "timeframe", "window", "scope", "points", "explain", "show_reason", "as", "gate", "clauses", "dist_points"],
        "filter": ["name", "timeframe", "window", "scope", "reason", "hard_penalty", "gate", "clauses"],
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
            if meta.out:
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
            
            # 检查语法问题
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
        
        # 检查常见语法错误
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
