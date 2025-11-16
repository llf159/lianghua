# -*- coding: utf-8 -*-
"""
规则编辑辅助工具模块
用于在 Streamlit UI 中提供可视化的规则配置界面
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import streamlit as st

try:
    from tdx_compat import translate_expression, evaluate_bool, evaluate
    from indicators import names_in_expr, REGISTRY
    from scoring_core import diagnose_expr
except ImportError:
    translate_expression = None
    evaluate_bool = None
    evaluate = None
    names_in_expr = lambda x: []
    REGISTRY = {}
    def diagnose_expr(expr): return {"ok": True, "error": None, "missing": [], "need_cols": []}


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
        "ranking": ["name", "timeframe", "score_windows", "scope", "points", "explain", "show_reason", "as", "gate", "trigger", "require", "clauses", "dist_points"],
        "filter": ["name", "timeframe", "score_windows", "scope", "reason", "hard_penalty", "gate", "trigger", "require", "clauses"],
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
        if REGISTRY:
            for indicator_name, meta in REGISTRY.items():
                if hasattr(meta, 'out') and meta.out:
                    for col in meta.out.keys():
                        self.available_indicators.add(col.lower())
                        self.available_columns.add(col.lower())
    
    def validate_rule(self, rule: Dict[str, Any], category: str = "ranking") -> StrategyValidationResult:
        """验证单个策略规则"""
        result = StrategyValidationResult()
        
        if not isinstance(rule, dict):
            result.add_error("规则必须是字典格式")
            return result
        
        # 检查必填字段
        # 注意：如果使用clauses，则不需要when字段
        has_clauses = "clauses" in rule and rule["clauses"]
        required_fields = self.REQUIRED_FIELDS.get(category, [])
        for field in required_fields:
            # 如果使用clauses且必填字段是when，则跳过检查
            if field == "when" and has_clauses:
                continue
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
        
        # 检查gate/trigger/require表达式（功能相同，字段名不同）
        for gate_field in ["gate", "trigger", "require"]:
            if gate_field in rule and rule[gate_field]:
                gate_value = rule[gate_field]
                if isinstance(gate_value, str):
                    expr_result = self._validate_expression(gate_value, gate_field)
                    if not expr_result["valid"]:
                        result.add_warning(f"{gate_field}表达式错误: {expr_result['error']}", gate_field)
                elif isinstance(gate_value, dict):
                    # 子规则对象格式
                    gate_result = self.validate_rule(gate_value, category)
                    if not gate_result.is_valid:
                        for error in gate_result.errors:
                            result.add_warning(f"{gate_field}[{error.get('field', '')}]: {error['message']}", f"{gate_field}.{error.get('field', '')}")
                elif isinstance(gate_value, list):
                    # 子句数组格式
                    for i, gate_clause in enumerate(gate_value):
                        if isinstance(gate_clause, dict):
                            gate_clause_result = self.validate_rule(gate_clause, category)
                            if not gate_clause_result.is_valid:
                                for error in gate_clause_result.errors:
                                    result.add_warning(f"{gate_field}[{i}][{error.get('field', '')}]: {error['message']}", f"{gate_field}[{i}].{error.get('field', '')}")
        
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
        
        # 检查score_windows（可选，但推荐使用）
        if "score_windows" in rule:
            score_windows = rule["score_windows"]
            if score_windows is not None and (not isinstance(score_windows, (int, float)) or score_windows <= 0):
                result.add_error(f"score_windows必须是正整数: {score_windows}", "score_windows")
        
        # 检查window（已废弃，但为了向后兼容仍支持）
        if "window" in rule:
            window = rule["window"]
            if not isinstance(window, (int, float)) or window <= 0:
                result.add_warning(f"window字段已废弃，请使用score_windows。window必须是正整数: {window}", "window")
        
        # 检查scope
        if "scope" in rule:
            scope = rule["scope"]
            if not isinstance(scope, str):
                result.add_error(f"scope必须是字符串: {scope}", "scope")
            else:
                # 检查scope格式
                scope_upper = scope.upper().strip()
                # 支持基本格式：ANY, LAST, ALL, EACH, RECENT, DIST, NEAR
                # 支持COUNT>=k格式
                # 支持CONSEC>=m格式
                scope_valid = False
                if scope_upper in self.SUPPORTED_SCOPES:
                    scope_valid = True
                elif scope_upper.startswith("COUNT>="):
                    try:
                        k = int(scope_upper.split(">=")[1])
                        if k > 0:
                            scope_valid = True
                    except (ValueError, IndexError):
                        pass
                elif scope_upper.startswith("CONSEC>="):
                    try:
                        m = int(scope_upper.split(">=")[1])
                        if m > 0:
                            scope_valid = True
                    except (ValueError, IndexError):
                        pass
                
                if not scope_valid:
                    result.add_warning(f"scope格式可能不正确: {scope}，支持格式：ANY/LAST/ALL/EACH/RECENT/DIST/NEAR/COUNT>=k/CONSEC>=m", "scope")
        
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
            if diagnose_expr:
                diag_result = diagnose_expr(expr)
                
                if not diag_result["ok"]:
                    result["error"] = diag_result["error"]
                    result["missing_columns"] = diag_result.get("missing", [])
                    return result
                
                # 检查缺失的列和指标
                need_cols = diag_result.get("need_cols", [])
                missing = diag_result.get("missing", [])
                result["missing_columns"] = missing
            else:
                diag_result = {"ok": True}
            
            # 检查语法
            syntax_issues = self._check_expression_syntax(expr)
            result["issues"] = syntax_issues
            
            # 检查指标依赖
            if names_in_expr:
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
        
        # 检查score_windows设置
        if "score_windows" in rule:
            score_windows = rule["score_windows"]
            if score_windows is not None:
                if score_windows > 100:
                    result.add_warning(f"score_windows值较大({score_windows})，可能影响性能", "score_windows")
                elif score_windows < 5:
                    result.add_warning(f"score_windows值较小({score_windows})，可能数据不足", "score_windows")
        
        # 检查window设置（已废弃，但为了向后兼容仍支持）
        if "window" in rule:
            window = rule["window"]
            if window > 100:
                result.add_warning(f"window字段已废弃，请使用score_windows。window值较大({window})，可能影响性能", "window")
            elif window < 5:
                result.add_warning(f"window字段已废弃，请使用score_windows。window值较小({window})，可能数据不足", "window")
        
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


def _read_md_file(path: str) -> str:
    """读取 Markdown 文件内容"""
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except Exception:
        # 兜底提示，避免页面报错
        return "⚠️ 未找到帮助文档：" + path


def _compile_rule_expression(rule: dict) -> dict:
    """编译单个规则表达式，返回翻译和分析结果"""
    result = {
        'original': '',
        'translated': '',
        'when': None,
        'check': None,
        'gate': None,
    }
    
    # 处理 when 表达式
    if 'when' in rule and rule['when']:
        original = rule['when'].strip()
        result['original'] = original
        result['when'] = original
        if translate_expression:
            try:
                translated = translate_expression(original)
                result['translated'] = translated
            except Exception as e:
                result['translated'] = f"翻译错误: {e}"
    
    # 处理 check 表达式
    if 'check' in rule and rule['check']:
        check_expr = rule['check'].strip()
        result['check'] = check_expr
        if not result['original']:
            result['original'] = f"[check]: {check_expr}"
        else:
            result['original'] += f"\n[check]: {check_expr}"
        if translate_expression:
            try:
                translated = translate_expression(check_expr)
                if result['translated']:
                    result['translated'] += f"\n[check翻译]: {translated}"
                else:
                    result['translated'] = f"[check翻译]: {translated}"
            except Exception as e:
                if result['translated']:
                    result['translated'] += f"\n[check翻译错误]: {e}"
                else:
                    result['translated'] = f"[check翻译错误]: {e}"
    
    # 处理 gate 表达式
    if 'gate' in rule and rule['gate']:
        gate_expr = rule['gate'].strip()
        result['gate'] = gate_expr
        if result['translated']:
            result['translated'] += f"\n[gate]: {gate_expr}"
        else:
            result['translated'] = f"[gate]: {gate_expr}"
    
    return result


def _analyze_rule_logic(rule: dict) -> dict:
    """分析规则逻辑，返回获取数据、计算数据、判断条件的说明"""
    when_expr = rule.get('when', '') or ''
    check_expr = rule.get('check', '') or ''
    gate_expr = rule.get('gate', '') or ''
    
    # 数据获取列表
    data_needed = []
    # 计算列表
    calculations = []
    # 判断条件
    conditions = []
    
    expr_to_analyze = when_expr or check_expr or ''
    
    if not expr_to_analyze:
        return {
            'data': '无表达式',
            'calc': '无',
            'judge': '无'
        }
    
    # 分析需要的数据
    if 'O' in expr_to_analyze or 'open' in expr_to_analyze.lower():
        data_needed.append('开盘价(O)')
    if 'C' in expr_to_analyze or 'close' in expr_to_analyze.lower():
        data_needed.append('收盘价(C)')
    if 'H' in expr_to_analyze or 'high' in expr_to_analyze.lower():
        data_needed.append('最高价(H)')
    if 'L' in expr_to_analyze or 'low' in expr_to_analyze.lower():
        data_needed.append('最低价(L)')
    if 'V' in expr_to_analyze or 'vol' in expr_to_analyze.lower():
        data_needed.append('成交量(V)')
    if 'duokong_long' in expr_to_analyze or 'duokong_short' in expr_to_analyze:
        data_needed.append('多空均线(duokong_long/duokong_short)')
    if 'diff' in expr_to_analyze.lower():
        data_needed.append('MACD的DIFF值')
    if 'j' in expr_to_analyze.lower():
        data_needed.append('KDJ的J值')
    if 'bbi' in expr_to_analyze.lower():
        data_needed.append('BBI均线')
    if 'z_score' in expr_to_analyze.lower():
        data_needed.append('Z-score值')
    if 'vr' in expr_to_analyze.lower():
        data_needed.append('VR指标值')
    
    # 分析计算
    if 'MA(' in expr_to_analyze or 'HHV(' in expr_to_analyze or 'LLV(' in expr_to_analyze:
        calculations.append('计算移动平均/最高价/最低价')
    if 'TS_RANK(' in expr_to_analyze:
        calculations.append('计算时间序列排名')
    if 'TS_PCT(' in expr_to_analyze:
        calculations.append('计算时间序列分位数')
    if 'SAFE_DIV(' in expr_to_analyze:
        calculations.append('计算安全除法(涨跌幅/比率)')
    if 'REF(' in expr_to_analyze:
        calculations.append('引用历史数据')
    if 'COUNT(' in expr_to_analyze:
        calculations.append('统计满足条件的次数')
    if 'CROSS(' in expr_to_analyze:
        calculations.append('判断交叉信号')
    if 'BARSLAST(' in expr_to_analyze:
        calculations.append('计算距离上次条件的周期数')
    if 'ATAN(' in expr_to_analyze or 'ANGLE' in expr_to_analyze:
        calculations.append('计算角度/斜率')
    if 'GET_LAST' in expr_to_analyze or 'REVERSE_PRICE' in expr_to_analyze:
        calculations.append('获取历史条件价格或反推价格')
    
    # 简化的判断条件说明
    if when_expr:
        # 根据表达式内容生成简化的判断说明
        if '> duokong_long' in when_expr or '> duokong_short' in when_expr:
            conditions.append('判断价格是否高于均线')
        if '< duokong_long' in when_expr or '< duokong_short' in when_expr:
            conditions.append('判断价格是否低于均线')
        if 'CROSS(' in when_expr:
            conditions.append('判断是否发生金叉/死叉')
        if 'TS_RANK' in when_expr:
            conditions.append('判断排名是否达到要求')
        if 'TS_PCT' in when_expr:
            conditions.append('判断分位数是否达到要求')
        if 'COUNT(' in when_expr:
            conditions.append('判断满足条件的次数')
        if '<= 13' in when_expr or 'j <' in when_expr or 'j <=' in when_expr:
            conditions.append('判断J值是否超卖')
        if 'SAFE_DIV(' in when_expr and '>=' in when_expr:
            conditions.append('判断涨跌幅或比率是否达到阈值')
        if 'REF(' in when_expr:
            conditions.append('对比历史数据')
    
    if gate_expr:
        conditions.append(f'额外条件(gate): {gate_expr}')
    
    return {
        'data': '、'.join(set(data_needed)) if data_needed else '基础价格和成交量数据',
        'calc': '、'.join(set(calculations)) if calculations else '直接使用原始数据',
        'judge': '；'.join(set(conditions)) if conditions else '直接判断表达式真假'
    }


def render_rule_editor():
    """
    渲染规则编辑辅助工具界面
    
    注意：此函数应该在 with tab: 上下文中调用
    """
    st.subheader("规则编辑辅助工具")
    st.info("通过可视化界面配置策略规则，自动生成规则配置")
    
    # 规则类型选择
    rule_type = st.selectbox(
        "选择规则类型",
        ["排名策略 (ranking)", "筛选策略 (filter)", "模拟策略 (prediction)", "持仓策略 (position)", "买点策略 (opportunity)"],
        help="选择要创建的规则类型，不同类型有不同的必填字段"
    )
    
    # 策略类型说明
    with st.expander("策略类型说明", expanded=False):
        st.markdown("""
    **策略类型详解：**
    
    - **排名策略 (ranking)**: 用于股票评分排名，使用 `when` 表达式判断条件，通过 `points` 字段加分
      - 配置项：name, timeframe, score_windows, scope, points, explain, show_reason, as, gate/trigger/require, clauses, dist_points
      - 注意：window字段已废弃，请使用score_windows
      
    - **筛选策略 (filter)**: 用于股票筛选过滤，使用 `when` 表达式判断条件，可设置 `hard_penalty` 硬性惩罚
      - 配置项：name, timeframe, score_windows, scope, reason, hard_penalty, gate/trigger/require, clauses
      - 注意：window字段已废弃，请使用score_windows
      
    - **模拟策略 (prediction)**: 用于市场场景模拟，使用 `check` 表达式判断条件，需要 `scenario` 场景名称
      - 配置项：name, scenario
      
    - **持仓策略 (position)**: 用于持仓股票检查，使用 `when` 表达式判断买卖时机
      - 配置项：name, explain
      
    - **买点策略 (opportunity)**: 用于寻找买入机会，使用 `when` 表达式判断买入条件
      - 配置项：name, explain
    """)
    
    # 预设模板选择
    col_template1, col_template2 = st.columns([3, 1])
    with col_template1:
        template_option = st.selectbox(
            "选择预设模板（可选）",
            ["自定义", "均线突破", "成交量放大", "价格回调", "趋势确认", "技术指标"],
            help="选择预设模板可以快速填充常用配置"
        )
    with col_template2:
        if st.button("🔄 清除模板", help="清除当前模板设置，恢复默认值"):
            # 清除所有模板相关的session_state
            for key in ['template_name', 'template_timeframe', 'template_window', 'template_score_windows',
                       'template_scope', 'template_scope_count', 'template_scope_consec', 
                       'template_points', 'template_explain', 'template_when',
                       'template_check', 'template_scenario']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # 提取规则类型
    rule_category = rule_type.split(" ")[1].strip("()")
    
    # 根据模板预设值
    if template_option != "自定义":
        if template_option == "均线突破":
            st.session_state.template_name = "均线突破"
            st.session_state.template_timeframe = "D"
            st.session_state.template_window = 20
            st.session_state.template_score_windows = 20
            st.session_state.template_scope = "EACH"
            st.session_state.template_points = 2
            st.session_state.template_explain = "价格突破均线，确认上涨趋势"
            st.session_state.template_when = "C > MA(C, 20)"
        elif template_option == "成交量放大":
            st.session_state.template_name = "成交量放大"
            st.session_state.template_timeframe = "D"
            st.session_state.template_window = 20
            st.session_state.template_score_windows = 20
            st.session_state.template_scope = "EACH"
            st.session_state.template_points = 1
            st.session_state.template_explain = "成交量显著放大，显示资金关注"
            st.session_state.template_when = "V > MA(V, 20) * 1.5"
        elif template_option == "价格回调":
            st.session_state.template_name = "价格回调"
            st.session_state.template_timeframe = "D"
            st.session_state.template_window = 10
            st.session_state.template_score_windows = 10
            st.session_state.template_scope = "LAST"
            st.session_state.template_points = -5
            st.session_state.template_explain = "短期价格回调，风险提示"
            st.session_state.template_when = "C < MA(C, 5)"
        elif template_option == "趋势确认":
            st.session_state.template_name = "趋势确认"
            st.session_state.template_timeframe = "D"
            st.session_state.template_window = 20
            st.session_state.template_score_windows = 20
            st.session_state.template_scope = "EACH"
            st.session_state.template_points = 3
            st.session_state.template_explain = "多重条件确认趋势"
            st.session_state.template_when = "C > MA(C, 20) AND MA(C, 5) > MA(C, 20) AND V > MA(V, 20)"
        elif template_option == "技术指标":
            st.session_state.template_name = "技术指标"
            st.session_state.template_timeframe = "D"
            st.session_state.template_window = 14
            st.session_state.template_score_windows = 14
            st.session_state.template_scope = "EACH"
            st.session_state.template_points = 2
            st.session_state.template_explain = "基于技术指标的信号"
            st.session_state.template_when = "RSI < 30 AND C > MA(C, 10)"
    
    # 为模拟策略添加特殊模板
    if rule_category == "prediction":
        if template_option == "均线突破":
            st.session_state.template_check = "C > MA(C, 20)"
            st.session_state.template_scenario = "均线突破场景"
        elif template_option == "成交量放大":
            st.session_state.template_check = "V > MA(V, 20) * 1.5"
            st.session_state.template_scenario = "成交量放大场景"
        elif template_option == "价格回调":
            st.session_state.template_check = "C < MA(C, 5)"
            st.session_state.template_scenario = "价格回调场景"
        elif template_option == "趋势确认":
            st.session_state.template_check = "C > MA(C, 20) AND MA(C, 5) > MA(C, 20) AND V > MA(V, 20)"
            st.session_state.template_scenario = "趋势确认场景"
        elif template_option == "技术指标":
            st.session_state.template_check = "RSI < 30 AND C > MA(C, 10)"
            st.session_state.template_scenario = "技术指标场景"
    
    # 初始化变量
    use_clauses = False
    when_expr = ""
    check_expr = ""
    scenario = ""
    scenario_config = None
    explain = ""
    rule_name = ""
    timeframe = "D"
    score_windows = 60
    scope = "ANY"
    scope_count_value = 1
    scope_consec_value = 1
    points = 0
    show_reason = True
    rule_as = "auto"
    gate = ""
    trigger = ""
    require = ""
    dist_points_config = ""
    use_dist_points = False
    hard_penalty = False
    reason = ""
    clauses_config = ""
    
    # 场景配置变量
    use_scenario = False
    price_mode = "close_pct"
    pct = 0.0
    gap_pct = 0.0
    hl_mode = "follow"
    range_pct = 1.5
    atr_mult = 1.0
    vol_mode = "same"
    vol_arg = 0.0
    lock_higher_than_open = False
    lock_inside_day = False
    warmup_days = 60
    
    # 根据策略类型显示不同的配置项
    if rule_category == "ranking":
        # 排名策略配置
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 基础配置")
            
            # 规则名称
            rule_name = st.text_input(
                "规则名称 (name)",
                value=st.session_state.get('template_name', ''),
                placeholder="例如：短期上涨趋势",
                help="规则的显示名称，用于识别和说明"
            )
            
            # 时间周期
            timeframe_options = ["D", "W", "M", "60MIN"]
            timeframe_index = timeframe_options.index(st.session_state.get('template_timeframe', 'D')) if st.session_state.get('template_timeframe', 'D') in timeframe_options else 0
            timeframe = st.selectbox(
                "时间周期 (timeframe)",
                timeframe_options,
                index=timeframe_index,
                help="数据的时间周期：D(日线)、W(周线)、M(月线)、60MIN(60分钟)"
            )
            
            # 计分窗口（score_windows）
            score_windows = st.number_input(
                "计分窗口 (score_windows)",
                min_value=1,
                max_value=500,
                value=st.session_state.get('template_score_windows', st.session_state.get('template_window', 60)),
                help="用于计分判断的历史数据条数，通常设置为5-100。注意：window字段已废弃，请使用score_windows"
            )
            
            # 命中口径
            scope_options = ["ANY", "LAST", "ALL", "EACH", "RECENT", "DIST", "NEAR", "CONSEC", "COUNT"]
            scope_index = scope_options.index(st.session_state.get('template_scope', 'ANY')) if st.session_state.get('template_scope', 'ANY') in scope_options else 0
            scope_base = st.selectbox(
                "命中口径 (scope)",
                scope_options,
                index=scope_index,
                help="规则命中的判断方式：ANY(任意)、LAST(最近)、ALL(全部)、EACH(每个)等"
            )
            
            # 处理COUNT和CONSEC格式
            if scope_base == "COUNT":
                scope_count_value = st.number_input(
                    "COUNT阈值",
                    min_value=1,
                    max_value=500,
                    value=st.session_state.get('template_scope_count', 1),
                    help="COUNT>=k格式，k的值"
                )
                scope = f"COUNT>={scope_count_value}"
            elif scope_base == "CONSEC":
                scope_consec_value = st.number_input(
                    "CONSEC连续天数",
                    min_value=1,
                    max_value=500,
                    value=st.session_state.get('template_scope_consec', 1),
                    help="CONSEC>=m格式，m的值"
                )
                scope = f"CONSEC>={scope_consec_value}"
            else:
                scope = scope_base
            
            # 分数
            points = st.number_input(
                "分数 (points)",
                value=st.session_state.get('template_points', 0),
                step=1,
                help="规则命中时的加分或减分，正数为加分，负数为减分"
            )
        
        with col_right:
            st.markdown("#### 高级配置")
            
            # 说明文字
            explain = st.text_area(
                "说明文字 (explain)",
                value=st.session_state.get('template_explain', ''),
                placeholder="例如：短期上涨趋势，价格突破短期均线",
                help="规则的详细说明，用于解释规则的作用"
            )
            
            # 是否显示理由
            show_reason = st.checkbox(
                "显示理由 (show_reason)",
                value=True,
                help="是否在结果中显示此规则的命中理由"
            )
            
            # 分类标签
            rule_as = st.selectbox(
                "分类标签 (as)",
                ["auto", "opportunity", "highlight", "drawback"],
                index=0,
                help="规则分类：auto(自动)、opportunity(机会)、highlight(高亮)、drawback(缺点)"
            )
            
            # 前置门槛（支持gate/trigger/require）
            gate_type = st.selectbox(
                "前置门槛类型",
                ["gate", "trigger", "require", "不使用"],
                index=3,
                help="前置门槛类型：gate/trigger/require功能相同，只是字段名不同"
            )
            
            if gate_type != "不使用":
                gate = st.text_input(
                    f"前置门槛 ({gate_type})",
                    placeholder="例如：C > MA(C, 5)",
                    help="规则生效的前置条件，必须满足才能执行此规则。支持字符串表达式、子规则对象或子句数组（JSON格式）"
                )
                trigger = gate if gate_type == "trigger" else ""
                require = gate if gate_type == "require" else ""
                if gate_type != "gate":
                    gate = ""
            else:
                gate = ""
                trigger = ""
                require = ""
            
            # 多子句组合
            use_clauses = st.checkbox(
                "使用多子句组合 (clauses)",
                help="使用clauses替代when字段，支持更复杂的逻辑组合"
            )
            
            # 分布分数（dist_points）- 用于RECENT/DIST/NEAR
            if scope_base in ["RECENT", "DIST", "NEAR"]:
                use_dist_points = st.checkbox(
                    "使用分布分数 (dist_points)",
                    help="根据最近一次命中的距离分段给分，仅用于RECENT/DIST/NEAR口径"
                )
                if use_dist_points:
                    dist_points_config = st.text_area(
                        "分布分数配置 (dist_points)",
                        placeholder='[[0,5,20], [6,10,10], [11,20,5]]\n或\n[{"min":0, "max":5, "points":20}, {"min":6, "max":10, "points":10}]',
                        help="JSON格式的列表，每个元素为[min, max, points]三元组或{min, max, points}对象"
                    )
                    with st.expander("分布分数配置说明", expanded=False):
                        st.markdown("""
                        **格式1：区间三元组**
                        ```json
                        [[0,5,20], [6,10,10], [11,20,5]]
                        ```
                        表示：距离0-5天给20分，6-10天给10分，11-20天给5分
                        
                        **格式2：显式对象**
                        ```json
                        [
                          {"min":0, "max":5, "points":20},
                          {"min":6, "max":10, "points":10},
                          {"min":11, "max":20, "points":5}
                        ]
                        ```
                        """)
            else:
                use_dist_points = False
                dist_points_config = ""
    
    elif rule_category == "filter":
        # 筛选策略配置
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 基础配置")
            
            # 规则名称
            rule_name = st.text_input(
                "规则名称 (name)",
                value=st.session_state.get('template_name', ''),
                placeholder="例如：基本面筛选",
                help="规则的显示名称，用于识别和说明"
            )
            
            # 时间周期
            timeframe_options = ["D", "W", "M", "60MIN"]
            timeframe_index = timeframe_options.index(st.session_state.get('template_timeframe', 'D')) if st.session_state.get('template_timeframe', 'D') in timeframe_options else 0
            timeframe = st.selectbox(
                "时间周期 (timeframe)",
                timeframe_options,
                index=timeframe_index,
                help="数据的时间周期：D(日线)、W(周线)、M(月线)、60MIN(60分钟)"
            )
            
            # 计分窗口（score_windows）
            score_windows = st.number_input(
                "计分窗口 (score_windows)",
                min_value=1,
                max_value=500,
                value=st.session_state.get('template_score_windows', st.session_state.get('template_window', 60)),
                help="用于计分判断的历史数据条数，通常设置为5-100。注意：window字段已废弃，请使用score_windows"
            )
            
            # 命中口径
            scope_options = ["ANY", "LAST", "ALL", "EACH", "RECENT", "DIST", "NEAR", "CONSEC", "COUNT"]
            scope_index = scope_options.index(st.session_state.get('template_scope', 'ANY')) if st.session_state.get('template_scope', 'ANY') in scope_options else 0
            scope_base = st.selectbox(
                "命中口径 (scope)",
                scope_options,
                index=scope_index,
                help="规则命中的判断方式：ANY(任意)、LAST(最近)、ALL(全部)、EACH(每个)等"
            )
            
            # 处理COUNT和CONSEC格式
            if scope_base == "COUNT":
                scope_count_value = st.number_input(
                    "COUNT阈值",
                    min_value=1,
                    max_value=500,
                    value=st.session_state.get('template_scope_count', 1),
                    help="COUNT>=k格式，k的值"
                )
                scope = f"COUNT>={scope_count_value}"
            elif scope_base == "CONSEC":
                scope_consec_value = st.number_input(
                    "CONSEC连续天数",
                    min_value=1,
                    max_value=500,
                    value=st.session_state.get('template_scope_consec', 1),
                    help="CONSEC>=m格式，m的值"
                )
                scope = f"CONSEC>={scope_consec_value}"
            else:
                scope = scope_base
        
        with col_right:
            st.markdown("#### 筛选配置")
            
            # 硬性惩罚
            hard_penalty = st.checkbox(
                "硬性惩罚 (hard_penalty)",
                help="是否启用硬性惩罚，启用后不符合条件的股票将被直接排除"
            )
            
            # 筛选原因
            reason = st.text_input(
                "筛选原因 (reason)",
                value=st.session_state.get('template_reason', ''),
                placeholder="例如：不符合基本面要求",
                help="筛选策略的拒绝原因说明"
            )
            
            # 前置门槛（支持gate/trigger/require）
            gate_type = st.selectbox(
                "前置门槛类型",
                ["gate", "trigger", "require", "不使用"],
                index=3,
                help="前置门槛类型：gate/trigger/require功能相同，只是字段名不同"
            )
            
            if gate_type != "不使用":
                gate = st.text_input(
                    f"前置门槛 ({gate_type})",
                    placeholder="例如：C > MA(C, 5)",
                    help="规则生效的前置条件，必须满足才能执行此规则。支持字符串表达式、子规则对象或子句数组（JSON格式）"
                )
                trigger = gate if gate_type == "trigger" else ""
                require = gate if gate_type == "require" else ""
                if gate_type != "gate":
                    gate = ""
            else:
                gate = ""
                trigger = ""
                require = ""
            
            # 多子句组合
            use_clauses = st.checkbox(
                "使用多子句组合 (clauses)",
                help="使用clauses替代when字段，支持更复杂的逻辑组合"
            )
    
    elif rule_category == "prediction":
        # 模拟策略配置
        st.markdown("#### 基础配置")
        
        # 规则名称
        rule_name = st.text_input(
            "规则名称 (name)",
            value=st.session_state.get('template_name', ''),
            placeholder="例如：上涨场景模拟",
            help="规则的显示名称，用于识别和说明"
        )
        
        st.markdown("#### 场景配置")
        
        # 场景配置开关
        use_scenario = st.checkbox(
            "使用场景配置 (scenario)",
            help="是否使用内置场景配置，否则使用默认场景"
        )
        
        if use_scenario:
            col_scenario1, col_scenario2 = st.columns([1, 1])
            
            with col_scenario1:
                st.markdown("##### 价格假设")
                
                # 价格模式
                price_mode = st.selectbox(
                    "价格模式 (mode)",
                    ["close_pct", "open_pct", "gap_then_close_pct", "limit_up", "limit_down", "flat"],
                    index=0,
                    help="价格变化模式：close_pct(收盘涨跌)、open_pct(开盘涨跌)、gap_then_close_pct(跳空后收盘涨跌)、limit_up(涨停)、limit_down(跌停)、flat(平盘)"
                )
                
                # 涨跌幅
                if price_mode in ["close_pct", "open_pct", "gap_then_close_pct"]:
                    pct = st.number_input(
                        "涨跌幅 (pct)",
                        value=0.0,
                        step=0.1,
                        format="%.1f",
                        help="涨跌幅百分比，正数为上涨，负数为下跌"
                    )
                
                # 跳空幅度
                if price_mode == "gap_then_close_pct":
                    gap_pct = st.number_input(
                        "跳空幅度 (gap_pct)",
                        value=0.0,
                        step=0.1,
                        format="%.1f",
                        help="跳空幅度百分比，开盘=昨收*(1+gap_pct)"
                    )
                
                st.markdown("##### 高低点生成")
                
                # 高低点模式
                hl_mode = st.selectbox(
                    "高低点模式 (hl_mode)",
                    ["follow", "atr_like", "range_pct"],
                    index=0,
                    help="高低点生成模式：follow(跟随)、atr_like(类ATR)、range_pct(固定振幅)"
                )
                
                if hl_mode == "range_pct":
                    range_pct = st.number_input(
                        "当日振幅 (range_pct)",
                        value=1.5,
                        step=0.1,
                        format="%.1f",
                        help="当日高低振幅百分比"
                    )
                elif hl_mode == "atr_like":
                    atr_mult = st.number_input(
                        "ATR倍数 (atr_mult)",
                        value=1.0,
                        step=0.1,
                        format="%.1f",
                        help="ATR倍数，从近N日高低均值估算"
                    )
            
            with col_scenario2:
                st.markdown("##### 成交量配置")
                
                # 成交量模式
                vol_mode = st.selectbox(
                    "成交量模式 (vol_mode)",
                    ["same", "pct", "mult"],
                    index=0,
                    help="成交量模式：same(相同)、pct(百分比变化)、mult(倍数变化)"
                )
                
                if vol_mode == "pct":
                    vol_arg = st.number_input(
                        "成交量变化 (vol_arg)",
                        value=0.0,
                        step=1.0,
                        format="%.1f",
                        help="成交量变化百分比，+10表示+10%"
                    )
                elif vol_mode == "mult":
                    vol_arg = st.number_input(
                        "成交量倍数 (vol_arg)",
                        value=1.0,
                        step=0.1,
                        format="%.1f",
                        help="成交量倍数，1.2表示放大20%"
                    )
                
                st.markdown("##### 约束条件")
                
                # 约束条件
                lock_higher_than_open = st.checkbox(
                    "收盘高于开盘 (lock_higher_than_open)",
                    help="强制收盘价≥开盘价"
                )
                
                lock_inside_day = st.checkbox(
                    "高低点覆盖开收盘 (lock_inside_day)",
                    help="强制H/L覆盖O/C"
                )
                
                # 指标重算窗口
                warmup_days = st.number_input(
                    "指标重算窗口 (warmup_days)",
                    min_value=10,
                    max_value=200,
                    value=60,
                    help="需要拼接多少历史天作warm-up，越大指标越准但越慢"
                )
            
            # 构建场景配置
            scenario_config = {
                "mode": price_mode,
                "pct": pct if price_mode in ["close_pct", "open_pct", "gap_then_close_pct"] else 0.0,
                "gap_pct": gap_pct if price_mode == "gap_then_close_pct" else 0.0,
                "hl_mode": hl_mode,
                "range_pct": range_pct if hl_mode == "range_pct" else 1.5,
                "atr_mult": atr_mult if hl_mode == "atr_like" else 1.0,
                "vol_mode": vol_mode,
                "vol_arg": vol_arg if vol_mode in ["pct", "mult"] else 0.0,
                "lock_higher_than_open": lock_higher_than_open,
                "lock_inside_day": lock_inside_day,
                "warmup_days": warmup_days
            }
            
            # 显示生成的场景配置
            with st.expander("场景配置预览", expanded=False):
                st.code(json.dumps(scenario_config, ensure_ascii=False, indent=2), language="json")
        else:
            scenario_config = None
    
    elif rule_category == "position":
        # 持仓策略配置
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 基础配置")
            
            # 规则名称
            rule_name = st.text_input(
                "规则名称 (name)",
                value=st.session_state.get('template_name', ''),
                placeholder="例如：止损策略",
                help="规则的显示名称，用于识别和说明"
            )
        
        with col_right:
            st.markdown("#### 策略配置")
            
            # 说明文字
            explain = st.text_area(
                "说明文字 (explain)",
                value=st.session_state.get('template_explain', ''),
                placeholder="例如：当价格跌破支撑位时止损",
                help="规则的详细说明，用于解释策略的作用"
            )
    
    elif rule_category == "opportunity":
        # 买点策略配置
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 基础配置")
            
            # 规则名称
            rule_name = st.text_input(
                "规则名称 (name)",
                value=st.session_state.get('template_name', ''),
                placeholder="例如：突破买点",
                help="规则的显示名称，用于识别和说明"
            )
        
        with col_right:
            st.markdown("#### 策略配置")
            
            # 说明文字
            explain = st.text_area(
                "说明文字 (explain)",
                value=st.session_state.get('template_explain', ''),
                placeholder="例如：价格突破阻力位时的买入机会",
                help="规则的详细说明，用于解释策略的作用"
            )
        
        use_clauses = st.checkbox(
            "使用多子句组合 (clauses)",
            help="使用clauses替代when字段，支持更复杂的逻辑组合"
        )
        
        if use_clauses:
            clauses_config = st.text_area(
                "子句配置 (clauses)",
                placeholder='[{"when": "C > MA(C, 20)", "points": 2}, {"when": "V > MA(V, 20)", "points": 1}]',
                help="JSON格式的多子句配置，每个子句包含when表达式和points分数"
            )
        
        # 子句配置提示
        with st.expander("多子句配置说明", expanded=False):
            st.markdown("""
            **多子句配置格式：**
            ```json
            [
                {
                    "when": "C > MA(C, 20)",
                    "points": 2,
                    "explain": "价格突破20日均线"
                },
                {
                    "when": "V > MA(V, 20)",
                    "points": 1,
                    "explain": "成交量放大"
                }
            ]
            ```
            
            **字段说明：**
            - `when`: 条件表达式（必填）
            - `points`: 分数（可选，默认0）
            - `explain`: 说明文字（可选）
            """)

    # 条件表达式配置 - 根据策略类型显示不同字段
    if rule_category == "prediction":
        st.markdown("#### 模拟策略表达式")
        check_expr = st.text_area(
            "检查表达式 (check) *",
            value=st.session_state.get('template_check', ''),
            placeholder="例如：C > MA(C, 20)",
            help="模拟策略的检查条件，用于判断是否满足特定场景"
        )
        scenario = st.text_input(
            "场景名称 (scenario)",
            value=st.session_state.get('template_scenario', ''),
            placeholder="例如：上涨场景",
            help="模拟策略的场景名称，用于标识不同的市场情况（可选）"
        )
    else:
        st.markdown("#### 条件表达式")
        
        if not use_clauses:
            # 单条件表达式
            when_expr = st.text_area(
                "条件表达式 (when) *",
                value=st.session_state.get('template_when', ''),
                placeholder="例如：C > MA(C, 20) AND V > MA(V, 20)",
                help=f"TDX风格的布尔表达式，用于{rule_category}策略的条件判断"
            )
        else:
            st.info("使用多子句组合时，条件表达式在clauses字段中配置")
            if rule_category in ["ranking", "filter"]:
                clauses_config = st.text_area(
                    "子句配置 (clauses)",
                    placeholder='[{"when": "C > MA(C, 20)", "points": 2}, {"when": "V > MA(V, 20)", "points": 1}]',
                    help="JSON格式的多子句配置，每个子句包含when表达式和points分数"
                )
                
                # 子句配置提示
                with st.expander("多子句配置说明", expanded=False):
                    st.markdown("""
                    **多子句配置格式：**
                    ```json
                    [
                        {
                            "when": "C > MA(C, 20)",
                            "points": 2,
                            "explain": "价格突破20日均线"
                        },
                        {
                            "when": "V > MA(V, 20)",
                            "points": 1,
                            "explain": "成交量放大"
                        }
                    ]
                    ```
                    
                    **字段说明：**
                    - `when`: 条件表达式（必填）
                    - `points`: 分数（可选，默认0）
                    - `explain`: 说明文字（可选）
                    """)

    # 表达式语法提示
    with st.expander("表达式语法提示", expanded=False):
        md_path_candidates = ["./手册/规则编辑方法.md"]
        for _p in md_path_candidates:
            md_text = _read_md_file(_p)
            if not md_text.startswith("⚠️ 未找到帮助文档"):
                break
        st.markdown(md_text)

    # 生成规则配置
    st.markdown("#### 规则预览")

    col_generate1, col_generate2 = st.columns([1, 1])
    with col_generate1:
        generate_btn = st.button("🔧 生成规则配置", type="primary")
    with col_generate2:
        validate_btn = st.button("✅ 验证规则", help="验证规则配置是否正确")

    if generate_btn or validate_btn:
        # 构建规则配置
        rule_config = {}
        
        # 根据策略类型处理字段
        if rule_category == "ranking":
            # 排名策略字段
            if rule_name:
                rule_config["name"] = rule_name
            if timeframe != "D":
                rule_config["timeframe"] = timeframe
            if score_windows != 60:
                rule_config["score_windows"] = score_windows
            if scope != "ANY":
                rule_config["scope"] = scope
            if points != 0:
                rule_config["points"] = points
            if explain:
                rule_config["explain"] = explain
            if not show_reason:
                rule_config["show_reason"] = show_reason
            if rule_as != "auto":
                rule_config["as"] = rule_as
            # 前置门槛（gate/trigger/require）
            if gate:
                rule_config["gate"] = gate
            elif trigger:
                rule_config["trigger"] = trigger
            elif require:
                rule_config["require"] = require
            # 分布分数（dist_points）
            if use_dist_points and dist_points_config:
                try:
                    dist_points_parsed = json.loads(dist_points_config)
                    if isinstance(dist_points_parsed, list) and len(dist_points_parsed) > 0:
                        rule_config["dist_points"] = dist_points_parsed
                except json.JSONDecodeError as e:
                    st.warning(f"dist_points配置格式错误：{str(e)}，将忽略此配置")
                
        elif rule_category == "filter":
            # 筛选策略字段
            if rule_name:
                rule_config["name"] = rule_name
            if timeframe != "D":
                rule_config["timeframe"] = timeframe
            if score_windows != 60:
                rule_config["score_windows"] = score_windows
            if scope != "ANY":
                rule_config["scope"] = scope
            if hard_penalty:
                rule_config["hard_penalty"] = hard_penalty
            if reason:
                rule_config["reason"] = reason
            # 前置门槛（gate/trigger/require）
            if gate:
                rule_config["gate"] = gate
            elif trigger:
                rule_config["trigger"] = trigger
            elif require:
                rule_config["require"] = require
                
        elif rule_category == "prediction":
            # 模拟策略字段
            if rule_name:
                rule_config["name"] = rule_name
            if scenario_config:
                rule_config["scenario"] = scenario_config
                
        elif rule_category == "position":
            # 持仓策略字段
            if rule_name:
                rule_config["name"] = rule_name
            if explain:
                rule_config["explain"] = explain
                
        elif rule_category == "opportunity":
            # 买点策略字段
            if rule_name:
                rule_config["name"] = rule_name
            if explain:
                rule_config["explain"] = explain
        
        # 条件表达式 - 根据策略类型处理
        if rule_category == "prediction":
            # 模拟策略使用check字段
            if check_expr:
                rule_config["check"] = check_expr
            if scenario:
                rule_config["scenario"] = scenario
            
            # 验证必填字段
            if not check_expr:
                st.error("❌ 缺少必填字段：检查表达式 (check)")
            else:
                # 显示生成的配置
                st.success("✅ 规则配置生成成功！")
                
                # 显示JSON格式
                st.markdown("**生成的规则配置：**")
                st.code(json.dumps(rule_config, ensure_ascii=False, indent=2), language="json")
                
                # 提供复制功能
                if st.button("📋 复制配置"):
                    st.code(json.dumps(rule_config, ensure_ascii=False, indent=2))
                    st.success("配置已复制到剪贴板（请手动复制）")
        else:
            # 其他策略类型使用when字段
            config_error = None
            
            if use_clauses:
                # 使用多子句组合
                if not clauses_config:
                    config_error = "❌ 缺少必填字段：子句配置 (clauses)"
                else:
                    try:
                        # 解析JSON格式的clauses配置
                        clauses_parsed = json.loads(clauses_config)
                        rule_config["clauses"] = clauses_parsed
                    except json.JSONDecodeError as e:
                        config_error = f"❌ 子句配置格式错误：{str(e)}"
            else:
                # 使用单条件表达式
                if not when_expr:
                    config_error = "❌ 缺少必填字段：条件表达式 (when)"
                else:
                    rule_config["when"] = when_expr
            
            # 显示配置或错误
            if config_error:
                st.error(config_error)
            else:
                # 显示生成的配置
                st.success("✅ 规则配置生成成功！")
                st.markdown("**生成的规则配置：**")
                st.code(json.dumps(rule_config, ensure_ascii=False, indent=2), language="json")
                
                # 提供复制功能
                copy_key = f"copy_config_{'clauses' if use_clauses else 'when'}"
                if st.button("📋 复制配置", key=copy_key):
                    st.code(json.dumps(rule_config, ensure_ascii=False, indent=2))
                    st.success("配置已复制到剪贴板（请手动复制）")
        
        # 规则解释功能
        if rule_config and (rule_config.get('when') or rule_config.get('check') or rule_config.get('gate') or rule_config.get('clauses')):
            st.markdown("---")
            st.markdown("#### 📖 规则解释")
            
            # 处理 clauses 情况
            if rule_config.get('clauses'):
                # 如果有 clauses，展示每个子句的解释
                clauses_data = rule_config.get('clauses', [])
                if isinstance(clauses_data, str):
                    try:
                        clauses_data = json.loads(clauses_data)
                    except:
                        clauses_data = []
                
                st.markdown("**多子句配置解释：**")
                for i, clause in enumerate(clauses_data, 1):
                    if isinstance(clause, dict):
                        clause_expr = clause.get('when', '') or ''
                        if clause_expr:
                            st.markdown(f"##### 子句 {i}")
                            
                            # 编译单个子句
                            clause_result = _compile_rule_expression({'when': clause_expr})
                            clause_logic = _analyze_rule_logic({'when': clause_expr})
                            
                            # 显示子句逻辑分析
                            col_c1, col_c2, col_c3 = st.columns(3)
                            with col_c1:
                                st.info(f"**获取数据**\n\n{clause_logic['data']}")
                            with col_c2:
                                st.info(f"**计算数据**\n\n{clause_logic['calc']}")
                            with col_c3:
                                st.info(f"**判断条件**\n\n{clause_logic['judge']}")
                            
                            # 显示子句表达式
                            with st.expander(f"📝 子句 {i} 表达式", expanded=False):
                                st.markdown("**when 表达式：**")
                                st.code(clause_expr, language="text")
                                if clause_result['translated'] and translate_expression:
                                    st.markdown("**翻译后的 Python 表达式：**")
                                    st.code(clause_result['translated'], language="python")
                                elif not translate_expression:
                                    st.warning("⚠️ 无法翻译表达式：tdx_compat 模块未导入")
                            
                            # 显示子句的其他属性
                            if clause.get('points'):
                                st.caption(f"分数: {clause['points']}")
                            if clause.get('explain'):
                                st.caption(f"说明: {clause['explain']}")
            else:
                # 编译表达式
                compile_result = _compile_rule_expression(rule_config)
                
                # 分析规则逻辑
                logic_analysis = _analyze_rule_logic(rule_config)
                
                # 显示规则逻辑分析
                col_logic1, col_logic2, col_logic3 = st.columns(3)
                with col_logic1:
                    st.info(f"**获取数据**\n\n{logic_analysis['data']}")
                with col_logic2:
                    st.info(f"**计算数据**\n\n{logic_analysis['calc']}")
                with col_logic3:
                    st.info(f"**判断条件**\n\n{logic_analysis['judge']}")
                
                # 显示原始表达式
                if compile_result['original']:
                    with st.expander("📝 原始表达式", expanded=False):
                        if compile_result.get('when'):
                            st.markdown("**when 表达式：**")
                            st.code(compile_result['when'], language="text")
                        if compile_result.get('check'):
                            st.markdown("**check 表达式：**")
                            st.code(compile_result['check'], language="text")
                        if compile_result.get('gate'):
                            st.markdown("**gate 表达式：**")
                            st.code(compile_result['gate'], language="text")
                
                # 显示翻译后的表达式
                if compile_result['translated'] and translate_expression:
                    with st.expander("🔤 翻译后的 Python 表达式", expanded=False):
                        st.code(compile_result['translated'], language="python")
                elif not translate_expression:
                    st.warning("⚠️ 无法翻译表达式：tdx_compat 模块未导入")
        
        # 验证规则
        if validate_btn:
            st.markdown("#### 规则验证结果")
            
            # 使用验证器进行完整验证
            validator = StrategyValidator()
            validation_result = validator.validate_rule(rule_config, rule_category)
            
            # 显示验证结果
            if validation_result.is_valid:
                st.success("✅ 规则验证通过")
            else:
                st.error("❌ 验证失败")
            
            # 显示错误
            if validation_result.errors:
                st.error("**错误信息：**")
                for error in validation_result.errors:
                    field_info = f" [{error.get('field', '')}]" if error.get('field') else ""
                    st.error(f"• {error['message']}{field_info}")
            
            # 显示警告
            if validation_result.warnings:
                st.warning("**警告信息：**")
                for warning in validation_result.warnings:
                    field_info = f" [{warning.get('field', '')}]" if warning.get('field') else ""
                    st.warning(f"• {warning['message']}{field_info}")
            
            # 显示建议
            if validation_result.suggestions:
                st.info("**改进建议：**")
                for suggestion in validation_result.suggestions:
                    field_info = f" [{suggestion.get('field', '')}]" if suggestion.get('field') else ""
                    st.info(f"• {suggestion['message']}{field_info}")
            
            # 显示缺失的列和指标
            if validation_result.missing_columns:
                st.warning(f"**缺失的列：** {', '.join(validation_result.missing_columns)}")
            
            if validation_result.missing_indicators:
                st.warning(f"**缺失的指标：** {', '.join(validation_result.missing_indicators)}")
            
            # 显示语法问题
            if validation_result.syntax_issues:
                st.warning("**语法问题：**")
                for issue in validation_result.syntax_issues:
                    st.warning(f"• {issue}")
            
            # 显示配置预览
            if not validation_result.errors:
                st.markdown("**配置预览：**")
                st.code(json.dumps(rule_config, ensure_ascii=False, indent=2), language="json")


def validate_strategy_file(file_path: str):
    """
    验证策略文件的语法和字段有效性
    
    Args:
        file_path: 策略文件路径
        
    Returns:
        StrategyValidationResult: 验证结果对象
    """
    import ast
    import importlib.util
    from pathlib import Path
    
    result = StrategyValidationResult()
    validator = StrategyValidator()
    
    try:
        # 读取文件内容
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            result.add_error(f"文件不存在: {file_path}")
            return result
        
        content = file_path_obj.read_text(encoding='utf-8')
        
        # 检查Python语法
        try:
            ast.parse(content)
        except SyntaxError as e:
            result.add_error(f"Python语法错误: {e.msg} (行 {e.lineno})")
            return result
        
        # 尝试加载模块
        spec = importlib.util.spec_from_file_location("strategy_module", file_path)
        if spec is None or spec.loader is None:
            result.add_error("无法加载策略文件模块")
            return result
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 验证各种策略规则列表
        rule_lists = {
            "RANKING_RULES": ("ranking", "排名策略"),
            "FILTER_RULES": ("filter", "筛选策略"),
            "PREDICTION_RULES": ("prediction", "模拟策略"),
            "POSITION_POLICIES": ("position", "持仓策略"),
            "OPPORTUNITY_POLICIES": ("opportunity", "买点策略")
        }
        
        total_rules = 0
        for list_name, (category, category_name) in rule_lists.items():
            if hasattr(module, list_name):
                rules = getattr(module, list_name)
                if isinstance(rules, list):
                    total_rules += len(rules)
                    for i, rule in enumerate(rules):
                        if isinstance(rule, dict):
                            rule_result = validator.validate_rule(rule, category)
                            if not rule_result.is_valid:
                                for error in rule_result.errors:
                                    result.add_error(
                                        f"{category_name}[{i}]: {error['message']}",
                                        f"{list_name}[{i}].{error.get('field', '')}"
                                    )
                            for warning in rule_result.warnings:
                                result.add_warning(
                                    f"{category_name}[{i}]: {warning['message']}",
                                    f"{list_name}[{i}].{warning.get('field', '')}"
                                )
                            for suggestion in rule_result.suggestions:
                                result.add_suggestion(
                                    f"{category_name}[{i}]: {suggestion['message']}",
                                    f"{list_name}[{i}].{suggestion.get('field', '')}"
                                )
                            result.missing_columns.extend(rule_result.missing_columns)
                            result.missing_indicators.extend(rule_result.missing_indicators)
                            result.syntax_issues.extend(rule_result.syntax_issues)
                        else:
                            result.add_error(f"{category_name}[{i}]: 规则必须是字典格式", f"{list_name}[{i}]")
        
        if total_rules == 0:
            result.add_warning("未找到任何策略规则，请检查策略文件是否包含RANKING_RULES、FILTER_RULES等列表")
        else:
            result.add_suggestion(f"共验证了 {total_rules} 条策略规则")
        
    except Exception as e:
        result.add_error(f"验证过程发生异常: {str(e)}")
    
    return result
