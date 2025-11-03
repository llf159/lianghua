# -*- coding: utf-8 -*-
"""
生成策略编译报告
从 strategies_repo.py 读取所有策略，编译表达式，并生成格式化的报告
"""
from strategies_repo import (
    RANKING_RULES, FILTER_RULES, PREDICTION_RULES, OPPORTUNITY_POLICIES
)
from tdx_compat import translate_expression, compile_script


def analyze_rule_logic(rule: dict) -> dict:
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
    
    # 注意：gate 表达式不显示在判断条件中，只在翻译后的 Python 表达式中显示
    
    return {
        'data': '、'.join(set(data_needed)) if data_needed else '基础价格和成交量数据',
        'calc': '、'.join(set(calculations)) if calculations else '直接使用原始数据',
        'judge': '；'.join(set(conditions)) if conditions else '直接判断表达式真假'
    }


def compile_rule_expression(rule: dict) -> dict:
    """编译单个规则表达式，返回翻译结果"""
    result = {
        'original': '',
        'translated': '',
        'when': None,
        'check': None,
        'gate': None,
        'gate_translated': None,
    }
    
    # 处理 when 表达式
    if 'when' in rule and rule['when']:
        original = rule['when'].strip()
        result['original'] = original
        result['when'] = original
        try:
            # 检查是否为多行表达式（包含变量赋值或分号）
            if ';' in original or ':=' in original:
                # 使用 compile_script 处理多行表达式
                compiled = compile_script(original)
                # 格式化编译结果
                compiled_lines = []
                for name, py_expr in compiled:
                    if name:
                        compiled_lines.append(f"{name} = {py_expr}")
                    else:
                        compiled_lines.append(str(py_expr))
                result['translated'] = '\n'.join(compiled_lines)
            else:
                # 使用 translate_expression 处理单行表达式
                translated = translate_expression(original)
                result['translated'] = translated
        except Exception as e:
            result['translated'] = f"翻译错误: {e}"
    
    # 处理 check 表达式
    if 'check' in rule and rule['check']:
        check_expr = rule['check'].strip()
        result['check'] = check_expr
        if not result['original']:
            result['original'] = check_expr
        else:
            result['original'] += f"\n[check]: {check_expr}"
        try:
            translated = translate_expression(check_expr)
            if result['translated']:
                result['translated'] += f"\n[check翻译]: {translated}"
            else:
                result['translated'] = translated
        except Exception as e:
            if result['translated']:
                result['translated'] += f"\n[check翻译错误]: {e}"
            else:
                result['translated'] = f"[check翻译错误]: {e}"
    
    # 处理 gate 表达式
    if 'gate' in rule and rule['gate']:
        gate_expr = rule['gate'].strip()
        result['gate'] = gate_expr
        try:
            gate_translated = translate_expression(gate_expr)
            result['gate_translated'] = gate_translated
        except Exception as e:
            result['gate_translated'] = f"翻译错误: {e}"
    
    return result


def get_strategy_type(rule: dict, rule_source: str) -> str:
    """根据规则和来源确定策略类型"""
    if rule.get('as') == 'opportunity':
        return '买点策略'
    elif rule_source == 'RANKING':
        return '排名策略'
    elif rule_source == 'FILTER':
        return '筛选策略'
    elif rule_source == 'PREDICTION':
        return '模拟策略'
    elif rule_source == 'OPPORTUNITY':
        return '买点策略'
    else:
        return '未知策略'


def generate_report():
    """生成策略编译报告"""
    report_lines = []
    
    # 报告头部
    report_lines.append("=" * 100)
    report_lines.append("策略表达式编译报告")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    strategy_count = 0
    ranking_count = 0
    filter_count = 0
    prediction_count = 0
    opportunity_count = 0
    
    # 收集所有策略
    all_strategies = []
    
    # 排名策略
    for rule in RANKING_RULES:
        all_strategies.append((rule, 'RANKING'))
    
    # 筛选策略
    for rule in FILTER_RULES:
        all_strategies.append((rule, 'FILTER'))
    
    # 模拟策略
    for rule in PREDICTION_RULES:
        all_strategies.append((rule, 'PREDICTION'))
    
    # 买点策略
    for rule in OPPORTUNITY_POLICIES:
        all_strategies.append((rule, 'OPPORTUNITY'))
    
    # 处理每个策略
    for rule, rule_source in all_strategies:
        strategy_count += 1
        strategy_type = get_strategy_type(rule, rule_source)
        
        if strategy_type == '排名策略':
            ranking_count += 1
        elif strategy_type == '筛选策略':
            filter_count += 1
        elif strategy_type == '模拟策略':
            prediction_count += 1
        elif strategy_type == '买点策略':
            opportunity_count += 1
        
        name = rule.get('name', '未命名策略')
        explain = rule.get('explain', '')
        
        # 编译表达式
        compile_result = compile_rule_expression(rule)
        
        # 分析逻辑
        logic_result = analyze_rule_logic(rule)
        
        # 生成策略报告
        report_lines.append(f"【策略 {strategy_count}】{name} ({strategy_type})")
        if explain:
            report_lines.append(f"说明: {explain}")
        report_lines.append("")
        
        report_lines.append("策略逻辑分析:")
        report_lines.append(f"  获取数据: {logic_result['data']}")
        report_lines.append(f"  计算数据: {logic_result['calc']}")
        report_lines.append(f"  判断条件: {logic_result['judge']}")
        report_lines.append("")
        
        # 原始表达式
        expr_type = 'when' if compile_result['when'] else 'check'
        if compile_result['when']:
            report_lines.append("原始表达式 (when):")
            report_lines.append(f"  {compile_result['when']}")
        elif compile_result['check']:
            report_lines.append("原始表达式 (check):")
            report_lines.append(f"  {compile_result['check']}")
        
        report_lines.append("")
        
        # 翻译后的 Python 表达式
        report_lines.append("翻译后的 Python 表达式:")
        if compile_result['translated']:
            # 处理多行表达式
            translated_lines = compile_result['translated'].split('\n')
            for line in translated_lines:
                if line.strip():
                    report_lines.append(f"  {line}")
        
        # Gate 表达式
        if compile_result['gate_translated']:
            report_lines.append(f"  [gate]: {compile_result['gate_translated']}")
        
        report_lines.append("")
        report_lines.append("-" * 100)
        report_lines.append("")
    
    # 统计信息
    report_lines.append("=" * 100)
    report_lines.append("统计信息")
    report_lines.append("=" * 100)
    report_lines.append(f"总策略数: {strategy_count}")
    report_lines.append(f"  排名策略: {ranking_count} 个")
    report_lines.append(f"  筛选策略: {filter_count} 个")
    report_lines.append(f"  模拟策略: {prediction_count} 个")
    report_lines.append(f"  买点策略: {opportunity_count} 个")
    
    return '\n'.join(report_lines)


if __name__ == '__main__':
    report = generate_report()
    
    # 保存报告
    output_file = 'strategy_compile_report.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"策略编译报告已生成: {output_file}")
    print(f"\n报告预览（前500字符）:")
    print(report[:500])
