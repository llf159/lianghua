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
    # {
    #     'name': '10日均线斜率0-10度',
    #     'timeframe': 'D',
    #     'when': '''
    #         MA10_CUR := MA(C, 10);
    #         MA10_10AGO := REF(MA10_CUR, 10);
    #         RATIO := SAFE_DIV(MA10_CUR, MA10_10AGO);
    #         SLOPE := (RATIO - 1) / 0.5;
    #         ANGLE_RAD := ATAN(SLOPE);
    #         ANGLE_DEG := ANGLE_RAD * 180 / 3.141592653589793;
    #         ANGLE_DEG > 0 AND ANGLE_DEG <= 10
    #     ''',
    #     'scope': 'LAST',
    #     'points': 12,
    #     'explain': '10日均线斜率对应的角度在0-10度之间'
    # },
    # {
    #     'name': '当日机会',
    #     'timeframe': 'D',
    #     'window': 2,
    #     'when': "TAG_HITS('opportunity') > 3",
    #     'scope': 'LAST',
    #     'points': 8,
    #     'explain': '大机会'
    # },
    # {
    #     'name': '前期涨幅温和',
    #     'timeframe': 'D',
    #     'window': 60,
    #     'when': 'SAFE_DIV(HHV(C,60) - LLV(C,60), LLV(C,60)) <= 0.20',
    #     'scope': 'LAST',
    #     'points': 5,
    #     'explain': '过去60日涨幅不大'
    # },
    # {
    #     'name': '前期涨幅强势',
    #     'timeframe': 'D',
    #     'window': 60,
    #     'when': 'SAFE_DIV(HHV(C,60) - LLV(C,60), LLV(C,60)) < 0.40 AND SAFE_DIV(HHV(C,60) - '
    #             'LLV(C,60), LLV(C,60)) > 0.20',
    #     'scope': 'LAST',
    #     'points': 10,
    #     'explain': '过去60日涨幅过大'
    # },
    # {
    #     'name': '短期过高',
    #     'timeframe': 'D',
    #     'window': 10,
    #     'when': 'SAFE_DIV(HHV(C,8) - LLV(C,8), LLV(C,8)) > 0.3 ',
    #     'scope': 'LAST',
    #     'points': -15,
    #     'explain': '短期过高'
    # },
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

# 自选榜策略组合配置
# 格式：列表，每个元素包含 name（组合名称）、rules（策略名列表）、agg_mode（聚合方法：OR/AND）、output_name（落盘名称）、explain（说明）、exclude_rules（排除策略列表）；可选 rule_groups（混合逻辑组）与 group_mode（组间聚合）
CUSTOM_COMBOS = [
    # {
    #     'name': '混合逻辑示例',
    #     'rules': ['b1买点', '量价共振_上涨放量', '底部放量', '长期成本线附近'],      # 规则列表
    #     'agg_mode': 'AND',     # 规则间聚合方式
    #     'rule_groups': [
    #         {'mode': 'OR', 'rules': ['b1买点', '量价共振_上涨放量']},    # 组内聚合方式
    #         {'mode': 'OR', 'rules': ['底部放量', '长期成本线附近']},
    #     ],
    #     'group_mode': 'AND',    # 组间聚合方式
    #     'output_name': '混合逻辑示例',
    #     'explain': '(b1买点 或 量价共振_上涨放量) 且 (底部放量 或 长期成本线附近)',
    # },
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
# {
#     'name': '岛型反转',
#     'timeframe': 'D',
#     'score_windows': 21,
#     'when': 'TS_PCT(REF(C,1),20) >= 0.9 AND TS_PCT(V,20) <= 0.7 AND SAFE_DIV(C - O, REF(C,1)) >= 0.3',
#     'scope': 'ANY',
#     'points': -30,
#     'explain': '岛型反转'
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
