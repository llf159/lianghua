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
]

# 个股持仓检查策略（仅用于个股页签的触发表展示，不生成动作）
POSITION_POLICIES = [
]

# 个股买点策略（用于给出买点价格来源；可选，不影响持仓检查）
OPPORTUNITY_POLICIES = [
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
