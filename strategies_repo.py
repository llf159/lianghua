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

# 自选榜策略组合配置
# 格式：列表，每个元素包含 name（组合名称）、rules（策略名列表）、agg_mode（聚合方法：OR/AND）、explain（说明）等字段
CUSTOM_COMBOS = [
]

# 个股买点策略（用于给出买点价格来源；可选，不影响持仓检查）
OPPORTUNITY_POLICIES = [
]
