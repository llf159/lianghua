#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试明日预测功能
"""
import sys
from pathlib import Path
import pandas as pd

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from predict_core import PredictionInput, Scenario, run_prediction, load_prediction_rules, simulate_next_day, eval_when_exprs

def debug_predict():
    """调试明日预测功能"""
    print("调试明日预测功能...")
    
    # 加载策略
    rules = load_prediction_rules()
    print(f"加载到 {len(rules)} 个预测策略:")
    for rule in rules:
        print(f"  - {rule.get('name', 'unnamed')}: {rule.get('check', 'no check')}")
    
    # 创建测试输入
    inp = PredictionInput(
        ref_date="20251010",  # 使用最新的日期
        universe=["000001.SZ", "000002.SZ", "000004.SZ"],  # 测试几只股票
        scenario=Scenario(mode="close_pct", pct=2.0),  # 预设涨2%
        rules=rules,  # 使用策略文件中的规则
        expr=None,   # 不使用表达式
        use_rule_scenario=True,  # 使用规则内置场景
        recompute_indicators=("kdj",),
        cache_dir="cache/sim_pred"
    )
    
    try:
        # 先测试模拟明日数据
        print("\n=== 测试模拟明日数据 ===")
        sim_result = simulate_next_day(
            inp.ref_date, 
            inp.universe, 
            inp.scenario,
            recompute_indicators=inp.recompute_indicators
        )
        print(f"模拟结果形状: {sim_result.df_concat.shape}")
        print(f"模拟日期: {sim_result.sim_date}")
        print(f"历史+模拟数据前5行:")
        print(sim_result.df_concat.head())
        
        # 检查明日数据
        tomorrow_data = sim_result.df_concat[sim_result.df_concat["trade_date"].astype(str) == sim_result.sim_date]
        print(f"\n明日数据形状: {tomorrow_data.shape}")
        print("明日数据:")
        print(tomorrow_data)
        
        # 测试表达式求值
        print("\n=== 测试表达式求值 ===")
        if rules:
            rule = rules[0]
            expr = rule.get("check", "")
            print(f"测试表达式: {expr}")
            
            # 直接测试 tdx.evaluate
            import tdx_compat as tdx
            from predict_core import _build_eval_ctx
            
            # 测试一只股票
            test_ts = "000001.SZ"
            sub = sim_result.df_concat[sim_result.df_concat["ts_code"].astype(str) == test_ts].copy()
            sub = sub.sort_values("trade_date")
            ctx_df = _build_eval_ctx(sub)
            print(f"\n测试股票 {test_ts} 的上下文数据:")
            print(ctx_df)
            
            # 设置环境
            tdx.EXTRA_CONTEXT.update({"TS": test_ts, "REF_DATE": sim_result.sim_date})
            
            # 直接调用 evaluate
            out = tdx.evaluate(expr, ctx_df)
            print(f"\nevaluate 返回值:")
            print(out)
            print(f"返回值类型: {type(out)}")
            if isinstance(out, dict):
                for k, v in out.items():
                    print(f"  {k}: {type(v)} = {v}")
            
            exprs = {rule.get("name", "test"): expr}
            hits = eval_when_exprs(sim_result.df_concat, sim_result.sim_date, exprs, for_ts_codes=inp.universe)
            print(f"\n表达式求值结果:")
            print(hits)
        
        # 运行完整预测
        print("\n=== 运行完整预测 ===")
        result = run_prediction(inp)
        print(f"\n预测结果形状: {result.shape}")
        print(f"列名: {list(result.columns)}")
        print(f"结果:")
        print(result)
        
        if not result.empty:
            print("✅ 明日预测功能正常工作！")
            print(f"命中股票数量: {len(result)}")
        else:
            print("❌ 明日预测功能返回空结果")
            
    except Exception as e:
        print(f"❌ 明日预测功能出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_predict()
