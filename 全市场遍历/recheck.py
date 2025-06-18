import os
import pandas as pd
import datetime
import config
import indicators
import utils
import stat_strategy
import judge_strategy

# === 目录配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAT_CONFIG_PATH = os.path.join(BASE_DIR, "stat_config.txt")
JUDGE_CONFIG_PATH = os.path.join(BASE_DIR, "judge_config.txt")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# === 加载配置 ===
stat_rules = stat_strategy.parse_stat_config(STAT_CONFIG_PATH)
judge_rules = judge_strategy.parse_judge_config(JUDGE_CONFIG_PATH)

# === 动态窗口提取函数 ===
def get_max_lookback(stat_rules):
    max_back = 0
    for rule in stat_rules:
        if rule["type"] == "point":
            max_back = max(max_back, rule["day"])
        elif rule["type"] == "window":
            max_back = max(max_back, rule["from"])
    return max_back

def get_max_lookforward(judge_rules):
    max_fwd = 0
    for rule in judge_rules:
        if rule["type"] == "point":
            max_fwd = max(max_fwd, rule["day"])
        elif rule["type"] == "window":
            max_fwd = max(max_fwd, rule["to"])
    return max_fwd

max_back = get_max_lookback(stat_rules)
max_fwd = get_max_lookforward(judge_rules)

records = []

# === 遍历所有股票文件 ===
for fname in os.listdir(config.DATA_DIR):
    if not fname.endswith(".csv"):
        continue
    path = os.path.join(config.DATA_DIR, fname)
    df = pd.read_csv(path, parse_dates=["trade_date"])
    df.rename(columns={"trade_date": "date"}, inplace=True)
    df = utils.ensure_datetime_index(df)

    # === 生成指标列 ===
    df["pct"] = df["close"].pct_change() * 100
    df["rsi"] = indicators.rsi(df["close"])
    df["k"], df["d"], df["j"] = indicators.kdj(df)
    # 可在此继续添加其他指标列

    # === 回测主循环 ===
    for i in range(max_back, len(df) - max_fwd):
        if not judge_strategy.judge(df, i, judge_rules):
            continue
        features = stat_strategy.extract_features(df, i, stat_rules)
        records.append(features)

# === 输出结果 ===
result_df = pd.DataFrame(records)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
output_path = os.path.join(RESULT_DIR, f"huice_{timestamp}.xlsx")
result_df.to_excel(output_path, index=False)
