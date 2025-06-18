import os
import pandas as pd
import numpy as np
import importlib.util
import datetime
import sys
sys.stdout.reconfigure(encoding='utf-8')
import importlib.util
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from indicators import cci, rsi, kdj, bupiao, volume_ratio


# ===== 动态导入 config.py / indicators.py / utils.py =====
def load_module(name, filename):
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

indicators = load_module("indicators", "indicators.py")
config = load_module("config", "config.py")
utils = load_module("utils", "utils.py")

# ======= 主回测分析函数 =======
def reverse_stats():
    records = []

    for fname in os.listdir(config.DATA_DIR):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(config.DATA_DIR, fname)
        df = pd.read_csv(path, parse_dates=["trade_date"])
        df.rename(columns={"trade_date": "date"}, inplace=True)
        df = utils.ensure_datetime_index(df)

        if not all(x in df.columns for x in ["open", "close", "high", "low", "vol"]):
            continue

        df["pct"] = df["close"].pct_change() * 100
        df["vr"] = indicators.volume_ratio(df)
        df["rsi"] = indicators.rsi(df["close"])
        df["cci"] = cci(df)
        df["k"], df["d"], df["j"] = indicators.kdj(df)
        short, _, _, long = indicators.bupiao(df)
        df["short"] = short
        df["long"] = long

        for i in range(5, len(df) - 1):
            if df.iloc[i]["pct"] <= 6:
                continue
            prev_1 = df.iloc[i - 1]
            prev_2 = df.iloc[i - 2]
            avg_pct = df.iloc[i - 5:i]["pct"].mean()

            records.append({
                "k": prev_2["k"],
                "d": prev_2["d"],
                "j": prev_2["j"],
                "rsi": prev_2["rsi"],
                "cci": prev_2["cci"],
                "bupiao_short": prev_2["short"],
                "bupiao_long": prev_2["long"],
                "volume": prev_2["vol"],
                "vr": prev_2["vr"],
                "pct_n-1": prev_1["pct"],
                "avg_pct_n-5_to_n-1": avg_pct
            })

    df_all = pd.DataFrame(records)
    return summarize_stats(df_all)

# ======= 汇总统计函数 =======
def summarize_stats(df):
    stats = {}
    for col in df.columns:
        s = df[col].dropna()
        stats[col] = {
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean(),
            "70%": s.quantile(0.7),
            "80%": s.quantile(0.8),
            "90%": s.quantile(0.9),
        }
    return pd.DataFrame(stats).T

# ======= 程序入口 =======
if __name__ == "__main__":
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    # 构造输出路径
    filename = f"huice_{timestamp}.xlsx"
    output_path = os.path.join(RESULTS_DIR, filename)

    # 执行回测并保存
    result = reverse_stats()
    result.to_excel(output_path)

    print(f"结果已保存到：{output_path}")
    print(result)
