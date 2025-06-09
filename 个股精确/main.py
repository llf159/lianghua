
import pandas as pd
import numpy as np
import argparse
import os
from strategy import buy_signal
from config import HOLD_DAYS, START_DATE, END_DATE
from utils import ensure_datetime_index
from config import DATA_DIR


def backtest(df, n_days):
    signals = buy_signal(df)
    buy_dates = df.index[signals]

    details = []
    results = []
    for buy_date in buy_dates:
        idx = df.index.get_loc(buy_date)
        if idx + n_days < len(df):
            buy_price = df.iloc[idx]['close']
            sell_price = df.iloc[idx + n_days]['close']
            ret = (sell_price - buy_price) / buy_price
            results.append(ret)

            details.append({
                'buy_date': buy_date.strftime('%Y-%m-%d'),
                'buy_price': round(buy_price, 2),
                'sell_date': df.index[idx + n_days].strftime('%Y-%m-%d'),
                'sell_price': round(sell_price, 2),
                'return': f"{ret:.2%}"
            })

    returns = np.array(results)
    summary = {
        "信号数": len(results),
        "平均收益": f"{returns.mean():.2%}" if len(returns) else "N/A",
        "胜率": f"{(returns > 0).mean():.2%}" if len(returns) else "N/A",
        "最大收益": f"{returns.max():.2%}" if len(returns) else "N/A",
        "最大亏损": f"{returns.min():.2%}" if len(returns) else "N/A",
    }

    return summary, pd.DataFrame(details)

from datetime import datetime

def run_backtest_on_file(file_path):
    df = pd.read_csv(file_path, parse_dates=['trade_date'])
    df.rename(columns={'trade_date': 'date'}, inplace=True)
    df = ensure_datetime_index(df)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]

    summary, detail_df = backtest(df, HOLD_DAYS)

    # 提取股票名称：例如 “000001.SZ_平安银行.csv” → “平安银行”
    fname = os.path.basename(file_path)
    if "_" in fname:
        stock_name = fname.split("_")[1].replace(".csv", "")
    else:
        stock_name = fname.replace(".csv", "")

    # 当前日期
    today_str = datetime.today().strftime('%Y-%m-%d')

    # 构建输出文件名
    output_name = f"{stock_name}_{today_str}_details.csv"
    output_path = os.path.join("results", output_name)

    # 保存明细
    detail_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 打印提示
    print(f"[成功] 回测完成：{stock_name}")
    print(f"[保存至] {output_path}")
    print(f"[明细前5条]：\n{detail_df.head().to_string(index=False)}")

    return stock_name, summary


def main():
    parser = argparse.ArgumentParser(description="回测单只股票并输出交易明细")
    from config import DATA_FILE
    from config import DATA_DIR  # 保留原有的目录配置

def main():
    if not os.path.exists(DATA_DIR):
        print(f"[错误] 数据目录不存在: {DATA_DIR}")
        return

    # 获取目录下的所有CSV文件
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    if not csv_files:
        print(f"[错误] 数据目录中找不到任何CSV文件: {DATA_DIR}")
        return

    # 自动选择第一个文件
    file_path = os.path.join(DATA_DIR, csv_files[0])
    print(f"[提示] 自动选中数据文件: {file_path}")

    fname, summary = run_backtest_on_file(file_path)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("程序发生错误：", e)
        input("按下回车关闭窗口...")
