import pandas as pd
import numpy as np
import argparse
import time
import os
from datetime import datetime
import openpyxl
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import FormulaRule
from openpyxl.formatting.rule import CellIsRule
from tqdm import tqdm
import sys
sys.stdout.reconfigure(encoding='utf-8')
from strategy import buy_signal, sell_signal
from config import HOLD_DAYS, START_DATE, END_DATE, BUY_MODE, SELL_MODE, DATA_DIR, FALLBACK_SELL_MODE, MAX_HOLD_DAYS
from utils import ensure_datetime_index

#初始化运行时间 & 文件路径
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, f"summary_{RUN_TIMESTAMP}.csv")
XLSX_PATH = CSV_PATH.replace(".csv", ".xlsx")

#逻辑部分#################################################################################################
def backtest(df, n_days, buy_mode):
    signals = buy_signal(df)
    buy_dates = df.index[signals]
    all_sell_signals = sell_signal(df)
    results, wins, losses = [], [], []

    for buy_date in buy_dates:
        idx = df.index.get_loc(buy_date)

        # === 买入逻辑 ===
        if buy_mode == 'open':
            if idx + 1 + n_days < len(df):
                prev_close = df.iloc[idx]['close']
                open_price = df.iloc[idx + 1]['open']
                limit_up_price = round(prev_close * 1.10, 2)
                if open_price >= limit_up_price:                          # 如果开盘价大于等于涨停价，则不买入
                    continue
                buy_price = open_price
                sell_start_idx = idx + 1 + n_days
            else:
                continue

        elif buy_mode == 'close':
            if idx + n_days < len(df):
                buy_price = df.iloc[idx]['close']
                sell_start_idx = idx + n_days
            else:
                continue

        elif buy_mode == 'signal_open':
            if idx >= len(df):
                continue
            buy_price = df.iloc[idx]['open']
            sell_start_idx = idx + n_days

        else:
            raise ValueError(f"不支持的买入模式: {buy_mode}")

        # === 卖出逻辑 ===
        if SELL_MODE == 'open':
            sell_price = df.iloc[sell_start_idx]['open']
            
        elif SELL_MODE == 'close':
            sell_price = df.iloc[sell_start_idx]['close']
        
        elif SELL_MODE == 'strategy':
            max_days = MAX_HOLD_DAYS if MAX_HOLD_DAYS != -1 else len(df)
            sell_signals = all_sell_signals.iloc[sell_start_idx : sell_start_idx + max_days + 1]
            sell_window = df.iloc[sell_start_idx : sell_start_idx + max_days + 1]
            sell_price = None
            for offset in range(1, max_days + 1):
                if offset >= len(sell_signals):
                    break
                if sell_signals.iloc[offset]['sell_by_open']:
                    sell_price = sell_window.iloc[offset]['open']
                    break
                elif sell_signals.iloc[offset]['sell_by_close']:
                    sell_price = sell_window.iloc[offset]['close']
                    break

            if sell_price is None:
                fallback_idx = sell_start_idx + max_days
                if fallback_idx >= len(df):
                    continue
                if FALLBACK_SELL_MODE == 'open':
                    sell_price = df.iloc[fallback_idx]['open']
                else:
                    sell_price = df.iloc[fallback_idx]['close']

        else:
            raise ValueError(f"不支持的 SELL_MODE: {SELL_MODE}")

        # === 收益计算 ===
        ret = (sell_price - buy_price) / buy_price
        results.append(ret)
        if ret > 0:
            wins.append(ret)
        elif ret < 0:
            losses.append(ret)

#统计部分#################################################################################################
    returns = np.array(results)
    win_avg = np.mean(wins) if wins else 0
    loss_avg = abs(np.mean(losses)) if losses else 0
    win_rate = (returns > 0).mean() if len(results) else 0
    avg_return = returns.mean() if len(results) else 0
    pl_ratio = win_avg / loss_avg if loss_avg else avg_return
    score = pl_ratio * win_rate

    summary = {
        "信号数": len(results),
        "平均收益": f"{avg_return:.2%}" if len(results) else "N/A",
        "胜率": f"{win_rate:.2%}" if len(results) else "N/A",
        "最大收益": f"{returns.max():.2%}" if len(results) else "N/A",
        "最大亏损": f"{returns.min():.2%}" if len(results) else "N/A",
        "盈亏比": round(pl_ratio, 3) if len(results) else None,
        "平均盈亏比": round(score, 3) if len(results) else None
    }

    return summary


def run_backtest_on_file(file_path):
    df = pd.read_csv(file_path, parse_dates=['trade_date'])
    df.rename(columns={'trade_date': 'date'}, inplace=True)
    df = ensure_datetime_index(df)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
    summary = backtest(df, HOLD_DAYS, BUY_MODE)
    return os.path.basename(file_path), summary

#主函数##################################################################################################
def main():
    start_time = time.time()
    
    all_results = []

    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    VERBOSE = 0
    # total = len(files)
    # for i, fpath in enumerate(files, 1):
    #     print(f"正在回测第 {i}/{total} 支股票：{os.path.basename(fpath)}")
    #     fname, summary = run_backtest_on_file(fpath)
    for fpath in tqdm(files, desc="回测中", unit="支", mininterval=0.5):
        fname, summary = run_backtest_on_file(fpath)

        if VERBOSE:
            print(f"【{fname}】结果：")
            for k, v in summary.items():
                print(f"  {k}: {v}")
        all_results.append(dict(filename=fname, **summary))
        
    end_time = time.time()

    # 保存 CSV/Excel 结果#################################################################################
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    # ==== 加权平均盈亏比 ====
    valid_rows = df_all[pd.to_numeric(df_all["盈亏比"], errors='coerce').notna()]
    valid_rows.loc[:, "信号数"] = pd.to_numeric(valid_rows["信号数"], errors='coerce')
    valid_rows.loc[:, "盈亏比"] = pd.to_numeric(valid_rows["盈亏比"], errors='coerce')

    total_signals = valid_rows["信号数"].sum()
    if total_signals > 0:
        weighted_pl_ratio = (valid_rows["盈亏比"] * valid_rows["信号数"]).sum() / total_signals
    else:
        weighted_pl_ratio = 0

    df_all.to_excel(XLSX_PATH, index=False)

    # 添加条件格式
    wb = openpyxl.load_workbook(XLSX_PATH)
    ws = wb.active
    red_fill = PatternFill(start_color="FF9999", end_color="FF9999", fill_type="solid")
    


    # 胜率 < 1 → G列标红
    ws.conditional_formatting.add("G2:G6000",
        CellIsRule(operator="lessThan", formula=["1"], stopIfTrue=True, fill=red_fill))
    # 平均盈亏比 < 1 → H列标红
    ws.conditional_formatting.add("H2:H6000",
        CellIsRule(operator="lessThan", formula=["1"], stopIfTrue=True, fill=red_fill))

    ws.insert_rows(1)
    ws.cell(row=1, column=1, value="加权平均盈亏比")
    ws.cell(row=1, column=2, value=round(weighted_pl_ratio, 4))

    wb.save(XLSX_PATH)
    
    print(f"\n总用时：{end_time - start_time:.2f} 秒")
    input("按下回车退出...")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("程序发生错误：", e)
        input("按下回车关闭窗口...")
