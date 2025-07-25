import pandas as pd
import os
from datetime import datetime
from strategy import buy_signal, sell_signal
from config import HOLD_DAYS, START_DATE, END_DATE, BUY_MODE, SELL_MODE, DATA_DIR, FALLBACK_SELL_MODE, MAX_HOLD_DAYS
from utils import ensure_datetime_index

def backtest_single_stock(df):
    signals = buy_signal(df)
    buy_dates = pd.to_datetime(df.index[signals])

    trades = []
    holding_until = None 

    for buy_date in buy_dates:
        if holding_until and buy_date <= holding_until:
            continue
        
        idx = df.index.get_loc(buy_date)

        # === 买入逻辑 ===
        if BUY_MODE == 'open':
            if idx + 1 + HOLD_DAYS >= len(df):
                continue
            prev_close = df.iloc[idx]['close']
            open_price = df.iloc[idx + 1]['open']
            limit_up_price = round(prev_close * 1.10, 2)
            if open_price >= limit_up_price:
                continue
            buy_price = open_price
            sell_start_idx = idx + 1 + HOLD_DAYS

        elif BUY_MODE == 'close':
            if idx + HOLD_DAYS >= len(df):
                continue
            buy_price = df.iloc[idx]['close']
            sell_start_idx = idx + HOLD_DAYS

        elif BUY_MODE == 'signal_open':
            if idx + HOLD_DAYS >= len(df):
                continue
            buy_price = df.iloc[idx]['open']
            sell_start_idx = idx + 1

        else:
            raise ValueError(f"不支持的买入模式: {BUY_MODE}")

        # === 卖出逻辑 ===
        if SELL_MODE in ['open', 'close']:
            max_days = HOLD_DAYS
            if sell_start_idx + max_days >= len(df):
                continue

            if SELL_MODE == 'open':
                sell_price = df.iloc[sell_start_idx]['open']
            else:
                sell_price = df.iloc[sell_start_idx]['close']
            real_sell_idx = sell_start_idx

        elif SELL_MODE == 'strategy':
            max_days = MAX_HOLD_DAYS if MAX_HOLD_DAYS != -1 else len(df)

            if sell_start_idx >= len(df):
                continue

            remaining_len = len(df) - sell_start_idx
            actual_window = min(max_days + 1, remaining_len)

            all_sell_signals = sell_signal(df, [buy_date])
            
            sell_signals = all_sell_signals.iloc[sell_start_idx : sell_start_idx + actual_window]
            sell_window  = df.iloc[sell_start_idx : sell_start_idx + actual_window]

            sell_price = None
            real_sell_idx = None

            for offset in range(0, len(sell_signals)):
                if sell_signals.iloc[offset]['sell_by_open']:
                    sell_price = sell_window.iloc[offset]['open']
                    real_sell_idx = sell_start_idx + offset
                    break
                elif sell_signals.iloc[offset]['sell_by_close']:
                    sell_price = sell_window.iloc[offset]['close']
                    real_sell_idx = sell_start_idx + offset
                    break

            if real_sell_idx is None or real_sell_idx >= len(df):
                continue

        else:
            raise ValueError(f"不支持的 SELL_MODE: {SELL_MODE}")

        holding_until = df.index[real_sell_idx]

#统计部分###############################################################################
        ret = (sell_price - buy_price) / buy_price
        hold_days = (df.index[real_sell_idx] - buy_date).days

        trades.append({
        "买入日期": str(buy_date.date()),
        "卖出日期": str(df.index[real_sell_idx].date()),
        "持股天数": (df.index[real_sell_idx] - buy_date).days,
        "买入价": round(buy_price, 2),
        "卖出价": round(sell_price, 2),
        "收益率": f"{ret:.2%}"
})


    return trades

def main():
    code = input("请输入股票代码（6位数字，如 000001）：").strip()
    if not code.isdigit() or len(code) != 6:
        print("无效的股票代码！请输入6位数字。")
        return

    # 搜索以该代码开头的文件（例如：000001.SZ_平安银行.csv）
    target_file = None
    for fname in os.listdir(DATA_DIR):
        if fname.startswith(code) and fname.endswith(".csv"):
            target_file = fname
            break

    if not target_file:
        print(f"未找到以 {code} 开头的CSV文件，请确认放在 {DATA_DIR} 目录下")
        return

    file_path = os.path.join(DATA_DIR, target_file)

    df = pd.read_csv(file_path, parse_dates=["trade_date"])
    df.rename(columns={"trade_date": "date"}, inplace=True)
    df = ensure_datetime_index(df)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]

    trades = backtest_single_stock(df)

    if not trades:
        print("没有产生任何交易信号。")
        return

    trades_df = pd.DataFrame(trades)
    output_path = os.path.join("results", f"{target_file.replace('.csv', '_trades.csv')}")
    trades_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n 交易明细已保存至：{output_path}")
    print(trades_df)

if __name__ == "__main__":
    main()
