import pandas as pd
import os
from datetime import datetime
from strategy import buy_signal, sell_signal
from config import HOLD_DAYS, START_DATE, END_DATE, BUY_MODE, SELL_MODE, DATA_DIR, FALLBACK_SELL_MODE, MAX_HOLD_DAYS
from utils import ensure_datetime_index

def backtest_single_stock(df):
    signals = buy_signal(df)
    buy_dates = df.index[signals]
    all_sell_signals = sell_signal(df)

    trades = []

    for buy_date in buy_dates:
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
            if idx >= len(df):
                continue
            buy_price = df.iloc[idx]['open']
            sell_start_idx = idx + HOLD_DAYS

        else:
            raise ValueError(f"不支持的买入模式: {BUY_MODE}")

        # === 卖出逻辑 ===
        if SELL_MODE == 'open':
            if sell_start_idx >= len(df): continue
            sell_price = df.iloc[sell_start_idx]['open']

        elif SELL_MODE == 'close':
            if sell_start_idx >= len(df): continue
            sell_price = df.iloc[sell_start_idx]['close']

        elif SELL_MODE == 'strategy':
            max_days = MAX_HOLD_DAYS if MAX_HOLD_DAYS != -1 else len(df)
            sell_signals = all_sell_signals.iloc[sell_start_idx : sell_start_idx + max_days + 1]
            sell_window = df.iloc[sell_start_idx : sell_start_idx + max_days + 1]
            sell_price = None
            for offset in range(1, max_days + 1):
                if offset >= len(sell_signals): break
                if sell_signals.iloc[offset]['sell_by_open']:
                    sell_price = sell_window.iloc[offset]['open']
                    sell_start_idx += offset
                    break
                elif sell_signals.iloc[offset]['sell_by_close']:
                    sell_price = sell_window.iloc[offset]['close']
                    sell_start_idx += offset
                    break

            if sell_price is None:
                fallback_idx = sell_start_idx + max_days
                if fallback_idx >= len(df): continue
                sell_price = df.iloc[fallback_idx]['open'] if FALLBACK_SELL_MODE == 'open' else df.iloc[fallback_idx]['close']
                sell_start_idx = fallback_idx

        else:
            raise ValueError(f"不支持的 SELL_MODE: {SELL_MODE}")

        ret = (sell_price - buy_price) / buy_price
        hold_days = (df.index[sell_start_idx] - buy_date).days

        trades.append({
            "买入日期": buy_date.strftime("%Y-%m-%d"),
            "卖出日期": df.index[sell_start_idx].strftime("%Y-%m-%d"),
            "持股天数": hold_days,
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

    print(f"\n✅ 交易明细已保存至：{output_path}")
    print(trades_df)

if __name__ == "__main__":
    main()
