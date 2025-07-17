# from indicators import bupiao
# import pandas as pd


# def buy_signal(df):
#     short, mid, midlong, long = bupiao(df)
#     signal = (short <= 20) & (long >= 80)
#     return signal.fillna(False)

# def sell_signal(df):
#     """
#     返回两个布尔序列：
#     - sell_by_open: 是否满足开盘价卖出条件
#     - sell_by_close: 是否满足收盘价卖出条件
#     """
#     short, mid, midlong, long = bupiao(df)
#     sell_by_open = (short <= 80) & (long <= 80)  # 示例条件
#     sell_by_close = (short <= 80) & (long <= 80)  # 示例条件

#     return pd.DataFrame({
#         'sell_by_open': sell_by_open.fillna(False),
#         'sell_by_close': sell_by_close.fillna(False)
#     })

# import pandas as pd
# from indicators import kdj

# def buy_signal(df):
#     k, d, j = kdj(df)
#     j_ma = j.rolling(5).mean()
#     signal = (j_ma < 0)
#     return signal.fillna(False)

# def sell_signal(df):
#     """
#     开盘卖出条件：
#     - KDJ 的 J 值 > 80
#     """
#     _, _, j = kdj(df)
#     sell_by_open = j > 80
#     sell_by_close = pd.Series(False, index=df.index)  # 不用收盘价卖出
#     return pd.DataFrame({
#         'sell_by_open': sell_by_open.fillna(False),
#         'sell_by_close': sell_by_close
#     })

# import pandas as pd
# from indicators import bupiao

# def buy_signal(df):
#     short, _, _, long = bupiao(df)
#     df["short"] = short
#     df["long"] = long

#     # 将所需数据向后/向前平移
#     day1_long = df["long"].shift(2)
#     day1_short = df["short"].shift(2)

#     day2_long = df["long"].shift(1)
#     day2_short = df["short"].shift(1)

#     day3_long = df["long"]
#     day3_short = df["short"]

#     # 构造条件
#     cond = (
#         (day1_long == 100) & (day1_short == 100) &
#         (day2_long > 80) & (day2_short < 25) &
#         (day3_long > 80) & (day3_short > 80)
#     )

#     return cond.fillna(False)


# from indicators import bbi

# def sell_signal(df):
#     close = df['close']
#     bbi_line = bbi(df)

#     below_bbi_today = close < bbi_line
#     below_bbi_yesterday = below_bbi_today.shift(1)

#     condition = below_bbi_today & below_bbi_yesterday

#     return pd.DataFrame({
#         'sell_by_open': pd.Series(False, index=df.index),
#         'sell_by_close': condition.fillna(False)
#     })

import pandas as pd
import numpy as np

def buy_signal(df):
    df = df.copy()

    # === 1. Z分数（基于开收涨跌幅） ===
    df['oc_return'] = (df['close'] - df['open']) / df['open'] * 100
    df['oc_mean'] = df['oc_return'].rolling(20).mean()
    df['oc_std'] = df['oc_return'].rolling(20).std()
    df['z_score'] = (df['oc_return'] - df['oc_mean']) / df['oc_std']

    # === 2. 成交量与放量判断 ===
    df['vma5'] = df['vol'].rolling(5).mean()
    df['vma5_max5'] = df['vma5'].shift(1).rolling(5).max()
    df['explosion'] = (df['z_score'] > 2) & (df['vol'] > df['vma5']) & (df['vma5'] > df['vma5_max5'])

    # === 3. 最近10日是否出现爆发 ===
    df['explosion_recent'] = df['explosion'].rolling(10).sum() >= 1

    # === 4. 今天缩量 ===
    df['shrink'] = df['vol'] < df['vma5'] * 0.6

    # === 5. 多空线趋势向上 ===
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['trend_line'] = df['ema10'].ewm(span=10, adjust=False).mean()
    df['trend_up'] = df['trend_line'] > df['trend_line'].shift(1)

    # === 6. KDJ (J < 12) ===
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100

    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    # === 最终信号 ===
    signal = (
        df['explosion_recent'] &
        df['shrink'] &
        df['trend_up'] &
        (df['j'] < 12)
    )

    return signal.fillna(False)

# import pandas as pd
# from indicators import z_score

# def sell_signal(df, buy_dates):
#     df = df.copy()
#     z = z_score(df)
#     df['z_score'] = z['z_score']
#     df['z_slope'] = z['z_slope']

#     sell_by_close = pd.Series(False, index=df.index)
#     sell_by_open = pd.Series(False, index=df.index)  # 可后续扩展

#     for buy_date in buy_dates:
#         if buy_date not in df.index:
#             continue

#         buy_idx = df.index.get_loc(buy_date)
#         if buy_idx + 2 >= len(df):
#             continue

#         buy_price = df.iloc[buy_idx]['close']
#         future_prices = df.iloc[buy_idx + 1 : buy_idx + 3]['close']
#         max_gain = (future_prices.max() - buy_price) / buy_price

#         if max_gain < 0.02:
#             # 情况一：1天不涨则第2天收盘卖出
#             sell_idx = buy_idx + 1
#             if sell_idx < len(df):
#                 sell_by_close.iloc[sell_idx] = True
#         else:
#             # 情况二：涨了，等待 z_slope < 0 的点卖出
#             for j in range(buy_idx + 1, len(df)):
#                 if df.iloc[j]['z_slope'] < 0:
#                     sell_by_close.iloc[j] = True
#                     break

#     return pd.DataFrame({
#         'sell_by_open': sell_by_open,
#         'sell_by_close': sell_by_close
#     })

# 
import pandas as pd
from indicators import z_score

def sell_signal(df, buy_dates):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    buy_dates = pd.to_datetime(buy_dates)

    # 获取波动率指标
    z = z_score(df)
    df['z_score'] = z['z_score'].ffill()
    df['z_slope'] = z['z_slope'].ffill()

    # 初始化卖出信号
    sell_by_close = pd.Series(False, index=df.index)
    sell_by_open = pd.Series(False, index=df.index)

    for buy_date in buy_dates:
        if buy_date not in df.index:
            continue

        buy_idx = df.index.get_loc(buy_date)
        if buy_idx + 2 >= len(df):
            continue

        # 第1步：判断是否2天内涨超2%
        buy_price = df.iloc[buy_idx]['close']
        future_2 = df.iloc[buy_idx+1 : buy_idx+3]['close']
        max_gain = (future_2.max() - buy_price) / buy_price

        if max_gain < 0.02:
            # 情况一：未涨超2%，第2天收盘卖
            sell_idx = buy_idx + 1
            if sell_idx < len(df):
                sell_by_close.iloc[sell_idx] = True
        else:
            # 情况二：从第2天起，等待 z_slope < 0 出现
            for j in range(buy_idx + 2, len(df)):
                if df.iloc[j]['z_slope'] < 0:
                    sell_by_close.iloc[j] = True
                    break

    return pd.DataFrame({
        'sell_by_open': sell_by_open,
        'sell_by_close': sell_by_close
    })
