import pandas as pd
import numpy as np
from indicators import z_score, kdj, bupiao, bbi, volume_ratio, four_line_zero_strategy, rsi, shuangjunxian
###############################################################################################################################
def buy_signal(df):
    df = df.copy()

    # === 1. Z分数（基于开收涨跌幅） ===
    z = z_score(df, window=20, smooth=3)
    df['z_score'] = z['z_score']
    df['z_slope'] = z['z_slope']
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


# def buy_signal(df):
#     k, d, j = kdj(df)
#     j_ma = j.rolling(5).mean()
#     signal = (j_ma < 0)
#     return signal.fillna(False)


# 单针下20
# def buy_signal(df):
#     short, mid, midlong, long = bupiao(df)
#     signal = (short <= 20) & (long >= 80)
#     return signal.fillna(False)
