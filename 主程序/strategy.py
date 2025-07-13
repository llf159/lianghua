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
#     """
#     收盘买入条件：
#     - KDJ 中 K, D 均线的 5日均线 < 0
#     """
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

import pandas as pd
from indicators import bupiao
from patterns import is_soft_W_pattern

def buy_signal(df):
    short, _, _, long = bupiao(df)
    df['short'] = short
    df['long'] = long

    cond1 = df['long'].rolling(5).min() > 80
    cond2 = pd.Series(False, index=df.index)

    n = 10
    for i in range(n, len(df)):
        if not cond1.iloc[i]:
            continue
        window_series = df['short'].iloc[i-n:i+1]
        if is_soft_W_pattern(window_series):
            cond2.iloc[i] = True

    return (cond1 & cond2).fillna(False)


from indicators import bbi

def sell_signal(df):
    close = df['close']
    bbi_line = bbi(df)

    below_bbi_today = close < bbi_line
    below_bbi_yesterday = below_bbi_today.shift(1)

    condition = below_bbi_today & below_bbi_yesterday

    return pd.DataFrame({
        'sell_by_open': pd.Series(False, index=df.index),
        'sell_by_close': condition.fillna(False)
    })

