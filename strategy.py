from indicators import bupiao
import pandas as pd


def buy_signal(df):
    short, mid, midlong, long = bupiao(df)
    signal = (short <= 20) & (long >= 80)
    return signal.fillna(False)

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

