import pandas as pd
import numpy as np
from indicators import z_score, kdj, bupiao, bbi, volume_ratio, four_line_zero_strategy, rsi, shuangjunxian
###############################################################################################################################
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
