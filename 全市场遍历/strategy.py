from indicators import bupiao
import pandas as pd


def buy_signal(df):
    short, mid, midlong, long = bupiao(df)
    signal = (short <= 20) & (long >= 80)
    return signal.fillna(False)

def sell_signal(df):
    """
    返回两个布尔序列：
    - sell_by_open: 是否满足开盘价卖出条件
    - sell_by_close: 是否满足收盘价卖出条件
    """
    short, mid, midlong, long = bupiao(df)
    sell_by_open = (short <= 80) & (long <= 80)  # 示例条件
    sell_by_close = (short <= 80) & (long <= 80)  # 示例条件

    return pd.DataFrame({
        'sell_by_open': sell_by_open.fillna(False),
        'sell_by_close': sell_by_close.fillna(False)
    })
