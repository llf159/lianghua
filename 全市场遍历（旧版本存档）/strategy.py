#单针下20
from indicators import bupiao

def buy_signal(df):
    short, mid, midlong, long = bupiao(df)

    signal = (short <= 20) & (long >= 80)

    return signal.fillna(False)

###############################################################################################

# #j负值后带量突破
# from indicators import kdj, volume_ratio

# def buy_signal(df):
#     # 计算 KDJ
#     k, d, j = kdj(df)

#     # 计算量比 VR
#     vr = volume_ratio(df)

#     # 将 J 值向后移动一位，代表“前一天的 J 值”
#     j_prev = j.shift(1)

#     # 构建买入信号条件
#     signal = (j_prev < 0) & (vr > 2)

#     return signal.fillna(False)

###############################################################################################

# from indicators import bupiao
# import pandas as pd

# def buy_signal(df):
#     short, mid, midlong, long = bupiao(df)
    
#     # 条件1：第一天 short < 20 and long > 80
#     cond1 = (short < 20) & (long > 80)

#     # 条件2：第二天 short == 100 and long == 100
#     short_next = short.shift(-1)
#     long_next = long.shift(-1)
#     cond2 = (short_next >= 99) & (long_next >= 99)

#     # 将满足第一天和第二天条件的位置作为信号（在第二天发出）
#     signal = cond1 & cond2

#     return signal.fillna(False)

