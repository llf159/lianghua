#单针下20
from indicators import momentum_multi

def buy_signal(df):
    short, mid, midlong, long = momentum_multi(df)

    signal = (short <= 20) & (long >= 80)

    return signal.fillna(False)

###############################################################################################

#j负值后带量突破
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