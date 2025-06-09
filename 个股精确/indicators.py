import pandas as pd

def moving_average(series, window):
    return series.rolling(window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def kdj(df, n=9, k_period=3, d_period=3):
    low_min = df['low'].rolling(n).min()
    high_max = df['high'].rolling(n).max()
    rsv = 100 * (df['close'] - low_min) / (high_max - low_min)
    k = rsv.ewm(com=(k_period - 1)).mean()
    d = k.ewm(com=(d_period - 1)).mean()
    j = 3 * k - 2 * d
    return k, d, j

def volume_ratio(df, n=20):
    """
    量比 VR = 当前成交量 / 过去n日平均成交量
    """
    avg_volume = df['volume'].rolling(n).mean()
    vr = df['volume'] / avg_volume
    return vr

def momentum_multi(df, n1=5, n2=30):
    """计算短期/中期/中长期/长期动量指标"""
    C = df['close']
    L = df['low']

    def calc_momentum(n):
        llv = L.rolling(n).min()
        hhv = C.rolling(n).max()
        return 100 * (C - llv) / (hhv - llv)

    short = calc_momentum(n1)
    mid = calc_momentum(10)
    midlong = calc_momentum(20)
    long = calc_momentum(n2)

    return short, mid, midlong, long

def four_line_zero_strategy(df, **params):
    """
    四线归零买入逻辑：
    - 若短期、中期、中长期、长期动量都 <= 6，则返回 True（买入信号）
    """
    n1 = params.get("n1", 5)   # 短期
    n2 = params.get("n2", 30)  # 长期

    # 获取四个动量值（你之前提供的公式）
    short, mid, midlong, long = momentum_multi(df, n1=n1, n2=n2)

    # 加入 df 可视化（非必要）
    df['short'] = short
    df['mid'] = mid
    df['midlong'] = midlong
    df['long'] = long

    # 四线都处于低位（如 <= 6）
    signal = (short <= 6) & (mid <= 6) & (midlong <= 6) & (long <= 6)

    return signal.fillna(False)
