# strategy.py
from indicators import momentum_multi

def buy_signal(df):
    short, mid, midlong, long = momentum_multi(df)

    signal = (short <= 20) & (long >= 80)

    return signal.fillna(False)
