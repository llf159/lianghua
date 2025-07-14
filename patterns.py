import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from indicators import bupiao

def is_soft_W_pattern(series, window=5, high_thresh=80, low_thresh=40):
    """
    检测是否在过去一段时间内构成宽松 W 形态
    """
    vals = series.values
    local_max = argrelextrema(vals, np.greater_equal, order=window)[0]
    local_min = argrelextrema(vals, np.less_equal, order=window)[0]

    # 至少两个高点、两个低点，才能构成 W
    if len(local_max) < 2 or len(local_min) < 2:
        return False

    # 取最近一段的点来判断结构
    pts = sorted(np.concatenate([local_max, local_min]))
    pts = pts[-5:]  # 保留最近5个极值点

    # 要求结构为：高 - 低 - 高 - 低 - 当前上涨
    if len(pts) < 4:
        return False

    p0, p1, p2, p3 = pts[-4:]

    v0, v1, v2, v3 = vals[p0], vals[p1], vals[p2], vals[p3]
    current = vals[-1]

    return (
        v0 > high_thresh and
        v1 < low_thresh and
        v2 > high_thresh and
        v3 < low_thresh and
        current > v3
    )
