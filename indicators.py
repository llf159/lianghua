# indicators.py
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union
import pandas as pd

EPS = 1e-9

@dataclass
class IndMeta:
    name: str
    out: Dict[str, int]                 # 输出列 -> 小数位，如 {"j":2}、{"z_slope":3, "z_score":3}
    tdx: Optional[str] = None           # 可选：TDX 脚本
    py_func: Optional[Callable] = None  # 可选：Python 兜底函数
    kwargs: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)  # 使用场景标签，如 ["product","prelaunch"]

REGISTRY: Dict[str, IndMeta] = {
    "kdj": IndMeta(
        name="kdj",
        out={"j": 2},
        tdx="""
            RSV := RSV(C, H, L, 9);
            K := SMA(RSV, 3, 1);
            D := SMA(K, 3, 1);
            J := 3*K - 2*D;
        """,
        py_func=lambda df, **kw: kdj(df, **kw),  # 复用你已有的 kdj(df) 实现
        kwargs={}, 
        tags=["product","prelaunch"]
    ),
    "volume_ratio": IndMeta(
        name="volume_ratio",
        out={"vr": 4},
        tdx="VR := SAFE_DIV(V, MA(V, 20));",
        py_func=lambda df, **kw: volume_ratio(df, **kw),
        kwargs={"n": 20},
        tags=["product","prelaunch"]
    ),
    "bbi": IndMeta(
        name="bbi",
        out={"bbi": 2},
        tdx="BBI := (MA(C,3) + MA(C,6) + MA(C,12) + MA(C,24)) / 4;",
        py_func=lambda df, **kw: bbi(df),
        tags=["product","prelaunch"]
    ),
    "rsi": IndMeta(
        name="rsi",
        out={"rsi": 2},  # 小数位数
        tdx="""
            N := 14;
            LC := REF(CLOSE, 1);
            RSI := SMA(MAX(CLOSE - LC, 0), N, 1) / SMA(ABS(CLOSE - LC), N, 1) * 100;
        """,
        py_func=lambda df, **kw: rsi(df['close'], **kw),
        kwargs={"period": 14},  # Python 兜底参数
        tags=["product","prelaunch"]
    ),
    "DIFF": IndMeta(
        name="DIFF",
        out={"diff": 2},
        tdx="""
            DIFF := EMA(CLOSE, 12) - EMA(CLOSE, 26);
        """,
        py_func=lambda df, **kw: macd_diff(df['close'], **kw),
        kwargs={"fast": 12, "slow": 26},
        tags=["product","prelaunch"]
    ),
}

# —— 统一计算入口：优先 TDX，失败回退 Python —— 
def compute(df: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    from tdx_compat import evaluate as tdx_eval
    out_df = df.copy()
    for name in names or []:
        meta = REGISTRY.get(name)
        if not meta:
            continue

        tdx_ok = False
        if meta.tdx:
            try:
                res = tdx_eval(meta.tdx, out_df)
                # 尝试用 TDX 结果填充声明的输出列
                filled = 0
                for col in meta.out.keys():
                    # 所有 key 做一份小写映射，避免大小写不一致
                    lower_map = {k.lower(): v for k, v in res.items()}
                    if col.lower() in lower_map:
                        out_df[col] = lower_map[col.lower()]
                        filled += 1
                # 若 TDX 把所有声明列都填好了，就视为成功
                if filled == len(meta.out):
                    tdx_ok = True
            except Exception:
                tdx_ok = False  # TDX 报错则回退

        # 若 TDX 未成功或未填全 → Python 兜底补齐缺列
        if (not tdx_ok) and meta.py_func:
            try:
                res = meta.py_func(out_df, **meta.kwargs)
                if isinstance(res, pd.DataFrame):
                    for col in meta.out.keys():
                        if col in res.columns:
                            out_df[col] = res[col]
                elif isinstance(res, (list, tuple)):
                    if len(res) == len(meta.out):
                        for col, series in zip(meta.out.keys(), res):
                            out_df[col] = series
                else:
                    only_col = next(iter(meta.out.keys()))
                    out_df[only_col] = res
            except Exception:
                # 保底：兜底也失败就跳过，不阻塞其它指标
                pass

    return out_df


def outputs_for(names: List[str]) -> Dict[str, int]:
    """返回这些指标的所有输出列及小数位，供统一 round 使用。"""
    out = {}
    for n in names or []:
        meta = REGISTRY.get(n)
        if meta:
            out.update(meta.out)
    return out


def names_by_tag(tag: str) -> List[str]:
    return [n for n,m in REGISTRY.items() if tag in m.tags]


# =================== python兼容层 ==========================
def moving_average(series, window):
    return series.rolling(window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd_diff(close, fast=12, slow=26, **kw):
    return ema(close, fast) - ema(close, slow)

def rsi(series, period=14):
    from tdx_compat import SMA
    s = pd.Series(series)
    lc = s.shift(1)
    up = (s - lc).clip(lower=0)      # = MAX(CLOSE - LC, 0)
    dn = (lc - s).clip(lower=0)      # = MAX(LC - CLOSE, 0) = ABS(CLOSE - LC) 的下行部分
    avg_up = SMA(up, period, 1)      # TDX: SMA(..., N, 1)
    avg_dn = SMA(dn, period, 1)
    rs = avg_up / (avg_dn + EPS)     # SAFE_DIV
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val

def kdj(df, n=9, k_period=3, d_period=3):
    low_min = df['low'].rolling(n).min()
    high_max = df['high'].rolling(n).max()
    denom = (high_max - low_min).replace(0, EPS)
    rsv = 100 * (df['close'] - low_min) / denom
    k = rsv.ewm(com=(k_period - 1)).mean()
    d = k.ewm(com=(d_period - 1)).mean()
    j = 3 * k - 2 * d
    return j

def volume_ratio(df, n=20):
    """
    量比 VR = 当前成交量 / 过去n日平均成交量
    """
    avg_volume = df['vol'].rolling(n).mean()
    vr = df['vol'] / avg_volume
    return vr

def cci(df, n=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: (abs(x - x.mean())).mean())
    return (tp - ma) / (0.015 * md)

def bbi(df):
    ma3 = df['close'].rolling(3).mean()
    ma6 = df['close'].rolling(6).mean()
    ma12 = df['close'].rolling(12).mean()
    ma24 = df['close'].rolling(24).mean()

    return (ma3 + ma6 + ma12 + ma24) / 4

