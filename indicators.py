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
    warmup: Optional[int] = None        # 该指标建议的预热天数
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
        tags=["product","prelaunch"],
        warmup=10,
    ),
    "volume_ratio": IndMeta(
        name="volume_ratio",
        out={"vr": 4},
        tdx="VR := SAFE_DIV(V, MA(V, 20));",
        py_func=lambda df, **kw: volume_ratio(df, **kw),
        kwargs={"n": 20},
        tags=["product","prelaunch"],
        warmup=20,
    ),
    "bbi": IndMeta(
        name="bbi",
        out={"bbi": 2},
        tdx="BBI := (MA(C,3) + MA(C,6) + MA(C,12) + MA(C,24)) / 4;",
        py_func=lambda df, **kw: bbi(df),
        tags=["product","prelaunch"],
        warmup=24,
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
        tags=["product","prelaunch"],
        warmup=14,
    ),
    "DIFF": IndMeta(
        name="DIFF",
        out={"diff": 2},
        tdx="""
            DIFF := EMA(CLOSE, 12) - EMA(CLOSE, 26);
        """,
        py_func=lambda df, **kw: macd_diff(df['close'], **kw),
        kwargs={"fast": 12, "slow": 26},
        tags=["product","prelaunch"],
        warmup=26,
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
        tdx_error = None
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
            except Exception as e:
                tdx_ok = False  # TDX 报错则回退
                tdx_error = e

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
            except Exception as e:
                # 指标计算失败，直接抛出异常以保证数据完整性
                error_msg = f"指标计算失败: {name}"
                if tdx_error:
                    error_msg += f"\n  TDX计算错误: {tdx_error}"
                error_msg += f"\n  Python计算错误: {e}"
                raise RuntimeError(error_msg) from e
        
        # 如果既没有TDX也没有Python实现，或者TDX失败但没有Python实现，直接报错
        if not tdx_ok and not meta.py_func:
            error_msg = f"指标计算失败: {name}"
            if tdx_error:
                error_msg += f"\n  TDX计算错误: {tdx_error}"
            error_msg += "\n  没有可用的Python实现作为兜底"
            raise RuntimeError(error_msg)

    # 应用精度控制
    decs = outputs_for(names)
    for col, n in decs.items():
        if col in out_df.columns and pd.api.types.is_numeric_dtype(out_df[col]):
            out_df[col] = out_df[col].round(n)

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


def get_all_indicator_names():
    """获取所有指标名称"""
    return list(REGISTRY.keys())

# def names_in_expr(expr: str) -> List[str]:
#     """
#     从一条 TDX 表达式里解析可能涉及的指标名（返回 REGISTRY 的 key 列表，去重）。
#     这里只做“足够用”的关键字映射，而不是完整解析器；不识别的符号会被忽略。
#     """
#     if not expr:
#         return []
#     import re
#     tokens = set(re.findall(r'[A-Z_][A-Z0-9_]*', str(expr).upper()))
#     # 避免把内置函数当成指标
#     blacklist = {
#         'OPEN','HIGH','LOW','CLOSE','V','VOL','AMOUNT','O','H','L','C',
#         'MA','EMA','SMA','LLV','HHV','REF','CROSS','COUNT','IF','MIN','MAX',
#         'ABS','STD','STDV','STDDEV','SUM','ZIG','FILTER','BACKSET','BARSLAST',
#         'AND','OR','NOT','TRUE','FALSE','N','M','K','D','RSV','WMA','VAR','EXP',
#         'SAFE_DIV','EPS','SLOPE','DIFF','SIGN','ROUND','FLOOR','CEIL','SQRT',
#         'MACD','DEA','DI','ADX','ADXR','HHVBARS','LLVBARS','CONST','BETWEEN'
#     }
#     tokens = {t for t in tokens if t not in blacklist}
#     # 关键字 → 指标名（REGISTRY 键）
#     mapping = {
#         'J': 'kdj',
#         'KDJ_J': 'kdj',
#         'VR': 'volume_ratio',
#         'BBI': 'bbi',
#         'RSI': 'rsi',
#         'BUPIAO_SHORT': 'bupiao',
#         'BUPIAO_LONG': 'bupiao',
#         'DUOKONG_SHORT': 'duokong_short',
#         'DUOKONG_LONG': 'duokong_long',
#         'Z_SCORE': 'z_score',
#     }
#     out = []
#     for t in tokens:
#         key = mapping.get(t)
#         if key and key in REGISTRY:
#             out.append(key)
#     # 去重并保持原顺序
#     seen=set(); unique=[]
#     for n in out:
#         if n not in seen:
#             seen.add(n); unique.append(n)
#     return unique


def names_in_expr(expr: str) -> list[str]:
    import re
    tokens = list(re.findall(r'[A-Z_][A-Z0-9_]*', str(expr).upper()))
    # 内置函数/符号黑名单（去掉 DIFF，避免误杀）
    blacklist = {
        'OPEN','HIGH','LOW','CLOSE','V','VOL','AMOUNT','O','H','L','C',
        'MA','EMA','SMA','LLV','HHV','REF','CROSS','COUNT','IF','MIN','MAX',
        'ABS','STD','STDV','STDDEV','SUM','ZIG','FILTER','BACKSET','BARSLAST',
        'AND','OR','NOT','TRUE','FALSE','N','M','K','D','RSV','WMA','VAR','EXP',
        'SAFE_DIV','EPS','SLOPE','SIGN','ROUND','FLOOR','CEIL','SQRT',
        'MACD','DEA','DI','ADX','ADXR','HHVBARS','LLVBARS','CONST','BETWEEN'
    }
    toks = [t for t in tokens if t not in blacklist]

    # 自动：由 REGISTRY 的 out 反推映射（输出列 -> 指标名）
    mapping = {}
    for key, meta in REGISTRY.items():
        for out_col in (meta.out or {}):
            mapping[out_col.upper()] = key

    # 手工特例（别名）
    mapping.update({
        'J': 'kdj',
        'KDJ_J': 'kdj',
        'VR': 'volume_ratio',
    })

    seen, out = set(), []
    for t in toks:
        key = mapping.get(t)
        if key and key in REGISTRY and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def warmup_for(names: Optional[Union[str, List[str]]]) -> int:
    """
    给一组指标名，返回需要的 warm-up 天数（取最大）。
    names 可以是 "all"、None、单个字符串或列表。
    """
    if names is None:
        return 0
    if isinstance(names, str):
        if names.lower().strip() == 'all':
            sel = list(REGISTRY.keys())
        else:
            sel = [n.strip() for n in names.split(',') if n.strip()]
    else:
        sel = list(names)
    w = 0
    for n in sel:
        meta = REGISTRY.get(n)
        if not meta:
            continue
        try:
            w = max(w, int(meta.warmup or 0))
        except Exception:
            pass
    return int(w)


def estimate_warmup(exprs: Optional[List[str]], recompute_indicators: Union[str, List[str], tuple]) -> int:
    """
    综合“将要重算哪些指标” + “表达式内涉及哪些指标”，估算需要的 warm-up 天数。
    - recompute_indicators: "none" | "all" | [name, ...]
    - exprs: 规则/临时表达式列表；可为 None/空
    返回：所需 warm-up 天数（至少 0）
    """
    # 不重算指标 → 不需要 warm-up
    if isinstance(recompute_indicators, str) and recompute_indicators.lower().strip() == 'none':
        return 0
    # 将要重算的指标集合
    if isinstance(recompute_indicators, str) and recompute_indicators.lower().strip() == 'all':
        need = set(REGISTRY.keys())
    else:
        need = {str(x).strip() for x in (list(recompute_indicators) if isinstance(recompute_indicators, (list, tuple)) else []) if str(x).strip()}
    # 从表达式里补齐隐式依赖
    for e in (exprs or []):
        for n in names_in_expr(e):
            need.add(n)
    # 返回最大 warm-up
    return warmup_for(sorted(need))

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
