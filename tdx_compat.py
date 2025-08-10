import re
import math
import numpy as np
import pandas as pd

EPS = 1e-12
# 统一把条件转成布尔并把 NaN 当 False
def _as_bool(cond):
    s = pd.Series(cond)
    # NaN 一律按 False 处理，避免“开头数据不全”把条件误判为 True
    return s.fillna(False).astype(bool)

def IF(cond, a, b):
    condb = _as_bool(cond)
    idx = a.index if isinstance(a, pd.Series) else (b.index if isinstance(b, pd.Series) else None)
    return pd.Series(np.where(condb, a, b), index=idx)

def COUNT(cond, n):
    cond_series = _as_bool(cond)
    # COUNT 在样本不足 n 时，按“已有样本”计数（TDX 的常见用法也是从起始可用）
    return cond_series.rolling(int(n), min_periods=1).sum()

def BARSLAST(cond):
    cond = _as_bool(cond)
    idx = pd.Series(np.where(cond, np.arange(len(cond)), np.nan), index=cond.index).ffill()
    return pd.Series(np.arange(len(cond)), index=cond.index) - idx

def MA(x, n):  
    return x.rolling(int(n), min_periods=1).mean()

def SUM(x, n): 
    return x.rolling(int(n), min_periods=1).sum()

def HHV(x, n): 
    return x.rolling(int(n), min_periods=1).max()

def LLV(x, n): 
    return x.rolling(int(n), min_periods=1).min()

def STD(x, n): 
    return x.rolling(int(n), min_periods=1).std(ddof=0)

def REF(x, n=1):
    return x.shift(int(n))

def EMA(x, n):
    return x.ewm(span=int(n), adjust=False).mean()

def SMA(x, n, m):
    alpha = float(m) / float(n)
    return x.ewm(alpha=alpha, adjust=False).mean()

# def MA(x, n):
#     return x.rolling(int(n)).mean()

# def SUM(x, n):
#     return x.rolling(int(n)).sum()

# def HHV(x, n):
#     return x.rolling(int(n)).max()

# def LLV(x, n):
#     return x.rolling(int(n)).min()

# def STD(x, n):
#     return x.rolling(int(n)).std(ddof=0)

# def IF(cond, a, b):
#     return pd.Series(np.where(cond.astype(bool), a, b), index=a.index if isinstance(a, pd.Series) else b.index)

# def COUNT(cond, n):
#     cond_series = cond.astype(bool)
#     return cond_series.rolling(int(n)).sum()

# def BARSLAST(cond):
#     cond = cond.astype(bool)
#     idx = pd.Series(np.where(cond, np.arange(len(cond)), np.nan), index=cond.index).ffill()
#     return pd.Series(np.arange(len(cond)), index=cond.index) - idx

def ABS(x):
    return np.abs(x)

def MAX(a, b):
    return np.maximum(a, b)

def MIN(a, b):
    return np.minimum(a, b)

def _ensure_series(x):
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)

def CROSS(a, b):
    a = _ensure_series(a)
    b = _ensure_series(b)
    return (a > b) & (a.shift(1) <= b.shift(1))

def SAFE_DIV(a, b):
    # 安全除法：b≈0 时避免 NaN/Inf
    return a / (b + EPS)

def RSV(C, H, L, n=9):
    # RSV = 100 * (C - LLV(L,n)) / (HHV(H,n) - LLV(L,n) + EPS)
    llv = LLV(L, n)
    hhv = HHV(H, n)
    return 100.0 * (C - llv) / (hhv - llv + EPS)

VAR_MAP = {
    "C": "df['close']",
    "CLOSE": "df['close']",
    "O": "df['open']",
    "OPEN": "df['open']",
    "H": "df['high']",
    "HIGH": "df['high']",
    "L": "df['low']",
    "LOW": "df['low']",
    "V": "df['vol']",
    "VOL": "df['vol']",
    "AMOUNT": "df['amount']",
    "REFDATE": "REFDATE",
    "J": "df['j']",
    "VR": "df['vr']",
    "Z_SLOPE": "df['z_slope']",
    "BBI": "df['bbi']",
    "BUPIAO_SHORT": "df['bupiao_short']",
    "BUPIAO_LONG": "df['bupiao_long']",
}

FUNC_MAP = {
    "REF": "REF",
    "MA": "MA",
    "EMA": "EMA",
    "SMA": "SMA",
    "SUM": "SUM",
    "HHV": "HHV",
    "LLV": "LLV",
    "STD": "STD",
    "ABS": "ABS",
    "MAX": "MAX",
    "MIN": "MIN",
    "IF": "IF",
    "COUNT": "COUNT",
    "CROSS": "CROSS",
    "BARSLAST": "BARSLAST",
}

IGNORE_FUNCS = {
    "STICKLINE",
    "DRAWKLINE",
    "DRAWTEXT",
    "DRAWICON",
    "COLORRED",
    "COLORGREEN",
    "COLORYELLOW",
    "LINETHICK1",
    "LINETHICK2",
    "LINETHICK3",
    "DOTLINE",
    "POINTDOT",
}

LOGICAL_MAP = {
    "AND": "&",
    "OR": "|",
    "NOT": "~",
}

ASSIGN_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:?=\s*(.+?)\s*$')
COMMENT_RE = re.compile(r'^\s*[{].*?[}]\s*$')
INLINE_COMMENT_RE = re.compile(r'\{.*?\}')
SEMICOL_SPLIT_RE = re.compile(r';(?=(?:[^"]*"[^"]*")*[^"]*$)')

def _replace_variables(expr):
    keys = sorted(VAR_MAP.keys(), key=len, reverse=True)
    for k in keys:
        expr = re.sub(rf'(?<![A-Za-z0-9_]){re.escape(k)}(?![A-Za-z0-9_])', VAR_MAP[k], expr)
    return expr

def _replace_functions(expr):
    for k, v in FUNC_MAP.items():
        expr = re.sub(rf'(?<![A-Za-z0-9_]){re.escape(k)}\s*\(', v + "(", expr)
    return expr

def _replace_logicals(expr):
    for k, v in LOGICAL_MAP.items():
        expr = re.sub(rf'(?<![A-Za-z0-9_]){k}(?![A-Za-z0-9_])', v, expr, flags=re.IGNORECASE)
    expr = expr.replace("&&", "&").replace("||", "|").replace("!", "~")
    return expr

def _replace_equality(expr):
    expr = re.sub(r'(?<![<>!:])=(?!=)', '==', expr)
    return expr

def _drop_ignored(expr):
    for name in IGNORE_FUNCS:
        expr = re.sub(rf'{name}\s*\([^()]*\)\s*,?', '', expr)
    return expr

def translate_expression(expr):
    expr = INLINE_COMMENT_RE.sub('', expr)
    expr = _drop_ignored(expr)
    expr = _replace_logicals(expr)
    expr = _replace_functions(expr)
    expr = _replace_variables(expr)
    expr = _replace_equality(expr)
    return expr.strip()

def compile_script(script):
    script = script.replace('\r', '')
    lines = [ln for ln in script.split('\n') if not COMMENT_RE.match(ln)]
    script = '\n'.join(lines)
    parts = [p.strip() for p in SEMICOL_SPLIT_RE.split(script) if p.strip()]
    program = []
    for part in parts:
        m = ASSIGN_RE.match(part)
        if m:
            name, rhs = m.group(1), m.group(2)
            py_expr = translate_expression(rhs)
            program.append((name, py_expr))
        else:
            py_expr = translate_expression(part)
            if py_expr:
                program.append((None, py_expr))
    return program

def evaluate(script, df, extra_context=None):
    program = compile_script(script)
    ctx = {
        "np": np,
        "pd": pd,
        "df": df,
        "REF": REF, "MA": MA, "EMA": EMA, "SMA": SMA, "SUM": SUM, "HHV": HHV, "LLV": LLV,
        "STD": STD, "ABS": ABS, "MAX": MAX, "MIN": MIN, "IF": IF, "COUNT": COUNT, "CROSS": CROSS,
        "BARSLAST": BARSLAST,
        "EPS": EPS,
        "SAFE_DIV": SAFE_DIV,
        "RSV": RSV,
    }
    if extra_context:
        ctx.update(extra_context)

    results = {}
    last_value = None
    for name, expr in program:
        try:
            val = eval(expr, {"__builtins__": {}}, ctx)
        except Exception as e:
            raise RuntimeError(f"Error evaluating expression: {expr}\n{e}")
        if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray, float, int, bool)):
            if isinstance(val, pd.Series):
                val.name = name or getattr(val, 'name', None)
            if name:
                ctx[name] = val
                results[name] = val
            else:
                last_value = val
        else:
            last_value = val
    if last_value is not None:
        results["LAST_EXPR"] = last_value
    return results

def tdx_to_python(script):
    return compile_script(script)
