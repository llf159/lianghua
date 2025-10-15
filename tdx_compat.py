import re
import math
import numpy as np
import pandas as pd

EPS = 1e-12
EXTRA_CONTEXT: dict = {}   # 运行时可注入自定义函数/变量，比如 TS/REF_DATE/RANK_*

COMP_RE = re.compile(r'(<=|>=|==|!=|<|>)')

def _wrap_comparisons_for_bitwise(expr: str) -> str:
    # 在顶层 & 和 | 处分段；分段内若含比较运算，自动加括号
    out, buf, depth = [], [], 0
    def flush():
        seg = ''.join(buf).strip()
        if seg and COMP_RE.search(seg):
            seg = f'({seg})'
        out.append(seg)
        buf.clear()

    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch in '&|':
            flush()
            out.append(ch)
        else:
            buf.append(ch)
    flush()
    return ''.join(out)

def _extract_bool_signal(res: dict, index, prefer_keys=("sig", "last_expr", "SIG", "LAST_EXPR")):
    """
    从 evaluate 的结果字典里，按优先级挑一个键并转成布尔 Series。
    若没有任何候选键，则返回全 False。
    """
    import pandas as pd

    for k in prefer_keys:
        if k in res:
            s = res[k]
            # s 可能是 ndarray / list / Series；统一成布尔 Series，并与 index 对齐
            return pd.Series(s, index=index).fillna(False).astype(bool)

    return pd.Series(False, index=index)

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
    if isinstance(b, pd.Series):
        b = b.reindex_like(a)         # 对齐索引
        b_prev = b.shift(1)
    else:
        # 标量/常数：让 pandas 做广播
        b_prev = b
    return (a > b) & (a.shift(1) <= b_prev)

def SAFE_DIV(a, b):
    # 安全除法：b≈0 时避免 NaN/Inf
    return a / (b + EPS)

def RSV(C, H, L, n=9):
    # RSV = 100 * (C - LLV(L,n)) / (HHV(H,n) - LLV(L,n) + EPS)
    llv = LLV(L, n)
    hhv = HHV(H, n)
    return 100.0 * (C - llv) / (hhv - llv + EPS)

def TS_PCT(x, n):
    """返回与 x 同索引的序列：每日对应“该日值在最近 n 天窗口内的百分位（0..1）”"""
    import numpy as np, pandas as pd
    s = pd.Series(x)
    def pct(arr):
        last = arr[-1]
        return float((arr <= last).sum()) / len(arr)
    return s.rolling(int(n), min_periods=1).apply(lambda a: pct(a.values), raw=False)

def TS_RANK(x, n):
    """返回每日对应“该日值在最近 n 天窗口内的名次（1..n）”"""
    import numpy as np, pandas as pd
    s = pd.Series(x)
    def rk(arr):
        import numpy as np
        last = arr[-1]
        return float((arr <= last).sum())
    return s.rolling(int(n), min_periods=1).apply(lambda a: rk(a.values), raw=False)

def _iter_custom_tag_series(pattern: str, df_index):
    """从 EXTRA_CONTEXT['CUSTOM_TAGS'] 里按名称匹配，取出与 df_index 对齐的布尔序列"""
    import re, pandas as pd
    tags = EXTRA_CONTEXT.get("CUSTOM_TAGS", {})
    if not isinstance(tags, dict) or not tags:
        return []
    pat = pattern.strip()
    is_regex = bool(re.search(r"[.^$*+?{}\[\]|()]", pat))
    if not is_regex and "|" in pat:
        keys = [k.strip() for k in pat.split("|") if k.strip()]
        names = [k for k in tags.keys() if any(kk.lower() in str(k).lower() for kk in keys)]
    else:
        if is_regex:
            rx = re.compile(pat, flags=re.IGNORECASE)
            names = [k for k in tags.keys() if rx.search(str(k))]
        else:
            names = [k for k in tags.keys() if pat.lower() in str(k).lower()]
    out = []
    for name in names:
        try:
            s = tags.get(name)
            if s is None: 
                continue
            s2 = _coerce_bool_series(s)
            if df_index is not None:
                # s2 = s2.reindex(df_index).fillna(False)
                s2 = s2.reindex(df_index, fill_value=False).infer_objects().astype(bool)
            out.append((name, s2))
        except Exception:
            continue
    return out

def _coerce_bool_series(x):
    import numpy as np, pandas as pd
    s = pd.Series(x)
    if s.dtype == bool:
        return s.fillna(False)
    # 数值/字符串等：非零/非空视为 True
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0) != 0
    return s.astype(object).apply(lambda v: bool(v) if v is not None and v == v else False)

# —— 根据“标签”文本，自动搜列名并 OR 到一起；shift 可表示引用历史（1=昨日）——
# def ANY_TAG(pattern: str, shift: int = 0):
#     """
#     pattern: 子串或正则（自动识别）。多个关键词可用竖线 '|' 写在一起（视为“或”）。
#     shift  : 0=当日，1=昨日，2=前日...
#     规则：在 df.columns 里寻找“列名包含 pattern（不区分大小写）”的列，逐列转布尔后做 OR。
#     """
#     import re, pandas as pd
#     df = EXTRA_CONTEXT.get("DF", None)
#     if df is None or getattr(df, "empty", True):
#         # 兜底：给一个全 False 的布尔序列
#         return pd.Series(False, index=getattr(df, "index", None))

#     pat = pattern.strip()
#     # 自动判断是否按正则：含有正则元字符就按正则匹配，否则按不区分大小写的子串匹配
#     is_regex = bool(re.search(r"[.^$*+?{}\[\]|()]", pat))
#     if not is_regex and "|" in pat:
#         # 多关键字 OR：拆开做子串匹配
#         keys = [k.strip() for k in pat.split("|") if k.strip()]
#         cols = [c for c in df.columns
#                 if any(k.lower() in str(c).lower() for k in keys)]
#     else:
#         if is_regex:
#             rx = re.compile(pat, flags=re.IGNORECASE)
#             cols = [c for c in df.columns if rx.search(str(c))]
#         else:
#             cols = [c for c in df.columns if pat.lower() in str(c).lower()]

#     if not cols:
#         return pd.Series(False, index=df.index)

#     custom = _iter_custom_tag_series(pat, getattr(df, 'index', None))
#     if not cols and not custom:
#         return pd.Series(False, index=df.index)
    
#     agg = None
#     for c in cols:
#         try:
#             s = _coerce_bool_series(df[c])
#         except Exception:
#             continue
#         agg = s if agg is None else (agg | s)
#     for name, s in custom:
#         try:
#             s = _coerce_bool_series(s)
#         except Exception:
#             continue
#         agg = s if agg is None else (agg | s)

#     if agg is None:
#         agg = pd.Series(False, index=df.index)

#     if int(shift) != 0:
#         agg = agg.shift(int(shift)).fillna(False)

#     return agg

def ANY_TAG(pattern: str, shift: int = 0):
    import re, pandas as pd
    df = EXTRA_CONTEXT.get("DF", None)
    if df is None or getattr(df, "empty", True):
        return pd.Series(False, index=getattr(df, "index", None))

    pat = pattern.strip()
    is_regex = bool(re.search(r"[.^$*+?{}\[\]|()]", pat))
    # 先确定 df 列匹配
    if not is_regex and "|" in pat:
        keys = [k.strip() for k in pat.split("|") if k.strip()]
        cols = [c for c in df.columns if any(k.lower() in str(c).lower() for k in keys)]
    else:
        cols = [c for c in df.columns if (re.compile(pat, re.I).search(str(c)) if is_regex else pat.lower() in str(c).lower())]

    # 再取自定义标签（与 df.index 对齐）
    custom = _iter_custom_tag_series(pat, getattr(df, 'index', None))

    # 两者都空，才返回空序列
    if not cols and not custom:
        return pd.Series(False, index=df.index)

    agg = None
    for c in cols:
        try:
            s = _coerce_bool_series(df[c])
        except Exception:
            continue
        agg = s if agg is None else (agg | s)
    for _, s in custom:
        try:
            s = _coerce_bool_series(s)
        except Exception:
            continue
        agg = s if agg is None else (agg | s)

    if agg is None:
        agg = pd.Series(False, index=df.index)
    if int(shift) != 0:
        agg = agg.shift(int(shift)).fillna(False)
    return agg

# 语义糖：专指“昨日任意匹配标签为 True”
def YDAY_ANY_TAG(pattern: str):
    return ANY_TAG(pattern, shift=1)

# —— 统计匹配标签的“当日命中个数”，以及“是否至少命中 k 个” —— 
def TAG_HITS(pattern: str, shift: int = 0):
    """
    计数“命中标签”的个数。优先按 bucket 精确匹配 CUSTOM_TAGS；
    若无则回落到列名匹配 / 自定义标签名匹配；最后支持 shift（1=昨日）。
    """
    import re, pandas as pd

    df = EXTRA_CONTEXT.get("DF", None)
    # 先拿到 index；若 DF 没列/为空，尝试从 CUSTOM_TAGS 的任一序列推断
    idx = getattr(df, "index", None)
    if idx is None or len(idx) == 0:
        tags = EXTRA_CONTEXT.get("CUSTOM_TAGS", {}) or {}
        for ser in tags.values():
            try:
                cand = getattr(ser, "index", None)
                if cand is not None and len(cand) > 0:
                    idx = cand
                    break
            except Exception:
                pass
    if idx is None or len(idx) == 0:
        return pd.Series(dtype=int)  # 实在没有 index 就返回空序列

    pat = pattern.strip()
    is_regex = bool(re.search(r"[.^$*+?{}\[\]|()]", pat))
    hits = None

    # ① bucket 精确匹配（pattern 为普通词且不含 '|'）
    if pat and not is_regex and "|" not in pat:
        tags = EXTRA_CONTEXT.get("CUSTOM_TAGS", {}) or {}
        want = pat.lower()
        for key, ser in tags.items():
            try:
                parts = str(key).split("::", 2)  # ["CFG_TAG", bucket, name]
                if len(parts) >= 2 and (parts[1] or "").lower() == want:
                    # s = _coerce_bool_series(ser).reindex(idx).fillna(0).astype(int)
                    s = (_coerce_bool_series(ser)
                        .reindex(idx, fill_value=False)
                        .infer_objects()
                        .astype(int))
                    hits = s if hits is None else (hits + s)
            except Exception:
                continue

    # ② 列名匹配（仅当 df 存在且有列时才尝试）
    if hits is None and df is not None and hasattr(df, "columns") and len(df.columns) > 0:
        cols = list(df.columns)
        if pat and "|" in pat and not is_regex:
            keys = [k.strip() for k in pat.split("|") if k.strip()]
            names = [c for c in cols if any(kk.lower() in str(c).lower() for kk in keys)]
        elif is_regex:
            rx = re.compile(pat, flags=re.IGNORECASE)
            names = [c for c in cols if rx.search(str(c))]
        else:
            names = [c for c in cols if pat.lower() in str(c).lower()] if pat else []
        if names:
            s = sum((_coerce_bool_series(df[n]).astype(int) for n in names), start=pd.Series(0, index=idx, dtype=int))
            hits = s

    # ③ 自定义标签名称匹配（regex/子串）
    if hits is None:
        parts = list(_iter_custom_tag_series(pat, idx))
        if parts:
            # s = sum((_coerce_bool_series(ser).reindex(idx).fillna(0).astype(int) for ser in parts),
            #         start=pd.Series(0, index=idx, dtype=int))
            s = sum(
                ( _coerce_bool_series(ser)
                    .reindex(idx, fill_value=False)
                    .infer_objects()
                    .astype(int)
                for _, ser in parts),
                start=pd.Series(0, index=idx, dtype=int)
                )
            hits = s

    # ④ 兜底 + 位移
    if hits is None:
        hits = pd.Series(0, index=idx, dtype=int)
    if int(shift) != 0:
        hits = hits.shift(int(shift)).fillna(0).astype(int)
    return hits


def ANY_TAG_AT_LEAST(pattern: str, k: int, shift: int = 0):
    k = int(k)
    return TAG_HITS(pattern, shift=shift) >= k

# 语义糖（昨日）
def YDAY_TAG_HITS(pattern: str):
    return TAG_HITS(pattern, shift=1)


def YDAY_ANY_TAG_AT_LEAST(pattern: str, k: int):
    return ANY_TAG_AT_LEAST(pattern, k, shift=1)

# 注册到额外上下文，供表达式直接调用
EXTRA_CONTEXT["TS_PCT"] = TS_PCT
EXTRA_CONTEXT["TS_RANK"] = TS_RANK
EXTRA_CONTEXT.update({
    "ANY_TAG": ANY_TAG,
    "YDAY_ANY_TAG": YDAY_ANY_TAG,
    "TAG_HITS": TAG_HITS,
    "ANY_TAG_AT_LEAST": ANY_TAG_AT_LEAST,
    "YDAY_TAG_HITS": YDAY_TAG_HITS,
    "YDAY_ANY_TAG_AT_LEAST": YDAY_ANY_TAG_AT_LEAST,
})

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
    "REFDATE": "REF_DATE",
    "J": "df['j']",
    "j": "df['j']",  # 添加小写版本
    "K": "df['k']",
    "k": "df['k']",  # 添加小写版本
    "D": "df['d']",
    "d": "df['d']",  # 添加小写版本
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
    "TS_RANK": "TS_RANK",
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

# ASSIGN_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:?=\s*(.+?)\s*$')
ASSIGN_RE = re.compile(r'^\s*([^\W\d]\w*)\s*:?=\s*(.+?)\s*$', flags=re.UNICODE)
COMMENT_RE = re.compile(r'^\s*[{].*?[}]\s*$')
INLINE_COMMENT_RE = re.compile(r'\{.*?\}')
SEMICOL_SPLIT_RE = re.compile(r';(?=(?:[^"]*"[^"]*")*[^"]*$)')

import logging
LOG = logging.getLogger("tdx")

def _extra_ctx_from_df_columns(df) -> dict:
    """
    为 df 的每一列建立两个别名：
      - 原列名（如 z_score）
      - 全大写（如 Z_SCORE）
    避免覆盖内置函数名、已存在的 VAR_MAP key 等。
    """
    reserved = set(FUNC_MAP.keys()) | set(VAR_MAP.keys()) | {"AND","OR","NOT"}
    ctx = {}
    for col in getattr(df, "columns", []):
        if not isinstance(col, str):
            continue
        series = df[col]
        for key in (col, col.upper()):
            if key in reserved:
                continue
            # 合法的标识符才挂（TDX 里你本来也写成这种）
            if key.isidentifier():
                # 如果上一步替换里已经把某些固定符号映射到 df['...']，这里就当补充别名
                ctx.setdefault(key, series)
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug("[TDX] alias count=%d (sample=%s)", len(ctx), list(ctx)[:10])
    return ctx

def evaluate_bool(script: str, df, prefer_keys=("sig", "last_expr", "SIG", "LAST_EXPR")):
    """
    便捷入口：直接返回一个布尔 Series（与 df.index 对齐）。
    自动将 df 中现有列注入上下文（原名 + 全大写）。
    """
    extra_ctx = _extra_ctx_from_df_columns(df)
    if EXTRA_CONTEXT:
        extra_ctx.update(EXTRA_CONTEXT)
    res = evaluate(script, df, extra_context=extra_ctx)  # evaluate 会把 extra_context 合入 ctx 使用
    return _extract_bool_signal(res, df.index, prefer_keys=prefer_keys)

def _replace_variables(expr):
    keys = sorted(VAR_MAP.keys(), key=len, reverse=True)
    for k in keys:
        # 只替换独立的变量名，不替换已经在 df['...'] 中的内容
        # 检查是否已经包含该变量的替换结果
        replacement = VAR_MAP[k]
        if replacement in expr:
            continue  # 如果已经替换过，跳过
        expr = re.sub(rf'(?<![A-Za-z0-9_]){re.escape(k)}(?![A-Za-z0-9_])', replacement, expr)
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
    expr = _wrap_comparisons_for_bitwise(expr)
    return expr.strip()


def compile_script(script):
    script = script.replace('\r', '')
    script = script.translate(str.maketrans({'；': ';', '：': ':', '（': '(', '）': ')', '，': ','}))
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
    EXTRA_CONTEXT["DF"] = df
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
        "TS_RANK": TS_RANK,
    }
    if extra_context:
        ctx.update(extra_context) 
    
    # 添加 DataFrame 列的动态访问支持
    # 这样 VAR_MAP 中的 "J": "df['j']" 就能正确工作
    for col in df.columns:
        if isinstance(col, str) and col.isidentifier():
            ctx[f"df['{col}']"] = df[col]
    
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
    results = {k.lower(): v for k, v in results.items()}
    return results


def tdx_to_python(script):
    return compile_script(script)
