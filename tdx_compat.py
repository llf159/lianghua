import re
import math
import logging
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Dict, Tuple, Any, Optional

EPS = 1e-12
EXTRA_CONTEXT: dict = {}   # 运行时可注入自定义函数/变量，比如 TS/REF_DATE/RANK_*

# 表达式编译缓存统计
_EXPR_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "size": 0
}

COMP_RE = re.compile(r'(<=|>=|==|!=|<|>)')
# 函数名模式（用于识别函数调用，如REF(、COUNT(等）
FUNC_NAME_RE = re.compile(r'[A-Z_][A-Z0-9_]*\s*\(')


def _wrap_comparisons_for_bitwise(expr: str) -> str:
    """
    目标：
      1) 在顶层把 & 和 | 两侧的段落一律包成 (...)，
      2) 在函数的"第一个参数"段里若含 &/|，也整体包成 (...).
      3) 确保每个比较表达式（>=, <=, ==, !=, <, >）都被正确包裹
    说明：这是保守做法（可能多一些括号），但能稳定避免 Python 的优先级/链式比较陷阱。
    """
    def wrap_comparisons_in_segment(seg: str) -> str:
        """在片段中包裹所有比较表达式，确保每个比较操作都被括号包裹"""
        # 使用正则表达式找到所有比较表达式模式：value >=/<=/==/!=/</> value
        # 但要注意函数调用中的参数
        import re
        
        # 先找到所有比较运算符的位置，但要在正确的上下文（不在字符串或嵌套函数参数中）
        result = []
        i = 0
        depth = 0
        in_string = False
        
        while i < len(seg):
            ch = seg[i]
            
            # 跟踪字符串
            if ch in ('"', "'") and (i == 0 or seg[i-1] != '\\'):
                in_string = not in_string
            
            if not in_string:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                
                # 找到比较运算符（在正确的深度）
                if depth == 0 and i > 0 and i < len(seg) - 1:
                    # 检查是否是 >=, <=, ==, !=
                    if i + 1 < len(seg):
                        two_char = seg[i:i+2]
                        if two_char in ('>=', '<=', '==', '!='):
                            # 找到比较运算符，需要包裹前后
                            # 向前找比较的左操作数（跳过空格）
                            left_start = i - 1
                            while left_start >= 0 and seg[left_start] in ' \t':
                                left_start -= 1
                            
                            # 向后找比较的右操作数（跳过空格）
                            right_end = i + 2
                            while right_end < len(seg) and seg[right_end] in ' \t':
                                right_end += 1
                            
                            # 如果这个比较表达式还没有被括号包裹，就包裹它
                            # 检查前面的 &| 和后面的 &| 来确定边界
                            # 简化：在 & 或 | 之间，每个比较表达式都应该被包裹
                            result.append(ch)
                            i += 2
                            continue
            
            result.append(ch)
            i += 1
        
        return ''.join(result)
    
    def wrap_top_level(s: str) -> str:
        out, seg, depth = [], [], 0
        in_string = False  # 标记是否在字符串中
        def flush():
            t = ''.join(seg).strip()
            if t:
                # 在片段中，确保每个比较表达式都被包裹
                # 但只在包含 & 或 | 时才需要额外包裹
                if '&' in t or '|' in t:
                    # 在每个 & 或 | 之前和之后包裹比较表达式
                    # 使用正则表达式找到所有比较表达式并包裹它们
                    import re
                    # 匹配模式：值 比较运算符 值（在 & 或 | 的上下文中）
                    # 简化为：在 & 或 | 之间包裹每个子表达式
                    parts = re.split(r'([&|])', t)
                    wrapped_parts = []
                    for i, part in enumerate(parts):
                        part = part.strip()
                        if part and part not in ('&', '|'):
                            # 如果 part 包含比较运算符且没有被括号包裹，就包裹它
                            if COMP_RE.search(part):
                                # 检查括号是否平衡
                                depth_check = 0
                                has_unbalanced = False
                                for ch in part:
                                    if ch == '(':
                                        depth_check += 1
                                    elif ch == ')':
                                        depth_check -= 1
                                    if depth_check < 0:
                                        has_unbalanced = True
                                        break
                                
                                # 如果括号不平衡或者没有被完整包裹，就包裹它
                                if depth_check != 0 or has_unbalanced or not (part.startswith('(') and part.endswith(')')):
                                    wrapped_parts.append(f'({part})')
                                else:
                                    wrapped_parts.append(part)
                            else:
                                wrapped_parts.append(part)
                        else:
                            wrapped_parts.append(part)
                    t = ''.join(wrapped_parts)
                out.append(f'({t})' if t else '')
            seg.clear()
        for ch in s:
            # 简单处理字符串引号
            if ch in ('"', "'"):
                in_string = not in_string
                seg.append(ch)
                continue
            
            if not in_string:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                
                # 只有在 depth == 0 时才分割（这意味着在顶层，不在任何函数调用内部）
                if depth == 0 and ch in '&|':
                    flush()
                    out.append(ch)
                else:
                    seg.append(ch)
            else:
                # 在字符串内，直接添加字符
                seg.append(ch)
        flush()
        return ''.join(out)

    # 先处理"函数第一个参数"里可能存在的 &/|
    # 注意：必须在 wrap_top_level 之前处理，因为 wrap_top_level 会在顶层分割 &/|
    # 如果先 wrap_top_level，函数参数内的 &/| 会被错误地移到外面
    out, i, n = [], 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch.isalpha() or ch == '_':
            # 读取可能的函数名
            j = i + 1
            while j < n and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            fn = expr[i:j]
            # 若后面紧跟 '('，进入函数参数处理
            if j < n and expr[j] == '(':
                out.append(fn)
                out.append('(')
                depth = 1
                k = j + 1
                # 抓取"第一个参数"直到逗号（或右括号）
                first_arg = []
                found_comma = False
                # 标记是否在字符串中（简化处理，假设表达式不包含复杂字符串字面量）
                in_string = False
                while k < n and depth > 0:
                    c = expr[k]
                    # 简单处理字符串引号（单引号和双引号）
                    if c in ('"', "'"):
                        # 切换字符串状态
                        in_string = not in_string
                        first_arg.append(c)
                        k += 1
                        continue
                    
                    if not in_string:
                        if c == '(':
                            depth += 1
                            first_arg.append(c)
                        elif c == ')':
                            depth -= 1
                            if depth == 0:
                                # 遇到右括号，说明没有第二个参数（这是单参数函数调用）
                                # 不添加右括号到 first_arg，因为我们会在后面统一处理
                                break
                            else:
                                first_arg.append(c)
                        elif depth == 1 and c == ',':
                            # 找到第一个参数的结束位置（逗号）
                            # 注意：这里的 depth == 1 确保逗号在函数调用的第一层，而不是嵌套函数中
                            found_comma = True
                            k += 1  # 跳过逗号
                            break
                        else:
                            first_arg.append(c)
                    else:
                        # 在字符串内，直接添加字符
                        first_arg.append(c)
                    k += 1
                
                # k 此时在：
                # - 如果找到逗号：在逗号后的第一个字符位置
                # - 如果没有逗号：在右括号的位置（depth == 0 时 break）
                
                # 如果第一个参数里含 &/|，整体再包一层
                fa = ''.join(first_arg).strip()
                if ('&' in fa) or ('|' in fa):
                    fa = f'({fa})'
                out.append(fa)
                
                # 如果有第二个参数，继续处理后续内容
                if found_comma:
                    # 找到了逗号，需要添加逗号并处理后续参数
                    out.append(',')
                    # 把余下直到配对右括号的内容复制
                    # 此时 depth 应该是 1（函数调用的深度），因为我们找到了逗号后 depth 仍然是 1
                    rest_depth = 1
                    in_string = False
                    while k < n:
                        c = expr[k]
                        # 简单处理字符串引号
                        if c in ('"', "'"):
                            in_string = not in_string
                            out.append(c)
                            k += 1
                            continue
                        
                        if not in_string:
                            if c == '(':
                                rest_depth += 1
                            elif c == ')':
                                rest_depth -= 1
                                if rest_depth == 0:
                                    out.append(c)  # 添加右括号
                                    k += 1
                                    break
                        out.append(c)
                        k += 1
                else:
                    # 没有找到逗号，说明这是单参数函数调用
                    # k 现在在右括号位置，我们需要添加右括号并继续
                    if k < n and expr[k] == ')':
                        out.append(')')
                        k += 1
                
                i = k
                continue
        # 非函数名起始，正常抄
        out.append(ch)
        i += 1
    s = ''.join(out)
    
    # 再做顶层包裹（此时函数参数内的 &/| 已经被正确处理）
    s = wrap_top_level(s)
    
    return s


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
            series = pd.Series(s, index=index, dtype=object)
            # 使用条件表达式处理 NaN 以避免 downcasting 警告
            series = series.where(series.notna(), False)
            return series.infer_objects(copy=False).astype(bool)

    return pd.Series(False, index=index)

# 统一把条件转成布尔并把 NaN 当 False
def _as_bool(cond):
    # 如果 cond 已经是布尔类型的 Series，直接返回
    if isinstance(cond, pd.Series) and cond.dtype == bool:
        return cond.fillna(False)
    # 如果 cond 是数值类型的 Series，先转换为布尔
    if isinstance(cond, pd.Series) and np.issubdtype(cond.dtype, np.number):
        return (cond != 0).fillna(False)
    # 处理 numpy 数组
    if isinstance(cond, np.ndarray):
        # 如果是布尔数组，直接转换
        if cond.dtype == bool:
            return pd.Series(cond).fillna(False)
        # 如果是数值数组，转换为布尔
        if np.issubdtype(cond.dtype, np.number):
            return pd.Series(cond != 0).fillna(False)
        # 其他类型，先转为对象类型再转换
        s = pd.Series(cond, dtype=object)
        s = s.where(s.notna(), False)
        return s.infer_objects(copy=False).astype(bool)
    # 处理标量值
    try:
        # 兼容不同版本的 numpy
        bool_types = (bool,)
        if hasattr(np, 'bool_'):
            bool_types = (bool, np.bool_)
        if isinstance(cond, bool_types):
            # 如果是单个布尔值，需要创建 Series（但通常不应该单独传入标量）
            # 这里返回一个只包含该值的 Series，但需要确保有正确的 index
            # 实际上，这种情况不应该发生，但为了健壮性，我们尝试从上下文获取 index
            df = EXTRA_CONTEXT.get("DF")
            if df is not None and not df.empty:
                return pd.Series([bool(cond)] * len(df), index=df.index, dtype=bool)
            return pd.Series([bool(cond)], dtype=bool)
    except Exception:
        pass
    # 通用处理：转为 Series 然后转为布尔
    try:
        s = pd.Series(cond, dtype=object)
        # NaN 一律按 False 处理，避免"开头数据不全"把条件误判为 True
        # 使用 where 而不是 replace/fillna 以避免 downcasting 警告
        s = s.where(s.notna(), False)
        return s.infer_objects(copy=False).astype(bool)
    except Exception:
        # 如果转换失败，尝试从上下文获取 DataFrame 并创建全 False 序列
        df = EXTRA_CONTEXT.get("DF")
        if df is not None and not df.empty:
            return pd.Series(False, index=df.index, dtype=bool)
        return pd.Series([False], dtype=bool)

def IF(cond, a, b):
    condb = _as_bool(cond)
    idx = a.index if isinstance(a, pd.Series) else (b.index if isinstance(b, pd.Series) else None)
    return pd.Series(np.where(condb, a, b), index=idx)

def COUNT(cond, n):
    try:
        # 先尝试转换条件为布尔类型
        # 如果 cond 是一个表达式的结果（可能包含 & 或 | 运算符），
        # 需要确保所有子表达式都被正确计算并转换为布尔类型
        try:
            cond_series = _as_bool(cond)
        except (TypeError, ValueError) as e:
            # 如果转换失败，可能是因为类型不匹配
            # 尝试手动处理：如果是 Series，确保是布尔类型
            if isinstance(cond, pd.Series):
                # 如果已经是布尔类型，直接使用
                if cond.dtype == bool:
                    cond_series = cond.fillna(False)
                # 如果是数值类型，转换为布尔
                elif np.issubdtype(cond.dtype, np.number):
                    cond_series = (cond != 0).fillna(False).astype(bool)
                else:
                    # 其他类型，尝试转换为布尔
                    cond_series = pd.Series(cond).fillna(False).astype(bool)
            elif isinstance(cond, np.ndarray):
                # numpy 数组
                if cond.dtype == bool:
                    cond_series = pd.Series(cond).fillna(False)
                else:
                    cond_series = pd.Series(cond != 0).fillna(False).astype(bool)
            else:
                # 其他类型，尝试转换为 Series 然后转为布尔
                df = EXTRA_CONTEXT.get("DF")
                if df is not None and not df.empty:
                    # 尝试将 cond 视为标量，创建全为 cond 的 Series
                    if isinstance(cond, (bool, np.bool_)):
                        cond_series = pd.Series([bool(cond)] * len(df), index=df.index, dtype=bool)
                    else:
                        # 尝试转换
                        cond_series = pd.Series(cond, index=df.index).fillna(False).astype(bool)
                else:
                    cond_series = pd.Series([False], dtype=bool)
        
        # 确保 cond_series 是一个有效的 Series
        if not isinstance(cond_series, pd.Series) or cond_series.empty:
            df = EXTRA_CONTEXT.get("DF")
            if df is not None and not df.empty:
                return pd.Series(0.0, index=df.index, dtype=float)
            return pd.Series([0.0], dtype=float)
        
        # COUNT 在样本不足 n 时，按"已有样本"计数（TDX 的常见用法也是从起始可用）
        n_int = int(n)
        if n_int <= 0:
            n_int = 1
        result = cond_series.rolling(n_int, min_periods=1).sum()
        # 确保结果是数值类型（rolling sum 应该是 float）
        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=cond_series.index, dtype=float)
        return result.astype(float)
    except Exception as e:
        # 如果出错，返回全 0 序列
        df = EXTRA_CONTEXT.get("DF")
        if df is not None and not df.empty:
            return pd.Series(0.0, index=df.index, dtype=float)
        return pd.Series([0.0], dtype=float)

def BARSLAST(cond):
    cond = _as_bool(cond)
    idx = pd.Series(np.where(cond, np.arange(len(cond)), np.nan), index=cond.index).ffill().infer_objects(copy=False)
    return pd.Series(np.arange(len(cond)), index=cond.index) - idx

def _rolling_strict(x, n, func):
    """严格 rolling：窗口内任意 NaN 则返回 NaN。"""
    s = pd.Series(x)
    n_int = max(int(n), 1)
    return s.rolling(n_int, min_periods=n_int).apply(
        lambda arr: func(arr) if not np.isnan(arr).any() else np.nan,
        raw=True
    )

def MA(x, n):  
    return _rolling_strict(x, n, lambda a: np.mean(a))

def SUM(x, n): 
    return _rolling_strict(x, n, lambda a: np.sum(a))

def HHV(x, n): 
    return _rolling_strict(x, n, lambda a: np.max(a))

def LLV(x, n): 
    return _rolling_strict(x, n, lambda a: np.min(a))

def STD(x, n): 
    return _rolling_strict(x, n, lambda a: np.std(a, ddof=0))

def REF(x, n=1):
    return x.shift(int(n))

def EMA(x, n):
    s = pd.Series(x)
    res = s.ewm(span=int(n), adjust=False).mean()
    if s.isna().any():
        res = res.mask(s.isna(), np.nan)
    return res

def SMA(x, n, m):
    alpha = float(m) / float(n)
    s = pd.Series(x)
    res = s.ewm(alpha=alpha, adjust=False).mean()
    if s.isna().any():
        res = res.mask(s.isna(), np.nan)
    return res

def ABS(x):
    return np.abs(x)

def MAX(a, b):
    return np.maximum(a, b)

def MIN(a, b):
    return np.minimum(a, b)

def ATAN(x):
    """计算反正切，返回弧度值"""
    return np.arctan(x)

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
    s = pd.Series(x)
    n_int = max(int(n), 1)
    def pct(arr):
        if np.isnan(arr).any():
            return np.nan
        last = arr[-1]
        return float((arr <= last).sum()) / len(arr)
    return s.rolling(n_int, min_periods=n_int).apply(lambda a: pct(a.values), raw=False)

def TS_RANK(x, n):
    """返回每日对应“该日值在最近 n 天窗口内的名次（1..n）”"""
    s = pd.Series(x)
    n_int = max(int(n), 1)
    def rk(arr):
        if np.isnan(arr).any():
            return np.nan
        last = arr[-1]
        return float((arr <= last).sum())
    return s.rolling(n_int, min_periods=n_int).apply(lambda a: rk(a.values), raw=False)

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
    # 先以原始类型创建 Series，避免强制转换为 object
    s = pd.Series(x)
    if s.dtype == bool:
        # 对于已经是 bool 类型的，确保为 object 再进行 where 避免警告
        s = s.astype(object)
        s = s.where(s.notna(), False)
        return s.infer_objects(copy=False)
    # 数值/字符串等：非零/非空视为 True
    if np.issubdtype(s.dtype, np.number):
        # 对于数值类型，先转为 object 避免警告
        s = s.astype(object)
        s = s.where(s.notna(), 0)
        return (s.infer_objects(copy=False) != 0)
    # 已经是 object 类型或其他类型
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
        agg = agg.shift(int(shift))
        agg = agg.where(agg.notna(), False)
        agg = agg.infer_objects(copy=False)
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
        hits = hits.shift(int(shift))
        hits = hits.where(hits.notna(), 0)
        hits = hits.infer_objects(copy=False).astype(int)
    return hits


def ANY_TAG_AT_LEAST(pattern: str, k: int, shift: int = 0):
    k = int(k)
    return TAG_HITS(pattern, shift=shift) >= k

# 语义糖（昨日）
def YDAY_TAG_HITS(pattern: str):
    return TAG_HITS(pattern, shift=1)


def YDAY_ANY_TAG_AT_LEAST(pattern: str, k: int):
    return ANY_TAG_AT_LEAST(pattern, k, shift=1)


def GET_LAST_CONDITION_PRICE(condition_expr: str, lookback: int = 100):
    """
    获取上一次满足指定条件的收盘价
    
    参数:
    - condition_expr: 条件表达式字符串，如 'j < 13' 或 'C > MA(C, 20)'
    - lookback: 回看天数，默认100天
    
    返回:
    - 收盘价序列，如果该日满足条件则返回当日收盘价，否则返回NaN
    """
    df = EXTRA_CONTEXT.get("DF", None)
    if df is None or df.empty:
        return pd.Series(np.nan, index=getattr(df, "index", None))
    
    # 获取收盘价列
    close_col = None
    for col in df.columns:
        if str(col).lower() in ['close', 'c']:
            close_col = col
            break
    
    if close_col is None:
        return pd.Series(np.nan, index=df.index)
    
    close_values = pd.to_numeric(df[close_col], errors="coerce")
    result = pd.Series(np.nan, index=df.index)
    
    # 按lookback窗口查找
    for i in range(lookback, len(df)):
        window_df = df.iloc[i-lookback:i+1].copy()
        window_df = window_df.reset_index(drop=True)
        
        # 设置EXTRA_CONTEXT为当前窗口
        original_df = EXTRA_CONTEXT.get("DF")
        EXTRA_CONTEXT["DF"] = window_df
        
        try:
            # 评估条件表达式
            # 使用 skip_var_replacement=False 以便变量替换正确处理表达式
            condition_result = evaluate_bool(condition_expr, window_df, skip_var_replacement=False)
            if len(condition_result) > 0 and condition_result.iloc[-1]:
                result.iloc[i] = close_values.iloc[i]
        except Exception as e:
            # 记录异常但继续处理
            pass
        finally:
            # 恢复原始DF
            EXTRA_CONTEXT["DF"] = original_df
    
    return result


def FIND_LAST_LOWEST_J(threshold: float = 13.0, lookback: int = 100):
    """
    查找上一次KDJ的J值最低点，且J值要低于指定阈值
    
    参数:
    - threshold: J值阈值，默认13.0
    - lookback: 回看天数，默认100天
    
    返回:
    - 收盘价序列，返回在lookback窗口内满足threshold条件的J值最低点对应的收盘价
      用于获取买点参考价格
    """
    df = EXTRA_CONTEXT.get("DF", None)
    if df is None or df.empty:
        return pd.Series(np.nan, index=getattr(df, "index", None))
    
    # 获取J值列和收盘价列
    j_col = None
    close_col = None
    
    for col in df.columns:
        if str(col).lower() in ['j', 'kdj_j']:
            j_col = col
        elif str(col).lower() in ['close', 'c']:
            close_col = col
    
    if j_col is None or close_col is None:
        return pd.Series(np.nan, index=df.index)
    
    j_values = pd.to_numeric(df[j_col], errors="coerce")
    close_values = pd.to_numeric(df[close_col], errors="coerce")
    result = pd.Series(np.nan, index=df.index)
    
    # 对于每一天，回看lookback天，找到历史上满足条件的J值最低点
    for i in range(lookback, len(j_values)):
        # 回看窗口：从i-lookback到i（不包含i+1）
        window = j_values.iloc[i-lookback:i]
        if window.isna().all():
            continue
            
        # 找到窗口内J值低于阈值的所有位置
        below_threshold = window < threshold
        if not below_threshold.any():
            continue
            
        # 在这些位置中找到最小值（历史最低点）
        valid_values = window[below_threshold]
        if valid_values.empty:
            continue
            
        min_j = valid_values.min()
        min_positions = valid_values[valid_values == min_j].index
        
        # 如果有多个位置都是最小值，取最近的一个（index最大的）
        if len(min_positions) > 0:
            latest_min_pos = max(min_positions)
            # 返回历史最低点对应的收盘价
            result.iloc[i] = close_values.iloc[latest_min_pos]
    
    return result

# =================== DIFF分析相关TDX函数 ==========================
def find_last_diff_high(df, lookback_days=60):
    """
    找到上次DIFF最高点的信息
    
    Args:
        df: 包含历史数据的DataFrame，需要包含close和diff列
        lookback_days: 回看天数
        
    Returns:
        包含最高点信息的字典，如果没有找到返回None
    """
    if df.empty or 'diff' not in df.columns:
        return None
        
    # 确保数据按日期排序
    df_sorted = df.sort_values('trade_date').copy()
    
    # 找到DIFF最高点的位置
    max_diff_idx = df_sorted['diff'].idxmax()
    if pd.isna(max_diff_idx):
        return None
        
    max_diff_row = df_sorted.loc[max_diff_idx]
    
    return {
        'date': str(max_diff_row['trade_date']),
        'price': float(max_diff_row['close']),
        'diff_value': float(max_diff_row['diff']),
        'index': max_diff_idx
    }


def get_last_diff_high_price(df, lookback_days=60):
    """
    获取上次DIFF最高点时的收盘价
    
    Args:
        df: 历史数据
        lookback_days: 回看天数
        
    Returns:
        上次DIFF最高点时的收盘价，如果未找到返回0
    """
    result = find_last_diff_high(df, lookback_days)
    return result['price'] if result else 0.0


def get_last_diff_high_value(df, lookback_days=60):
    """
    获取上次DIFF最高点的数值
    
    Args:
        df: 历史数据
        lookback_days: 回看天数
        
    Returns:
        上次DIFF最高点的数值，如果未找到返回0
    """
    result = find_last_diff_high(df, lookback_days)
    return result['diff_value'] if result else 0.0


def GET_LAST_DIFF_HIGH_PRICE(lookback_days=60):
    """
    获取上次DIFF最高点时的收盘价
    
    Args:
        lookback_days: 回看天数
        
    Returns:
        上次DIFF最高点时的收盘价，如果未找到返回0
    """
    try:
        df = EXTRA_CONTEXT.get("DF", None)
        if df is not None:
            return get_last_diff_high_price(df, lookback_days)
        return 0.0
    except Exception:
        return 0.0


def GET_LAST_DIFF_HIGH_VALUE(lookback_days=60):
    """
    获取上次DIFF最高点的数值
    
    Args:
        lookback_days: 回看天数
        
    Returns:
        上次DIFF最高点的数值，如果未找到返回0
    """
    try:
        df = EXTRA_CONTEXT.get("DF", None)
        if df is not None:
            return get_last_diff_high_value(df, lookback_days)
        return 0.0
    except Exception:
        return 0.0


def _calc_limit_up_pct(ts_code: Optional[str], df=None) -> float:
    """
    按板块推断涨停幅度：
    - 创业板/科创板：20%（300/301/688/689）
    - 北交所：30%（后缀 .BJ）
    - ST 股：5%（名称包含 ST / *ST，优先级最高）
    - 其余主板：10%
    """
    try:
        ts = (ts_code or "").strip()
        ts_upper = ts.upper()
        core = ts_upper.split(".")[0] if ts_upper else ""
        suffix = ""
        if "." in ts_upper:
            _, _, suffix = ts_upper.partition(".")
            suffix = suffix.lower()

        pct = 0.095
        if suffix == "bj":
            pct = 0.295
        elif core.startswith(("300", "301", "688", "689")):
            pct = 0.195
        else:
            pct = 0.095

        try:
            if df is not None:
                st_pattern = re.compile(r'^\s*\*?ST', flags=re.IGNORECASE)
                for col in ("name", "ts_name", "ts_fullname", "stock_name"):
                    if col in df.columns:
                        ser = df[col]
                        if ser.dropna().astype(str).str.contains(st_pattern).any():
                            pct = 0.05
                            break
        except Exception:
            pass

        return float(pct)
    except Exception:
        return 0.10


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
    "FIND_LAST_LOWEST_J": FIND_LAST_LOWEST_J,
    "GET_LAST_DIFF_HIGH_PRICE": GET_LAST_DIFF_HIGH_PRICE,
    "GET_LAST_DIFF_HIGH_VALUE": GET_LAST_DIFF_HIGH_VALUE,
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
    "TOR": "df['tor']",
    "tor": "df['tor']",
    "Z_SLOPE": "df['z_slope']",
    "BBI": "df['bbi']",
    "BUPIAO_SHORT": "df['bupiao_short']",
    "BUPIAO_LONG": "df['bupiao_long']",
    "DUOKONG_SHORT": "df['duokong_short']",
    "duokong_short": "df['duokong_short']",
    "DUOKONG_LONG": "df['duokong_long']",
    "duokong_long": "df['duokong_long']",
    "ZHANG": "ZHANG",
    "zhang": "ZHANG",
    
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
    "ATAN": "ATAN",
    "FIND_LAST_LOWEST_J": "FIND_LAST_LOWEST_J",
    "GET_LAST_CONDITION_PRICE": "GET_LAST_CONDITION_PRICE",
    "GET_LAST_DIFF_HIGH_PRICE": "GET_LAST_DIFF_HIGH_PRICE",
    "GET_LAST_DIFF_HIGH_VALUE": "GET_LAST_DIFF_HIGH_VALUE",
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

from log_system import get_logger
LOG = get_logger("tdx_compat")

# 用于缓存已记录的列别名信息，避免重复日志
_logged_aliases = set()

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
    # 只在列别名信息发生变化时记录日志，避免重复日志
    if len(ctx) > 0:
        alias_key = tuple(sorted(ctx.keys()))
        if alias_key not in _logged_aliases:
            _logged_aliases.add(alias_key)
            if LOG.logger.isEnabledFor(logging.INFO):
                LOG.info(f"[TDX] 创建了 {len(ctx)} 个列别名 (示例: {list(ctx)[:5]})")
    return ctx


def evaluate_bool(script: str, df, prefer_keys=("sig", "last_expr", "SIG", "LAST_EXPR"), skip_var_replacement=False):
    """
    便捷入口：直接返回一个布尔 Series（与 df.index 对齐）。
    自动将 df 中现有列注入上下文（原名 + 全大写）。
    使用LRU缓存优化重复表达式编译。
    
    Args:
        skip_var_replacement: 如果为True，跳过变量替换步骤（用于处理字符串字面量中的表达式）
    """
    extra_ctx = _extra_ctx_from_df_columns(df)
    if EXTRA_CONTEXT:
        extra_ctx.update(EXTRA_CONTEXT)
    
    res = evaluate(script, df, extra_context=extra_ctx)  # evaluate 会把 extra_context 合入 ctx 使用
    return _extract_bool_signal(res, df.index, prefer_keys=prefer_keys)


def _replace_variables(expr):
    # 保护字符串字面量，避免在字符串内进行替换
    strings = {}
    placeholders = []
    counter = 0

    def replace_string(match):
        nonlocal counter
        placeholder = f"__STRING_{counter}__"
        strings[placeholder] = match.group(0)
        placeholders.append(placeholder)
        counter += 1
        return placeholder

    # 匹配 '...' 或 "..." 形式的字符串，支持转义引号
    pattern = r"""('[^'\\]*(?:\\.[^'\\]*)*')|("[^"\\]*(?:\\.[^"\\]*)*")"""
    expr_protected = re.sub(pattern, replace_string, expr)

    # 构造忽略大小写的变量替换
    key_map = {k.upper(): v for k, v in VAR_MAP.items()}
    if not key_map:
        return expr
    joined = "|".join(sorted(map(re.escape, key_map.keys()), key=len, reverse=True))
    regex = re.compile(rf'(?<![A-Za-z0-9_])({joined})(?![A-Za-z0-9_])', flags=re.IGNORECASE)

    def repl(match):
        return key_map.get(match.group(1).upper(), match.group(0))

    expr_protected = regex.sub(repl, expr_protected)

    # 恢复字符串字面量
    for placeholder in placeholders:
        expr_protected = expr_protected.replace(placeholder, strings[placeholder])

    return expr_protected


def _replace_functions(expr):
    func_map = {k.upper(): v for k, v in FUNC_MAP.items()}
    if not func_map:
        return expr
    joined = "|".join(sorted(map(re.escape, func_map.keys()), key=len, reverse=True))
    regex = re.compile(rf'(?<![A-Za-z0-9_])({joined})\s*\(', flags=re.IGNORECASE)

    def repl(match):
        return func_map.get(match.group(1).upper(), match.group(0)) + "("

    return regex.sub(repl, expr)


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


def _normalize_script_for_cache(script: str) -> str:
    """
    标准化脚本用于缓存，确保相同逻辑的脚本能命中相同的缓存
    """
    if not script:
        return ""
    
    # 移除 \r
    script = script.replace('\r', '')
    
    # 替换中文标点
    script = script.translate(str.maketrans({'；': ';', '：': ':', '（': '(', '）': ')', '，': ','}))
    
    # 移除注释行
    lines = [ln for ln in script.split('\n') if not COMMENT_RE.match(ln)]
    script = '\n'.join(lines)
    
    # 移除多余的空格，但保留单词之间的分隔
    script = re.sub(r'\s+', ' ', script).strip()
    
    # 先处理逻辑运算符（按长度从长到短）
    logical_ops = ['AND', 'OR', 'NOT']
    logical_ops.sort(key=len, reverse=True)
    for op in logical_ops:
        script = re.sub(rf'\b{re.escape(op.upper())}\b', f' {op.upper()} ', script)
        script = re.sub(rf'\b{re.escape(op.lower())}\b', f' {op.upper()} ', script)
    
    # 在运算符前后添加空格（按长度从长到短处理，避免误匹配）
    # 这样可以让相同逻辑的表达式标准化为相同格式
    
    # 分两步处理：
    # 1. 先用占位符标记组合运算符
    combined_ops = ['<=', '>=', '==', '!=', ':=', '&&', '||']
    placeholders = {}
    for idx, op in enumerate(combined_ops):
        placeholder = f"__OP{idx}__"
        script = script.replace(op, f' {placeholder} ')
        placeholders[placeholder] = op
    
    # 2. 处理单个字符的运算符（使用正则表达式确保只匹配独立的字符）
    single_ops = ['>', '<', '&', '|', '~', '=', '+', '-', '*', '/']
    for op in single_ops:
        # 使用正则表达式确保只匹配非字母数字的运算符
        script = re.sub(rf'(?<![A-Za-z0-9_]){re.escape(op)}(?![A-Za-z0-9_])', f' {op} ', script)
    
    # 3. 恢复组合运算符（加上空格）
    for placeholder, op in placeholders.items():
        script = script.replace(placeholder, f' {op} ')
    
    # 再次标准化空格
    script = re.sub(r'\s+', ' ', script).strip()
    
    return script


# LRU缓存：表达式文本 -> 编译后的程序
@lru_cache(maxsize=1000)
def _compile_script_cached(script: str) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
    """
    编译表达式脚本，返回可序列化的元组格式用于缓存
    返回: ((name, py_expr), ...) 的元组
    """
    global _EXPR_CACHE_STATS
    _EXPR_CACHE_STATS["misses"] += 1
    
    # 记录编译日志（缓存未命中时）
    if LOG.logger.isEnabledFor(logging.DEBUG):
        # 简化表达式用于日志显示（避免过长）
        expr_preview = script[:50] + "..." if len(script) > 50 else script
        LOG.debug(f"[表达式编译] 编译表达式: {expr_preview}")
    
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
    
    # 更新缓存统计
    _EXPR_CACHE_STATS["size"] = _compile_script_cached.cache_info().currsize
    
    return tuple(program)


def compile_script(script):
    """编译表达式脚本，使用LRU缓存优化重复编译"""
    global _EXPR_CACHE_STATS
    
    # 先标准化脚本，确保相同逻辑的表达式能命中缓存
    normalized_script = _normalize_script_for_cache(script)
    
    # 获取编译前的缓存信息
    cache_info_before = _compile_script_cached.cache_info()
    hits_before = cache_info_before.hits
    misses_before = _EXPR_CACHE_STATS["misses"]
    
    # 使用缓存版本
    cached_program = _compile_script_cached(normalized_script)
    
    # 检查缓存命中
    cache_info = _compile_script_cached.cache_info()
    is_cache_hit = cache_info.hits > hits_before
    
    if is_cache_hit:
        # 统计缓存命中（使用LRU缓存的累计命中数，保持一致性）
        _EXPR_CACHE_STATS["hits"] = cache_info.hits
        # 记录缓存命中日志
        expr_preview = normalized_script[:50] + "..." if len(normalized_script) > 50 else normalized_script
        LOG.debug(f"[表达式缓存] 命中缓存: {expr_preview}")
    else:
        # 未命中，_compile_script_cached 内部已经增加了 misses
        # 这里保持统计同步（misses 已在 _compile_script_cached 中递增）
        pass
    
    # 转换为列表格式（保持向后兼容）
    return list(cached_program)


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
        "ATAN": ATAN,
        "FIND_LAST_LOWEST_J": FIND_LAST_LOWEST_J,
        "GET_LAST_CONDITION_PRICE": GET_LAST_CONDITION_PRICE,
        "GET_LAST_DIFF_HIGH_PRICE": GET_LAST_DIFF_HIGH_PRICE,
        "GET_LAST_DIFF_HIGH_VALUE": GET_LAST_DIFF_HIGH_VALUE,
    }
    if extra_context:
        ctx.update(extra_context) 

    # 根据当前标的推断涨停幅度，暴露 ZHANG 变量（10%/20%/30%/5%ST）
    zhang_pct = _calc_limit_up_pct(EXTRA_CONTEXT.get("TS"), df)
    ctx["ZHANG"] = zhang_pct
    ctx["zhang"] = zhang_pct
    
    # 添加 DataFrame 列的动态访问支持
    # 这样 VAR_MAP 中的 "J": "df['j']" 就能正确工作
    for col in df.columns:
        if isinstance(col, str) and col.isidentifier():
            ctx[f"df['{col}']"] = df[col]
    
    results = {}
    last_value = None
    for name, expr in program:
        val = None  # 初始化 val
        try:
            val = eval(expr, {"__builtins__": {}}, ctx)
        except TypeError as e:
            # 处理类型不匹配错误，特别是位运算符 & 和 | 的类型不匹配
            if "rand_" in str(e) or "Cannot perform" in str(e):
                # 尝试修复：将表达式中的比较操作结果强制转换为布尔类型
                try:
                    # 创建一个辅助函数来包装比较操作，确保返回布尔类型
                    def _ensure_bool_for_bitwise(val):
                        """确保值可以用于位运算（& 和 |）"""
                        if isinstance(val, pd.Series):
                            if val.dtype == bool:
                                return val
                            # 如果是数值类型，转换为布尔
                            if np.issubdtype(val.dtype, np.number):
                                return (val != 0).astype(bool)
                            # 其他类型，尝试转换为布尔
                            return val.astype(bool)
                        elif isinstance(val, np.ndarray):
                            if val.dtype == bool:
                                return val
                            return (val != 0).astype(bool)
                        elif isinstance(val, (bool, np.bool_)):
                            return val
                        else:
                            # 标量值，转换为布尔
                            return bool(val)
                    
                    # 在上下文中添加辅助函数
                    ctx_with_helper = ctx.copy()
                    ctx_with_helper['_ensure_bool'] = _ensure_bool_for_bitwise
                    
                    # 修改表达式，在所有比较操作后添加 _ensure_bool 包装
                    # 这个修复比较复杂，先尝试捕获并给出更友好的错误信息
                    raise RuntimeError(
                        f"Error evaluating expression: {expr}\n"
                        f"Type mismatch in bitwise operation (& or |). "
                        f"This usually happens when comparing non-boolean values. "
                        f"Please ensure all comparison operations return boolean values.\n"
                        f"Original error: {e}"
                    )
                except RuntimeError:
                    raise
            else:
                # 如果不是我们处理的 TypeError，重新抛出
                raise
        except Exception as e:
            raise RuntimeError(f"Error evaluating expression: {expr}\n{e}")
        
        # 如果 val 没有被赋值（因为异常），跳过后续处理
        if val is None:
            continue
            
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


def get_expr_cache_stats() -> Dict[str, Any]:
    """获取表达式编译缓存统计信息"""
    global _EXPR_CACHE_STATS
    cache_info = _compile_script_cached.cache_info()
    
    total_requests = cache_info.hits + cache_info.misses
    hit_rate = cache_info.hits / total_requests if total_requests > 0 else 0.0
    
    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "hit_rate": hit_rate,
        "cache_size": cache_info.currsize,
        "max_size": cache_info.maxsize
    }


def clear_expr_cache():
    """清理表达式编译缓存"""
    global _EXPR_CACHE_STATS
    _compile_script_cached.cache_clear()
    _EXPR_CACHE_STATS = {"hits": 0, "misses": 0, "size": 0}


def enable_expr_cache(enable: bool = True):
    """启用或禁用表达式缓存"""
    # 表达式缓存已通过 @lru_cache 装饰器实现
    # 此函数主要用于显示状态信息
    stats = get_expr_cache_stats()
    max_size = _compile_script_cached.cache_info().maxsize
    return {
        "enabled": True,
        "max_size": max_size,
        "current_size": stats["cache_size"],
        "hits": stats["hits"],
        "misses": stats["misses"],
        "hit_rate": stats["hit_rate"]
    }
