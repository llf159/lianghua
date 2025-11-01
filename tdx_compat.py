import re
import math
import logging
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Dict, Tuple, Any

EPS = 1e-12
EXTRA_CONTEXT: dict = {}   # 运行时可注入自定义函数/变量，比如 TS/REF_DATE/RANK_*

# 表达式编译缓存统计
_EXPR_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "size": 0
}

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
            series = pd.Series(s, index=index, dtype=object)
            # 使用条件表达式处理 NaN 以避免 downcasting 警告
            series = series.where(series.notna(), False)
            return series.infer_objects(copy=False).astype(bool)

    return pd.Series(False, index=index)

# 统一把条件转成布尔并把 NaN 当 False
def _as_bool(cond):
    s = pd.Series(cond, dtype=object)
    # NaN 一律按 False 处理，避免"开头数据不全"把条件误判为 True
    # 使用 where 而不是 replace/fillna 以避免 downcasting 警告
    s = s.where(s.notna(), False)
    return s.infer_objects(copy=False).astype(bool)

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
    idx = pd.Series(np.where(cond, np.arange(len(cond)), np.nan), index=cond.index).ffill().infer_objects(copy=False)
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
    def pct(arr):
        last = arr[-1]
        return float((arr <= last).sum()) / len(arr)
    return s.rolling(int(n), min_periods=1).apply(lambda a: pct(a.values), raw=False)

def TS_RANK(x, n):
    """返回每日对应“该日值在最近 n 天窗口内的名次（1..n）”"""
    s = pd.Series(x)
    def rk(arr):
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


def reverse_price_to_diff_value(df, target_diff_value, method="optimize"):
    """
    反推到指定DIFF值时的价格（改进版：先反推价格，再重算DIFF验证）
    
    Args:
        df: 历史数据
        target_diff_value: 目标DIFF值
        method: 求解方法
        
    Returns:
        反推的收盘价，如果失败返回当前收盘价
    """
    try:
        from predict_core import PriceSolver, PriceBounds
        
        # 设置价格约束
        last_close = float(df['close'].iloc[-1])
        price_bounds = PriceBounds(
            open_min=last_close * 0.5,
            open_max=last_close * 2.0,
            high_min=last_close * 0.5,
            high_max=last_close * 2.0,
            low_min=last_close * 0.5,
            low_max=last_close * 2.0,
            close_min=last_close * 0.5,
            close_max=last_close * 2.0
        )
        
        # 创建求解器
        solver = PriceSolver(
            max_iterations=1000,
            tolerance=1e-6,
            verbose=False
        )
        
        # 求解价格
        result = solver.solve_price(
            condition="diff",  # DIFF指标
            target_value=target_diff_value,
            historical_data=df,
            price_bounds=price_bounds,
            method=method
        )
        
        if result.success:
            # 改进：用反推的价格重新计算DIFF指标进行验证
            reverse_price = result.prices['close']
            verified_price = _verify_and_adjust_diff_price(df, reverse_price, target_diff_value)
            return verified_price
        else:
            return last_close
            
    except Exception as e:
        print(f"反推价格失败: {e}")
        return float(df['close'].iloc[-1])


def _verify_and_adjust_diff_price(df, initial_price, target_diff_value, max_iterations=10, tolerance=1e-4):
    """
    验证并调整反推价格，确保DIFF值准确
    
    Args:
        df: 历史数据
        initial_price: 初始反推价格
        target_diff_value: 目标DIFF值
        max_iterations: 最大调整次数
        tolerance: 容差
        
    Returns:
        调整后的价格
    """
    try:
        current_price = initial_price
        
        for i in range(max_iterations):
            # 构造包含新价格的完整数据
            test_data = df.copy()
            test_data = test_data.iloc[:-1]  # 移除最后一行
            
            # 添加新的价格行
            new_row = test_data.iloc[-1].copy()
            new_row['close'] = current_price
            new_row['open'] = current_price * 0.99  # 简单设置开盘价
            new_row['high'] = current_price * 1.01  # 简单设置最高价
            new_row['low'] = current_price * 0.99   # 简单设置最低价
            
            test_data = pd.concat([test_data, pd.DataFrame([new_row])], ignore_index=True)
            
            # 重新计算DIFF指标
            test_data = _calculate_macd_diff_for_verification(test_data)
            
            # 获取最新的DIFF值
            actual_diff = test_data['diff'].iloc[-1]
            
            # 检查是否达到目标
            diff_error = abs(actual_diff - target_diff_value)
            if diff_error <= tolerance:
                return current_price
            
            # 调整价格（简单的线性调整）
            if abs(target_diff_value) > 1e-10:  # 避免除零
                if actual_diff < target_diff_value:
                    # DIFF值太小，需要提高价格
                    current_price *= (1 + diff_error / abs(target_diff_value) * 0.1)
                else:
                    # DIFF值太大，需要降低价格
                    current_price *= (1 - diff_error / abs(target_diff_value) * 0.1)
            else:
                # 目标DIFF值接近0，使用固定调整
                if actual_diff < target_diff_value:
                    current_price *= 1.01
                else:
                    current_price *= 0.99
            
            # 确保价格在合理范围内
            last_close = float(df['close'].iloc[-1])
            current_price = max(last_close * 0.5, min(last_close * 2.0, current_price))
        
        return current_price
        
    except Exception as e:
        print(f"价格验证调整失败: {e}")
        return initial_price


def _calculate_macd_diff_for_verification(df):
    """
    为验证目的计算MACD和DIFF指标
    """
    close = df['close']
    
    # 计算EMA
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    
    # 计算DIFF
    diff = ema12 - ema26
    
    # 计算DEA
    dea = diff.ewm(span=9).mean()
    
    # 计算MACD
    macd = (diff - dea) * 2
    
    df['diff'] = diff
    df['dea'] = dea
    df['macd'] = macd
    
    return df


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


def REVERSE_PRICE_TO_DIFF(target_diff_value, method="optimize"):
    """
    反推到指定DIFF值时的价格
    
    Args:
        target_diff_value: 目标DIFF值
        method: 求解方法
        
    Returns:
        反推的收盘价，如果失败返回当前收盘价
    """
    try:
        df = EXTRA_CONTEXT.get("DF", None)
        if df is not None:
            return reverse_price_to_diff_value(df, target_diff_value, method)
        return float(df['close'].iloc[-1]) if df is not None and not df.empty else 0.0
    except Exception:
        return 0.0


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
    "REVERSE_PRICE_TO_DIFF": REVERSE_PRICE_TO_DIFF,
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
    "DUOKONG_SHORT": "df['duokong_short']",
    "duokong_short": "df['duokong_short']",
    "DUOKONG_LONG": "df['duokong_long']",
    "duokong_long": "df['duokong_long']",
    
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
    "REVERSE_PRICE_TO_DIFF": "REVERSE_PRICE_TO_DIFF",
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
    keys = sorted(VAR_MAP.keys(), key=len, reverse=True)
    
    # 保护字符串字面量，避免在字符串内进行替换
    # 使用占位符临时替换字符串内容
    import re
    strings = {}
    placeholders = []
    counter = 0
    
    # 找到所有引号内的内容（支持单引号和双引号，处理转义字符）
    def replace_string(match):
        nonlocal counter
        placeholder = f"__STRING_{counter}__"
        strings[placeholder] = match.group(0)
        placeholders.append(placeholder)
        counter += 1
        return placeholder
    
    # 使用更健壮的正则表达式匹配字符串字面量
    # 匹配 '...' 或 "..." 形式的字符串，支持转义引号
    pattern = r"""('[^'\\]*(?:\\.[^'\\]*)*')|("[^"\\]*(?:\\.[^"\\]*)*")"""
    expr_protected = re.sub(pattern, replace_string, expr)
    
    # 在非字符串部分进行变量替换
    for k in keys:
        replacement = VAR_MAP[k]
        if replacement in expr_protected:
            continue  # 如果已经替换过，跳过
        expr_protected = re.sub(rf'(?<![A-Za-z0-9_]){re.escape(k)}(?![A-Za-z0-9_])', replacement, expr_protected)
    
    # 恢复字符串字面量
    for placeholder in placeholders:
        expr_protected = expr_protected.replace(placeholder, strings[placeholder])
    
    return expr_protected


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
        "REVERSE_PRICE_TO_DIFF": REVERSE_PRICE_TO_DIFF,
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
