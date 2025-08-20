import numpy as np
import pandas as pd
from typing import Callable, Tuple, List, Dict, Optional
from indicators import *
from tdx_compat import evaluate
from config import (
    TDX_BUY_PATH,
    TDX_SELL_OPEN_PATH,
    TDX_SELL_CLOSE_PATH,
    TDX_SELL_PATH,
)

def backtest(df, *, HOLD_DAYS, BUY_MODE, SELL_MODE, MAX_HOLD_DAYS, FALLBACK_SELL_MODE, buy_signal, sell_signal, record_trades=False) -> Tuple[Dict, Optional[List[Dict]]]:

    signals = buy_signal(df)
    buy_dates = pd.to_datetime(df.index[signals])

    results, wins, losses = [], [], []
    trades: List[Dict] = []
    
    other_future_max_gains: List[float] = []
    other_future_days_to_max: List[int] = []

    holding_until = None  # avoid overlapping positions

    all_sell_signals = None
    if SELL_MODE == 'strategy':
        all_sell_signals = sell_signal(df, [])
        if all_sell_signals is None or all_sell_signals.empty:
            # 防御：没有任何卖出信号时提供全 False 的占位，避免后续空判断分支过多
            all_sell_signals = pd.DataFrame({
                "sell_by_open":  pd.Series(False, index=df.index),
                "sell_by_close": pd.Series(False, index=df.index),
            })

    for buy_date in buy_dates:
        if holding_until is not None and buy_date <= holding_until:
            continue

        idx = df.index.get_loc(buy_date)

        # === 买入逻辑 ===
        if BUY_MODE == 'open':
            if idx + 1 < len(df):
                prev_close = df.iloc[idx]['close']
                open_price = df.iloc[idx + 1]['open']
                limit_up_price = round(prev_close * 1.099, 2)
                if open_price >= limit_up_price:                          # 如果开盘价大于等于涨停价，则不买入
                    continue
                buy_price = open_price
                sell_start_idx = idx + 1
            else:
                continue

        elif BUY_MODE == 'close':
            if idx + 1 < len(df):
                buy_price = df.iloc[idx]['close']
                sell_start_idx = idx + 1
            else:
                continue

        elif BUY_MODE == 'signal_open':
            prev_close = df.iloc[idx]['close']
            open_price = df.iloc[idx]['open']
            limit_up_price = round(prev_close * 1.099, 2)
            if open_price >= limit_up_price:
                continue
            if SELL_MODE in ('open', 'close') and idx + HOLD_DAYS >= len(df):
                continue
            buy_price = df.iloc[idx]['open']
            sell_start_idx = idx + 1

        else:
            raise ValueError(f"不支持的买入模式:  {BUY_MODE}")

        # === 卖出逻辑 ===
        if SELL_MODE in ['open', 'close']:
            if SELL_MODE == 'open':
                # 开盘卖出使用 +HOLD_DAYS，确保至少跨越一个交易日
                sell_idx = sell_start_idx + HOLD_DAYS
                if sell_idx >= len(df):
                    continue
                sell_price = df.iloc[sell_idx]['open']
                real_sell_idx = sell_idx
            elif SELL_MODE == 'close':
                # 收盘卖出仍然使用 +HOLD_DAYS-1
                sell_idx = sell_start_idx + HOLD_DAYS - 1
                if sell_idx >= len(df):
                    continue
                sell_price = df.iloc[sell_idx]['close']
                real_sell_idx = sell_idx      
    
        elif SELL_MODE == 'strategy':
            remaining_len = len(df) - sell_start_idx
            actual_window = min(MAX_HOLD_DAYS, remaining_len)
            if actual_window <= 0:
                continue
            # 直接切片使用预先计算的 all_sell_signals
            sell_signals = all_sell_signals.iloc[sell_start_idx : sell_start_idx + actual_window]
            sell_window  = df.iloc[sell_start_idx : sell_start_idx + actual_window]
            sell_price = None
            for offset in range(len(sell_signals)):
                if sell_signals.iloc[offset]['sell_by_open']:
                    sell_price = sell_window.iloc[offset]['open']
                    real_sell_idx = sell_start_idx + offset
                    break
                if sell_signals.iloc[offset]['sell_by_close']:
                    sell_price = sell_window.iloc[offset]['close']
                    real_sell_idx = sell_start_idx + offset
                    break

            if sell_price is None:
                if FALLBACK_SELL_MODE == 'open':
                    fallback_idx = sell_start_idx + MAX_HOLD_DAYS
                    if fallback_idx >= len(df): 
                        continue
                    sell_price = df.iloc[fallback_idx]['open']
                    real_sell_idx = fallback_idx
                elif FALLBACK_SELL_MODE == 'close':
                    fallback_idx = sell_start_idx + MAX_HOLD_DAYS - 1
                    if fallback_idx >= len(df): 
                        continue
                    sell_price = df.iloc[fallback_idx]['close']
                    real_sell_idx = fallback_idx
                else:
                    continue

            if sell_signals.empty or sell_window.empty:
                continue
        
        elif SELL_MODE == 'other':
            # 仅做“前瞻窗口”评估，不进行真实卖出/收益统计
            remaining_len = len(df) - sell_start_idx
            actual_window = min(MAX_HOLD_DAYS, remaining_len)
            if actual_window <= 0:
                continue

            window = df.iloc[sell_start_idx : sell_start_idx + actual_window]

            # 用未来窗口中的最高价来计算“可达到的最大涨幅”
            highs = window['high'].values
            if highs.size == 0:
                continue

            max_idx_in_window = int(np.argmax(highs))      # 第一次达到峰值的偏移（从 0 开始）
            max_high = float(highs[max_idx_in_window])

            future_max_gain = (max_high - float(buy_price)) / float(buy_price)
            other_future_max_gains.append(future_max_gain)
            # “达峰用时”按交易日计，从 1 开始计数更直观
            other_future_days_to_max.append(max_idx_in_window + 1)

            # 为了避免后续买点与本窗口重叠，视为“持有到窗口末端”再释放
            real_sell_idx = sell_start_idx + actual_window - 1
            holding_until = df.index[real_sell_idx]

            # 不做真实交易统计（不写入 results/wins/losses/trades）
            continue

        else:
            raise ValueError(f"不支持的 SELL_MODE: {SELL_MODE}")
        
        holding_until = df.index[real_sell_idx]

        # === stats ===
        ret = (sell_price - buy_price) / buy_price
        results.append(ret)
        if ret > 0:
            wins.append(ret)
        elif ret < 0:
            losses.append(ret)

        if record_trades:
            trades.append({
                "买入日期": str(buy_date.date()),
                "卖出日期": str(df.index[real_sell_idx].date()),
                "持股天数": (df.index[real_sell_idx] - buy_date).days,
                "持股K线数": int(real_sell_idx - idx),
                "买入价": round(buy_price, 2),
                "卖出价": round(sell_price, 2),
                "收益率": f"{ret:.2%}"
            })

    returns = np.array(results)
    win_avg = np.mean(wins) if wins else 0
    loss_avg = abs(np.mean(losses)) if losses else 0
    win_rate = (returns > 0).mean() if len(results) else 0
    avg_return = returns.mean() if len(results) else 0
    pl_ratio = win_avg + 1 if loss_avg == 0 else win_avg / loss_avg
    
    k = 5
    n = len(results)
    adjusted_pl_ratio = pl_ratio * n / (n + k)
    
    score = adjusted_pl_ratio * win_rate

    if len(results) == 0:
        max_loss = None
    elif (returns >= 0).all():
        max_loss = 0
    else:
        max_loss = returns.min()

    summary = {
        "信号数": len(results),
        "平均收益": f"{avg_return:.2%}" if len(results) else None,
        "胜率": f"{win_rate:.2%}" if len(results) else None,
        "最大收益": f"{returns.max():.2%}" if len(results) else None,
        "最大亏损": f"{max_loss:.2%}" if max_loss is not None else None,
        "盈亏比": round(pl_ratio, 3) if len(results) else None,
        "平均盈亏比": round(score, 3) if len(results) else None
    }
    
    if SELL_MODE == 'other':
        summary["信号数"] = len(other_future_max_gains)
    # Extra summary for SELL_MODE == 'other'
    if SELL_MODE == 'other':
        if other_future_max_gains:
            avg_future_max = float(np.mean(other_future_max_gains))
            med_future_max = float(np.median(other_future_max_gains))
            avg_days_to_max = float(np.mean(other_future_days_to_max))
            summary.update({
                "未来窗口最大涨幅(均值)": f"{avg_future_max:.2%}",
                "未来窗口最大涨幅(中位数)": f"{med_future_max:.2%}",
                "达峰平均用时(天)": round(avg_days_to_max, 2),
                "评估窗口(交易日)": int(MAX_HOLD_DAYS),
            })
        else:
            summary.update({
                "未来窗口最大涨幅(均值)": None,
                "未来窗口最大涨幅(中位数)": None,
                "达峰平均用时(天)": None,
                "评估窗口(交易日)": int(MAX_HOLD_DAYS),
            })
    return (summary, trades) if record_trades else (summary, None)


def _read_tdx(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _eval_tdx_bool(script_path: str, df: pd.DataFrame) -> pd.Series:
    """读取并执行 TDX 文本规则，返回布尔 Series（与 df.index 对齐）"""
    script = _read_tdx(script_path)
    # res = evaluate(script, df)
    # sig = res.get("sig") or res.get("last_expr") or res.get("SIG") or res.get("LAST_EXPR")
    # return pd.Series(sig, index=df.index).fillna(False).astype(bool)
    from tdx_compat import evaluate_bool
    return evaluate_bool(script, df)


def tdxsell(df: pd.DataFrame) -> pd.DataFrame:
    """
    分别用两份 TDX 文本规则生成：
      - sell_by_open: 开盘卖出信号（布尔序列）
      - sell_by_close: 收盘卖出信号（布尔序列）
    两份规则**格式完全一致**，互不耦合。
    """
    # 向后兼容：如果没配置新路径，则退化为老的 TDX_SELL_PATH -> 全部当作收盘卖出
    if TDX_SELL_OPEN_PATH and TDX_SELL_CLOSE_PATH:
        open_sig  = _eval_tdx_bool(TDX_SELL_OPEN_PATH,  df)
        close_sig = _eval_tdx_bool(TDX_SELL_CLOSE_PATH, df)
    elif TDX_SELL_PATH:
        open_sig  = pd.Series(False, index=df.index)
        close_sig = _eval_tdx_bool(TDX_SELL_PATH, df)
    else:
        # 都没配：全 False
        open_sig  = pd.Series(False, index=df.index)
        close_sig = pd.Series(False, index=df.index)

    # 可选：避免同一天既开盘又收盘都为 True 的歧义（优先开盘）
    close_sig = close_sig & ~open_sig

    return pd.DataFrame({
        "sell_by_open":  open_sig,
        "sell_by_close": close_sig,
    })


def buy_signal(df: pd.DataFrame) -> pd.Series:
    """
    返回布尔 Series（索引与 df.index 对齐）。
    """
    script = _read_tdx(TDX_BUY_PATH)
    from tdx_compat import evaluate_bool
    return evaluate_bool(script, df)


def sell_signal(df: pd.DataFrame, buy_dates: list) -> pd.DataFrame:
    # 接口保持不变，buy_dates 暂不使用但必须保留
    return tdxsell(df)
