"""Unified back‑testing entry script.

Usage
-----
# 批量回测整个 DATA_DIR
python main.py

# 仅回测指定股票
python main.py --code 000001
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import ensure_datetime_index
import pandas as pd
from tqdm import tqdm

from backtest_core import backtest

# 👉 请根据自己项目实际情况实现下面两个函数 / 模块
from strategy import buy_signal, sell_signal  # type: ignore
from config import (  # type: ignore
    DATA_DIR,
    HOLD_DAYS,
    BUY_MODE,
    SELL_MODE,
    MAX_HOLD_DAYS,
    FALLBACK_SELL_MODE,
    START_DATE,
    END_DATE,
)

N_WORKERS = max(1, os.cpu_count() - 1)

def load_file_by_code(code: str) -> str:
    """根据股票代码查找文件路径。"""
    pattern = os.path.join(DATA_DIR, f"*{code}*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"找不到股票 {code} 的历史数据文件：{pattern}")
    # 如果有多于一个匹配，取文件名最短/最早的一个
    return sorted(matches)[0]


def run_backtest_on_file(file_path: str) -> tuple[str, dict]:
    """子进程回测函数。"""
    df = pd.read_csv(file_path)
    df = ensure_datetime_index(df)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]

    summary, _ = backtest(
        df,
        HOLD_DAYS=HOLD_DAYS,
        BUY_MODE=BUY_MODE,
        SELL_MODE=SELL_MODE,
        MAX_HOLD_DAYS=MAX_HOLD_DAYS,
        FALLBACK_SELL_MODE=FALLBACK_SELL_MODE,
        buy_signal=buy_signal,
        sell_signal=sell_signal,
        record_trades=False,
    )
    code = os.path.splitext(os.path.basename(file_path))[0]
    return code, summary


def run_single(code: str) -> None:
    file_path = load_file_by_code(code)
    df = pd.read_csv(file_path, parse_dates=['trade_date'])
    df = ensure_datetime_index(df)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]

    summary, trades = backtest(
        df,
        HOLD_DAYS=HOLD_DAYS,
        BUY_MODE=BUY_MODE,
        SELL_MODE=SELL_MODE,
        MAX_HOLD_DAYS=MAX_HOLD_DAYS,
        FALLBACK_SELL_MODE=FALLBACK_SELL_MODE,
        buy_signal=buy_signal,
        sell_signal=sell_signal,
        record_trades=True,
    )

    print("\n===== 概览 =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if trades:
        trades_df = pd.DataFrame(trades)
        print("\n===== 交易明细 =====")
        print(trades_df.to_string(index=False))
    else:
        print("\n没有产生任何交易信号。")


def run_batch() -> None:
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        print(f"DATA_DIR = {DATA_DIR} 下没有找到任何数据文件！", file=sys.stderr)
        sys.exit(1)

    results: list[tuple[str, dict]] = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_backtest_on_file, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="回测进度"):
            code, summary = fut.result()
            summary["代码"] = code
            results.append((code, summary))

    # 汇总结果
    results = [{**s, "代码": c} for c, s in results]
    # df = pd.DataFrame(results).set_index("代码").sort_values("平均收益", ascending=False)
    df = pd.DataFrame(results).set_index("代码").sort_index()



    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"summary_{timestamp}.csv")
    df.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n已保存汇总结果 → {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="量化回测入口脚本")
    parser.add_argument("--code", help="只回测指定股票代码，例如 000001")

    args = parser.parse_args()

    if args.code:
        run_single(args.code)
    else:
        run_batch()


if __name__ == "__main__":
    main()
