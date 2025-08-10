"""
批量回测整个 DATA_DIR
python main.py

仅回测指定股票
python main.py --code 000001
"""
def mute_duckdb_progress():
    import duckdb
    stmts = [
        "PRAGMA disable_progress_bar",      # 新写法
        "PRAGMA disable_print_progress_bar",# 旧别名
        "SET enable_progress_bar = false",  # SET 形式
    ]
    for s in stmts:
        try:
            duckdb.sql(s)
        except Exception:
            pass

import os
import sys
from pathlib import Path
os.environ["TQDM_DISABLE"] = "0"
os.environ.setdefault("TQDM_DISABLE", "0")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
mute_duckdb_progress()
ROOT = Path(__file__).resolve().parent  # 当前脚本所在的根目录
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parquet_viewer as pv
import argparse
import glob
from datetime import datetime, date
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import ensure_datetime_index
import pandas as pd
from tabulate import tabulate
from backtest_core import backtest, buy_signal, sell_signal 
from config import *
import re
from contextlib import redirect_stdout, redirect_stderr
import multiprocessing
# from rich.console import Console
# RICH_CONSOLE = Console(force_terminal=True, stderr=False, highlight=False)
import colorama
colorama.just_fix_windows_console()
from functools import partial
import time
from tqdm.rich import tqdm
_TQDM_BAR = None
PROGRESS_LOCK = multiprocessing.RLock()
GLOBAL_LOCK = PROGRESS_LOCK
CPU_COUNT = os.cpu_count() or 1
N_WORKERS = max(1, CPU_COUNT - 1)  # 保留一个核心用于系统任务

START_TS  = pd.to_datetime(START_DATE)
END_TS    = pd.to_datetime(END_DATE)
START_STR = START_TS.strftime("%Y%m%d")   # 供 Parquet 分区读取
END_STR   = END_TS.strftime("%Y%m%d")

def _pbar_update(done: int, total: int, start_ts: float, desc: str = "回测进度") -> None:
    """主进程用 tqdm 展示进度；子进程已禁用"""
    global _TQDM_BAR
    # 禁用条件：显式禁用或非交互终端
    disable = (os.environ.get("TQDM_DISABLE", "0") == "1") or (not sys.stderr.isatty())
    if _TQDM_BAR is None and not disable:
        _TQDM_BAR = tqdm(
            total=total,
            desc=desc,
            dynamic_ncols=True,
            unit="项",
            smoothing=0.1,
            disable=disable,
        )
    if _TQDM_BAR is not None:
        # 将进度设置为 done（而不是每次 +1，防止越界）
        delta = max(0, done - _TQDM_BAR.n)
        if delta:
            _TQDM_BAR.update(delta)
            
def _pbar_finish() -> None:
    global _TQDM_BAR
    if _TQDM_BAR is not None:
        _TQDM_BAR.close()
        _TQDM_BAR = None
    if sys.stderr.isatty():
        sys.stderr.write("\n")
        sys.stderr.flush()

def _child_init(lock=None):
    os.environ["TQDM_DISABLE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # 提前静默标准流（尽量早）
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    # 统一彻底禁用 DuckDB 进度条（多条指令，兼容不同版本）
    try:
        import duckdb
        for s in (
            "PRAGMA disable_progress_bar",
            "PRAGMA disable_print_progress_bar",
            "SET enable_progress_bar = false",
        ):
            try:
                duckdb.sql(s)
            except Exception:
                pass
    except Exception:
        pass

def as_yyyymmdd(x) -> str:
    """把任意日期输入统一成 'YYYYMMDD' 字符串，供 Parquet 分区查询用。"""
    if isinstance(x, (datetime, date, pd.Timestamp)):
        return pd.to_datetime(x).strftime("%Y%m%d")
    if isinstance(x, str):
        s = x.strip()
        if re.fullmatch(r"\d{8}", s):  # 例如 '20050101'
            return s
        return pd.to_datetime(s).strftime("%Y%m%d")
    return pd.to_datetime(x).strftime("%Y%m%d")

def as_timestamp(x) -> pd.Timestamp:
    """统一成 pandas Timestamp，供 DataFrame 过滤用。"""
    return pd.to_datetime(x)

def make_pbar(total, desc="回测进度"):
    return tqdm(
        total=total,
        desc=desc,
        leave=True,
        position=0,
        refresh_per_second=2,
        ascii=True, 
        disable=not sys.stderr.isatty(),
        file=sys.stderr,
    )

def normalize_ts(ts_input: str, asset: str = "stock") -> str:
    ts = (ts_input or "").strip()
    if asset == "stock" and len(ts) == 6 and ts.isdigit():
        if ts.startswith("8"):
            market = ".BJ"
        elif ts[0] in {"5", "6", "9"}:
            market = ".SH"
        else:
            market = ".SZ"
        ts = ts + market
    return ts.upper()

def load_df_from_parquet(ts_code: str) -> pd.DataFrame:
    base_adj = "qfq" if "qfq" in PARQUET_ADJ else ("hfq" if "hfq" in PARQUET_ADJ else "daily")
    with_ind = "indicators" in PARQUET_ADJ
    cols = None

    try:
        df = pv.read_by_symbol(PARQUET_BASE, base_adj, ts_code, with_indicators=with_ind)
    except Exception:
        df = pv.read_range(PARQUET_BASE, "stock", PARQUET_ADJ,
                           ts_code, START_STR, END_STR, columns=cols, limit=None)

    df = ensure_datetime_index(df)
    df = df[(df.index >= START_TS) & (df.index <= END_TS)]
    return df

def load_file_by_code(code: str) -> str:
    """根据股票代码查找文件路径。"""
    pattern = os.path.join(DATA_DIR, f"*{code}*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"找不到股票 {code} 的历史数据文件：{pattern}")
    # 如果有多于一个匹配，取文件名最短/最早的一个
    return sorted(matches)[0]

def run_backtest_on_file(file_path: str) -> tuple[str, dict]:
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        df = pd.read_csv(file_path, parse_dates=['trade_date'])
        df = ensure_datetime_index(df)
        start_ts = START_TS
        end_ts   = END_TS
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]

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

def run_backtest_on_ts(ts_code: str) -> tuple[str, dict]:
    with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
        df = load_df_from_parquet(ts_code)
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
        return ts_code, summary

def run_single(code: str) -> None:
    if USE_PARQUET:
        ts_code = normalize_ts(code, asset="stock")
        df = load_df_from_parquet(ts_code)
    else:
        file_path = load_file_by_code(code)
        df = pd.read_csv(file_path, parse_dates=['trade_date'])
        df = ensure_datetime_index(df)
        start_ts = START_TS
        end_ts   = END_TS
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]

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
        trades_df.columns = ["Buy Date", "Sell Date", "Hold Days", "Buy Price", "Sell Price", "Return Rate"]
        print("\n===== 交易明细 =====")
        print(tabulate(trades_df, headers='keys', tablefmt='github', showindex=False))
    else:
        print("\n没有产生任何交易信号。")

def run_batch() -> None:
    results: list[tuple[str, dict]] = []
    if USE_PARQUET:
        # 1) 列出一个最近交易日的所有标的
        latest_dates = pv.list_trade_dates(pv.asset_root(PARQUET_BASE, "stock", PARQUET_ADJ))
        if not latest_dates:
            print("没有可用分区"); sys.exit(1)
        latest = latest_dates[-1]
        # codes_df = pv.read_day(PARQUET_BASE, "stock", PARQUET_ADJ, latest, limit=None)
        # codes = sorted(codes_df["ts_code"].unique()) if not codes_df.empty else []
        
        codes_df = pv.read_range(
            PARQUET_BASE, "stock", PARQUET_ADJ,
            ts_code=None, start=latest, end=latest,
            columns=["ts_code"], limit=None
        )
        codes = sorted(codes_df["ts_code"].dropna().unique().tolist()) if not codes_df.empty else []
        if not codes:
            print("没有可用标的"); sys.exit(1)

        # res_list = []
        # for code in tqdm(codes, desc="回测进度", file=sys.stderr):
        #     res_list.append(run_backtest_on_ts(code))
        
        # 2) 并行回测（Parquet）
        start_ts = time.time()
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_child_init
        ) as ex:
            futures = [ex.submit(run_backtest_on_ts, code) for code in codes]
            done = 0
            res_list = []
            for fut in as_completed(futures):
                res_list.append(fut.result())
                done += 1
                _pbar_update(done, len(futures), start_ts, desc="回测进度")
        _pbar_finish()

        results.extend(res_list)

    else:
        # 旧 CSV 路径：扫目录并行
        files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if not files:
            print(f"[WARN] 在 {DATA_DIR} 未发现 CSV 文件，跳过。")
            return

        start_ts = time.time()
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_child_init
        ) as ex:
            futures = [ex.submit(run_backtest_on_file, f) for f in files]
            done = 0
            res_list = []
            for fut in as_completed(futures):
                res_list.append(fut.result())
                done += 1
                _pbar_update(done, len(futures), start_ts, desc="回测进度")
        _pbar_finish()

        results.extend(res_list)


    # === 汇总结果到 Excel（保持你的原有逻辑） ===
    rows = [dict(ts_code=code, **summary) for code, summary in results]
    df = pd.DataFrame(rows)

    # 确保需要的列存在且为数值
    if "信号数" in df.columns:
        df["信号数"] = pd.to_numeric(df["信号数"], errors="coerce")
    else:
        df["信号数"] = pd.NA

    if "盈亏比" in df.columns:
        df["盈亏比"] = pd.to_numeric(df["盈亏比"], errors="coerce")
    else:
        df["盈亏比"] = pd.NA

    valid_rows = df[df["信号数"].notna()].copy()
    total_signals = valid_rows["信号数"].sum(min_count=1)
    weighted_pl_ratio = (
        (valid_rows["盈亏比"] * valid_rows["信号数"]).sum(min_count=1) / total_signals
        if pd.notna(total_signals) and total_signals > 0 else pd.NA
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    xlsx_path = os.path.join(out_dir, f"summary_{timestamp}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        sheet = "by_symbol"
        # 主表从第 2 行（索引 1）开始
        df.to_excel(writer, sheet_name=sheet, index=False, startrow=1)

        ws = writer.sheets[sheet]
        # 可选：加粗格式
        bold = writer.book.add_format({"bold": True})

        # 第一行写关键指标（你可按需增减）
        ws.write(0, 0, "加权盈亏比", bold)
        ws.write(0, 1, (float(weighted_pl_ratio)
                        if pd.notna(weighted_pl_ratio) else None))
        ws.write(0, 3, "总信号数", bold)
        ws.write(0, 4, (int(total_signals)
                        if pd.notna(total_signals) else 0))
        ws.write(0, 6, f"样本标的数: {len(df)}")

def main() -> None:
    parser = argparse.ArgumentParser(description="量化回测入口脚本")
    parser.add_argument("--code", help="只回测指定股票代码，例如 000001")

    args = parser.parse_args()

    if args.code:
        run_single(args.code)
    else:
        run_batch()
             
if __name__ == "__main__":
    import multiprocessing as mp
    try:
        if sys.platform.startswith("win"):
            mp.set_start_method("spawn", force=True)

    except RuntimeError:
        pass
    mp.freeze_support()
    main()
