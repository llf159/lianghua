#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
用途
----
FAST_INIT 用于“首次全历史快速回补”。它按股票多线程抓取 Tushare 日线数据，并立刻产出两类数据：
1) 单股票成品(含指标、带增量 warm-up) → 便于回测直接读取
2) (可选)原始K线的按日分区 → 便于后续离线加工或校验

关键配置
--------
- TOKEN:           Tushare Token(必填)
- DATA_ROOT:       本地数据根目录
- START_DATE:      起始日期，如 "20050101"
- END_DATE:        结束日期，支持 "today"
- FAST_INIT_MODE:  True=启用 FAST_INIT；False=走日常增量
- API_ADJ:         原始接口的复权选择："raw" / "qfq" / "hfq"
- FAST_INIT_THREADS: 下载并发线程
- CALLS_PER_MIN:   限频窗口(滑动计数，避免被限流)
- WRITE_SYMBOL_INDICATORS: True 时写“单股票成品(含指标)”
- SYMBOL_PRODUCT_INDICATORS: 指标配置，"all" 或逗号分隔
- SYMBOL_PRODUCT_WARMUP_DAYS: 指标增量重算的 warm-up 天数(默认 60 天)
- DUCK_MERGE_DAY_LAG: 仅当按日分区的最大 trade_date 落后超过 LAG 天才触发合并
- DUCKDB_THREADS / DUCKDB_MEMORY_LIMIT / DUCKDB_TEMP_DIR: DuckDB 并行、内存与临时目录参数

生成目录结构
------------
(1) 临时/缓存：每只股票的原始下载文件
    {DATA_ROOT}/fast_init_symbol/{API_ADJ}/{ts_code}.parquet

(2) 单股票成品(含指标，带 warm-up 增量)
    {DATA_ROOT}/stock/by_symbol_{adj}/{ts_code}.parquet
    说明：{adj} 由 API_ADJ 决定，"daily" / "qfq" / "hfq"
          文件内字段：基础行情 + 已计算的各类指标(精度按 INDICATOR_DECIMALS 控制)

(3) 按日分区(含指标；增量 COPY)
    {DATA_ROOT}/stock/{adj}_indicators/trade_date=YYYYMMDD/part-*.parquet
    产出方式：调用 duckdb_merge_symbol_products_to_daily()，
    将 (2) 中 “by_symbol_{adj}/*.parquet” 增量 COPY 到该分区目录

(4) (可选)原始K线的按日分区(不含指标)
    {DATA_ROOT}/stock/daily[/qfq/hfq]/trade_date=YYYYMMDD/part-*.parquet
    产出方式：STREAM_MERGE(流式合并 fast_init_symbol/{raw|qfq|hfq} → 对应 daily* 目录)

执行流程
--------
A. 下载阶段(多线程、限频、重试)
   - 对股票清单逐只调用 ts.pro_bar(..., adj=API_ADJ)，写入 (1)
   - 若开启 WRITE_SYMBOL_INDICATORS：对每只股票
       · 读取“旧成品”尾部 warm-up 区间 + 新数据 → 合并后统一重算指标 → 写入 (2)

B. 合并阶段(默认开启)
   - 运行 duckdb_merge_symbol_products_to_daily()：
     从 (2) 增量 COPY 新 trade_date 到 (3) 的按日分区

C. (可选)原始K线流式合并
   - 将 (1) 的 raw/qfq/hfq 分别流式合并到 (4)

断点与幂等
----------
- FAST_INIT 下载：若 {fast_init_symbol}/{API_ADJ}/{ts_code}.parquet 已存在，
  且其最大 trade_date ≥ (END_DATE - CHECK_SKIP_ALLOW_LAG_DAYS)，则跳过
- 成品写入：单股票文件按 trade_date 去重；指标 warm-up 仅重算尾部窗口
- 合并到按日分区：使用 DuckDB 的 COPY + PARTITION_BY(trade_date) + OVERWRITE_OR_IGNORE

常见搭配
--------
1) “最快可用”方案(接口直接 qfq)
   FAST_INIT_MODE=True, API_ADJ="qfq", WRITE_SYMBOL_INDICATORS=True, ADJ_MODE="none"
   → 立即得到带指标的 by_symbol_qfq 与 {qfq}_indicators

2) “更稳健的本地复权”方案(推荐)
   FAST_INIT_MODE=True, API_ADJ="raw", WRITE_SYMBOL_INDICATORS=True, ADJ_MODE="both"
   → 先拉原始，再本地构建 qfq/hfq，并同步生成对应的 *_indicators

与回测对齐
----------
与viewer一致

使用建议：
| 使用场景            | FAST\_INIT\_MODE | API\_ADJ | ADJ\_MODE                               |
| --------------------|------------------| -------- | --------------------------------------- |
| 快速获取接口前复权   | True             | `"qfq"`  | `"none"`                                |
| 本地构建复权(推荐) | True             | `"raw"`  | `"both"`                                |
| 日常增量更新         | False            | 无用     | `"qfq"` / `"hfq"` / `"both"` / `"none"` |


注意：
  - 请先填写 TOKEN
  - 若使用 Windows 路径请确保编码 UTF-8
"""

from __future__ import annotations
import os
import sys
import time
import json
import glob
from pathlib import Path
import random
import logging
import datetime as dt
from typing import List, Optional, Callable, Dict, Tuple
from logging.handlers import TimedRotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="tushare.pro.data_pro",
    message=".*fillna.*method.*deprecated.*"
)

from config import *
from tdx_compat import evaluate as tdx_eval
import indicators as ind
from utils import normalize_trade_date


# ------------- 基本检查 -------------
if TOKEN.startswith("在这里"):
    sys.stderr.write("请先在 TOKEN 中填写你的 Tushare Token.\n")
    sys.exit(1)

try:
    import tushare as ts
except ImportError:
    sys.stderr.write("未安装 tushare: pip install tushare\n")
    sys.exit(1)

ts.set_token(TOKEN)
pro = ts.pro_api()

os.makedirs(DATA_ROOT, exist_ok=True)

# ========== 限频与重试 ==========
_CALL_TS: List[float] = []
_CALL_TS_LOCK = threading.Lock()


# === 全局日志策略 ======================================================
# 1.  INFO 及以上 → fast_init.log(每天轮换，保留 7 份)
# 2.  WARNING 及以上 → 终端 stdout(与 tqdm 共存)
# =======================================================================

root = logging.getLogger()
root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# 文件 Handler(轮换)
file_hdl = TimedRotatingFileHandler(
    "fast_init.log",
    when="midnight",
    backupCount=7,
    encoding="utf-8"
)
file_fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
file_hdl.setFormatter(file_fmt)
root.addHandler(file_hdl)

# 终端 Handler
console_hdl = logging.StreamHandler(sys.stdout)
console_hdl.setLevel(logging.INFO)
console_hdl.setFormatter(file_fmt)
root.addHandler(console_hdl)
log_dir = os.path.join("./log", "log")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "download.log")
dl_hdl = logging.FileHandler(log_path, encoding="utf-8")
dl_hdl.setLevel(logging.INFO)
dl_hdl.setFormatter(file_fmt)
root.addHandler(dl_hdl)

def _parse_indicators(arg: str):
    if not arg:
        return []
    if arg.lower().strip() == "all":
        return list(ind.REGISTRY.keys())
    return [x.strip() for x in arg.split(",") if x.strip()]

def _decide_symbol_adj_for_fast_init() -> str:
    aj = (API_ADJ).lower()
    return aj 

def _maybe_compact(dirpath: str):
    mode = str(DUCKDB_ENABLE_COMPACT_AFTER).lower()
    if mode in ("false", "0", "off", "none"):
        return
    # "if_needed" 的判定：仅当某些日期的 part 数超过阈值才做
    if mode in ("if_needed", "auto"):
        # 粗略检查：命中任何一个 trade_date 目录超过阈值就触发
        over = False
        for d in os.listdir(dirpath):
            p = os.path.join(dirpath, d)
            if d.startswith("trade_date=") and os.path.isdir(p):
                cnt = sum(1 for x in os.listdir(p) if x.endswith(".parquet"))
                if cnt > COMPACT_MAX_FILES_PER_DATE:
                    over = True
                    break
        if not over:
            return
    # 其余视作强制压实
    compact_daily_partitions(base_dir=dirpath)

def _update_fast_init_cache(ts_code: str, df: pd.DataFrame, adj: str):
    """
    将 df 合并进 fast_init_symbol/<adj>/<ts_code>.parquet，按 trade_date 去重。
    adj: 'daily' | 'qfq' | 'hfq' 的语义对应目录 raw/qfq/hfq
    """
    # adj -> 子目录名
    sub = {"daily":"raw", "qfq":"qfq", "hfq":"hfq"}.get(adj, "raw")
    symbol_dir = os.path.join(FAST_INIT_STOCK_DIR, sub)
    os.makedirs(symbol_dir, exist_ok=True)
    fpath = os.path.join(symbol_dir, f"{ts_code}.parquet")

    df2 = df.copy()
    df2 = normalize_trade_date(df2)
    df2 = df2.sort_values("trade_date").drop_duplicates("trade_date", keep="last")

    if os.path.exists(fpath):
        try:
            old = pd.read_parquet(fpath)
            old = normalize_trade_date(old)

            old = normalize_trade_date(old)
            both = pd.concat([old, df2], ignore_index=True)
            both = both.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
            both.to_parquet(fpath, index=False)
            return
        except Exception as e:
            logging.warning("[FAST_CACHE] 读取旧缓存失败 %s -> 覆盖写新: %s", ts_code, e)
    df2.to_parquet(fpath, index=False)

def _WRITE_SYMBOL_INDICATORS(ts_code: str, df: pd.DataFrame, end_date: str):
    """
    把该 ts_code 的 DataFrame 计算指标后写入 by_symbol 成品(带 warm-up 增量)。
    要求 df 至少包含: trade_date, open, high, low, close, vol[, amount, pre_close]
    """
    if not (WRITE_SYMBOL_PLAIN or WRITE_SYMBOL_INDICATORS):
        return
        # 1) 选择输出目录（按你要求的命名）
    adj = _decide_symbol_adj_for_fast_init()  # 'raw' | 'qfq' | 'hfq'
    base_dir = DATA_ROOT

    single_plain_dir = os.path.join(base_dir, "stock", "single", f"single_{adj}")
    single_ind_dir   = os.path.join(base_dir, "stock", "single", f"single_{adj}_indicators")
    single_plain_dir_csv = os.path.join(base_dir, "stock", "single", "csv", adj)
    single_ind_dir_csv   = os.path.join(base_dir, "stock", "single", "csv", f"{adj}_indicators")
    if WRITE_SYMBOL_PLAIN:
        os.makedirs(single_plain_dir, exist_ok=True)
        os.makedirs(single_plain_dir_csv, exist_ok=True)
    if WRITE_SYMBOL_INDICATORS:
        os.makedirs(single_ind_dir, exist_ok=True)
        os.makedirs(single_ind_dir_csv, exist_ok=True)

    plain_parquet = os.path.join(single_plain_dir, f"{ts_code}.parquet")
    plain_csv     = os.path.join(single_plain_dir_csv, f"{ts_code}.csv")
    ind_out_path_parquet = os.path.join(single_ind_dir, f"{ts_code}.parquet")
    ind_out_path_csv     = os.path.join(single_ind_dir_csv, f"{ts_code}.csv")

    
    # 2) 规范、排序、去重
    df2 = df.copy()
    if "trade_date" not in df2.columns:
        raise ValueError("df 缺少 trade_date 列")
    
    df2 = df2.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    df2 = normalize_trade_date(df2)
    price_cols = ["open", "high", "low", "close", "pre_close", "change"]

    # ---------- plain：不带指标 ----------
    if WRITE_SYMBOL_PLAIN:
        df_plain = df2.copy()
        for col in price_cols:
            if col in df_plain.columns and pd.api.types.is_numeric_dtype(df_plain[col]):
                df_plain[col] = df_plain[col].round(2)
        df_plain = df_plain.sort_values("trade_date").drop_duplicates("trade_date", keep="last")

        plain_parquet = os.path.join(single_plain_dir, f"{ts_code}.parquet")
        plain_csv     = os.path.join(single_plain_dir_csv, f"{ts_code}.csv")

        if "parquet" in SYMBOL_PRODUCT_FORMATS.get("plain", []):
            df_plain.to_parquet(plain_parquet, index=False, engine=PARQUET_ENGINE)
        if "csv" in SYMBOL_PRODUCT_FORMATS.get("plain", []):
            df_plain.to_csv(plain_csv, index=False, encoding="utf-8-sig")

    # ---------- ind：带指标 ----------
    if WRITE_SYMBOL_INDICATORS:
        # warm-up 增量（沿用原逻辑）
        ind_out_path_parquet = os.path.join(single_ind_dir, f"{ts_code}.parquet")
        ind_out_path_csv     = os.path.join(single_ind_dir_csv, f"{ts_code}.csv")

        warm_df = df2.copy()
        warmup_start = None
        if os.path.exists(ind_out_path_parquet) or os.path.exists(ind_out_path_csv):
            try:
                # 优先读 parquet；若不存在再读 csv
                if os.path.exists(ind_out_path_parquet):
                    old = pd.read_parquet(ind_out_path_parquet)
                else:
                    old = pd.read_csv(ind_out_path_csv, dtype=str)
                if not old.empty:
                    old_td = pd.to_datetime(old["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
                    last = old_td.max()
                    warmup_start = (last - pd.Timedelta(days=SYMBOL_PRODUCT_WARMUP_DAYS))

                    keep_old = old.loc[old_td < warmup_start].copy()
                    keep_old = normalize_trade_date(keep_old)

                    new_part = df2.loc[pd.to_datetime(df2["trade_date"].astype(str), format="%Y%m%d", errors="coerce") >= warmup_start].copy()
                    new_part = normalize_trade_date(new_part)
                    warm_df = pd.concat([keep_old, new_part], ignore_index=True)
            except Exception as e:
                logging.warning("[PRODUCT][%s] 读取旧(带指标)失败，按全量重算：%s", ts_code, e)

        # 统一指标引擎计算 + 四舍五入
        try:
            names = (SYMBOL_PRODUCT_INDICATORS or "all")
            names = list(ind.REGISTRY.keys()) if str(names).lower() == "all" else [x.strip() for x in str(names).split(",") if x.strip()]
            warm_df = ind.compute(warm_df, names)
        except Exception as e:
            logging.exception("[PRODUCT][%s] 指标计算失败：%s", ts_code, e)
            raise

        decimals = ind.outputs_for(names)
        for col, n in decimals.items():
            if col in warm_df.columns and pd.api.types.is_numeric_dtype(warm_df[col]):
                warm_df[col] = warm_df[col].round(n)
        for col in price_cols:
            if col in warm_df.columns and pd.api.types.is_numeric_dtype(warm_df[col]):
                warm_df[col] = warm_df[col].round(2)

        warm_df = normalize_trade_date(warm_df)
        warm_df = warm_df.sort_values("trade_date").drop_duplicates("trade_date", keep="last")

        if "parquet" in SYMBOL_PRODUCT_FORMATS.get("ind", []):
            warm_df.to_parquet(ind_out_path_parquet, index=False, engine=PARQUET_ENGINE)
        if "csv" in SYMBOL_PRODUCT_FORMATS.get("ind", []):
            warm_df.to_csv(ind_out_path_csv, index=False, encoding="utf-8-sig")
        logging.debug("[PRODUCT][%s] 成品已写出 plain_dir=%s ind_dir=%s", ts_code, single_plain_dir, single_ind_dir)

def _rate_limit():
    now = time.time()
    while True:
        with _CALL_TS_LOCK:
            # 清理过期
            while _CALL_TS and now - _CALL_TS[0] > 60:
                _CALL_TS.pop(0)
            if len(_CALL_TS) < CALLS_PER_MIN:
                _CALL_TS.append(now)
                return
            sleep_for = 60 - (now - _CALL_TS[0]) + 0.01
        time.sleep(sleep_for)
        now = time.time()
        
def _retry(fn: Callable[[], pd.DataFrame], desc: str, retries: int = RETRY_TIMES) -> pd.DataFrame:
    """
    固定延迟序列重试：15s -> 10s -> 5s -> 5s ...
    失败后等待时加入轻微随机抖动，减少多线程同时再次打接口。
    :param fn: 无参调用(外部用 lambda 封装)
    :param desc: 日志标识
    :param retries: 最大尝试次数(包含第一次)
    """
    import random
    last_msg = ""
    for attempt in range(1, retries + 1):
        try:
            _rate_limit()          # 如果你后面换了 _rate_control_point()，这里对应改
            return fn()
        except Exception as e:
            last_msg = str(e)
            logging.warning("%s 失败 (%s) 尝试 %d/%d", desc, last_msg, attempt, retries)
            if attempt == retries:
                break  # 出循环抛错
            # 计算基础等待
            if attempt <= len(RETRY_DELAY_SEQUENCE):
                base_wait = RETRY_DELAY_SEQUENCE[attempt - 1]
            else:
                base_wait = RETRY_DELAY_SEQUENCE[-1]
            # 抖动
            jitter = random.uniform(RETRY_JITTER_RANGE[0], RETRY_JITTER_RANGE[1])
            wait_sec = max(0.1, base_wait + jitter)
            if RETRY_LOG_LEVEL.upper() == "INFO":
                logging.info("%s 重试前等待 %.2fs (base=%ds, jitter=%.2f)",
                             desc, wait_sec, base_wait, jitter)
            else:
                logging.debug("%s 重试前等待 %.2fs (base=%ds, jitter=%.2f)",
                              desc, wait_sec, base_wait, jitter)
            time.sleep(wait_sec)
    raise RuntimeError(f"{desc} 最终失败: {last_msg}")

def _trade_dates(start: str, end: str) -> List[str]:
    cal = _retry(
        lambda: pro.trade_cal(
            exchange="SSE",
            start_date=start,
            end_date=end,
            is_open=1,
            fields="cal_date"
        ),
        "trade_cal"
    )
    if cal.empty:
        raise RuntimeError(f"trade_cal 返回为空({start}~{end})")
    return cal["cal_date"].astype(str).tolist()

def _last_partition_date(root: str) -> Optional[str]:
    if not os.path.exists(root):
        return None
    dates = [d.split("=")[-1] for d in os.listdir(root) if d.startswith("trade_date=")]
    return max(dates) if dates else None

def _save_partition(df: pd.DataFrame, root: str):
    if df is None or df.empty:
        return
    for dt, sub in df.groupby("trade_date"):
        pdir = os.path.join(root, f"trade_date={dt}")
        os.makedirs(pdir, exist_ok=True)
        fname = os.path.join(pdir, f"part-{int(time.time()*1e6)}.parquet")
        sub.to_parquet(fname, index=False, engine=PARQUET_ENGINE)

def _tqdm_iter(seq, desc: str, unit="日"):
    return tqdm(seq,
                total=len(seq),
                desc=desc,
                ncols=110,
                unit=unit,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

def compact_daily_partitions(base_dir: str | None = None) -> None:
    """
    压实按日期分区目录：如果某日期下 parquet 文件数超过阈值，
    以 DuckDB 读取后重写(尽量生成少量新文件)。
    """
    import duckdb, glob, shutil

    daily_dir = os.path.join(DATA_ROOT, "stock", "daily")
    if base_dir is not None:
        daily_dir = base_dir
    else:
        candidates = [
            os.path.join(DATA_ROOT, "stock", "daily", "daily"),
            os.path.join(DATA_ROOT, "stock", "daily"),
        ]
        daily_dir = next((p for p in candidates if os.path.isdir(p)), candidates[0])

    if not os.path.isdir(daily_dir):
        logging.warning("[COMPACT] daily 目录不存在，跳过")
        return

    # 找出 trade_date=XXXX 目录
    date_dirs = [d for d in os.listdir(daily_dir)
                 if d.startswith("trade_date=") and os.path.isdir(os.path.join(daily_dir, d))]

    if not date_dirs:
        logging.info("[COMPACT] 无日期目录，跳过")
        return

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute("PRAGMA preserve_insertion_order=false;")
    os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")

    compacted = 0
    for d in date_dirs:
        full = os.path.join(daily_dir, d)
        parts = [p for p in os.listdir(full) if p.endswith(".parquet")]
        if len(parts) <= COMPACT_MAX_FILES_PER_DATE:
            continue

        # 计算合并批次数(近似控制输出文件数)
        # DuckDB 的 COPY 不直接指定目标 part 个数，只能一次性写 -> 得到 1 个或少量 part
        tmp_out = os.path.join(daily_dir, f"__tmp_compact_{d}")
        shutil.rmtree(tmp_out, ignore_errors=True)
        os.makedirs(tmp_out, exist_ok=True)

        pattern = os.path.join(full, "*.parquet").replace("\\", "/")
        logging.info("[COMPACT] %s 文件数=%d -> 压实", d, len(parts))
        sql = f"""
        COPY (
          SELECT * FROM read_parquet('{pattern}')
        ) TO '{tmp_out}'
        (FORMAT PARQUET);
        """
        try:
            con.execute(sql)
        except Exception as e:
            logging.error("[COMPACT] %s 压实失败: %s", d, e)
            shutil.rmtree(tmp_out, ignore_errors=True)
            continue

        # 删除旧 part，移动新 part
        for p in parts:
            try:
                os.remove(os.path.join(full, p))
            except Exception:
                pass
        # 可能 tmp_out 下生成多个文件，将其移动进原日期目录
        for newp in os.listdir(tmp_out):
            shutil.move(os.path.join(tmp_out, newp), os.path.join(full, newp))
        shutil.rmtree(tmp_out, ignore_errors=True)
        compacted += 1

    con.close()
    logging.info("[COMPACT] 完成 压实日期数=%d", compacted)

def _need_duck_merge(daily_dir: str) -> bool:
    """
    返回 True 表示需要触发 duckdb 合并
    规则：① parquet 最新 trade_date - duckdb 表最新 ≥ DUCK_MERGE_DAY_LAG
         ② 或新增行数 ≥ DUCK_MERGE_MIN_ROWS
    """
    import duckdb, glob, os, datetime as dt

    # parquet 端最新日期
    last_parquet = _last_partition_date(daily_dir)
    if last_parquet is None:
        return True          # 本地还没任何 parquet，后面流程会自动全量建

    con = duckdb.connect()    # 内存连接足够
    try:
        glob_path = os.path.join(daily_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
        last_duck = con.sql(
            f"SELECT max(trade_date) FROM parquet_scan('{glob_path}')"
        ).fetchone()[0]
    except duckdb.Error:
        last_duck = None

    if last_duck is not None:
        pattern_new = (
            f"{Path(daily_dir).as_posix()}/trade_date={int(last_duck)+1}%/part-*.parquet"
        )
        next_str = str(int(last_duck) + 1).zfill(8)
        try:                                              # ← 加入 try‑except
            new_rows = con.sql(
                f"SELECT count(*) FROM parquet_scan('{daily_dir}/trade_date={next_str}%/part-*.parquet')"
            ).fetchone()[0] or 0
        except duckdb.IOException:                        # 分区还没生成 → 说明需要合并
            return True
    else:
        new_rows = 0

    # ① 日期差 ≥ N 天
    if last_duck is None:
        return True                             # duckdb 还没建过
    lp = dt.datetime.strptime(str(last_parquet), "%Y%m%d")
    ld = dt.datetime.strptime(str(last_duck), "%Y%m%d")
    day_lag = (lp - ld).days
    if day_lag >= DUCK_MERGE_DAY_LAG:
        return True

    # ② 或新增行数 ≥ 阈值
    return new_rows >= DUCK_MERGE_MIN_ROWS

# ========== 按交易日批量模式(原有日常增量) ==========
def sync_index_daily_fast(start: str, end: str, whitelist: List[str], threads: int = 8):
    """
    按“指数代码”一次性拉取区间数据(并发)，写出到 data/index/daily/trade_date=YYYYMMDD/part-*.parquet
    比原来“按交易日 × 指数”快一个数量级以上。
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    root = os.path.join(DATA_ROOT, "index", "daily")
    os.makedirs(root, exist_ok=True)

    # 增量起始(沿用你现有的分区规则)
    last = _last_partition_date(root)
    actual_start = start if last is None else str(int(last) + 1)
    if actual_start > end:
        logging.info("[index][FAST] 日线已最新 (last=%s)", last)
        return

    lock_io = threading.Lock()
    ok = skip = err = empty = 0

    def write_partition(df: pd.DataFrame):
        """把单只指数的全量/增量数据，按 trade_date 分区写盘(带简单缓冲)。"""
        # 只保留需要的列可以减小体积(按需裁剪)
        # cols = ["ts_code","trade_date","open","high","low","close","pre_close","vol","amount"]
        # df = df[cols]
        for dt, sub in df.groupby("trade_date"):
            if dt < actual_start or dt > end:
                continue
            pdir = os.path.join(root, f"trade_date={dt}")
            os.makedirs(pdir, exist_ok=True)
            fname = os.path.join(pdir, f"part-{int(time.time()*1e6)}.parquet")
            # IO 上锁，避免极端情况下文件名撞车
            with lock_io:
                sub.to_parquet(fname, index=False, engine=PARQUET_ENGINE)

    def fetch_one(code: str):
        # 先尝试 pro_bar(更稳)，失败再退回 pro.index_daily
        def call_bar():
            return ts.pro_bar(
                ts_code=code,
                start_date=actual_start,
                end_date=end,
                freq='D',
                asset='I'   # 关键：指数
            )
        def call_daily():
            return pro.index_daily(
                ts_code=code,
                start_date=actual_start,
                end_date=end
            )

        try:
            df = _retry(lambda: call_bar(), f"index_pro_bar_{code}")
            if df is None or df.empty:
                # 兜底再试一次 index_daily
                df = _retry(lambda: call_daily(), f"index_daily_{code}")
            if df is None or df.empty:
                return code, "empty"
            df = df.sort_values("trade_date")
            write_partition(df)
            return code, "ok"
        except Exception as e:
            return code, f"err:{e}"

    with ThreadPoolExecutor(max_workers=threads) as exe:
        futs = {exe.submit(fetch_one, code): code for code in whitelist}
        pbar = tqdm(as_completed(futs), total=len(futs), desc="指数(日线)并发拉取", ncols=120)
        for fut in pbar:
            code, st = fut.result()
            if st == "ok": ok += 1
            elif st == "empty": empty += 1
            else: err += 1
            pbar.set_postfix(ok=ok, empty=empty, err=err)
        pbar.close()

    logging.info("[index][FAST] 完成 ok=%d empty=%d err=%d 区间=%s~%s 起点=%s",
                 ok, empty, err, start, end, actual_start)

def sync_stock_daily_fast(start: str, end: str, threads: int = 8):
    """
    按“股票代码(ts_code)”一次性并发拉取区间(日线)，
    写出到 data/stock/daily/trade_date=YYYYMMDD/part-*.parquet
    """
    adj = _decide_symbol_adj_for_fast_init()  # 返回 'raw' | 'qfq' | 'hfq'
    adj_dir_map = {
        "raw": os.path.join(DATA_ROOT, "stock", "daily", "daily_raw"),
        "qfq": os.path.join(DATA_ROOT, "stock", "daily", "daily_qfq"),
        "hfq": os.path.join(DATA_ROOT, "stock", "daily", "daily_hfq"),
    }
    dst_dir = adj_dir_map[adj]
    os.makedirs(dst_dir, exist_ok=True)
    

    # 增量起点沿用你现有分区规则
    last = _last_partition_date(dst_dir)
    actual_start = start if last is None else str(int(last) + 1)
    if actual_start > end:
        logging.info("[stock][FAST] 日线已最新 (last=%s)", last)
        return

    # 文件名并发写入时做轻量互斥，避免极端时间戳撞名
    lock_io = threading.Lock()
    ok = empty = err = 0

    def write_partition(df: pd.DataFrame):
        for dt, sub in df.groupby("trade_date"):
            if dt < actual_start or dt > end:
                continue
            pdir = os.path.join(dst_dir, f"trade_date={dt}")
            os.makedirs(pdir, exist_ok=True)
            fname = os.path.join(pdir, f"part-{int(time.time()*1e6)}.parquet")
            with lock_io:
                sub.to_parquet(fname, index=False, engine=PARQUET_ENGINE)

    # 取股票清单（你已有缓存的封装）
    stocks = _fetch_stock_list()   # fast_init 已用到它，带本地缓存。:contentReference[oaicite:3]{index=3}
    codes = stocks.ts_code.astype(str).tolist()
    logging.info("[stock][FAST] 准备并发拉取 股票数=%d 区间=%s~%s 起点=%s",
                 len(codes), start, end, actual_start)

    def fetch_one(ts_code: str):
        # 优先 pro_bar（区间拉取），失败再兜底 pro.daily 分天拼
        def call_bar():
            return ts.pro_bar(
                ts_code=ts_code,
                start_date=actual_start,
                end_date=end,
                adj=None,       # NORMAL 模式落 raw 到 stock/daily
                freq='D',
                asset='E'
            )
        try:
            df = _retry(lambda: call_bar(), f"stock_pro_bar_{ts_code}")
            if df is None or df.empty:
                all_df = []
                for d in _trade_dates(actual_start, end):  # 你已有交易日获取。
                    try:
                        df_d = _retry(lambda dd=d: pro.daily(trade_date=dd, ts_code=ts_code),
                                    f"stock_daily_{ts_code}_{d}")
                        if df_d is not None and not df_d.empty:
                            all_df.append(df_d)
                    except Exception:
                        continue
                df = pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()
            if df is None or df.empty:
                return ts_code, "empty"
            df = df.sort_values("trade_date")
            write_partition(df)
            return ts_code, "ok"
        except Exception as e:
            return ts_code, f"err:{e}"

    with ThreadPoolExecutor(max_workers=threads) as exe:
        futs = {exe.submit(fetch_one, code): code for code in codes}
        pbar = tqdm(as_completed(futs), total=len(futs), desc="股票(日线)并发拉取", ncols=120)
        for fut in pbar:
            code, st = fut.result()
            if st == "ok": ok += 1
            elif st == "empty": empty += 1
            else: err += 1
            pbar.set_postfix(ok=ok, empty=empty, err=err)
        pbar.close()

    logging.info("[stock][FAST] 完成 ok=%d empty=%d err=%d 区间=%s~%s 起点=%s",
                 ok, empty, err, start, end, actual_start)

# ========== FAST INIT (按股票多线程) ==========
def _fetch_stock_list() -> pd.DataFrame:
    cache = os.path.join(DATA_ROOT, "stock_list.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype=str)
    df = _retry(lambda: pro.stock_basic(exchange='', list_status='L',
                                        fields='ts_code,name,list_date'),
                "stock_basic_full")
    df.to_csv(cache, index=False, encoding='utf-8')
    return df

def fast_init_download(end_date: str):
    """
    第一阶段：多线程按股票全量下载到 fast_init_symbol 目录(一个股票一个文件)。
    下载完自动执行一次失败股票补抓(若开启 FAILED_RETRY_ONCE)。
    """
    os.makedirs(FAST_INIT_STOCK_DIR, exist_ok=True)
    stocks = _fetch_stock_list()
    codes = stocks.ts_code.tolist()
    logging.info("[FAST_INIT] 股票数=%d", len(codes))

    lock_stats = threading.Lock()
    ok = skip = empty = err = 0
    failed = []   # 初次失败列表 (含异常信息)

    def task(ts_code: str):
        # 根据复权类型切换子目录
        adj_folder = API_ADJ  
        sub_dir = os.path.join(FAST_INIT_STOCK_DIR, adj_folder)
        os.makedirs(sub_dir, exist_ok=True)
        fpath = os.path.join(sub_dir, f"{ts_code}.parquet")

        if os.path.exists(fpath) and CHECK_SKIP_MIN_MAX:
            need_redownload = False
            min_d = max_d = None
            try:
                df_meta = pd.read_parquet(fpath, columns=CHECK_SKIP_READ_COLUMNS)
                if df_meta.empty:
                    need_redownload = True
                else:
                    # 仅用于日志，不参与判定
                    min_d = str(df_meta.trade_date.min())
                    max_d = str(df_meta.trade_date.max())

                    # ---- 尾部允许滞后判断 ----
                    # 允许滞后 N 天：只要文件最大日期 >= (end_date - N天) 就视为完整
                    lag_threshold = (
                        dt.datetime.strptime(end_date, "%Y%m%d") -
                        dt.timedelta(days=CHECK_SKIP_ALLOW_LAG_DAYS)
                    ).strftime("%Y%m%d")

                    if max_d < lag_threshold:
                        need_redownload = True

            except Exception as e:
                need_redownload = True
                logging.warning("Skip 检查读取失败 %s -> 强制重下 (%s)", ts_code, e)

            if not need_redownload:
                # 可选调试
                # logging.debug("[SKIP] %s min=%s max=%s lag_thr=%s", ts_code, min_d, max_d, lag_threshold)
                return (ts_code, 'skip', None)
            else:
                logging.info("文件尾部滞后 %s min=%s max=%s 需要>=%s(=end - %d天) -> 重下",
                            ts_code, min_d, max_d, lag_threshold, CHECK_SKIP_ALLOW_LAG_DAYS)


        def _call():
            return ts.pro_bar(
                ts_code=ts_code,
                start_date=START_DATE,
                end_date=end_date,
                adj = None if API_ADJ == "raw" else API_ADJ,
                freq = 'D',
                asset='E'
            )
        try:
            df = _retry(_call, f"pro_bar_{ts_code}")
            if df is None or df.empty:
                return (ts_code, 'empty', None)
            df = df.sort_values("trade_date")
            df.to_parquet(fpath, index=False)
            if API_ADJ != "raw":
                _WRITE_SYMBOL_INDICATORS(ts_code, df, end_date)
            return (ts_code, 'ok', None)
        except Exception as e:
            return (ts_code, 'err', str(e))

    with ThreadPoolExecutor(max_workers=FAST_INIT_THREADS) as exe:
        futs = {exe.submit(task, c): c for c in codes}
        pbar = tqdm(as_completed(futs), total=len(futs), desc="FAST_INIT 下载", ncols=120,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for fut in pbar:
            ts_code, status, msg = fut.result()
            with lock_stats:
                if status == 'ok':
                    ok += 1
                elif status == 'skip':
                    skip += 1
                elif status == 'empty':
                    empty += 1
                else:
                    err += 1
                    failed.append((ts_code, msg))
                    pbar.close()
                    # 立即取消还在排队/执行的任务
                    exe.shutdown(cancel_futures=True)
                    raise RuntimeError(f"FAIL-FAST 触发：{ts_code} 指标/写盘报错：{msg}")
            pbar.set_postfix(ok=ok, skip=skip, empty=empty, err=err)

        pbar.close()

    logging.info("[FAST_INIT] 初轮完成 ok=%d skip=%d empty=%d err=%d", ok, skip, empty, err)

    failed_codes = [c for c,_ in failed]
    if failed_codes:
        # 写第一轮失败
        with open(os.path.join(DATA_ROOT, "fast_init_failed_round1.txt"), "w", encoding="utf-8") as f:
            for c,m in failed:
                f.write(f"{c},{m}\n")

    # ====== 自动失败补抓(一次) ======
    if FAILED_RETRY_ONCE and failed_codes:
        logging.info("[FAST_INIT] 等待 %ds 后开始失败补抓，失败数=%d", FAILED_RETRY_WAIT, len(failed_codes))
        time.sleep(FAILED_RETRY_WAIT)

        def retry_task(ts_code: str):
            # fpath = os.path.join(FAST_INIT_STOCK_DIR, f"{ts_code}.parquet")
            fpath = os.path.join(FAST_INIT_STOCK_DIR, API_ADJ, f"{ts_code}.parquet")
            if os.path.exists(fpath):
                return (ts_code, 'exists')
            def _call():
                return ts.pro_bar(
                    ts_code=ts_code,
                    start_date=START_DATE,
                    end_date=end_date,
                    adj = None if API_ADJ == "raw" else API_ADJ,
                    freq = 'D',
                    asset = 'E'
                )
            try:
                df = _retry(_call, f"retry_pro_bar_{ts_code}")
                if df is None or df.empty:
                    return (ts_code, 'empty')
                df = df.sort_values("trade_date")
                df.to_parquet(fpath, index=False)
                if API_ADJ != "raw":
                    _WRITE_SYMBOL_INDICATORS(ts_code, df, end_date)
                return (ts_code, 'ok')
            except Exception as e:
                return (ts_code, f"err:{e}")

        ok2 = empty2 = err2 = exists2 = 0
        with ThreadPoolExecutor(max_workers=FAILED_RETRY_THREADS) as exe:
            futs2 = {exe.submit(retry_task, c): c for c in failed_codes}
            pbar2 = tqdm(as_completed(futs2), total=len(futs2), desc="失败补抓", ncols=120)
            for fut in pbar2:
                c, st = fut.result()
                if st == 'ok':
                    ok2 += 1
                elif st == 'empty':
                    empty2 += 1
                elif st == 'exists':
                    exists2 += 1
                else:
                    err2 += 1
                pbar2.set_postfix(ok=ok2, empty=empty2, exists=exists2, err=err2)
            pbar2.close()
        logging.info("[FAST_INIT] 补抓完成 ok=%d empty=%d exists=%d err=%d", ok2, empty2, exists2, err2)

        if err2 > 0:
            final_failed = []
            for c in failed_codes:
                fpath = os.path.join(FAST_INIT_STOCK_DIR, API_ADJ, f"{c}.parquet")
                if not os.path.exists(fpath):
                    final_failed.append(c)
            with open(os.path.join(DATA_ROOT, "fast_init_failed_final.txt"), "w", encoding="utf-8") as f:
                for c in final_failed:
                    f.write(c + "\n")
            logging.warning("[FAST_INIT] 仍失败股票数=%d -> fast_init_failed_final.txt", len(final_failed))
    else:
        logging.info("[FAST_INIT] 无需补抓或无失败股票。")

def duckdb_merge_symbol_products_to_daily(batch_days:int=30):
    import duckdb, os
    from math import ceil
    from tqdm import tqdm

    adj = _decide_symbol_adj_for_fast_init()
    src_dir = os.path.join(DATA_ROOT, "stock", "single", f"single_{adj}_indicators")
    dst_dir = os.path.join(DATA_ROOT, "stock", "daily", f"daily_{adj}_indicators")
    if not os.path.isdir(src_dir):
        logging.warning("[DUCK MERGE IND] 源目录不存在：%s", src_dir)
        return
    os.makedirs(dst_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")

    # 目标端已有的最大日期
    try:
        glob_dst = os.path.join(dst_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
        last_duck = con.sql(f"SELECT max(trade_date) FROM parquet_scan('{glob_dst}')").fetchone()[0] or 0
    except duckdb.Error:
        last_duck = 0
    last_duck_str = str(last_duck).zfill(8)

    # 找出需要合并的“新增日期”清单
    src_posix = src_dir.replace("\\", "/")
    df_dates = con.sql(f"""
        SELECT DISTINCT trade_date
        FROM parquet_scan('{src_posix}/*.parquet')
        WHERE trade_date > '{last_duck_str}'
        ORDER BY trade_date
    """).df()

    if df_dates.empty:
        logging.info("[DUCK MERGE IND] 已最新，跳过")
        con.close()
        return

    dates = df_dates.trade_date.astype(str).tolist()
    total_batches = ceil(len(dates)/batch_days)
    pbar = tqdm(range(total_batches), desc=f"DUCK MERGE IND ({adj})", ncols=120)

    for i in pbar:
        chunk = dates[i*batch_days:(i+1)*batch_days]
        mn, mx = chunk[0], chunk[-1]
        sql = f"""
        COPY (
          SELECT *
          FROM parquet_scan('{src_posix}/*.parquet')
          WHERE trade_date BETWEEN '{mn}' AND '{mx}'
        )
        TO '{dst_dir}'
        (FORMAT PARQUET, PARTITION_BY (trade_date), OVERWRITE_OR_IGNORE 1);
        """
        con.execute(sql)
        pbar.set_postfix(batch=f"{i+1}/{total_batches}", days=len(chunk), rng=f"{mn}~{mx}")

    con.close()
    logging.info("[DUCK MERGE IND] 合并完成 新增日期数=%d", len(dates))
    _maybe_compact(dst_dir)

# ====== 增量重算：把新增日期涉及的股票，补齐“按股票成品(含指标)”并合并到按日分区 ======
def _with_api_adj(temp_api_adj: str, fn, *args, **kwargs):
    """
    临时切换 API_ADJ(仅用于复用 _WRITE_SYMBOL_INDICATORS/duckdb_merge_* 内的 by_symbol 目录判定)，
    调用结束后恢复，避免侵入式改函数签名。
    """
    global API_ADJ
    old = API_ADJ
    API_ADJ = temp_api_adj  # 'raw' | 'qfq' | 'hfq'
    try:
        return fn(*args, **kwargs)
    finally:
        API_ADJ = old

def recalc_symbol_products_for_increment(start: str, end: str, threads: int = 0):
    """
    NORMAL(日常增量)的核心补全：
    1) 找出 stock/<target_adj> 新增日期涉及的 ts_code
    2) 以 warm-up 窗口回看，抽取这批 ts_code 的历史数据
    3) 用统一的 _WRITE_SYMBOL_INDICATORS() 重算单股成品(含指标)
    4) 回灌 fast_init 缓存；再合并到 <adj>_indicators 分区
    """
    import duckdb, os
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed

    target_adj = _decide_symbol_adj_for_fast_init()     # raw / qfq / hfq
    src_dir = os.path.join(DATA_ROOT, "stock", "daily", f"daily_{target_adj}")
    dst_dir = os.path.join(DATA_ROOT, "stock", "daily", f"daily_{target_adj}_indicators")
    if not os.path.isdir(src_dir):
        logging.warning("[INC_IND] 源目录不存在：%s(可能尚未构建 %s)", src_dir, target_adj)
        return

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")

    # ① 目标端(指标分区)最新日期
    try:
        glob_dst = os.path.join(dst_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
        last_duck = con.sql(
            f"SELECT max(trade_date) FROM parquet_scan('{glob_dst}')"
        ).fetchone()[0] or 0
    except duckdb.Error:
        last_duck = 0
    last_duck_str = str(last_duck).zfill(8)

    # ② 源端新增日期涉及的 ts_code
    glob_src = os.path.join(src_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
    try:
        df_codes = con.sql(f"""
            SELECT DISTINCT ts_code
            FROM parquet_scan('{glob_src}')
            WHERE trade_date > '{last_duck_str}'
        """).df()
    except duckdb.Error:
        df_codes = pd.DataFrame(columns=["ts_code"])

    if df_codes.empty:
        logging.info("[INC_IND] 指标分区已最新(last=%s，源端无新增日期)。", last_duck)
        con.close()
        return

    ts_list = df_codes["ts_code"].dropna().astype(str).unique().tolist()
    logging.info("[INC_IND] 需要补的股票数=%d (last_ind_date=%s)", len(ts_list), last_duck_str)

    # ③ 为每个 ts_code 提取“历史 + warm-up”窗数据
    #    为避免全量扫，限定提取窗口：从 (end - SYMBOL_PRODUCT_WARMUP_DAYS - 40) 起拿
    #    多回看 40 天只为冗余保险（遇到节假日/停牌）
    def fetch_code_df(ts_code: str) -> Optional[pd.DataFrame]:
        end_dt = dt.datetime.strptime(end, "%Y%m%d")
        start_dt = end_dt - pd.Timedelta(days=SYMBOL_PRODUCT_WARMUP_DAYS + 40)
        start_win = start_dt.strftime("%Y%m%d")
        try:
            # 每个线程单独建立连接
            loc = duckdb.connect()
            loc.execute(f"PRAGMA threads={max(1, DUCKDB_THREADS // 2)};")
            loc.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
            os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
            loc.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")
            qbase = f"FROM parquet_scan('{glob_src}') WHERE ts_code = '{ts_code}'"

            ind_parquet = os.path.join(DATA_ROOT, "stock", "single", f"single_{target_adj}_indicators", f"{ts_code}.parquet")
            ind_csv     = os.path.join(DATA_ROOT, "stock", "single", "csv", f"{target_adj}_indicators", f"{ts_code}.csv")
            no_single_ind_file = not (os.path.exists(ind_parquet) or os.path.exists(ind_csv))

            if no_single_ind_file:
                # 首次构建该股票成品 → 直接全量
                df = loc.sql(f"SELECT * {qbase}").df()
            else:
                # 只回看 warm-up 窗口
                df = loc.sql(
                    f"SELECT * {qbase} AND trade_date >= '{start_win}' AND trade_date <= '{end}'"
                ).df()
                if df is None or df.empty:
                    df = loc.sql(f"SELECT * {qbase}").df()


            loc.close()

            if df is None or df.empty:
                return None

            td = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
            df['trade_date'] = td.dt.strftime('%Y%m%d')
            df = df.sort_values('trade_date')
            return df

        except duckdb.Error as e:
            try:
                loc.close()
            except Exception:
                pass
            logging.warning('[INC_IND] 读取 %s 失败: %s', ts_code, e)
            
            return None


    # ④ 并发重算单股成品 + 回灌 fast_init 缓存
    def work(ts_code: str) -> Tuple[str, str]:
        df = fetch_code_df(ts_code)
        if df is None or df.empty:
            return ts_code, "empty"
        # 保持 by_symbol 的 adj 目录规则，且在调用期间临时切换 API_ADJ 以重用目录判定
        temp_api_adj = {"daily":"raw", "qfq":"qfq", "hfq":"hfq"}[target_adj]
        try:
            _with_api_adj(temp_api_adj, _WRITE_SYMBOL_INDICATORS, ts_code, df, end)
            _update_fast_init_cache(ts_code, df, target_adj)  # 回灌 fast_init 缓存
            return ts_code, "ok"
        except Exception as e:
            logging.exception("[INC_IND] %s 成品重算失败: %s", ts_code, e)
            return ts_code, f"err:{e}"

    max_workers = threads or max(4, min(16, os.cpu_count() or 8))
    ok = empty = err = 0
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futs = {exe.submit(work, c): c for c in ts_list}
        pbar = tqdm(as_completed(futs), total=len(futs), desc="INC 重算成品(含指标)", ncols=120)
        for fut in pbar:
            code, st = fut.result()
            if st == "ok": ok += 1
            elif st == "empty": empty += 1
            else: err += 1
            pbar.set_postfix(ok=ok, empty=empty, err=err)
        pbar.close()

    con.close()
    logging.info("[INC_IND] 单股成品重算完成 ok=%d empty=%d err=%d", ok, empty, err)

    # ⑤ 把 by_symbol_<adj> 的新增日期 COPY 到 <adj>_indicators（与 FAST 一致）
    if WRITE_SYMBOL_INDICATORS:
        _with_api_adj(
            {"daily":"raw", "qfq":"qfq", "hfq":"hfq"}[target_adj],
            duckdb_merge_symbol_products_to_daily
        )

def duckdb_partition_merge(batch_days:int=30):
    """
    增量合并 fast_init_symbol/{raw|qfq|hfq} 中 *新增* trade_date 到
    stock/daily/{daily|qfq|hfq}/，按日期微批复制并显示进度。
    """
    import duckdb, os
    from math import ceil
    from tqdm import tqdm

    merge_tasks = [
        ("raw", os.path.join(FAST_INIT_STOCK_DIR, "raw"),
                os.path.join(DATA_ROOT, "stock", "daily", "daily_raw")),
        ("qfq", os.path.join(FAST_INIT_STOCK_DIR, "qfq"),
                os.path.join(DATA_ROOT, "stock", "daily", "daily_qfq")),
        ("hfq", os.path.join(FAST_INIT_STOCK_DIR, "hfq"),
                os.path.join(DATA_ROOT, "stock", "daily", "daily_hfq")),
    ]

    for mode, src_dir, dst_dir in merge_tasks:
        if not os.path.isdir(src_dir):
            logging.warning("[DUCK MERGE][%s] 源目录不存在，跳过", mode)
            continue
        os.makedirs(dst_dir, exist_ok=True)

        con = duckdb.connect()
        con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
        con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
        os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")

        # 目标端已有的最大 trade_date
        try:
            glob_dst = os.path.join(dst_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
            last_duck = con.sql(
                f"SELECT max(trade_date) FROM parquet_scan('{glob_dst}')"
            ).fetchone()[0] or 0
        except duckdb.Error:
            last_duck = 0
        last_duck_str = str(last_duck).zfill(8)

        # 计算新增日期清单
        src_posix = src_dir.replace("\\", "/")
        df_dates = con.sql(f"""
            SELECT DISTINCT trade_date
            FROM parquet_scan('{src_posix}/*.parquet')
            WHERE trade_date > '{last_duck_str}'
            ORDER BY trade_date
        """).df()

        if df_dates.empty:
            logging.info("[DUCK MERGE][%s] 已最新，跳过", mode)
            con.close()
            continue

        dates = df_dates.trade_date.astype(str).tolist()
        total_batches = ceil(len(dates) / batch_days)
        pbar = tqdm(range(total_batches), desc=f"DUCK MERGE [{mode}]", ncols=120)

        for i in pbar:
            chunk = dates[i*batch_days:(i+1)*batch_days]
            mn, mx = chunk[0], chunk[-1]
            sql = f"""
            COPY (
              SELECT *
              FROM parquet_scan('{src_posix}/*.parquet')
              WHERE trade_date BETWEEN '{mn}' AND '{mx}'
            )
            TO '{dst_dir}'
            (FORMAT PARQUET, PARTITION_BY (trade_date), OVERWRITE_OR_IGNORE 1);
            """
            con.execute(sql)
            pbar.set_postfix(batch=f"{i+1}/{total_batches}", days=len(chunk), rng=f"{mn}~{mx}")

        con.close()
        logging.info("[DUCK MERGE][%s] 合并完成 新增日期数=%d", mode, len(dates))
        _maybe_compact(dst_dir)  # 合并完成日志之后

def streaming_merge_after_download():
    """
    第二阶段：将  下每个股票 parquet 文件流式合并到
    data/stock/daily/ 和 data/stock/qfq/ 目录，分别处理原始和前复权数据。
    """
    merge_tasks = [
    ("raw", os.path.join(FAST_INIT_STOCK_DIR, "raw"), os.path.join(DATA_ROOT, "stock", "daily", "daily_raw")),
    ("qfq", os.path.join(FAST_INIT_STOCK_DIR, "qfq"), os.path.join(DATA_ROOT, "stock", "daily", "daily_qfq")),
    ("hfq", os.path.join(FAST_INIT_STOCK_DIR, "hfq"), os.path.join(DATA_ROOT, "stock", "daily", "daily_hfq")),
    ]

    for mode_label, symbol_dir, daily_root in merge_tasks:
        stock_files = glob.glob(os.path.join(symbol_dir, "*.parquet"))
        total_files = len(stock_files)
        if total_files == 0:
            logging.warning("[STREAM_MERGE][%s] 无股票文件可归并", mode_label)
            continue

        os.makedirs(daily_root, exist_ok=True)
        buffer: Dict[str, List[pd.DataFrame]] = {}
        processed = 0
        last_flush_time = time.time()

        def flush(reason: str):
            nonlocal buffer
            if not buffer:
                return
            for dt, lst in buffer.items():
                if not lst:
                    continue
                df_day = pd.concat(lst, ignore_index=True)
                pdir = os.path.join(daily_root, f"trade_date={dt}")
                os.makedirs(pdir, exist_ok=True)
                fname = os.path.join(pdir, f"part-merge-{int(time.time()*1e6)}.parquet")
                df_day.to_parquet(fname, index=False)
            logging.info("[STREAM_MERGE][%s] flush: %s 写出日期数=%d", mode_label, reason, len(buffer))
            buffer = {}

        pbar = tqdm(stock_files, desc=f"STREAM 合并 [{mode_label}]", ncols=120)
        for f in pbar:
            try:
                df = pd.read_parquet(f)
            except Exception as e:
                logging.warning("[STREAM_MERGE][%s] 读失败 %s (%s)", mode_label, f, e)
                continue
            if df is None or df.empty:
                continue
            for dt, sub in df.groupby("trade_date"):
                buffer.setdefault(dt, []).append(sub)
            processed += 1

            if len(buffer) >= STREAM_FLUSH_DATE_BATCH:
                flush(f"date_batch>= {STREAM_FLUSH_DATE_BATCH}")
            elif processed % STREAM_FLUSH_STOCK_BATCH == 0:
                flush(f"stock_batch {STREAM_FLUSH_STOCK_BATCH}")
            elif time.time() - last_flush_time > 60:
                flush("time>60s")
                last_flush_time = time.time()

            if processed % STREAM_LOG_EVERY == 0:
                pbar.set_postfix(proc=processed, pct=f"{processed/total_files*100:.1f}%")

        flush("final")
        pbar.close()
        logging.info("[STREAM_MERGE][%s] 完成 处理股票文件=%d", mode_label, processed)

# ========== 主入口 ==========
def main():
    global end_date
    end_date = dt.date.today().strftime("%Y%m%d") if END_DATE.lower() == "today" else END_DATE
    assets = {a.lower() for a in ASSETS}

    logging.info(
        "=== 启动 mode=%s assets=%s 区间=%s-%s 原始数据复权=%s 本地复权=%s===",
        "FAST_INIT" if FAST_INIT_MODE else "NORMAL",
        sorted(assets), START_DATE, end_date, API_ADJ, ADJ_MODE
    )

    if FAST_INIT_MODE:
        fast_init_download(end_date)   # 这里 end_date 已经算好
        if DUCK_MERGE_DAY_LAG >= 0:      # 简单开关，可设 -1 跳过
            duckdb_partition_merge()
        if WRITE_SYMBOL_INDICATORS:
            duckdb_merge_symbol_products_to_daily()
    else:
        # ======= 日常增量模式 =======
        # 优先把 fastinit 缓存合并进 daily，避免全历史重拉
        if any(os.path.isdir(os.path.join(FAST_INIT_STOCK_DIR, d)) and
            len(glob.glob(os.path.join(FAST_INIT_STOCK_DIR, d, "*.parquet"))) > 0
            for d in ("raw","qfq","hfq")):
            duckdb_partition_merge()  # 只 copy 新增 trade_date，速度很快

        if "stock" in assets:
            sync_stock_daily_fast(START_DATE, end_date, threads=STOCK_INC_THREADS)
        if "index" in assets:
            sync_index_daily_fast(START_DATE, end_date, INDEX_WHITELIST)
        recalc_symbol_products_for_increment(START_DATE, end_date, threads=0)

    # 写元数据
    meta = {
        "run_mode": "FAST_INIT" if FAST_INIT_MODE else "NORMAL",
        "api_adj": API_ADJ if FAST_INIT_MODE else None,
        "rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": START_DATE,
        "end_date": end_date,
        "adj_mode": ADJ_MODE,
        "assets": sorted(list(assets))
    }
    with open(os.path.join(DATA_ROOT, "_META.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("=== 结束 ===")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("手动中断程序。")
