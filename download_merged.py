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
- WRITE_SYMBOL_PRODUCT: True 时写“单股票成品(含指标)”
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
    说明：{adj} 由 API_ADJ 决定，"daily" / "daily_qfq" / "daily_hfq"
          文件内字段：基础行情 + 已计算的各类指标(精度按 INDICATOR_DECIMALS 控制)

(3) 按日分区(含指标；增量 COPY)
    {DATA_ROOT}/stock/{adj}_indicators/trade_date=YYYYMMDD/part-*.parquet
    产出方式：调用 duckdb_merge_symbol_products_to_daily()，
    将 (2) 中 “by_symbol_{adj}/*.parquet” 增量 COPY 到该分区目录

(4) (可选)原始K线的按日分区(不含指标)
    {DATA_ROOT}/stock/daily[/daily_qfq/daily_hfq]/trade_date=YYYYMMDD/part-*.parquet
    产出方式：STREAM_MERGE(流式合并 fast_init_symbol/{raw|qfq|hfq} → 对应 daily* 目录)

执行流程
--------
A. 下载阶段(多线程、限频、重试)
   - 对股票清单逐只调用 ts.pro_bar(..., adj=API_ADJ)，写入 (1)
   - 若开启 WRITE_SYMBOL_PRODUCT：对每只股票
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
   FAST_INIT_MODE=True, API_ADJ="qfq", WRITE_SYMBOL_PRODUCT=True, ADJ_MODE="none"
   → 立即得到带指标的 by_symbol_daily_qfq 与 {daily_qfq}_indicators

2) “更稳健的本地复权”方案(推荐)
   FAST_INIT_MODE=True, API_ADJ="raw", WRITE_SYMBOL_PRODUCT=True, ADJ_MODE="both"
   → 先拉原始，再本地构建 qfq/hfq，并同步生成对应的 *_indicators

与回测对齐
----------
- 回测若读取 “含指标的按日分区”，请指向 {DATA_ROOT}/stock/{adj}_indicators/...
- 若回测读取的是“原始K线按日分区”，请确认指标在读取端现算或从 by_symbol_* 读取
- 强烈建议回测端的 PARQUET_ADJ 与这里的 {adj} 保持一致

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
from pathlib import Path
import random
import logging
import datetime as _dt
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
import threading

# ===================== 配置区 =====================
TOKEN = ""  # <-- 必填
# DATA_ROOT = "./data"             # 下载数据目录(可改为绝对路径)
DATA_ROOT = r"E:\stock_data"
ASSETS = ["stock", "index"]      # 可选: ["stock"], ["index"], ["stock","index"]
START_DATE = "20050101"
END_DATE = "today"               # 或具体日期 'YYYYMMDD'
INDEX_WHITELIST = [
    "000001.SH","399001.SZ","399300.SZ","399905.SZ","399006.SZ","000016.SH","000852.SH"
]


# 通用限频：你的权限额度
CALLS_PER_MIN = 475
RETRY_TIMES = 5
PARQUET_ENGINE = "pyarrow"
LOG_LEVEL = "INFO"

# -------- FAST INIT (按股票多线程全历史回补) 开关 --------
FAST_INIT_MODE = True                     # 首次全历史快速抓取
FAST_INIT_THREADS = 1                    # 并发线程数
FAST_INIT_STOCK_DIR = os.path.join(DATA_ROOT, "fast_init_symbol")
API_ADJ = "qfq"                           # qfq/hfq/raw
# 若 FAST_INIT_MODE=True，可通过设置 API_ADJ 控制接口返回的复权方式：
ADJ_MODE = "none"      # 本地复权模式: 'none' | 'qfq' | 'hfq' | 'both'

# -------- 按股票下载后同步生成成品(含指标) --------
WRITE_SYMBOL_PRODUCT = True                # 打开后：每只股票下载成功就同步写成品
SYMBOL_PRODUCT_INDICATORS = "all"  # 或 "all"
SYMBOL_PRODUCT_WARMUP_DAYS = 60          # 增量重算指标的 warm-up 天数
SYMBOL_PRODUCT_OUT = None                 # None → 自动写到 <base>/stock/by_symbol_<adj>

# ===== 重试策略配置 (固定序列 + 抖动) =====
RETRY_DELAY_SEQUENCE = [10, 10, 5]   # 固定序列；超过长度后都用最后一个值(5)
RETRY_JITTER_RANGE = (-0.5, 0.5)     # 每次等待加的随机抖动秒数范围 (可调为 (-1,1))
RETRY_LOG_LEVEL = "INFO"             # 等待日志级别：INFO / DEBUG
# ==========================================

# ==== Streaming Merge 配置 ================
STREAM_FLUSH_DATE_BATCH = 80      # 缓冲多少个不同 trade_date 就刷盘一次
STREAM_FLUSH_STOCK_BATCH = 200    # 处理多少只股票后强制刷盘(避免长时间不落盘)
STREAM_LOG_EVERY = 300            # 每处理多少只股票打印一次进度日志
FAILED_RETRY_ONCE = True          # 第一次下载后自动对失败股票再跑一轮
FAILED_RETRY_THREADS = 8          # 失败补抓的线程数(可低一些)
FAILED_RETRY_WAIT = 5             # 下载结束到补抓之间的等待秒(缓冲限频)
# ==========================================

# ====== Skip 文件完整性快速检查参数 ======
CHECK_SKIP_MIN_MAX = True                 # 是否启用跳过前检查
CHECK_SKIP_READ_COLUMNS = ["trade_date"]  # 读取的列，尽量最少减少 IO
CHECK_SKIP_ALLOW_LAG_DAYS = 1           # 允许已有文件的最大日期距离 end_date 的“滞后”天数 (0=必须等于 end_date)
SKIP_CHECK_START_ENABLED = False          # 是否启用开始日期检查(如果不需要可以关闭，减少接口调用)
# ==========================================

# ==== DuckDB 分批归并配置 =================
DUCKDB_BATCH_SIZE = 300          # 每批处理的“单股票文件”数量(内存紧 → 降到 150/100)
DUCKDB_THREADS = 12              # DuckDB 并行线程 (2~8 之间；太大内存峰值上升)
DUCKDB_MEMORY_LIMIT = "16GB"      # 给 DuckDB 的内存上限(小机器可设 "4GB")
DUCKDB_TEMP_DIR = "duckdb_tmp"   # Spill 目录(磁盘剩余空间要够)
DUCKDB_CLEAR_DAILY_BEFORE = False # 首次构建或要完全重建设 True，会清空 daily 目录
DUCKDB_COLUMNS = "*"             # 列裁剪：可改成 "ts_code,trade_date,open,high,low,close,vol,amount"
DUCKDB_ENABLE_COMPACT_AFTER = "if_needed"       # 启用压实
COMPACT_MAX_FILES_PER_DATE = 12          # 超过 12 个 part 的日期执行压实
COMPACT_TMP_DIR = "compact_tmp"          # 若 compact 函数里需要临时目录(当前版本没用到)
DUCK_MERGE_DAY_LAG = 5          # parquet 最大日期距离 duck 表 > 5 天才触发合并
DUCK_MERGE_MIN_ROWS = 1_000_000 # 或过去 5 天新增行数 > 100 万行才触发
# ==========================================

# ==== 复权因子多线程下载配置 ===============
ADJ_MT_ENABLE = True          # 启用多线程因子下载
ADJ_THREADS = 6               # 并发线程数(3~8 合理)
ADJ_CALLS_PER_MIN = 466       # 给因子下载预留的每分钟调用预算(与 CALLS_PER_MIN 总和不要超过额度)
ADJ_RETRY = 4                 # 每个日期的轻量重试次数
ADJ_BACKOFF_BASE = 1.5        # 指数回退基数
# ==========================================
# ==========================================

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
root.setLevel(logging.INFO)          # 全局级别

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

# 终端 Handler(仅 WARNING+)
console_hdl = logging.StreamHandler(sys.stdout)
console_hdl.setLevel(logging.INFO)
console_hdl.setFormatter(file_fmt)
root.addHandler(console_hdl)

dl_hdl = logging.FileHandler(os.path.join(DATA_ROOT, "download.log"), encoding="utf-8")
dl_hdl.setLevel(logging.INFO)
dl_hdl.setFormatter(file_fmt)
root.addHandler(dl_hdl)

# ===== 本地指标引擎(合并自 preprocess_symbols.py 的必要部分) =====
# 说明：为了避免外部依赖 preprocess_symbols.py，这里内置了最小可用的指标注册和计算函数。
# 仍依赖项目同目录下的 indicators.py 中的具体指标实现。
INDICATOR_REGISTRY = {
    "z_score": {
        "kind": "df",
        "out": "z_slope0",  # 直接把 DataFrame 合并进去(包含 z_score, z_slope)
    },
    "kdj": {
        "kind": "df",
        "out": ["j"],
    },
    "bbi": {
        "kind": "df",
        "out": "bbi",
    },
    "volume_ratio": {
        "kind": "df",
        "out": "vr",
        "kwargs": {"n": 20},
    },
    "bupiao": {
        "kind": "df",
        "out": ["bupiao_short","bupiao_long"],
        "kwargs": {"n1": 3, "n2": 21},
    },
    "shuangjunxian": {
        "kind": "df",
        "out": "bar_color",
        "kwargs": {"n": 5},
    },
}

def _parse_indicators(arg: str):
    if not arg:
        return []
    if arg.lower().strip() == "all":
        return list(INDICATOR_REGISTRY.keys())
    return [x.strip() for x in arg.split(",") if x.strip()]

def _add_indicators(df, names):
    """
    把所选指标计算后追加为新列；依赖 indicators.py
    支持两类：
      - kind='df'：指标函数接收整个 df，返回 DataFrame 或 Series
      - kind='series_close'：指标函数接收 close 序列
    """
    import pandas as pd
    import indicators as ind  # 项目同目录

    out = df.copy()
    for name in names or []:
        spec = INDICATOR_REGISTRY.get(name)
        if not spec:
            continue
        kind = spec.get("kind", "df")
        out_names = spec.get("out")
        kwargs = spec.get("kwargs", {})

        try:
            if kind == "df":
                res = getattr(ind, name)(out, **kwargs)
            elif kind == "series_close":
                res = getattr(ind, name)(out["close"], **kwargs)
            else:
                res = getattr(ind, name)(out, **kwargs)
        except Exception as e:
            logging.warning("[PRODUCT][IND] 指标 %s 计算失败：%s", name, e)
            continue

        # 落地
        if isinstance(res, pd.DataFrame):
            # 如果在注册表里指定了 out，就只保留这些列
            if out_names:
                keep = out_names if isinstance(out_names, (list, tuple)) else [out_names]
                res = res[[c for c in res.columns if c in keep]]
            for col in res.columns:
                out[col] = res[col]
        elif isinstance(res, (list, tuple)):
            if not isinstance(out_names, (list, tuple)) or len(out_names) != len(res):
                logging.warning("[PRODUCT][IND] 指标 %s 返回 %d 列，但 'out' 未提供匹配列名，跳过。", name, len(res))
                continue
            for col_name, series in zip(out_names, res):
                out[col_name] = series
        else:
            col_name = out_names if isinstance(out_names, str) else name
            out[col_name] = res

    return out

def _decide_symbol_adj_for_fast_init() -> str:
    # FAST_INIT 下基于 API_ADJ 决定写到哪个 by_symbol 目录
    aj = (API_ADJ or "raw").lower()
    if aj == "qfq":
        return "daily_qfq"
    if aj == "hfq":
        return "daily_hfq"
    return "daily"

def _write_symbol_product(ts_code: str, df: pd.DataFrame, end_date: str):
    """
    把该 ts_code 的 DataFrame 计算指标后写入 by_symbol 成品(带 warm-up 增量)。
    要求 df 至少包含: trade_date, open, high, low, close, vol[, amount, pre_close]
    """
    if not WRITE_SYMBOL_PRODUCT:
        return

    # 1) 选择输出目录
    adj = _decide_symbol_adj_for_fast_init()
    base_dir = DATA_ROOT
    out_dir = SYMBOL_PRODUCT_OUT or os.path.join(base_dir, "stock", f"by_symbol_{adj}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ts_code}.parquet")

    # 2) 规范、排序、去重
    df2 = df.copy()
    if "trade_date" not in df2.columns:
        raise ValueError("df 缺少 trade_date 列")
    df2["trade_date"] = (
        pd.to_datetime(df2["trade_date"].astype(str))   # 保留日期计算能力
        .dt.strftime("%Y%m%d")                       # 立即转回字符串
    )
    df2 = df2.sort_values("trade_date").drop_duplicates("trade_date", keep="last")

    # 3) 增量 warm-up(如已有旧文件：保留旧数据<warmup起点，warm-up 段与新数据一起重算指标)
    warmup_start = None
    if os.path.exists(out_path):
        try:
            old = pd.read_parquet(out_path)
            if not old.empty:
                last_dt = pd.to_datetime(old["trade_date"].astype(str)).max()
                warmup_start = (last_dt - pd.Timedelta(days=SYMBOL_PRODUCT_WARMUP_DAYS))
                # 拼接：旧数据的非 warmup 段 + 新数据(截到 end_date)
                old["trade_date"] = pd.to_datetime(old["trade_date"].astype(str))
                keep_old = old[old["trade_date"] < warmup_start]
                df2 = pd.concat([keep_old, df2[df2["trade_date"] >= warmup_start]], ignore_index=True)
        except Exception as e:
            logging.warning("[PRODUCT][%s] 读取旧文件失败，按全量重算：%s", ts_code, e)

    # 4) 直接使用内置指标引擎(不再依赖 preprocess_symbols.py)
    try:
        indicators = _parse_indicators(SYMBOL_PRODUCT_INDICATORS) if SYMBOL_PRODUCT_INDICATORS else []
        df2 = _add_indicators(df2, indicators)
    except Exception as e:
        logging.exception("[PRODUCT][%s] 指标计算失败：%s", ts_code, e)
        return

    # 5) 最终落盘(再次按日期去重)
    df2 = df2.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    price_cols = ["open", "high", "low", "close", "pre_close", "change"]
    for col in price_cols:
        if col in df2.columns:
            df2[col] = df2[col].round(2)
            
    INDICATOR_DECIMALS = {
        "j": 3,               # KDJ-J
        "vr": 4,              # volume_ratio
        "z_score": 6          # z_score
    }
    for col, n in INDICATOR_DECIMALS.items():
        if col in df2.columns:
            df2[col] = df2[col].round(n)
    df2.to_parquet(out_path, index=False, engine=PARQUET_ENGINE)
    logging.debug("[PRODUCT][%s] 成品已写出：%s(rows=%d, warmup=%s)",
                 ts_code, out_path, len(df2), SYMBOL_PRODUCT_WARMUP_DAYS if warmup_start is not None else 0)

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

# ========== 公共函数 ==========
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

def compact_daily_partitions():
    """
    压实按日期分区目录：如果某日期下 parquet 文件数超过阈值，
    以 DuckDB 读取后重写(尽量生成少量新文件)。
    """
    import duckdb, glob, shutil

    daily_dir = os.path.join(DATA_ROOT, "stock", "daily")
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
    day_lag = int(last_parquet) - int(last_duck)
    if day_lag >= DUCK_MERGE_DAY_LAG:
        return True

    # ② 或新增行数 ≥ 阈值
    return new_rows >= DUCK_MERGE_MIN_ROWS

# ========== 按交易日批量模式(原有日常增量) ==========
def sync_stock_daily(start: str, end: str):
    root = os.path.join(DATA_ROOT, "stock", "daily")
    os.makedirs(root, exist_ok=True)
    last_dt = _last_partition_date(root)
    actual_start = start if last_dt is None else str(int(last_dt) + 1)
    if actual_start > end:
        logging.info("[stock] 日线已最新 (last=%s)", last_dt)
        return
    dates = _trade_dates(actual_start, end)
    pbar = _tqdm_iter(dates, desc="股票日线")
    try:
        for d in pbar:
            pbar.set_postfix_str(d)
            try:
                df = _retry(lambda dd=d: pro.daily(trade_date=dd), f"stock_daily_{d}")
            except Exception:
                continue
            _save_partition(df, root)
    except KeyboardInterrupt:
        pbar.close()
        logging.warning("[stock] 中断 %d/%d", pbar.n, len(dates))
        raise
    finally:
        pbar.close()

def sync_index_daily(start: str, end: str, whitelist: List[str]):
    root = os.path.join(DATA_ROOT, "index", "daily")
    os.makedirs(root, exist_ok=True)
    last_dt = _last_partition_date(root)
    actual_start = start if last_dt is None else str(int(last_dt) + 1)
    if actual_start > end:
        logging.info("[index] 日线已最新 (last=%s)", last_dt)
        return
    dates = _trade_dates(actual_start, end)
    wl = set(whitelist)
    pbar = _tqdm_iter(dates, desc="指数日线")
    try:
        for d in pbar:
            pbar.set_postfix_str(d)
            all_df = []
            for code in whitelist:
                try:
                    df = _retry(lambda dd=d, ts_code=code: pro.index_daily(trade_date=dd, ts_code=ts_code),
                                f"index_daily_{d}_{code}")
                    all_df.append(df)
                except Exception:
                    continue
            if all_df:
                df_all = pd.concat(all_df, ignore_index=True)
                _save_partition(df_all, root)

    except KeyboardInterrupt:
        pbar.close()
        logging.warning("[index] 中断 %d/%d", pbar.n, len(dates))
        raise
    finally:
        pbar.close()

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
    last_dt = _last_partition_date(root)
    actual_start = start if last_dt is None else str(int(last_dt) + 1)
    if actual_start > end:
        logging.info("[index][FAST] 日线已最新 (last=%s)", last_dt)
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

def sync_adj_factor(start: str, end: str):
    root = os.path.join(DATA_ROOT, "stock", "adj_factor")
    os.makedirs(root, exist_ok=True)
    last_dt = _last_partition_date(root)
    actual_start = start if last_dt is None else str(int(last_dt) + 1)
    if actual_start > end:
        logging.info("[adj_factor] 已最新 (last=%s)", last_dt)
        return
    dates = _trade_dates(actual_start, end)
    pbar = _tqdm_iter(dates, desc="复权因子")
    try:
        for d in pbar:
            pbar.set_postfix_str(d)
            try:
                df = _retry(lambda dd=d: pro.adj_factor(trade_date=dd), f"adj_factor_{d}")
            except Exception:
                continue
            _save_partition(df, root)
    except KeyboardInterrupt:
        pbar.close()
        logging.warning("[adj_factor] 中断 %d/%d", pbar.n, len(dates))
        raise
    finally:
        pbar.close()

def sync_adj_factor_mt(start: str, end: str):
    """
    多线程按交易日抓取复权因子：
      - 按日期分区 trade_date=YYYYMMDD
      - 已存在分区跳过
      - 并发 + 限频 + 轻量重试
    """
    root = os.path.join(DATA_ROOT, "stock", "adj_factor")
    os.makedirs(root, exist_ok=True)

    # 增量起始
    last_dt = _last_partition_date(root)
    actual_start = start if last_dt is None else str(int(last_dt) + 1)
    if actual_start > end:
        logging.info("[adj_factor][MT] 已最新 (last=%s)", last_dt)
        return

    dates_all = _trade_dates(actual_start, end)
    # 逆序(新到旧)优先：可更快满足构建最近复权需求
    dates_all.reverse()

    def work(trade_date: str) -> Tuple[str, str]:
        pdir = os.path.join(root, f"trade_date={trade_date}")
        fpath = os.path.join(pdir, "part-0.parquet")
        if os.path.exists(fpath):
            return trade_date, "skip"
        for attempt in range(1, ADJ_RETRY + 1):
            try:
                _rate_limit()
                df = pro.adj_factor(trade_date=trade_date)
                if df is None or df.empty:
                    return trade_date, "empty"
                os.makedirs(pdir, exist_ok=True)
                # 精简列(若未来想减肥，可保留 ts_code,trade_date,adj_factor)
                df.to_parquet(fpath, index=False, engine=PARQUET_ENGINE)
                return trade_date, "ok"
            except Exception as e:
                if attempt == ADJ_RETRY:
                    return trade_date, f"err:{e}"
                # 指数回退
                time.sleep(ADJ_BACKOFF_BASE ** (attempt - 1))
        return trade_date, "err:unknown"

    ok=skip=empty=err=0
    with ThreadPoolExecutor(max_workers=ADJ_THREADS) as exe:
        futs = {exe.submit(work, d): d for d in dates_all}
        pbar = tqdm(as_completed(futs), total=len(futs), desc="adj_factor[MT]", ncols=120)
        for fut in pbar:
            d, st = fut.result()
            if st == "ok": ok += 1
            elif st == "skip": skip += 1
            elif st == "empty": empty += 1
            else: err += 1
            pbar.set_postfix(ok=ok, skip=skip, empty=empty, err=err)
        pbar.close()
    logging.info("[adj_factor][MT] 完成 ok=%d skip=%d empty=%d err=%d", ok, skip, empty, err)

def validate_completeness(start: str, end: str):
    daily_root  = os.path.join(DATA_ROOT, "stock", "daily")
    factor_root = os.path.join(DATA_ROOT, "stock", "adj_factor")
    dates = set(_trade_dates(start, end))
    have_daily = set(d.split("=")[1] for d in os.listdir(daily_root)
                     if d.startswith("trade_date=")) if os.path.exists(daily_root) else set()
    have_factor = set(d.split("=")[1] for d in os.listdir(factor_root)
                      if d.startswith("trade_date=")) if os.path.exists(factor_root) else set()
    miss_daily = sorted(list(dates - have_daily))
    miss_factor = sorted(list(dates - have_factor))
    logging.info("[check] total=%d daily_ok=%d factor_ok=%d miss_daily=%d miss_factor=%d",
                 len(dates), len(have_daily), len(have_factor), len(miss_daily), len(miss_factor))
    if miss_daily or miss_factor:
        rep = {
            "range": [start, end],
            "missing_daily": miss_daily,
            "missing_factor": miss_factor
        }
        with open(os.path.join(DATA_ROOT, "completeness_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        logging.warning("[check] 存在缺口 -> completeness_report.json")

# ========== 本地复权构建 (反向最新因子) ==========
def build_adjusted_prices(start: str, end: str, mode: str):
    assert mode in ("qfq","hfq","both")
    daily_root  = os.path.join(DATA_ROOT, "stock", "daily")
    factor_root = os.path.join(DATA_ROOT, "stock", "adj_factor")
    out_qfq = os.path.join(DATA_ROOT, "stock", "daily_qfq")
    out_hfq = os.path.join(DATA_ROOT, "stock", "daily_hfq")
    if mode in ("qfq","both"): os.makedirs(out_qfq, exist_ok=True)
    if mode in ("hfq","both"): os.makedirs(out_hfq, exist_ok=True)

    dates = _trade_dates(start, end)

    # 1) 反向收集最新因子
    latest_factor: Dict[str, float] = {}
    for d in reversed(dates):
        fdir = os.path.join(factor_root, f"trade_date={d}")
        if not os.path.exists(fdir):
            continue
        for fn in os.listdir(fdir):
            if not fn.endswith(".parquet"): continue
            df_f = pd.read_parquet(os.path.join(fdir, fn))
            for code, fac in zip(df_f.ts_code, df_f.adj_factor):
                if pd.isna(fac): continue
                if code not in latest_factor:
                    latest_factor[code] = fac  # 第一次(反向)即最终最新

    first_factor: Dict[str, float] = {}
    missing_factor_rows = 0
    rows_qfq = 0
    rows_hfq = 0

    pbar = _tqdm_iter(dates, desc="构建复权")
    try:
        for d in pbar:
            pbar.set_postfix_str(d)
            ddir = os.path.join(daily_root,  f"trade_date={d}")
            fdir = os.path.join(factor_root, f"trade_date={d}")
            if not (os.path.exists(ddir) and os.path.exists(fdir)):
                continue
            daily_files  = [os.path.join(ddir,x) for x in os.listdir(ddir) if x.endswith(".parquet")]
            factor_files = [os.path.join(fdir,x) for x in os.listdir(fdir) if x.endswith(".parquet")]
            if not daily_files or not factor_files:
                continue
            df_daily  = pd.concat([pd.read_parquet(f) for f in daily_files], ignore_index=True)
            df_factor = pd.concat([pd.read_parquet(f) for f in factor_files], ignore_index=True)
            merged = df_daily.merge(df_factor, on=["ts_code","trade_date"], how="left")

            recs_q = []
            recs_h = []
            for _, row in merged.iterrows():
                fac = row.adj_factor
                if pd.isna(fac):
                    missing_factor_rows += 1
                    continue
                code = row.ts_code
                if code not in first_factor:
                    first_factor[code] = fac
                base = row.to_dict()
                if mode in ("qfq","both"):
                    lf = latest_factor.get(code)
                    if lf:
                        ratio_q = fac / lf
                        for c in ["open","high","low","close","pre_close"]:
                            base[c+"_qfq"] = row[c] * ratio_q
                    recs_q.append(base.copy())
                if mode in ("hfq","both"):
                    ff = first_factor.get(code)
                    if ff:
                        ratio_h = fac / ff
                        for c in ["open","high","low","close","pre_close"]:
                            base[c+"_hfq"] = row[c] * ratio_h
                    recs_h.append(base.copy())

            if recs_q and mode in ("qfq","both"):
                out_df = pd.DataFrame(recs_q)
                qdir = os.path.join(out_qfq, f"trade_date={d}")
                os.makedirs(qdir, exist_ok=True)
                out_df.to_parquet(os.path.join(qdir, f"part-qfq-{int(time.time()*1e6)}.parquet"),
                                  index=False, engine=PARQUET_ENGINE)
                rows_qfq += len(out_df)

            if recs_h and mode in ("hfq","both"):
                out_df = pd.DataFrame(recs_h)
                hdir = os.path.join(out_hfq, f"trade_date={d}")
                os.makedirs(hdir, exist_ok=True)
                out_df.to_parquet(os.path.join(hdir, f"part-hfq-{int(time.time()*1e6)}.parquet"),
                                  index=False, engine=PARQUET_ENGINE)
                rows_hfq += len(out_df)

        logging.info("[adjust] 完成 qfq=%d 行 hfq=%d 行 缺因子行=%d 股票数(first=%d latest=%d)",
                     rows_qfq, rows_hfq, missing_factor_rows,
                     len(first_factor), len(latest_factor))
        report = {
            "range": [start, end],
            "rows_qfq": rows_qfq,
            "rows_hfq": rows_hfq,
            "missing_factor_rows": missing_factor_rows,
            "stock_count_first": len(first_factor),
            "stock_count_latest": len(latest_factor)
        }
        with open(os.path.join(DATA_ROOT, "adjust_build_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except KeyboardInterrupt:
        pbar.close()
        logging.warning("[adjust] 中断 当前日期=%s", d)
        raise
    finally:
        pbar.close()

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
                        _dt.datetime.strptime(end_date, "%Y%m%d") -
                        _dt.timedelta(days=CHECK_SKIP_ALLOW_LAG_DAYS)
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
            _write_symbol_product(ts_code, df, end_date)
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
            fpath = os.path.join(FAST_INIT_STOCK_DIR, f"{ts_code}.parquet")
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
                _write_symbol_product(ts_code, df, end_date)
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
                fpath = os.path.join(FAST_INIT_STOCK_DIR, f"{c}.parquet")
                if not os.path.exists(fpath):
                    final_failed.append(c)
            with open(os.path.join(DATA_ROOT, "fast_init_failed_final.txt"), "w", encoding="utf-8") as f:
                for c in final_failed:
                    f.write(c + "\n")
            logging.warning("[FAST_INIT] 仍失败股票数=%d -> fast_init_failed_final.txt", len(final_failed))
    else:
        logging.info("[FAST_INIT] 无需补抓或无失败股票。")

def duckdb_merge_symbol_products_to_daily():
    """
    将 stock/by_symbol_<adj> 下的“单股票成品(含指标)”按 trade_date 增量 COPY
    到 stock/<adj>_indicators/trade_date=YYYYMMDD/ 下。
    <adj> 取自 FAST_INIT 的 API_ADJ：daily / daily_qfq / daily_hfq
    """
    import duckdb, os

    adj = _decide_symbol_adj_for_fast_init()  # daily / daily_qfq / daily_hfq
    src_dir = os.path.join(DATA_ROOT, "stock", f"by_symbol_{adj}")
    dst_dir = os.path.join(DATA_ROOT, "stock", f"{adj}_indicators")

    if not os.path.isdir(src_dir):
        logging.warning("[DUCK MERGE IND] 源目录不存在：%s", src_dir)
        return
    os.makedirs(dst_dir, exist_ok=True)

    # —— 是否需要合并(沿用你的触发规则)——
    if not _need_duck_merge(dst_dir):
        logging.info("[DUCK MERGE IND] 已最新，跳过")
        return

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")

    # ① 读取目标端(已分区)能看到的最大 trade_date
    try:
        glob_path = os.path.join(dst_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
        last_duck = con.sql(
            f"SELECT max(trade_date) FROM parquet_scan('{glob_path}')"
        ).fetchone()[0] or 0
    except duckdb.Error:
        last_duck = 0

    # ② 从 by_symbol_<adj> 复制“新增日期”到按日分区
    src_posix = src_dir.replace("\\", "/")
    last_duck_str = str(last_duck).zfill(8)

    sql = f"""
    COPY (
      SELECT *
      FROM parquet_scan('{src_posix}/*.parquet')
      WHERE trade_date > '{last_duck_str}'
    )
    TO '{dst_dir}'
    (FORMAT PARQUET,
     PARTITION_BY (trade_date),
     OVERWRITE_OR_IGNORE 1);
    """
    con.execute(sql)
    con.close()
    logging.info("[DUCK MERGE IND] 完成，写入新日期 > %s 到 %s", last_duck, dst_dir)

# ====== 增量重算：把新增日期涉及的股票，补齐“按股票成品(含指标)”并合并到按日分区 ======
def _pick_target_adj_for_normal() -> str:
    """
    基于 NORMAL 模式下的 ADJ_MODE 选择指标来源：
    - 'qfq' or 'both' -> daily_qfq
    - 'hfq'           -> daily_hfq
    - else            -> daily
    """
    mode = (ADJ_MODE or "none").lower()
    if mode in ("qfq", "both"):
        return "daily_qfq"
    if mode == "hfq":
        return "daily_hfq"
    return "daily"

def _with_api_adj(temp_api_adj: str, fn, *args, **kwargs):
    """
    临时切换 API_ADJ(仅用于复用 _write_symbol_product/duckdb_merge_* 内的 by_symbol 目录判定)，
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
    NORMAL(日常增量)在写完 stock/daily(+qfq/hfq) 后调用：
    1) 找出 stock/<target_adj> 新增的 trade_date
    2) 提取这些日期涉及到的 ts_code 集合
    3) 仅对这些股票重算 by_symbol_<adj> 成品(含指标，带 warm-up)
    4) 合并到 stock/<adj>_indicators/trade_date=YYYYMMDD/
    """
    import duckdb, os
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    target_adj = _pick_target_adj_for_normal()     # daily / daily_qfq / daily_hfq
    src_dir = os.path.join(DATA_ROOT, "stock", target_adj)
    dst_dir = os.path.join(DATA_ROOT, "stock", f"{target_adj}_indicators")
    if not os.path.isdir(src_dir):
        logging.warning("[INC_IND] 源目录不存在：%s(可能尚未构建 %s)", src_dir, target_adj)
        return

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")

    # ① 目标端(已分区)最新 trade_date
    try:
        glob_dst = os.path.join(dst_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
        last_duck = con.sql(
            f"SELECT max(trade_date) FROM parquet_scan('{glob_dst}')"
        ).fetchone()[0] or 0
    except duckdb.Error:
        last_duck = 0
    last_duck_str = str(last_duck).zfill(8)

    # ② 源端(daily/daily_qfq/daily_hfq)新增日期涉及的股票
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
        c

def duckdb_partition_merge():
    """
    增量合并 fast_init_symbol/raw, qfq 中 *新增* trade_date 到
    stock/daily/ 与 stock/daily_qfq/。
    """
    import duckdb, os, shutil

    merge_tasks = [
        ("raw",  os.path.join(FAST_INIT_STOCK_DIR, "raw"),
                 os.path.join(DATA_ROOT, "stock", "daily")),
        ("qfq",  os.path.join(FAST_INIT_STOCK_DIR, "qfq"),
                 os.path.join(DATA_ROOT, "stock", "daily_qfq")),
        ("hfq", os.path.join(FAST_INIT_STOCK_DIR, "hfq"),
         os.path.join(DATA_ROOT, "stock", "daily_hfq")),
    ]

    for mode, src_dir, dst_dir in merge_tasks:
        if not os.path.isdir(src_dir):
            logging.warning("[DUCK MERGE][%s] 源目录不存在，跳过", mode)
            continue
        os.makedirs(dst_dir, exist_ok=True)

        # --- 判断是否需要合并 ---
        if not _need_duck_merge(dst_dir):
            logging.info("[DUCK MERGE][%s] 已最新，跳过", mode)
            continue

        logging.info("[DUCK MERGE][%s] 开始增量 COPY ...", mode)
        con = duckdb.connect()
        con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
        con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
        os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")

        # ① 找 duckdb 已有的最大 trade_date
        try:
            glob_path = os.path.join(dst_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
            last_duck = con.sql(
                f"SELECT max(trade_date) FROM parquet_scan('{glob_path}')"
            ).fetchone()[0] or 0

        except duckdb.Error:
            last_duck = 0

        # ② 直接 COPY 新增行到目标分区目录
        src_posix = src_dir.replace("\\", "/")
        last_duck_str = str(last_duck).zfill(8)    # 保证 '20250718' 这种 8 位

        sql = f"""
        COPY (
        SELECT *
        FROM parquet_scan('{src_posix}/*.parquet')
        WHERE trade_date > '{last_duck_str}'
        )
        TO '{dst_dir}'
        (FORMAT PARQUET,
        PARTITION_BY (trade_date),
        OVERWRITE_OR_IGNORE 1);
        """

        con.execute(sql)
        con.close()
        logging.info("[DUCK MERGE][%s] 完成，写入新日期 > %s", mode, last_duck)
    
def streaming_merge_after_download():
    """
    第二阶段：将 fast_init_symbol 下每个股票 parquet 文件流式合并到
    data/stock/daily/ 和 data/stock/daily_qfq/ 目录，分别处理原始和前复权数据。
    """
    import glob
    merge_tasks = [
        ("raw", os.path.join(FAST_INIT_STOCK_DIR, "raw"), os.path.join(DATA_ROOT, "stock", "daily")),
        ("qfq", os.path.join(FAST_INIT_STOCK_DIR, "qfq"), os.path.join(DATA_ROOT, "stock", "daily_qfq")),
        ("hfq", os.path.join(FAST_INIT_STOCK_DIR, "hfq"),
         os.path.join(DATA_ROOT, "stock", "daily_hfq")),

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
    end_date = _dt.date.today().strftime("%Y%m%d") if END_DATE.lower() == "today" else END_DATE
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
        if WRITE_SYMBOL_PRODUCT:
            duckdb_merge_symbol_products_to_daily()

        if ADJ_MODE in ("qfq", "hfq", "both"):
            logging.info("[ADJ] 开始下载复权因子并构建本地复权价格: mode=%s", ADJ_MODE)
            if ADJ_MT_ENABLE:
                sync_adj_factor_mt(START_DATE, end_date)
            else:
                sync_adj_factor(START_DATE, end_date)
            build_adjusted_prices(START_DATE, end_date, ADJ_MODE)
        else:
            logging.info("[ADJ] ADJ_MODE=none -> 跳过本地复权构建")


    else:
        # ======= 日常增量模式 =======
        if "stock" in assets:
            sync_stock_daily(START_DATE, end_date)
        if "index" in assets:
            sync_index_daily_fast(START_DATE, end_date, INDEX_WHITELIST)
        if "stock" in assets and ADJ_MODE in ("qfq", "hfq", "both"):
            sync_adj_factor(START_DATE, end_date)
            validate_completeness(START_DATE, end_date)
            build_adjusted_prices(START_DATE, end_date, ADJ_MODE)
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
