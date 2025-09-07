#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import sys
import time
import json
import glob
from pathlib import Path
import random
import logging
import pyarrow.dataset as ds
import pyarrow as pa
import datetime as dt
from typing import List, Optional, Callable, Dict, Tuple
from logging.handlers import TimedRotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import duckdb
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

import threading
_TLS = threading.local()
from utils import normalize_trade_date, normalize_ts

# --- 不走系统代理：仅对本进程生效 ---
for k in ("http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
    os.environ.pop(k, None)

# 想让哪些域名/本地地址强制直连（可按需增删）
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,api.tushare.pro,github.com,raw.githubusercontent.com,pypi.org,files.pythonhosted.org,huggingface.co,cdn-lfs.huggingface.co")
os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])

EPS = 1e-12
# ------------- 基本检查 -------------
if TOKEN.startswith("在这里"):
    logging.debug('[BRANCH] <module> | IF TOKEN.startswith("在这里") -> taken')
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

# root = logging.getLogger()
# root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.DEBUG))
root = logging.getLogger()
# 始终开 DEBUG，让各 Handler 决定往哪儿输出；控制台仍用 LOG_LEVEL 过滤
root.setLevel(logging.DEBUG)


# 文件 Handler(轮换)
LOG_DIR = os.path.join(".", "log")
log_file_path = os.path.join(LOG_DIR, "fast_init.log")
os.makedirs(LOG_DIR, exist_ok=True)
file_hdl = TimedRotatingFileHandler(
    log_file_path,
    when="midnight",
    backupCount=7,
    encoding="utf-8"
)
file_fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
file_hdl.setFormatter(file_fmt)
file_hdl.setLevel(logging.DEBUG)
root.addHandler(file_hdl)

# 终端 Handler
console_hdl = logging.StreamHandler(sys.stdout)
console_hdl.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))  # ← 控制台仍按配置
console_hdl.setFormatter(file_fmt)
root.addHandler(console_hdl)
log_dir = os.path.join(".", "log")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "download.log")
dl_hdl = logging.FileHandler(log_path, encoding="utf-8")
dl_hdl.setLevel(logging.DEBUG)
dl_hdl.setFormatter(file_fmt)
root.addHandler(dl_hdl)

# === 申万行业：本地缓存与合并 =================================================

import os, time, tempfile
import pandas as pd

SW_DATA_FILE = os.getenv("SW_DATA_FILE", "./stock/sw_industry.parquet")

def _ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _read_parquet_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "ts_code","l1_code","l1_name","l2_code","l2_name",
            "l3_code","l3_name","in_date","out_date","is_new","snapshot_ts"
        ])
    return pd.read_parquet(path)


def _atomic_write_parquet(df: pd.DataFrame, path: str):
    _ensure_parent_dir(path)
    # 写临时文件 -> 原子替换
    dir_ = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(dir=dir_, delete=False, suffix=".parquet") as tmp:
        tmp_path = tmp.name
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass


def persist_sw_to_parquet(ts_code: str, info: dict):
    """
    把下载到的行业信息落到 ./stock/sw_industry.parquet
    info 需要包含：
      sw_l1_code/sw_l1_name/sw_l2_code/sw_l2_name/sw_l3_code/sw_l3_name
      sw_in_date/sw_out_date/sw_is_new
    """
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "ts_code": ts_code,
        "l1_code": info.get("sw_l1_code") or "",
        "l1_name": info.get("sw_l1_name") or "",
        "l2_code": info.get("sw_l2_code") or "",
        "l2_name": info.get("sw_l2_name") or "",
        "l3_code": info.get("sw_l3_code") or "",
        "l3_name": info.get("sw_l3_name") or "",
        "in_date": info.get("sw_in_date") or "",
        "out_date": info.get("sw_out_date") or "",
        "is_new":  (info.get("sw_is_new") or "Y"),
        "snapshot_ts": now,
    }])

    df = _read_parquet_safe(SW_DATA_FILE)

    # 1) 历史维度去重：以 (ts_code, in_date) 唯一，保留 snapshot_ts 最新
    df = pd.concat([df, row], ignore_index=True)
    df["snapshot_ts_key"] = pd.to_datetime(df["snapshot_ts"], errors="coerce")
    df.sort_values(["ts_code","in_date","snapshot_ts_key"], ascending=[True,True,False], inplace=True)
    df = df.drop_duplicates(subset=["ts_code","in_date"], keep="first")
    df.drop(columns=["snapshot_ts_key"], inplace=True)

    # 2) 最新标识规范化：若新行 is_new='Y'，将该 ts_code 其他行的 is_new 强制置为 'N'
    if row.iloc[0]["is_new"] == "Y":
        mask_same = df["ts_code"].eq(ts_code)
        df.loc[mask_same, "is_new"] = df.loc[mask_same].apply(
            lambda r: "Y" if (r["in_date"] == row.iloc[0]["in_date"]) else "N", axis=1
        )

    _atomic_write_parquet(df, SW_DATA_FILE)


def _normalize_ts_code(code: str) -> str:
    if not isinstance(code, str):
        return code
    code = code.strip().upper().replace("SHSE", "SH").replace("SZSE", "SZ")
    if "." in code:
        return code
    if len(code) == 6:
        if code.startswith(("6","9")):
            return f"{code}.SH"
        else:
            return f"{code}.SZ"
    return code


def _attach_sw_industry_columns(ts_code: str, df: pd.DataFrame) -> pd.DataFrame:
    try:
        ts_code = _normalize_ts_code(ts_code)
        data = _read_parquet_safe(SW_DATA_FILE)
        if data.empty:
            return df
        row = data[(data["ts_code"] == ts_code) & (data["is_new"] == "Y")]
        if row.empty:
            return df
        row = row.iloc[0]
        df["l1_code"] = row.get("l1_code", "")
        df["l1_name"] = row.get("l1_name", "")
        df["l2_code"] = row.get("l2_code", "")
        df["l2_name"] = row.get("l2_name", "")
        df["l3_code"] = row.get("l3_code", "")
        df["l3_name"] = row.get("l3_name", "")
        df["sw_in_date"]  = row.get("in_date", "")
        df["sw_out_date"] = row.get("out_date", "")
    except Exception as e:
        logging.debug("[SW] 附加行业列失败 %s: %s", ts_code, e)
    return df


_SW_IO_LOCK = threading.Lock()
def _fetch_and_persist_sw_once(ts_code: str):
    """仅在 sw_industry.parquet 中没有该 ts_code 的“最新行”时，才打一次 index_member_all。"""
    ts_code = _normalize_ts_code(ts_code)
    # 先看本地是否已有最新
    with _SW_IO_LOCK:
        df_local = _read_parquet_safe(SW_DATA_FILE)
        has_latest = (
            not df_local.empty and
            (df_local["ts_code"] == ts_code).any() and
            (df_local.loc[df_local["ts_code"] == ts_code, "is_new"] == "Y").any()
        )
    if has_latest:
        return  # 已有最新，跳过 API

    # 拉一次 TuShare（默认 is_new='Y' 即最新；要历史可去掉 is_new 参数）
    df_sw = _retry(lambda: pro.index_member_all(ts_code=ts_code), f"index_member_all_{ts_code}")
    if df_sw is None or df_sw.empty:
        return

    # 只保留必要列并落盘（逐行 append；persist_sw_to_parquet 内部做了“同 ts_code+in_date 去重 + 最新行归一化”）
    cols = ["l1_code","l1_name","l2_code","l2_name","l3_code","l3_name",
            "ts_code","in_date","out_date","is_new"]
    df_sw = df_sw[[c for c in cols if c in df_sw.columns]].copy()

    for _, r in df_sw.iterrows():
        persist_sw_to_parquet(ts_code, {
            "sw_l1_code": r.get("l1_code",""),
            "sw_l1_name": r.get("l1_name",""),
            "sw_l2_code": r.get("l2_code",""),
            "sw_l2_name": r.get("l2_name",""),
            "sw_l3_code": r.get("l3_code",""),
            "sw_l3_name": r.get("l3_name",""),
            "sw_in_date": r.get("in_date",""),
            "sw_out_date": r.get("out_date",""),
            "sw_is_new":  r.get("is_new","Y") or "Y",
        })

# =======================================================================

def _parse_indicators(arg: str):
    if not arg:
        logging.debug('[BRANCH] def _parse_indicators | IF not arg -> taken')
        return []
    if arg.lower().strip() == "all":
        logging.debug('[BRANCH] def _parse_indicators | IF arg.lower().strip() == "all" -> taken')
        return list(ind.REGISTRY.keys())
    return [x.strip() for x in arg.split(",") if x.strip()]


def _decide_symbol_adj_for_fast_init() -> str:
    # Prefer per-thread override set by _with_api_adj; fallback to global API_ADJ
    try:
        aj = getattr(_TLS, "adj_override", None)
        if aj:
            return aj
    except Exception:
        pass
    return (API_ADJ).lower()
def _maybe_compact(dirpath: str):
    mode = str(DUCKDB_ENABLE_COMPACT_AFTER).lower()
    if mode in ("false", "0", "off", "none"):
        logging.debug('[BRANCH] def _maybe_compact | IF mode in ("false", "0", "off", "none") -> taken')
        return
    # "if_needed" 的判定：仅当某些日期的 part 数超过阈值才做
    if mode in ("if_needed", "auto"):
        # 粗略检查：命中任何一个 trade_date 目录超过阈值就触发
        logging.debug('[BRANCH] def _maybe_compact | IF mode in ("if_needed", "auto") -> taken')
        over = False
        for d in os.listdir(dirpath):
            p = os.path.join(dirpath, d)
            if d.startswith("trade_date=") and os.path.isdir(p):
                logging.debug('[BRANCH] def _maybe_compact | IF d.startswith("trade_date=") and os.path.isdir(p) -> taken')
                cnt = sum(1 for x in os.listdir(p) if x.endswith(".parquet"))
                if cnt > COMPACT_MAX_FILES_PER_DATE:
                    logging.debug('[BRANCH] def _maybe_compact | IF cnt > COMPACT_MAX_FILES_PER_DATE -> taken')
                    over = True
                    break
        if not over:
            logging.debug('[BRANCH] def _maybe_compact | IF not over -> taken')
            return
    # 其余视作强制压实
    compact_daily_partitions(base_dir=dirpath)


def _update_fast_init_cache(ts_code: str, df: pd.DataFrame, adj: str):
    """
    将 df 合并进 fast_init_symbol/<adj>/<ts_code>.parquet，按 trade_date 去重。
    adj: 'raw' | 'qfq' | 'hfq' 的语义对应目录 raw/qfq/hfq
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
        logging.debug('[BRANCH] def _update_fast_init_cache | IF os.path.exists(fpath) -> taken')
        try:
            old = pd.read_parquet(fpath)
            old = normalize_trade_date(old)
            both = pd.concat([old, df2], ignore_index=True)
            both = both.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
            both.to_parquet(fpath, index=False)
            return
        except Exception as e:
            logging.warning("[FAST_CACHE] 读取旧缓存失败 %s -> 覆盖写新: %s", ts_code, e)
    df2.to_parquet(fpath, index=False)


def _WRITE_SYMBOL_INDICATORS(ts_code: str, df: pd.DataFrame, end_date: str, prewarmed: bool = False):
    """
    把该 ts_code 的 DataFrame 计算指标后写入 by_symbol 成品(带 warm-up 增量)。
    要求 df 至少包含: trade_date, open, high, low, close, vol[, amount, pre_close]
    """
    if not (WRITE_SYMBOL_PLAIN or WRITE_SYMBOL_INDICATORS):
        logging.debug('[BRANCH] def _WRITE_SYMBOL_INDICATORS | IF not (WRITE_SYMBOL_PLAIN or WRITE_SYMBOL_INDICATORS) -> taken')
        return
        # 1) 选择输出目录（按你要求的命名）
    adj = _decide_symbol_adj_for_fast_init()  # 'raw' | 'qfq' | 'hfq'
    base_dir = DATA_ROOT

    single_plain_dir = os.path.join(base_dir, "stock", "single", f"single_{adj}")
    single_ind_dir   = os.path.join(base_dir, "stock", "single", f"single_{adj}_indicators")
    single_plain_dir_csv = os.path.join(base_dir, "stock", "single", "csv", adj)
    single_ind_dir_csv   = os.path.join(base_dir, "stock", "single", "csv", f"{adj}_indicators")
    if WRITE_SYMBOL_PLAIN:
        logging.debug('[BRANCH] def _WRITE_SYMBOL_INDICATORS | IF WRITE_SYMBOL_PLAIN -> taken')
        os.makedirs(single_plain_dir, exist_ok=True)
        os.makedirs(single_plain_dir_csv, exist_ok=True)
    if WRITE_SYMBOL_INDICATORS:
        logging.debug('[BRANCH] def _WRITE_SYMBOL_INDICATORS | IF WRITE_SYMBOL_INDICATORS -> taken')
        os.makedirs(single_ind_dir, exist_ok=True)
        os.makedirs(single_ind_dir_csv, exist_ok=True)

    plain_parquet = os.path.join(single_plain_dir, f"{ts_code}.parquet")
    plain_csv     = os.path.join(single_plain_dir_csv, f"{ts_code}.csv")
    ind_out_path_parquet = os.path.join(single_ind_dir, f"{ts_code}.parquet")
    ind_out_path_csv     = os.path.join(single_ind_dir_csv, f"{ts_code}.csv")

    
    # 2) 规范、排序、去重
    df2 = df.copy()
    if "trade_date" not in df2.columns:
        logging.debug('[BRANCH] def _WRITE_SYMBOL_INDICATORS | IF "trade_date" not in df2.columns -> taken')
        raise ValueError("df 缺少 trade_date 列")
    
    df2 = df2.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    df2 = normalize_trade_date(df2)
    # —— 统一数值化，确保可计算
    for c in ["open","high","low","close","vol","amount","pre_close","change","pct_chg"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # —— 无条件从 close 推导（覆盖空值/错误值）
    df2 = df2.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    df2["pre_close"] = df2["close"].shift(1)
    df2["change"]    = df2["close"] - df2["pre_close"]
    df2["pct_chg"]   = (df2["change"] / (df2["pre_close"] + EPS)) * 100.0

    # 审计日志（可留可去）
    try:
        logging.debug("[PRODUCT][%s] 价格三列回填检查: head=%s tail=%s",
                      ts_code,
                      df2[["trade_date","pre_close","change","pct_chg"]].head(2).to_dict("records"),
                      df2[["trade_date","pre_close","change","pct_chg"]].tail(2).to_dict("records"))
    except Exception:
        pass

    price_cols = ["open", "high", "low", "close", "pre_close", "change"]
    # 只保留常用基础行情列（存在即取），作为不带指标的成品
    base_cols = ["ts_code","trade_date","open","high","low","close","pre_close","change","pct_chg","vol","amount"]
    cols = [c for c in base_cols if c in df2.columns]
    df_plain = df2[cols].copy() if cols else df2.copy()

    # ---------- plain：不带指标 ----------
    old_plain = None
    try:
        if os.path.exists(plain_parquet):
            old_plain = pd.read_parquet(plain_parquet)
        elif os.path.exists(plain_csv):
            old_plain = pd.read_csv(plain_csv, dtype=str)            
            for c in ["open","high","low","close","pre_close","change","pct_chg","vol","amount"]:
                if c in old_plain.columns:
                    old_plain[c] = pd.to_numeric(old_plain[c], errors="coerce")
    except Exception as e:
        logging.warning("[PRODUCT][%s] 读取旧 plain 失败，按覆盖写: %s", ts_code, e)
        old_plain = None

    merged_plain = df_plain
    if isinstance(old_plain, pd.DataFrame) and not old_plain.empty:
        logging.debug('[BRANCH] def _WRITE_SYMBOL_INDICATORS | IF isinstance(old_plain, pd.DataFrame) and not old_plain.empty -> taken')
        old_plain = normalize_trade_date(old_plain)
        merged_plain = pd.concat([old_plain, df_plain], ignore_index=True)
        merged_plain = merged_plain.sort_values("trade_date").drop_duplicates("trade_date", keep="last")

    merged_plain = merged_plain.sort_values("trade_date").reset_index(drop=True)
    merged_plain["pre_close"] = merged_plain["close"].shift(1)
    merged_plain["change"]    = merged_plain["close"] - merged_plain["pre_close"]
    merged_plain["pct_chg"]   = (merged_plain["change"] / (merged_plain["pre_close"] + EPS)) * 100.0

    for col in ["open","high","low","close","pre_close","change"]:
        if col in merged_plain.columns and pd.api.types.is_numeric_dtype(merged_plain[col]):
            merged_plain[col] = merged_plain[col].round(2)

    # --- 附加申万行业（从本地缓存/akshare 自动构建） ---
    merged_plain = _attach_sw_industry_columns(ts_code, merged_plain)

    if "parquet" in SYMBOL_PRODUCT_FORMATS.get("plain", []):
        logging.debug('[BRANCH] def _WRITE_SYMBOL_INDICATORS | IF "parquet" in SYMBOL_PRODUCT_FORMATS.get("plain", []) -> taken')
        merged_plain.to_parquet(plain_parquet, index=False, engine=PARQUET_ENGINE)
    if "csv" in SYMBOL_PRODUCT_FORMATS.get("plain", []):
        logging.debug('[BRANCH] def _WRITE_SYMBOL_INDICATORS | IF "csv" in SYMBOL_PRODUCT_FORMATS.get("plain", []) -> taken')
        merged_plain.to_csv(plain_csv, index=False, encoding="utf-8-sig")

    if WRITE_SYMBOL_INDICATORS:
        logging.debug("[PRODUCT][%s] 写带指标成品… prewarmed=%s", ts_code, prewarmed)
        ind_out_path_parquet = os.path.join(single_ind_dir, f"{ts_code}.parquet")
        ind_out_path_csv     = os.path.join(single_ind_dir_csv, f"{ts_code}.csv")

        old_ind = None
        old_last = None
        if os.path.exists(ind_out_path_parquet) or os.path.exists(ind_out_path_csv):
            try:
                old_ind = pd.read_parquet(ind_out_path_parquet) if os.path.exists(ind_out_path_parquet) \
                          else pd.read_csv(ind_out_path_csv, dtype=str)
                if isinstance(old_ind, pd.DataFrame) and not old_ind.empty:
                    old_ind = normalize_trade_date(old_ind)
                    old_last = pd.to_datetime(old_ind["trade_date"].astype(str), errors="coerce").max()
            except Exception as e:
                logging.warning("[PRODUCT][%s] 读取旧(ind)失败：%s -> 本次按无旧处理", ts_code, e)
                old_ind, old_last = None, None

        # 无增量直接跳过
        new_last = pd.to_datetime(df2["trade_date"].astype(str), errors="coerce").max()
        if old_last is not None and (new_last is not None) and new_last <= old_last:
            logging.debug("[PRODUCT][%s] 无新增 trade_date（old_last=%s, new_last=%s），跳过指标重算。", 
                         ts_code, old_last.strftime("%Y%m%d"), new_last.strftime("%Y%m%d"))
            return

        # 计算指标：两条路径
        names = (SYMBOL_PRODUCT_INDICATORS or "all")
        names = list(ind.REGISTRY.keys()) if str(names).lower() == "all" else [x.strip() for x in str(names).split(",") if x.strip()]
        if prewarmed:
            df2 = normalize_trade_date(df2).sort_values("trade_date")
            start_dt = df2["trade_date"].min()

            # 选一段旧成品(或旧 plain)作 warm-up；用配置里的天数
            WARM = max(120, SYMBOL_PRODUCT_WARMUP_DAYS)  # 建议 ≥150 更稳
            tail_old = None
            if isinstance(old_ind, pd.DataFrame) and not old_ind.empty:
                old_ind = normalize_trade_date(old_ind)
                tail_old = old_ind[old_ind["trade_date"] < start_dt].tail(WARM)
            # 若没有旧 ind，就用 old_plain 的尾巴（如你前面已读到 old_plain）
            if (tail_old is None or tail_old.empty) and isinstance(old_plain, pd.DataFrame) and not old_plain.empty:
                old_plain = normalize_trade_date(old_plain)
                tail_old = old_plain[old_plain["trade_date"] < start_dt].tail(WARM)

            # 参与计算的输入：旧尾巴 + 新增窗口
            if tail_old is not None and not tail_old.empty:
                calc_in = pd.concat([tail_old, df2], ignore_index=True)
            else:
                calc_in = df2.copy()

            # 一起算 → 再裁掉 warm-up
            calc_all = ind.compute(calc_in, names)
            # round 指标列
            decs = ind.outputs_for(names)
            for col, n in decs.items():
                if col in calc_all.columns and pd.api.types.is_numeric_dtype(calc_all[col]):
                    calc_all[col] = calc_all[col].round(n)
            for col in ["open","high","low","close","pre_close","change","pct_chg","vol","amount"]:
                if col in calc_all.columns and pd.api.types.is_numeric_dtype(calc_all[col]):
                    calc_all[col] = calc_all[col].round(2)

            # 仅保留真正新增区间
            calc_df = calc_all[calc_all["trade_date"] >= start_dt].copy()

            # 和老段拼起来写出
            if isinstance(old_ind, pd.DataFrame) and not old_ind.empty:
                keep_old = old_ind[old_ind["trade_date"] < start_dt].copy()
                warm_df  = pd.concat([keep_old, calc_df], ignore_index=True)
            else:
                warm_df = calc_df
        else:
            # 原有 warm-up 逻辑保持不变（向后取窗口再整段计算）
            warm_df = df2.copy()
            try:
                if isinstance(old_ind, pd.DataFrame) and not old_ind.empty:
                    old_td = pd.to_datetime(old_ind["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
                    last   = old_td.max()
                    warmup_start = (last - pd.Timedelta(days=SYMBOL_PRODUCT_WARMUP_DAYS))
                    keep_old = old_ind.loc[old_td < warmup_start].copy()
                    keep_old = normalize_trade_date(keep_old)
                    new_part = df2.loc[pd.to_datetime(df2["trade_date"].astype(str), format="%Y%m%d", errors="coerce") >= warmup_start].copy()
                    new_part = normalize_trade_date(new_part)
                    warm_df  = pd.concat([keep_old, new_part], ignore_index=True)
            except Exception as e:
                logging.warning("[PRODUCT][%s] warm-up 读取旧失败，按全量重算：%s", ts_code, e)
            # 计算
            warm_df = ind.compute(warm_df, names)
            decs = ind.outputs_for(names)
            for col, n in decs.items():
                if col in warm_df.columns and pd.api.types.is_numeric_dtype(warm_df[col]):
                    warm_df[col] = warm_df[col].round(n)
            for col in price_cols:
                if col in warm_df.columns and pd.api.types.is_numeric_dtype(warm_df[col]):
                    warm_df[col] = warm_df[col].round(2)

        warm_df = normalize_trade_date(warm_df)
        warm_df = warm_df.sort_values("trade_date").drop_duplicates("trade_date", keep="last")

        # --- 附加申万行业（从本地缓存/akshare 自动构建） ---
        warm_df = _attach_sw_industry_columns(ts_code, warm_df)
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
                logging.debug('[BRANCH] def _rate_limit | IF len(_CALL_TS) < CALLS_PER_MIN -> taken')
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
                logging.debug('[BRANCH] def _retry | IF attempt == retries -> taken')
                break  # 出循环抛错
            # 计算基础等待
            if attempt <= len(RETRY_DELAY_SEQUENCE):
                logging.debug('[BRANCH] def _retry | IF attempt <= len(RETRY_DELAY_SEQUENCE) -> taken')
                base_wait = RETRY_DELAY_SEQUENCE[attempt - 1]
            else:
                logging.debug('[BRANCH] def _retry | ELSE of IF attempt <= len(RETRY_DELAY_SEQUENCE) -> taken')
                base_wait = RETRY_DELAY_SEQUENCE[-1]
            # 抖动
            jitter = random.uniform(RETRY_JITTER_RANGE[0], RETRY_JITTER_RANGE[1])
            wait_sec = max(0.1, base_wait + jitter)
            if RETRY_LOG_LEVEL.upper() == "INFO":
                logging.debug('[BRANCH] def _retry | IF RETRY_LOG_LEVEL.upper() == "INFO" -> taken')
                logging.info("%s 重试前等待 %.2fs (base=%ds, jitter=%.2f)",
                             desc, wait_sec, base_wait, jitter)
            else:
                logging.debug('[BRANCH] def _retry | ELSE of IF RETRY_LOG_LEVEL.upper() == "INFO" -> taken')
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
        logging.debug('[BRANCH] def _trade_dates | IF cal.empty -> taken')
        raise RuntimeError(f"trade_cal 返回为空({start}~{end})")
    return cal["cal_date"].astype(str).tolist()


def _last_partition_date(root: str) -> Optional[str]:
    if not os.path.exists(root):
        logging.debug('[BRANCH] def _last_partition_date | IF not os.path.exists(root) -> taken')
        return None
    dates = [d.split("=")[-1] for d in os.listdir(root) if d.startswith("trade_date=")]
    return max(dates) if dates else None


def _save_partition(df: pd.DataFrame, root: str):
    if df is None or df.empty:
        logging.debug('[BRANCH] def _save_partition | IF df is None or df.empty -> taken')
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
        logging.debug('[BRANCH] def compact_daily_partitions | IF base_dir is not None -> taken')
        daily_dir = base_dir
    else:
        logging.debug('[BRANCH] def compact_daily_partitions | ELSE of IF base_dir is not None -> taken')
        candidates = [
            os.path.join(DATA_ROOT, "stock", "daily", "daily"),
            os.path.join(DATA_ROOT, "stock", "daily"),
        ]
        daily_dir = next((p for p in candidates if os.path.isdir(p)), candidates[0])

    if not os.path.isdir(daily_dir):
        logging.debug('[BRANCH] def compact_daily_partitions | IF not os.path.isdir(daily_dir) -> taken')
        logging.warning("[COMPACT] daily 目录不存在，跳过")
        return

    # 找出 trade_date=XXXX 目录
    date_dirs = [d for d in os.listdir(daily_dir)
                 if d.startswith("trade_date=") and os.path.isdir(os.path.join(daily_dir, d))]

    if not date_dirs:
        logging.debug('[BRANCH] def compact_daily_partitions | IF not date_dirs -> taken')
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
            logging.debug('[BRANCH] def compact_daily_partitions | IF len(parts) <= COMPACT_MAX_FILES_PER_DATE -> taken')
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


def read_tail_from_single(ts_code: str, start_date: int, end_date: int,
                          root_single_dir: str, columns: list[str]):
    """
    从按股票存储的 Parquet 读取指定日期窗口与列。
    start_date/end_date 形如 20240101 的 int。
    """
    path = os.path.join(root_single_dir, f"{ts_code}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)

    # 懒加载/缺库兜底：环境缺少 pyarrow 时优雅退出
    if ds is None or pa is None:
        raise RuntimeError("需要 pyarrow 才能从单票 Parquet 读取（请先安装 pyarrow）")

    dataset = ds.dataset(path, format="parquet")
    filt = (ds.field("trade_date") >= pa.scalar(start_date, pa.int32())) & \
           (ds.field("trade_date") <= pa.scalar(end_date,   pa.int32()))
    table = dataset.to_table(filter=filt, columns=columns, use_threads=True)
    return table.to_pandas(types_mapper=pd.ArrowDtype)


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
        logging.debug('[BRANCH] def _need_duck_merge | IF last_parquet is None -> taken')
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
        logging.debug('[BRANCH] def _need_duck_merge | IF last_duck is not None -> taken')
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
        logging.debug('[BRANCH] def _need_duck_merge | ELSE of IF last_duck is not None -> taken')
        new_rows = 0

    # ① 日期差 ≥ N 天
    if last_duck is None:
        logging.debug('[BRANCH] def _need_duck_merge | IF last_duck is None -> taken')
        return True                             # duckdb 还没建过
    lp = dt.datetime.strptime(str(last_parquet), "%Y%m%d")
    ld = dt.datetime.strptime(str(last_duck), "%Y%m%d")
    day_lag = (lp - ld).days
    if day_lag >= DUCK_MERGE_DAY_LAG:
        logging.debug('[BRANCH] def _need_duck_merge | IF day_lag >= DUCK_MERGE_DAY_LAG -> taken')
        return True

    # ② 或新增行数 ≥ 阈值
    return new_rows >= DUCK_MERGE_MIN_ROWS


# === 批量增量缓存器：一次性把所有股票的增量读入内存，避免逐股 parquet_scan ===
def _build_cutoff_df(ts_codes: list[str], target_adj: str | None = None) -> pd.DataFrame:
    if target_adj is None:
        target_adj = _decide_symbol_adj_for_fast_init()  # 'raw'/'qfq'/'hfq'

    """
    生成每只股票的 'last_date'（single 尾部最后交易日）。
    优先从 Parquet 读 trade_date 列的最后若干行；没有 single 文件者标记为 None。
    """
    rows = []
    base_single_parquet_dir = os.path.join(DATA_ROOT, "stock", "single", f"single_{target_adj}")
    base_single_csv_dir = os.path.join(DATA_ROOT, "stock", "single", "csv", target_adj)

    for ts in ts_codes:
        last_date = None
        p_parquet = os.path.join(base_single_parquet_dir, f"{ts}.parquet")
        p_csv = os.path.join(base_single_csv_dir, f"{ts}.csv")
        try:
            if os.path.exists(p_parquet):
                # 只读 trade_date 列 + 取尾部（避免全表载入）
                df_tail = pd.read_parquet(p_parquet, columns=["trade_date"])
                if df_tail is not None and not df_tail.empty:
                    last_date = pd.to_datetime(df_tail["trade_date"].astype(str), errors="coerce").max()
            elif os.path.exists(p_csv):
                # CSV 慢，尽量避免（建议把 single 都转成 parquet）
                df_tail = pd.read_csv(p_csv, usecols=["trade_date"])
                if df_tail is not None and not df_tail.empty:
                    last_date = pd.to_datetime(df_tail["trade_date"].astype(str), errors="coerce").max()
        except Exception as e:
            logging.debug("[INC_IND] 读取 single 尾部失败 %s：%s", ts, e)

        rows.append({
            "ts_code": ts,
            "last_date": None if last_date is None or pd.isna(last_date) else last_date.strftime("%Y%m%d")
        })

    cutoff = pd.DataFrame(rows)
    # 拆分是否有基线日
    cutoff["has_single"] = cutoff["last_date"].notna()
    return cutoff


class BatchIncCache:
    """
    把本轮所有股票的增量一次性查出来缓存：
      1) 计算每股 last_date；
      2) 以 min(last_date)+1 作为全局下界，扫最近分区；
      3) 用 (ts_code, last_date) JOIN 进行逐股过滤；
      4) 结果按 ts_code 切片缓存为 dict。
    """
    def __init__(self, ts_codes: list[str], target_adj: str | None = None):
        self.ts_codes = ts_codes
        self.target_adj = target_adj or _decide_symbol_adj_for_fast_init()
        self.cutoff = _build_cutoff_df(ts_codes, self.target_adj)

        self.inc_dict: dict[str, pd.DataFrame] = {}
        self._loaded = False

    def load_once(self):
        if self._loaded:
            return
        # 对于没有 single 的股票，先不进“增量批量通道”，避免把全历史拉进来
        have_base = self.cutoff[self.cutoff["has_single"]].copy()
        if have_base.empty:
            self._loaded = True
            return

        min_last = have_base["last_date"].min()
        # 增量下界 = min(last_date)+1
        start_inc = (pd.to_datetime(min_last) + pd.Timedelta(days=1)).strftime("%Y%m%d")

        # 注册到 DuckDB
        loc = duckdb.connect()
        try:
            loc.execute(f"PRAGMA threads={max(1, DUCKDB_THREADS)};")
            loc.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
            os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
            loc.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")
            loc.execute("PRAGMA enable_object_cache;")  # 允许文件元数据缓存

            glob_src = os.path.join(
                DATA_ROOT, "stock", "daily", f"daily_{self.target_adj}", "trade_date=*/part-*.parquet"
            ).replace("\\", "/")

            # 只扫最近分区 + 逐股 last_date 过滤
            loc.register("cutoff_df", have_base)

            # 只取指标需要的基础列（按需增删）
            base_cols = ["ts_code","trade_date","open","high","low","close","pre_close","change","pct_chg","vol","amount"]
            cols_sql = ", ".join(f"p.{c}" for c in base_cols)

            inc_all = loc.sql(f"""
                SELECT {cols_sql}
                FROM parquet_scan('{glob_src}', hive_partitioning=1) AS p
                JOIN cutoff_df AS c ON p.ts_code = c.ts_code
                WHERE CAST(p.trade_date AS BIGINT) >= CAST('{start_inc}' AS BIGINT)  -- 强制同型比较
                  AND CAST(p.trade_date AS BIGINT) >  CAST(c.last_date AS BIGINT)    -- 逐股精确过滤
            """).df()

            if inc_all is None or inc_all.empty:
                self._loaded = True
                return

            # 规范化
            inc_all["trade_date"] = pd.to_datetime(inc_all["trade_date"].astype(str), errors="coerce").dt.strftime("%Y%m%d")
            inc_all = inc_all.sort_values(["ts_code","trade_date"])

            # 切片缓存
            for ts, g in inc_all.groupby("ts_code", sort=False):
                self.inc_dict[ts] = g.reset_index(drop=True)

            self._loaded = True
        finally:
            try:
                loc.close()
            except Exception:
                pass

    def get_inc(self, ts_code: str) -> Optional[pd.DataFrame]:
        self.load_once()
        return self.inc_dict.get(ts_code)


def _recalc_increment_inmem(ts_list, last_duck_str: str, end: str, threads: int = 0):
    """
    一次性把本次涉及股票 + 各自 warm-up 窗口的数据拉到内存；
    在内存里 groupby(ts_code) 统一计算指标；然后：
      - 写回 single_* 与 single_*_indicators（prewarmed=True，跳过旧读）
      - 增量 COPY 到按日分区（subset）
    """
    import duckdb, os, pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    target_adj = _decide_symbol_adj_for_fast_init()     # 'raw'/'qfq'/'hfq'
    src_dir = os.path.join(DATA_ROOT, "stock", "daily", f"daily_{target_adj}")
    dst_dir = os.path.join(DATA_ROOT, "stock", "daily", f"daily_{target_adj}_indicators")

    if not os.path.isdir(src_dir):
        logging.warning("[INC_IND][MEM] 源目录不存在：%s", src_dir)
        return

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
        con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
        os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")
        con.execute("PRAGMA enable_object_cache;")

        # ① 估每股 last_date（只从“指标分区”回看近 N 天，避免全库扫）
        con.register("ts_list", pd.DataFrame({"ts_code": ts_list}))
        glob_dst = os.path.join(dst_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
        cutoff_back = int(INC_INMEM_CUTOFF_BACK_DAYS or 365)
        _base = pd.to_datetime(str(last_duck_str), format="%Y%m%d", errors="coerce")
        if pd.isna(_base):  # 指标分区尚未建成，或拿到'00000000'
            _base = pd.Timestamp("2005-01-01")
        cutoff_start = (_base - pd.Timedelta(days=cutoff_back)).strftime("%Y%m%d")
        # 统一成整数，避免 DuckDB 的 BIGINT vs VARCHAR 比较
        _cutoff_i = int(cutoff_start)
        _end_i    = int(end)

        # （可选防呆）
        if _cutoff_i > _end_i:
            _cutoff_i = _end_i
        try:
            # —— df_cut 的 SQL：改为整数比较 ——  （原来用 d.trade_date >= '{cutoff_start}'）
            df_cut = con.sql(f"""
                SELECT d.ts_code, max(d.trade_date) AS last_date
                FROM parquet_scan('{glob_dst}', hive_partitioning=1) AS d
                JOIN ts_list t ON d.ts_code = t.ts_code
                WHERE CAST(d.trade_date AS BIGINT) >= {_cutoff_i}
                GROUP BY d.ts_code
            """).df()
            
        except duckdb.Error:
            df_cut = pd.DataFrame(columns=["ts_code","last_date"])

        cutoff = pd.DataFrame({"ts_code": ts_list})
        mp = dict(zip(df_cut["ts_code"], df_cut["last_date"]))
        cutoff["last_date"] = cutoff["ts_code"].map(mp).fillna(0).astype(int)

        # 每股 warm-up 下界（last_date - SYMBOL_PRODUCT_WARMUP_DAYS - PAD）
        pad = int(INC_INMEM_PADDING_DAYS or 5)
        cutoff["warm_lower"] = (
            pd.to_datetime(cutoff["last_date"].astype(str).str.zfill(8), errors="coerce")
            .fillna(pd.Timestamp("2005-01-01"))
            - pd.Timedelta(days=SYMBOL_PRODUCT_WARMUP_DAYS + pad)
        ).dt.strftime("%Y%m%d").astype(int)
        con.register("cutoff_df", cutoff)

        # ② 只读本次涉及股票 + 各自窗口的基础列（DuckDB 一把拉进内存）
        glob_src = os.path.join(src_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
        base_cols = ["ts_code","trade_date","open","high","low","close","vol","amount","pre_close","pct_chg","change"]
        cols_sql = ", ".join(f"p.{c}" for c in base_cols)
        # —— df_all 的 SQL：两端统一按整数比较 ——  （原来 warm_lower 是 str，end 也是 str）
        df_all = con.sql(f"""
            SELECT {cols_sql}
            FROM parquet_scan('{glob_src}', hive_partitioning=1) AS p
            JOIN cutoff_df AS c ON p.ts_code = c.ts_code
            WHERE CAST(p.trade_date AS BIGINT) >= c.warm_lower
            AND CAST(p.trade_date AS BIGINT) <= {_end_i}
        """).df()

    finally:
        try:
            con.close()
        except Exception:
            pass

    if df_all.empty:
        logging.info("[INC_IND][MEM] 本次无新增数据。")
        return

    df_all["trade_date"] = pd.to_datetime(df_all["trade_date"].astype(str), errors="coerce").dt.strftime("%Y%m%d")
    df_all = df_all.sort_values(["ts_code","trade_date"]).reset_index(drop=True)

    workers = max(1, int(threads or (os.cpu_count() or 4) - 1))
    logging.info("[INC_IND][MEM] 使用线程数：%d", workers)

    def _one(ts: str, g: pd.DataFrame):
        try:
            _with_api_adj(target_adj, _WRITE_SYMBOL_INDICATORS, ts, g, end, prewarmed=True)
            _update_fast_init_cache(ts, g, target_adj)
            return ts, None
        except Exception as e:
            return ts, str(e)

    futs = []
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ts, g in df_all.groupby("ts_code", sort=False):
            futs.append(ex.submit(_one, ts, g))
        for f in tqdm(as_completed(futs), total=len(futs), dynamic_ncols=True, desc="[INC_IND][MEM] 重算"):
            ts, err = f.result()
            if err:
                logging.warning("[INC_IND][MEM][%s] 失败：%s", ts, err); fail += 1
            else:
                ok += 1
    logging.info("[INC_IND][MEM] 单股写入完成：OK=%d FAIL=%d", ok, fail)

    # ④ 增量 COPY 到 <adj>_indicators（只合并这批股票）
    if WRITE_SYMBOL_INDICATORS and ts_list:
        _with_api_adj(target_adj, duckdb_merge_symbol_products_to_daily_subset, ts_list)


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
        logging.debug('[BRANCH] def sync_index_daily_fast | IF actual_start > end -> taken')
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
                logging.debug('[BRANCH] def sync_index_daily_fast > def write_partition | IF dt < actual_start or dt > end -> taken')
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
                logging.debug('[BRANCH] def sync_index_daily_fast > def fetch_one | IF df is None or df.empty -> taken')
                df = _retry(lambda: call_daily(), f"index_daily_{code}")
            if df is None or df.empty:
                logging.debug('[BRANCH] def sync_index_daily_fast > def fetch_one | IF df is None or df.empty -> taken')
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
            if st == "ok": logging.debug('[BRANCH] def sync_index_daily_fast | IF st == "ok" -> taken'); ok += 1
            elif st == "empty": logging.debug('[BRANCH] def sync_index_daily_fast | IF st == "empty" -> taken'); empty += 1
            else: logging.debug('[BRANCH] def sync_index_daily_fast | ELSE of IF st == "empty" -> taken'); err += 1
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
        logging.debug('[BRANCH] def sync_stock_daily_fast | IF actual_start > end -> taken')
        logging.info("[stock][FAST] 日线已最新 (last=%s)", last)
        return

    # 文件名并发写入时做轻量互斥，避免极端时间戳撞名
    lock_io = threading.Lock()
    ok = empty = err = 0

    def write_partition(df: pd.DataFrame):
        for dt, sub in df.groupby("trade_date"):
            if dt < actual_start or dt > end:
                logging.debug('[BRANCH] def sync_stock_daily_fast > def write_partition | IF dt < actual_start or dt > end -> taken')
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
                logging.debug('[BRANCH] def sync_stock_daily_fast > def fetch_one | IF df is None or df.empty -> taken')
                all_df = []
                for d in _trade_dates(actual_start, end):  # 你已有交易日获取。
                    try:
                        df_d = _retry(lambda dd=d: pro.daily(trade_date=dd, ts_code=ts_code),
                                    f"stock_daily_{ts_code}_{d}")
                        if df_d is not None and not df_d.empty:
                            logging.debug('[BRANCH] def sync_stock_daily_fast > def fetch_one | IF df_d is not None and not df_d.empty -> taken')
                            all_df.append(df_d)
                    except Exception:
                        continue
                df = pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()
            if df is None or df.empty:
                logging.debug('[BRANCH] def sync_stock_daily_fast > def fetch_one | IF df is None or df.empty -> taken')
                return ts_code, "empty"
            df = df.sort_values("trade_date")
            # —— 统一数值化（仅转型，不计算三列）
            for c in ["open","high","low","close","vol","amount","pre_close","change","pct_chg"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # —— 只给“增量段第一天”做 warm-up：从既有日分区里取上一交易日的 close
            last_close = None
            try:
                import duckdb, os
                con = duckdb.connect()
                con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
                con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
                con.execute("PRAGMA preserve_insertion_order=false;")
                os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
                con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}'")

                glob_prev = os.path.join(dst_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
                sql = f"""
                    SELECT max_by(close, trade_date) AS last_close
                    FROM parquet_scan('{glob_prev}')
                    WHERE ts_code = '{ts_code}' AND trade_date < '{actual_start}'
                """
                row = con.sql(sql).fetchone()
                con.close()
                last_close = row[0] if row else None
            except Exception:
                # 查询不到（新股/首次建库）或 parquet 尚不存在时，保持 NaN，语义正确
                last_close = None

            # —— 若有 last_close，则 seed 一行放到最前面（trade_date 取 actual_start 的前一天即可，稍后会丢掉）
            seeded = False
            if (last_close is not None) and not df.empty:
                seed = df.iloc[[0]].copy()
                seed['trade_date'] = (pd.to_datetime(actual_start) - pd.Timedelta(days=1)).strftime('%Y%m%d')   # 或替换为真实上一个交易日
                seed["close"] = last_close
                df = pd.concat([seed, df], ignore_index=True)
                seeded = True

            # —— 统一只计算一次三列
            df["pre_close"] = df["close"].shift(1)
            df["change"]    = df["close"] - df["pre_close"]
            df["pct_chg"]   = (df["change"] / (df["pre_close"] + EPS)) * 100.0

            # —— 丢掉 seed 行（只在 seeded 时）
            if seeded:
                df = df.iloc[1:].copy()

            write_partition(df)

            return ts_code, "ok"
        except Exception as e:
            return ts_code, f"err:{e}"

    with ThreadPoolExecutor(max_workers=threads) as exe:
        futs = {exe.submit(fetch_one, code): code for code in codes}
        pbar = tqdm(as_completed(futs), total=len(futs), desc="股票(日线)并发拉取", ncols=120)
        for fut in pbar:
            code, st = fut.result()
            if st == "ok": logging.debug('[BRANCH] def sync_stock_daily_fast | IF st == "ok" -> taken'); ok += 1
            elif st == "empty": logging.debug('[BRANCH] def sync_stock_daily_fast | IF st == "empty" -> taken'); empty += 1
            else: logging.debug('[BRANCH] def sync_stock_daily_fast | ELSE of IF st == "empty" -> taken'); err += 1
            pbar.set_postfix(ok=ok, empty=empty, err=err)
        pbar.close()

    logging.info("[stock][FAST] 完成 ok=%d empty=%d err=%d 区间=%s~%s 起点=%s",
                 ok, empty, err, start, end, actual_start)


# ========== FAST INIT (按股票多线程) ==========
def _fetch_stock_list() -> pd.DataFrame:
    cache = os.path.join(DATA_ROOT, "stock_list.csv")
    if os.path.exists(cache):
        logging.debug('[BRANCH] def _fetch_stock_list | IF os.path.exists(cache) -> taken')
        return pd.read_csv(cache, dtype=str)
    df = _retry(lambda: pro.stock_basic(exchange='', list_status='L',
                                        fields='ts_code,name,list_date'),
                "stock_basic_full")
    df.to_csv(cache, index=False, encoding='utf-8-sig')
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
            logging.debug('[BRANCH] def fast_init_download > def task | IF os.path.exists(fpath) and CHECK_SKIP_MIN_MAX -> taken')
            need_redownload = False
            min_d = max_d = None
            try:
                df_meta = pd.read_parquet(fpath, columns=CHECK_SKIP_READ_COLUMNS)
                if df_meta.empty:
                    logging.debug('[BRANCH] def fast_init_download > def task | IF df_meta.empty -> taken')
                    need_redownload = True
                else:
                    # 仅用于日志，不参与判定
                    logging.debug('[BRANCH] def fast_init_download > def task | ELSE of IF df_meta.empty -> taken')
                    min_d = str(df_meta.trade_date.min())
                    max_d = str(df_meta.trade_date.max())

                    # ---- 尾部允许滞后判断 ----
                    # 允许滞后 N 天：只要文件最大日期 >= (end_date - N天) 就视为完整
                    lag_threshold = (
                        dt.datetime.strptime(end_date, "%Y%m%d") -
                        dt.timedelta(days=CHECK_SKIP_ALLOW_LAG_DAYS)
                    ).strftime("%Y%m%d")

                    if max_d < lag_threshold:
                        logging.debug('[BRANCH] def fast_init_download > def task | IF max_d < lag_threshold -> taken')
                        need_redownload = True

            except Exception as e:
                need_redownload = True
                logging.warning("Skip 检查读取失败 %s -> 强制重下 (%s)", ts_code, e)

            if not need_redownload:
                # 可选调试
                # logging.debug("[SKIP] %s min=%s max=%s lag_thr=%s", ts_code, min_d, max_d, lag_threshold)
                logging.debug('[BRANCH] def fast_init_download > def task | IF not need_redownload -> taken')
                return (ts_code, 'skip', None)
            else:
                logging.debug('[BRANCH] def fast_init_download > def task | ELSE of IF not need_redownload -> taken')
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
                logging.debug('[BRANCH] def fast_init_download > def task | IF df is None or df.empty -> taken')
                return (ts_code, 'empty', None)
            df = df.sort_values("trade_date")
            df.to_parquet(fpath, index=False)
            try:
                _fetch_and_persist_sw_once(ts_code)
            except Exception as e:
                logging.warning("[SW][%s] 拉取/落盘失败：%s", ts_code, e)

            if API_ADJ != "raw":
                logging.debug('[BRANCH] def fast_init_download > def task | IF API_ADJ != "raw" -> taken')
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
                    logging.debug("[BRANCH] def fast_init_download | IF status == 'ok' -> taken")
                    ok += 1
                elif status == 'skip':
                    logging.debug("[BRANCH] def fast_init_download | IF status == 'skip' -> taken")
                    logging.debug("[BRANCH] def fast_init_download | ELIF status == 'skip' -> taken")
                    skip += 1
                elif status == 'empty':
                    logging.debug("[BRANCH] def fast_init_download | IF status == 'empty' -> taken")
                    logging.debug("[BRANCH] def fast_init_download | ELIF status == 'empty' -> taken")
                    empty += 1
                else:
                    logging.debug("[BRANCH] def fast_init_download | ELSE of IF status == 'empty' -> taken")
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
        logging.debug('[BRANCH] def fast_init_download | IF failed_codes -> taken')
        with open(os.path.join(DATA_ROOT, "fast_init_failed_round1.txt"), "w", encoding="utf-8") as f:
            for c,m in failed:
                f.write(f"{c},{m}\n")

    # ====== 自动失败补抓(一次) ======
    if FAILED_RETRY_ONCE and failed_codes:
        logging.debug('[BRANCH] def fast_init_download | IF FAILED_RETRY_ONCE and failed_codes -> taken')
        logging.info("[FAST_INIT] 等待 %ds 后开始失败补抓，失败数=%d", FAILED_RETRY_WAIT, len(failed_codes))
        time.sleep(FAILED_RETRY_WAIT)

        def retry_task(ts_code: str):
            # fpath = os.path.join(FAST_INIT_STOCK_DIR, f"{ts_code}.parquet")
            fpath = os.path.join(FAST_INIT_STOCK_DIR, API_ADJ, f"{ts_code}.parquet")
            if os.path.exists(fpath):
                logging.debug('[BRANCH] def fast_init_download > def retry_task | IF os.path.exists(fpath) -> taken')
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
                    logging.debug('[BRANCH] def fast_init_download > def retry_task | IF df is None or df.empty -> taken')
                    return (ts_code, 'empty')
                df = df.sort_values("trade_date")
                df.to_parquet(fpath, index=False)
                try:
                    _fetch_and_persist_sw_once(ts_code)
                except Exception as e:
                    logging.warning("[SW][%s] 拉取/落盘失败：%s", ts_code, e)

                if API_ADJ != "raw":
                    _WRITE_SYMBOL_INDICATORS(ts_code, df, end_date)
                if API_ADJ != "raw":
                    logging.debug('[BRANCH] def fast_init_download > def retry_task | IF API_ADJ != "raw" -> taken')
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
                    logging.debug("[BRANCH] def fast_init_download | IF st == 'ok' -> taken")
                    ok2 += 1
                elif st == 'empty':
                    logging.debug("[BRANCH] def fast_init_download | IF st == 'empty' -> taken")
                    logging.debug("[BRANCH] def fast_init_download | ELIF st == 'empty' -> taken")
                    empty2 += 1
                elif st == 'exists':
                    logging.debug("[BRANCH] def fast_init_download | IF st == 'exists' -> taken")
                    logging.debug("[BRANCH] def fast_init_download | ELIF st == 'exists' -> taken")
                    logging.debug("[BRANCH] def fast_init_download | IF st == 'exists' -> taken")
                    logging.debug("[BRANCH] def fast_init_download | ELIF st == 'exists' -> taken")
                    exists2 += 1
                else:
                    logging.debug("[BRANCH] def fast_init_download | ELSE of IF st == 'exists' -> taken")
                    logging.debug("[BRANCH] def fast_init_download | ELSE of IF st == 'exists' -> taken")
                    logging.debug("[BRANCH] def fast_init_download | ELSE of IF st == 'exists' -> taken")
                    logging.debug("[BRANCH] def fast_init_download | ELSE of IF st == 'exists' -> taken")
                    err2 += 1
                pbar2.set_postfix(ok=ok2, empty=empty2, exists=exists2, err=err2)
            pbar2.close()
        logging.info("[FAST_INIT] 补抓完成 ok=%d empty=%d exists=%d err=%d", ok2, empty2, exists2, err2)

        if err2 > 0:
            logging.debug('[BRANCH] def fast_init_download | IF err2 > 0 -> taken')
            final_failed = []
            for c in failed_codes:
                fpath = os.path.join(FAST_INIT_STOCK_DIR, API_ADJ, f"{c}.parquet")
                if not os.path.exists(fpath):
                    logging.debug('[BRANCH] def fast_init_download | IF not os.path.exists(fpath) -> taken')
                    final_failed.append(c)
            with open(os.path.join(DATA_ROOT, "fast_init_failed_final.txt"), "w", encoding="utf-8") as f:
                for c in final_failed:
                    f.write(c + "\n")
            logging.warning("[FAST_INIT] 仍失败股票数=%d -> fast_init_failed_final.txt", len(final_failed))
    else:
        logging.debug('[BRANCH] def fast_init_download | ELSE of IF FAILED_RETRY_ONCE and failed_codes -> taken')
        logging.info("[FAST_INIT] 无需补抓或无失败股票。")

def duckdb_merge_symbol_products_to_daily(batch_days:int=30):
    import duckdb, os
    from math import ceil
    from tqdm import tqdm

    adj = _decide_symbol_adj_for_fast_init()
    src_dir = os.path.join(DATA_ROOT, "stock", "single", f"single_{adj}_indicators")
    dst_dir = os.path.join(DATA_ROOT, "stock", "daily", f"daily_{adj}_indicators")
    if not os.path.isdir(src_dir):
        logging.debug('[BRANCH] def duckdb_merge_symbol_products_to_daily | IF not os.path.isdir(src_dir) -> taken')
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
        glob_dst_sql = glob_dst.replace("'", "''")
        last_duck = con.sql(f"SELECT max(trade_date) FROM parquet_scan('{glob_dst_sql}')").fetchone()[0] or 0
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
        logging.debug('[BRANCH] def duckdb_merge_symbol_products_to_daily | IF df_dates.empty -> taken')
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


def duckdb_merge_symbol_products_to_daily_subset(ts_codes: list[str], batch_days: int = 30):
    """
    只从给定 ts_codes 的单股成品(含指标)里抽取新增日期，合并到 daily_<adj>_indicators。
    减少 read_parquet 的文件枚举范围。
    """
    import duckdb
    from math import ceil
    adj = _decide_symbol_adj_for_fast_init()
    src_dir = os.path.join(DATA_ROOT, "stock", "single", f"single_{adj}_indicators")
    dst_dir = os.path.join(DATA_ROOT, "stock", "daily", f"daily_{adj}_indicators")
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

    # 只拼接这些文件的绝对路径
    files = [
        os.path.join(src_dir, f"{c}.parquet").replace("\\", "/")
        for c in ts_codes
        if os.path.exists(os.path.join(src_dir, f"{c}.parquet"))
    ]
    if not files:
        con.close()
        logging.info("[DUCK MERGE IND][subset] 本次无可合并文件")
        return

    # 先取这些文件的“新增日期清单”
    file_list_sql = ", ".join([f"'{p}'" for p in files])
    df_dates = con.sql(f"""
        SELECT DISTINCT trade_date
        FROM read_parquet([{file_list_sql}])
        WHERE trade_date > '{last_duck_str}'
        ORDER BY trade_date
    """).df()
    if df_dates.empty:
        con.close()
        logging.info("[DUCK MERGE IND][subset] 已最新，跳过")
        return

    dates = df_dates.trade_date.astype(str).tolist()
    total_batches = ceil(len(dates) / batch_days)
    for i in range(total_batches):
        chunk = dates[i*batch_days:(i+1)*batch_days]
        mn, mx = chunk[0], chunk[-1]
        sql = f"""
        COPY (
          SELECT * FROM read_parquet([{file_list_sql}])
          WHERE trade_date BETWEEN '{mn}' AND '{mx}'
        )
        TO '{dst_dir}'
        (FORMAT PARQUET, PARTITION_BY (trade_date), OVERWRITE_OR_IGNORE 1);
        """
        con.execute(sql)

    con.close()
    logging.info("[DUCK MERGE IND][subset] 完成，新增日期数=%d 涉及股票数=%d", len(dates), len(files))


# ====== 增量重算：把新增日期涉及的股票，补齐“按股票成品(含指标)”并合并到按日分区 ======
def _with_api_adj(temp_api_adj: str, fn, *args, **kwargs):
    """
    Temporarily set an *effective* API_ADJ for the current thread using thread-local storage,
    so concurrent threads don't race on the global value.
    """
    prev = getattr(_TLS, "adj_override", None)
    _TLS.adj_override = (temp_api_adj or "").lower()
    try:
        return fn(*args, **kwargs)
    finally:
        if prev is None:
            try:
                delattr(_TLS, "adj_override")
            except Exception:
                pass
        else:
            _TLS.adj_override = prev
def recalc_symbol_products_for_increment(start: str, end: str, threads: int = 0):
    """
    NORMAL(日常增量)的核心补全：
    1) 找出 stock/<target_adj> 新增日期涉及的 ts_code
    2) 以 warm-up 窗口回看，抽取这批 ts_code 的历史数据
    3) 用统一的 _WRITE_SYMBOL_INDICATORS() 重算单股成品(含指标)
    4) 回灌 fast_init 缓存；再合并到 <adj>_indicators 分区
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    target_adj = _decide_symbol_adj_for_fast_init()     # raw / qfq / hfq
    src_dir = os.path.join(DATA_ROOT, "stock", "daily", f"daily_{target_adj}")
    dst_dir = os.path.join(DATA_ROOT, "stock", "daily", f"daily_{target_adj}_indicators")
    if not os.path.isdir(src_dir):
        logging.debug('[BRANCH] def recalc_symbol_products_for_increment | IF not os.path.isdir(src_dir) -> taken')
        logging.warning("[INC_IND] 源目录不存在：%s(可能尚未构建 %s)", src_dir, target_adj)
        return

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP_DIR}';")
    
    # ① 目的端已有最大交易日
    try:
        glob_dst = os.path.join(dst_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
        glob_dst_sql = glob_dst.replace("'", "''")  # 防止路径中含有单引号
        val = con.sql(f"SELECT max(trade_date) FROM parquet_scan('{glob_dst_sql}')").fetchone()[0]
        last_duck = int(val) if val is not None else 0
    except duckdb.Error as e:
        print("DuckDB error (dst max trade_date):", e)
        last_duck = 0

    # 统一成 8 位字符串用于比较（如果 trade_date 是字符串类型）
    last_duck_str = str(last_duck).zfill(8)

    # 如果目的端还没有数据，用源端的最新日 - 1 与 start 取最大值做下限
    if last_duck <= 0:
        glob_src = os.path.join(src_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
        glob_src_sql = glob_src.replace("'", "''")
        try:
            val = con.sql(f"SELECT max(trade_date) FROM parquet_scan('{glob_src_sql}')").fetchone()[0]
            last_src = int(val) if val is not None else 0
        except duckdb.Error as e:
            print("DuckDB error (src max trade_date):", e)
            last_src = 0
        last_duck = max(last_src - 1, int(start))
        last_duck_str = str(last_duck).zfill(8)

    # ② 源端新增日期涉及的 ts_code
    glob_src = os.path.join(src_dir, "trade_date=*/part-*.parquet").replace("\\", "/")
    glob_src_sql = glob_src.replace("'", "''")

    try:
        df_codes = con.sql(f"""
            SELECT DISTINCT ts_code
            FROM parquet_scan('{glob_src_sql}')
            WHERE trade_date > '{last_duck_str}' AND trade_date <= '{end}'
        """).df()
    except duckdb.Error as e:
        print("DuckDB error (scan src for ts_code):", e)
        df_codes = pd.DataFrame(columns=["ts_code"])

    if df_codes.empty:
        logging.debug('[BRANCH] def recalc_symbol_products_for_increment | IF df_codes.empty -> taken')
        logging.info("[INC_IND] 指标分区已最新(last=%s，源端无新增日期)。", last_duck)
        con.close()
        return

    ts_list = df_codes["ts_code"].dropna().astype(str).unique().tolist()
    logging.info("[INC_IND] 需要补的股票数=%d (last_ind_date=%s)", len(ts_list), last_duck_str)

    if INC_IND_ALL_INMEM and ts_list:
        con.close()
        return _recalc_increment_inmem(ts_list, last_duck_str, end, threads)


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
            logging.debug('[BRANCH] def duckdb_partition_merge | IF not os.path.isdir(src_dir) -> taken')
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
            logging.debug('[BRANCH] def duckdb_partition_merge | IF df_dates.empty -> taken')
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
            logging.debug('[BRANCH] def streaming_merge_after_download | IF total_files == 0 -> taken')
            logging.warning("[STREAM_MERGE][%s] 无股票文件可归并", mode_label)
            continue

        os.makedirs(daily_root, exist_ok=True)
        buffer: Dict[str, List[pd.DataFrame]] = {}
        processed = 0
        last_flush_time = time.time()

        def flush(reason: str):
            nonlocal buffer
            if not buffer:
                logging.debug('[BRANCH] def streaming_merge_after_download > def flush | IF not buffer -> taken')
                return
            for dt, lst in buffer.items():
                if not lst:
                    logging.debug('[BRANCH] def streaming_merge_after_download > def flush | IF not lst -> taken')
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
                logging.debug('[BRANCH] def streaming_merge_after_download | IF df is None or df.empty -> taken')
                continue
            for dt, sub in df.groupby("trade_date"):
                buffer.setdefault(dt, []).append(sub)
            processed += 1

            if len(buffer) >= STREAM_FLUSH_DATE_BATCH:
                logging.debug('[BRANCH] def streaming_merge_after_download | IF len(buffer) >= STREAM_FLUSH_DATE_BATCH -> taken')
                flush(f"date_batch>= {STREAM_FLUSH_DATE_BATCH}")
            elif processed % STREAM_FLUSH_STOCK_BATCH == 0:
                logging.debug('[BRANCH] def streaming_merge_after_download | IF processed % STREAM_FLUSH_STOCK_BATCH == 0 -> taken')
                logging.debug('[BRANCH] def streaming_merge_after_download | ELIF processed % STREAM_FLUSH_STOCK_BATCH == 0 -> taken')
                flush(f"stock_batch {STREAM_FLUSH_STOCK_BATCH}")
            elif time.time() - last_flush_time > 60:
                logging.debug('[BRANCH] def streaming_merge_after_download | IF time.time() - last_flush_time > 60 -> taken')
                logging.debug('[BRANCH] def streaming_merge_after_download | ELIF time.time() - last_flush_time > 60 -> taken')
                logging.debug('[BRANCH] def streaming_merge_after_download | IF time.time() - last_flush_time > 60 -> taken')
                logging.debug('[BRANCH] def streaming_merge_after_download | ELIF time.time() - last_flush_time > 60 -> taken')
                flush("time>60s")
                last_flush_time = time.time()

            if processed % STREAM_LOG_EVERY == 0:
                logging.debug('[BRANCH] def streaming_merge_after_download | IF processed % STREAM_LOG_EVERY == 0 -> taken')
                pbar.set_postfix(proc=processed, pct=f"{processed/total_files*100:.1f}%")

        flush("final")
        pbar.close()
        logging.info("[STREAM_MERGE][%s] 完成 处理股票文件=%d", mode_label, processed)


# ========== 主入口 ==========
def main():
    global end_date
    end_date = dt.date.today().strftime("%Y%m%d") if END_DATE.lower() == "today" else END_DATE
    assets = {a.lower() for a in ASSETS}
    is_first_download = input("是否是第一次下载？(y/n)") == "y"
    logging.info(
        "=== 启动 mode=%s assets=%s 区间=%s-%s 原始数据复权=%s ===",
        "FAST_INIT" if is_first_download else "NORMAL",
        sorted(assets), START_DATE, end_date, API_ADJ
    )

    if is_first_download:
        logging.debug('[BRANCH] def main | IF is_first_download -> taken')
        fast_init_download(end_date)   # 这里 end_date 已经算好
        if DUCK_MERGE_DAY_LAG >= 0:  # 简单开关，可设 -1 跳过
            logging.debug('[BRANCH] def main | IF DUCK_MERGE_DAY_LAG >= 0 -> taken')
            duckdb_partition_merge()
        if WRITE_SYMBOL_INDICATORS:
            logging.debug('[BRANCH] def main | IF WRITE_SYMBOL_INDICATORS -> taken')
            duckdb_merge_symbol_products_to_daily()
        if "index" in assets:
            sync_index_daily_fast(START_DATE, end_date, INDEX_WHITELIST)
    else:
        # ======= 日常增量模式 =======
        # 优先把 fastinit 缓存合并进 daily，避免全历史重拉
        logging.debug('[BRANCH] def main | ELSE of IF is_first_download -> taken')
        if any(
            os.path.isdir(os.path.join(FAST_INIT_STOCK_DIR, d))
            and len(glob.glob(os.path.join(FAST_INIT_STOCK_DIR, d, "*.parquet"))) > 0
            for d in ("raw", "qfq", "hfq")
        ):
            logging.debug('[BRANCH] def main | IF any(fast_init has parquet) -> taken')
            duckdb_partition_merge()

        if "stock" in assets:
            logging.debug('[BRANCH] def main | IF "stock" in assets -> taken')
            sync_stock_daily_fast(START_DATE, end_date, threads=STOCK_INC_THREADS)
        if "index" in assets:
            logging.debug('[BRANCH] def main | IF "index" in assets -> taken')
            sync_index_daily_fast(START_DATE, end_date, INDEX_WHITELIST)
        auto_workers = (os.cpu_count() or 4) * 2
        workers = INC_RECALC_WORKERS or auto_workers

        recalc_symbol_products_for_increment(START_DATE, end_date, threads=workers)

    # 写元数据
    meta = {
        "run_mode": "FAST_INIT" if is_first_download else "NORMAL",
        "api_adj": API_ADJ if is_first_download else None,
        "rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": START_DATE,
        "end_date": end_date,
        "assets": sorted(list(assets))
    }
    with open(os.path.join(DATA_ROOT, "_META.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("=== 结束 ===")


if __name__ == "__main__":
    logging.debug('[BRANCH] <module> | IF __name__ == "__main__" -> taken')
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("手动中断程序。")
