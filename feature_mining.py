
"""
运行说明
========
本脚本用于提取 A 股股票的“启动前特征快照”，并导出启动日和前若干天的指标数据。

依赖条件：
1. 已在 config.py 中正确配置：
   - PARQUET_BASE  : 数据根目录(download_merged.py 生成的 parquet 数据路径)
   - PARQUET_ADJ   : 复权方式 ("daily", "daily_qfq", "daily_hfq")
   - START_DATE    : 起始日期 (YYYYMMDD)
   - END_DATE      : 结束日期 (YYYYMMDD)
2. 数据目录必须符合 parquet_viewer.py 约定的结构。
3. 已安装依赖：pandas, numpy, tqdm, duckdb, pyarrow。

运行方式：
----------
1) 全市场模式(多线程批量处理)
   python feature_mining.py --max-workers 20 --out feature_out

2) 单股模式(处理一只股票)
   python feature_mining.py --code 000001.SZ --out feature_out
   # 股票代码支持 6 位纯数字(自动加后缀)或带交易所后缀的 TS 代码

参数说明：
----------
--code         仅处理指定股票(可选)
--max-workers  并发线程数(默认 50)
--log-file     日志文件名(默认 feature_mining.log)
--log-level    日志等级(DEBUG/INFO/WARN/ERROR，默认 INFO)
--out          输出目录(默认 feature_out)

输出文件：
----------
1. launch_dates_XXXXXX.csv / .parquet
   - 启动日信息(股票代码、启动日期、当日收盘、量比、距离55日高点、未来最大涨幅、未来最大回撤)
2. prelaunch_features_XXXXXX.parquet
   - 启动日前若干天(LOOKBACK_K)的指标快照
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import PARQUET_BASE, PARQUET_ADJ, START_DATE as START_STR, END_DATE as END_STR
import parquet_viewer as pv
from utils import ensure_datetime_index
import indicators as ind

# ——— 参数(可调)———
N_FUTURE = 10        # 启动判定窗口
M_DD     = 3        # 回撤检测窗口
RET_MIN  = 0.2      # 启动未来涨幅阈值
DD_MAX   = 0.1      # 允许的最大回撤
VR_MIN   = 1.2       # 启动当天量比
RES_PAD  = 0.02      # 距55日高点容差2%
LOOKBACK_K = 5       # 回看天数，统计 t-1..t-K
MAX_WORKERS = 8

def load_all_codes():
    # 取最新日期的全市场代码列表
    latest_dates = pv.list_trade_dates(pv.asset_root(PARQUET_BASE, "stock", PARQUET_ADJ))
    latest = latest_dates[-1]
    df = pv.read_range(PARQUET_BASE, "stock", PARQUET_ADJ,
        ts_code=None, start=latest, end=latest,
        columns=["ts_code"], limit=None
    )
    return sorted(df["ts_code"].dropna().unique().tolist())

def read_df(code):
    # 根据 PARQUET_ADJ 推断 single 目录的 adj 和是否带指标
    base_adj = "qfq" if "qfq" in PARQUET_ADJ else ("hfq" if "hfq" in PARQUET_ADJ else "daily")
    with_ind = "indicators" in PARQUET_ADJ

    try:
        # ① 优先读取单股成品（速度快，直接拿到已算好的指标）
        df = pv.read_by_symbol(PARQUET_BASE, base_adj, code, with_indicators=with_ind)
    except Exception:
        # ② 回退到按日分区（兼容老目录/老数据）
        df = pv.read_range(
            PARQUET_BASE, "stock", PARQUET_ADJ,
            ts_code=code, start=START_STR, end=END_STR, columns=None, limit=None
        )

    if df is None or df.empty:
        return None
    df = ensure_datetime_index(df)
    df = df.sort_index()
    return df

def label_launch(df: pd.DataFrame):
    C = df["close"].astype(float)
    H_future = C.rolling(N_FUTURE).max().shift(-N_FUTURE+1)      # 未来N天最高收
    L_future = df["low"].astype(float).rolling(M_DD).min().shift(-M_DD+1)  # 未来M天最低价
    ret_fut = (H_future - C) / C
    dd_fut = (L_future - C) / C

    # 压力位：55日最高收
    hhv55 = C.rolling(55).max()
    near_res_ok = (C <= hhv55 * (1 + RES_PAD))

    vr_series = df.get("vr", pd.Series(index=df.index, dtype=float)).fillna(0)
    vr_ok = vr_series >= VR_MIN

    cond = (ret_fut >= RET_MIN) & (dd_fut >= -DD_MAX) & vr_ok & near_res_ok
    df["is_launch"] = cond.astype(int)

    # ——保留用于导出查看的列——
    df["fut_ret_maxN"] = ret_fut
    df["fut_dd_minM"] = dd_fut
    df["vr_now"] = vr_series
    df["hhv55"] = hhv55
    df["gap_to_hhv55"] = (C / hhv55) - 1.0  # 离55日最高的相对距离
    return df

def collect_features(df: pd.DataFrame):
    """
    从 df 中收集 prelaunch 标签的指标快照：
    1) 自动计算缺失的指标(调用 indicators.compute)
    2) 添加派生特征(j_diff, bbi_diff, z_turn_up)
    3) 截取启动日前 LOOKBACK_K 天的特征快照
    """
    # 1) 获取需要的指标名
    need_names = ind.names_by_tag("prelaunch")
    decimals = ind.outputs_for(need_names)

    # 2) 计算指标(缺啥补啥)
    df = ind.compute(df, need_names)

    # 3) 统一 round
    for col, n in decimals.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(n)

    # 4) 派生特征
    if "j" in df.columns:
        df["j_diff"] = df["j"].diff()
    else:
        df["j_diff"] = np.nan
    if "bbi" in df.columns:
        df["bbi_diff"] = df["bbi"].diff()
    else:
        df["bbi_diff"] = np.nan
    if "z_slope" in df.columns:
        df["z_turn_up"] = (df["z_slope"] > 0) & (df["z_slope"].shift(1) <= 0)
    else:
        df["z_turn_up"] = False
        
    # 5) 启动日前 LOOKBACK_K 天的快照
    rows = []
    idxs = np.where(df["is_launch"] == 1)[0]
    for i in idxs:
        for k in range(1, LOOKBACK_K+1):
            j = i - k
            if j < 0: continue
            snap = dict(
                ts=str(df["ts_code"].iloc[j]) if "ts_code" in df.columns else "",
                trade_date=str(df.index[j].date()),
                rel_day=-k,
                j=df["j"].iloc[j],
                j_diff=df["j_diff"].iloc[j],
                z_slope=df["z_slope"].iloc[j],
                z_turn_up=bool(df["z_turn_up"].iloc[j]),
                vr=df["vr"].iloc[j],
                bbi=df["bbi"].iloc[j],
                bbi_diff=df["bbi_diff"].iloc[j],
                bupiao_s=df["bupiao_short"].iloc[j],
                bupiao_l=df["bupiao_long"].iloc[j],
            )
            rows.append(snap)
    return pd.DataFrame(rows)

def process_one(code: str):
    try:
        df = read_df(code)
        if df is None or df.empty:
            return None, None
        df = label_launch(df)

        if "ts_code" not in df.columns:
            df["ts_code"] = code

        launches = df[df["is_launch"] == 1].copy()
        if not launches.empty:
            launches = launches.assign(
                launch_date=launches.index.date,
                close_now=launches["close"].astype(float),
            )[[
                "ts_code", "launch_date", "close_now", "vr_now",
                "gap_to_hhv55", "fut_ret_maxN", "fut_dd_minM",
            ]]
        else:
            launches = None

        snap = collect_features(df)
        if snap is not None and not snap.empty:
            snap["ts_code"] = code
        else:
            snap = None
        return snap, launches
    except Exception as e:
        # 避免单只股票异常阻断整体任务
        return None, None

import argparse
import logging
from logging.handlers import RotatingFileHandler
from time import time
from tqdm import tqdm

LOG = logging.getLogger("feature")

def _setup_logging(log_level:str="INFO", log_file:str|None=None):
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(level)
    root.addHandler(ch)

    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        root.addHandler(fh)

def _run_single(ts_code:str, out_dir:Path):
    LOG.info("单股特征提取开始: %s", ts_code)
    df = read_df(ts_code)
    if df is None or df.empty:
        LOG.warning("无数据: %s", ts_code)
        return
    df = label_launch(df)
    snap = collect_features(df)
    # 启动日导出
    launches = df[df["is_launch"] == 1].copy()
    if "ts_code" not in launches.columns:
        launches["ts_code"] = ts_code
    if not launches.empty:
        launches = launches.assign(
            launch_date=launches.index.date,
            close_now=launches["close"].astype(float),
        )[["ts_code","launch_date","close_now","vr_now","gap_to_hhv55","fut_ret_maxN","fut_dd_minM"]]
        launches.to_csv(out_dir / f"launch_dates_{ts_code.replace('.','_')}.csv", index=False, encoding="utf-8-sig")
        launches.to_parquet(out_dir / f"launch_dates_{ts_code.replace('.','_')}.parquet", index=False)
        LOG.info("启动日导出完成 rows=%d", len(launches))

    if snap is not None and not snap.empty:
        snap["ts_code"] = ts_code
        snap.to_parquet(out_dir / f"prelaunch_features_{ts_code.replace('.','_')}.parquet", index=False)
        LOG.info("特征快照导出完成 rows=%d", len(snap))
    LOG.info("单股特征提取结束: %s", ts_code)

def main():
    global MAX_WORKERS
    parser = argparse.ArgumentParser(description="启动前特征挖掘")
    parser.add_argument("--code", help="仅处理单个股票(支持6位或TS代码，例如 000001 或 000001.SZ)")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="线程数")
    parser.add_argument("--log-file", default="feature_mining.log", help="日志文件")
    parser.add_argument("--log-level", default="INFO", help="日志级别: DEBUG/INFO/WARN/ERROR")
    parser.add_argument("--out", default="feature_out", help="输出目录")
    args = parser.parse_args()

    _setup_logging(args.log_level, args.log_file)

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)

    # 线程数动态化
    MAX_WORKERS = max(1, int(args.max_workers))

    start_ts = time()

    if args.code:
        # 规范化代码 (借用 main 的规则: 6位补后缀)
        def _normalize(ts: str) -> str:
            ts = (ts or "").strip().upper()
            if len(ts) == 6 and ts.isdigit():
                if ts.startswith("8"):
                    suf = ".BJ"
                elif ts[0] in {"5","6","9"}:
                    suf = ".SH"
                else:
                    suf = ".SZ"
                ts += suf
            return ts
        ts_code = _normalize(args.code)
        _run_single(ts_code, out_dir)
        LOG.info("总耗时 %.2fs", time() - start_ts)
        return

    # ===== 批量模式 =====
    LOG.info("批量特征回测开始")
    try:
        codes = load_all_codes()
    except Exception as e:
        LOG.exception("获取代码列表失败: %s", e)
        return

    all_feat = []
    all_launch = []
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_one, code): code for code in codes}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="提取进度", unit="股"):
            code = futures[fut]
            try:
                snap, launches = fut.result()
            except Exception as e:
                errors += 1
                LOG.warning("处理失败 %s: %s", code, e)
                continue
            if snap is not None and not snap.empty:
                all_feat.append(snap)
            if launches is not None and not launches.empty:
                all_launch.append(launches)
    
    # for code in codes:
    #     df = read_df(code)
    #     if df is None or df.empty: continue
    #     df = label_launch(df)
        
    #     # ——导出启动日清单(逐只股票)——
    #     if "ts_code" not in df.columns:
    #         df["ts_code"] = code
    #     launches = df[df["is_launch"] == 1].copy()
    #     if not launches.empty:
    #         launches = launches.assign(
    #             launch_date = launches.index.date,
    #             close_now   = launches["close"].astype(float),
    #         )[[
    #             "ts_code","launch_date","close_now","vr_now",
    #             "gap_to_hhv55","fut_ret_maxN","fut_dd_minM"
    #         ]]
    #         all_launch.append(launches)
            
    #     snap = collect_features(df)
    #     if not snap.empty:
    #         snap["ts_code"] = code
    #         all_feat.append(snap)

    if not all_feat:
        print("no data"); return
    feats = pd.concat(all_feat, ignore_index=True)
    feats.to_parquet(out_dir / "prelaunch_features.parquet", index=False)
    # 简单统计输出
    g = feats.groupby("rel_day").agg(
        j_median=("j", "median"),
        j_q25=("j", lambda s: s.quantile(0.25)),
        j_q75=("j", lambda s: s.quantile(0.75)),
        j_diff_median=("j_diff", "median"),
        z_slope=("z_slope", "median"),
        z_turn_up=("z_turn_up", "mean"),
        vr=("vr", "median"),
        bbi_diff=("bbi_diff", "median"),
        bupiao_s=("bupiao_s", "median"),
        bupiao_l=("bupiao_l", "median"),
    )
    g.to_csv(out_dir / "prelaunch_summary.csv", encoding="utf-8-sig")
    
    if all_launch:
            launch_df = pd.concat(all_launch, ignore_index=True)
            # 按日期、代码排序，查阅更方便
            launch_df = launch_df.sort_values(["launch_date","ts_code"])
            launch_df.to_csv(out_dir / "launch_dates.csv", index=False, encoding="utf-8-sig")
            # 如需 Parquet 版本：
            launch_df.to_parquet(out_dir / "launch_dates.parquet", index=False)

    
    LOG.info("批量完成 codes=%d errors=%d 耗时=%.2fs", len(codes), errors, time() - start_ts)

if __name__ == "__main__":
    main()
