
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

import os, argparse, logging
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from time import time
from tqdm import tqdm
from types import SimpleNamespace
import parquet_viewer as pv
from utils import ensure_datetime_index
import indicators as ind
from config import *
try:
    from backtest_core import buy_signal
except Exception:
    buy_signal = None

# ========================= 用户可改参数区 =========================
USER_PARAMS = {
    # —— 特征/打标参数 ——
    "LOOKBACK_K": 5,         # 启动日往前回溯多少天取特征快照（例：5 表示取 -1 到 -5 天的数据）
    "SURGE_P": 60,           # Z-score 滚动窗口长度（计算平均值和标准差的周期）
    "SURGE_N": 15,           # 未来观察窗口长度（向前看多少天找最大连续子区间 z-score 和）
    "SURGE_Z_THR": 17.0,      # Z-score 阈值（达到或超过则标记为启动点）
    "SURGE_COOLDOWN": 5,     # 冷却天数（命中启动点后，多少天内不再标记新的启动点）

    # —— 并发/输出/日志 ——
    "MAX_WORKERS": 8,
    "OUT_DIR": "./output/feature_mining",
    "LOG_FILE": "./log/feature_mining.log",
    "LOG_LEVEL": "WARN",   # 可选: DEBUG / INFO / WARN / ERROR
    # —— 运行范围（留空则跑全市场）（会自动补后缀）——
    "CODES": [],
    # —— 启动前画像窗口列表（可多组）——
    "W_LIST": [5, 10, 20],
    # —— 买点规则文件（仅做可读性自检；无需环境变量）——
    "BUY_RULES_PATH": "./buy_rules.txt",
}

# 同步到模块级常量，便于下方引用（不想用也可直接全用 USER_PARAMS[...]）
LOOKBACK_K     = int(USER_PARAMS["LOOKBACK_K"])
SURGE_P        = int(USER_PARAMS["SURGE_P"])
SURGE_N        = int(USER_PARAMS["SURGE_N"])
SURGE_Z_THR    = float(USER_PARAMS["SURGE_Z_THR"])
SURGE_COOLDOWN = int(USER_PARAMS["SURGE_COOLDOWN"])
MAX_WORKERS    = int(USER_PARAMS["MAX_WORKERS"])
OUT_DIR        = str(USER_PARAMS["OUT_DIR"])
LOG_FILE       = str(USER_PARAMS["LOG_FILE"])
LOG_LEVEL      = str(USER_PARAMS["LOG_LEVEL"])
W_LIST         = list(USER_PARAMS["W_LIST"])
BUY_RULES_PATH = str(USER_PARAMS["BUY_RULES_PATH"])
# ================================================================

def _check_buy_rules(buy_path: str | Path) -> dict:
    """
    检查 buy 规则文件是否可读；统计非注释/非空行；给出 SHA1。
    注释行定义：以 //、#、-- 或 'N1:=' 这类行首空白后跟注释/空行视为注释或空。
    """
    p = Path(buy_path)
    info = {"path": str(p.resolve()), "exists": p.exists(), "readable": False,
            "non_comment_lines": 0, "sha1": None}

    if not p.exists():
        return info

    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        # 统计非注释/非空行（通达信/脚本常见注释符：//、#，也顺带跳过空行）
        non_comment = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("//") or s.startswith("#"):
                continue
            # 常见脚本里也可能有以 “-- ” 开头的注释
            if s.startswith("-- "):
                continue
            non_comment.append(line)
        info["non_comment_lines"] = len(non_comment)
        info["sha1"] = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
        info["readable"] = True
        return info
    except Exception:
        return info

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
    base_adj = "qfq" if "qfq" in PARQUET_ADJ else ("hfq" if "hfq" in PARQUET_ADJ else "daily")
    # ① 先强行尝试 single_<adj>_indicators
    try:
        LOG.debug("[read_df] try with_indicators=True base_adj=%s code=%s", base_adj, code)
        df = pv.read_by_symbol(PARQUET_BASE, base_adj, code, with_indicators=True)
        if df is not None and not df.empty:
            LOG.info("[read_df] loaded single_%s_indicators for %s", base_adj, code)

    except Exception:
        # ② 退到 single_<adj>（不带指标）
        LOG.debug("[read_df] with_indicators=True failed, fallback to with_indicators=False")
        try:
            df = pv.read_by_symbol(PARQUET_BASE, base_adj, code, with_indicators=False)
            if df is not None and not df.empty:
                LOG.info("[read_df] loaded single_%s (no indicators) for %s", base_adj, code)

        except Exception:
            # ③ 再退到按日分区
            LOG.debug("[read_df] fallback to daily range read (partitioned)")
            df = pv.read_range(PARQUET_BASE, "stock", PARQUET_ADJ,
                               ts_code=code, start=str(STRATEGY_START_DATE),
                               end=str(STRATEGY_END_DATE), columns=None, limit=None)
            if df is not None and not df.empty:
                LOG.info("[read_df] loaded daily range %s for %s", PARQUET_ADJ, code)

    if df is None or df.empty:
        LOG.warning("[read_df] no data for %s", code)
        return None
    df = ensure_datetime_index(df).sort_index()
    return df

def collect_features(df: pd.DataFrame):
    """
    从 df 中收集 prelaunch 标签的指标快照：
    1) 自动计算缺失的指标(调用 indicators.compute)
    2) 添加派生特征(j_diff, bbi_diff, z_turn_up)
    3) 截取启动日前 LOOKBACK_K 天的特征快照
    """
    # need_names = ind.names_by_tag("prelaunch")
    # decimals = ind.outputs_for(need_names)

    # df = ind.compute(df, need_names)

    # for col, n in decimals.items():
    #     if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
    #         df[col] = df[col].round(n)

    rows = []
    idxs = np.where(df["is_surge_dyn"] == 1)[0]
    exclude = {
        "ts_code","trade_date","open","high","low","close","vol","amount","pre_close",
        # 这些是标签/辅助列，通常不算“特征列”
    }

    for i in idxs:
        for k in range(1, LOOKBACK_K+1):
            j = i - k
            if j < 0: 
                continue
            # 在这添加需要的特征
            snap = {
                "ts": str(df["ts_code"].iloc[j]) if "ts_code" in df.columns else "",
                "trade_date": str(df.index[j].date()),
                "rel_day": -k,
            }
            # 动态枚举：把剩余列全带上
            for col in df.columns:
                if col not in exclude and col in df.columns:
                    snap[col] = df[col].iloc[j]
            rows.append(snap)
    return pd.DataFrame(rows)

def last_true_before(i: int, cond: pd.Series) -> int | None:
    """
    在位置 i 之前（严格小于 i），向前寻找 cond==True 的最近位置。
    返回索引号（整数位置），找不到则 None。
    """
    if i <= 0:
        return None
    # 用到的都是顺序索引，提升性能可用向量化写法；这里用简洁做法
    prev = cond.iloc[:i].to_numpy().nonzero()[0]
    if prev.size == 0:
        return None
    return int(prev[-1])  # 最近一次

# ====== 1) 动态“可观涨幅”打标（z-score累计） ======
def label_surge_dynamic(df: pd.DataFrame,
                        P: int | None = None,
                        N: int | None = None,
                        Z_THR: float | None = None,
                        cooldown: int | None = None,
                        col_close: str = "close") -> pd.DataFrame:
    """
    用未来 N 天内“任意连续子区间”的 z-score 最大和做动态打标：
      1) r_t = pct_change(close)
      2) z_t = (r_t - rolling_mean(P)) / rolling_std(P)
      3) zmax_fut_t = max_{1<=k<=N} sum_{j=1..k} z_{t+j}     ← 连续子段最大值（Kadane）
      4) zmax_fut_t >= Z_THR → surge 起点
      5) 命中之间施加 cooldown 天冷却
    """
    P = int(P or SURGE_P)
    N = int(N or SURGE_N)
    Z_THR = float(Z_THR if Z_THR is not None else SURGE_Z_THR)
    cooldown = int(cooldown or SURGE_COOLDOWN)

    df = df.copy()
    C = df[col_close].astype(float)
    LOG.debug("[label_surge_dynamic] params: P=%s N=%s Z_THR=%.3f cooldown=%s", P, N, Z_THR, cooldown)
    # --- 基础 z：优先使用指标层的新 z_score；没有则回退旧算法 ---
    z = None
    try:
        # 若 df 已带指标列(你 loader 会优先读 *_indicators 分区)
        if "z_score" not in df.columns:
            # 现场补算（优先 TDX，失败回退 Python 兜底）
            LOG.debug("[label_surge_dynamic] z_score not in df.columns -> compute via indicators.compute()")
            tmp = ind.compute(df, ["z_score"])
            if "z_score" in tmp.columns:
                df = tmp
        if "z_score" in df.columns:
            z = pd.to_numeric(df["z_score"], errors="coerce")
            LOG.info("[label_surge_dynamic] using NEW z_score from indicators")
    except Exception:
        z = None
        LOG.warning("[label_surge_dynamic] indicators z_score compute failed; will fallback")
    if z is None:
        # 旧算法兜底：基于收益的滚动 z
        LOG.info("[label_surge_dynamic] fallback to legacy rolling z computation")
        r = C.pct_change()
        m = r.rolling(P, min_periods=max(10, P // 3)).mean()
        s = r.rolling(P, min_periods=max(10, P // 3)).std().replace(0, np.nan)
        z = (r - m) / s
    # --- 未来 N 天“连续子区间最大和” ---
    z_values = z.to_numpy()
    T = len(z_values)
    zmax_fut = np.full(T, np.nan)

    LOG.debug("[label_surge_dynamic] start Kadane forward scan for zmax_fut; T=%d", T)
    for t in range(T):
        start = t + 1
        end = min(T, t + 1 + N)
        if start >= end:
            continue
        window = z_values[start:end]
        # NaN 不参与：置为极小值，确保不会被选中
        window = np.nan_to_num(window, nan=-1e12)

        # Kadane 最大子段和（允许全负，取最大单值）
        best = window[0]
        cur = window[0]
        for v in window[1:]:
            cur = v if (cur + v) < v else (cur + v)
            best = best if best >= cur else cur
        zmax_fut[t] = best

    df["zmax_fut"] = zmax_fut

    # --- 触发 + 冷却（优先保留每段第一个命中） ---
    raw_hit = (df["zmax_fut"] >= Z_THR).astype(int).to_numpy()
    keep = np.zeros(T, dtype=int)
    last_kept = -10**9
    hit_idx = np.where(raw_hit == 1)[0]
    LOG.debug("[label_surge_dynamic] raw hits=%d (threshold=%.3f)", len(hit_idx), Z_THR)
    for i in hit_idx:
        if i - last_kept >= cooldown:
            keep[i] = 1
            last_kept = i

    df["is_surge_dyn"] = keep
    LOG.info("[label_surge_dynamic] kept hits after cooldown=%d", int(keep.sum()))
    return df

# ====== 2) 买入信号布尔序列（复用你的回测买点）======
def compute_buy_cond(df: pd.DataFrame) -> pd.Series:
    # 需要 backtest_core.buy_signal(df)；若未提供，暂用保守占位符
    try:
        cond = buy_signal(df)
        LOG.debug("[compute_buy_cond] buy_signal(df) executed via backtest_core.buy_signal")
        cond = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    except Exception:
        LOG.warning("[compute_buy_cond] buy_signal unavailable -> return all False")
        cond = pd.Series(False, index=df.index)  # 占位：先全 False，待你提供源码后替换
    return cond

# ====== 3) 窗口画像聚合 ======
def aggregate_window_features(
    df: pd.DataFrame,
    surge_flag_col: str = "is_surge_dyn",
    W: int = 10,                       # 画像窗口长度（起点往前 W 天，不含起点）
) -> pd.DataFrame:
    """
    对每个启动起点，统计“起点前 W 天”的关键信息（信号密度/次数/连续段/最近-最远间隔等）。
    输出列（示例）：
      - ts_code, surge_date, win_start, win_end, W
      - signal_count, signal_days_ratio, signal_streak_max, signal_last_lag, signal_first_lag
      - zcum_fut_at_surge（可观强度快照）
    """
    LOG.debug("aggregate_window_features: buy_signal=%s buy_rules_file=%s",
            ("OK" if callable(buy_signal) else "MISSING"),
            Path(BUY_RULES_PATH).resolve())

    if df is None or df.empty or surge_flag_col not in df.columns:
        LOG.warning("[aggregate_window_features] invalid df or missing '%s'", surge_flag_col)
        return pd.DataFrame()

    idx = df.index
    buy_cond = compute_buy_cond(df)
    flags = df[surge_flag_col].fillna(0).astype(int)

    rows = []
    hit_pos = np.where(flags.to_numpy() == 1)[0]
    LOG.info("[aggregate_window_features] W=%d surge_count=%d", W, len(hit_pos))
    for pos in hit_pos:
        win_l = max(0, pos - W)  # [win_l, pos)  不含 pos
        win_r = pos
        if win_r <= win_l:
            LOG.debug("[aggregate_window_features] skip empty window pos=%d", pos)
            continue
        win_idx = idx[win_l:win_r]
        sig = buy_cond.reindex(win_idx).fillna(False).to_numpy()

        # 1) 次数与占比
        count = int(sig.sum())
        ratio = float(count) / len(sig)

        # 2) 最长连续段
        streak = 0
        cur = 0
        for v in sig:
            if v:
                cur += 1
                streak = max(streak, cur)
            else:
                cur = 0

        # 3) 最近/最早信号距起点的“天数”
        last_lag = None
        first_lag = None
        if count > 0:
            hit_days = np.where(sig)[0]           # 在窗口内的相对位置（0 是最早那天）
            last_lag  = (len(sig) - 1) - hit_days[-1]  # 距起点最近一次
            first_lag = (len(sig) - 1) - hit_days[0]   # 距起点最早一次

        rows.append(dict(
            ts_code = df.get("ts_code", pd.Series(index=idx, dtype=object)).reindex([idx[pos]]).iat[0]
                      if "ts_code" in df.columns else None,
            surge_date = idx[pos].date(),
            win_start  = win_idx[0].date(),
            win_end    = win_idx[-1].date(),
            W = int(W),
            signal_count = int(count),
            signal_days_ratio = float(ratio),
            signal_streak_max = int(streak),
            signal_last_lag   = (None if last_lag is None else int(last_lag)),
            signal_first_lag  = (None if first_lag is None else int(first_lag)),
            zcum_fut_at_surge = float(df["zcum_fut"].iat[pos]) if "zcum_fut" in df.columns else None,
        ))

    return pd.DataFrame(rows)

def process_one(code: str):
    try:
        df = read_df(code)
        if df is None or df.empty:
            LOG.debug("[process_one] skip %s due to empty df", code)
            return None, None
        df = label_surge_dynamic(df, P=SURGE_P, N=SURGE_N, Z_THR=SURGE_Z_THR, cooldown=SURGE_COOLDOWN)
        LOG.debug("[process_one] %s label_surge_dynamic done", code)

        if "ts_code" not in df.columns:
            df["ts_code"] = code

        launches = df[df["is_surge_dyn"] == 1].copy()
        if not launches.empty:
            LOG.info("[process_one] %s surge points=%d", code, len(launches))
            launches = launches.assign(
                launch_date=launches.index.date,
                close_now=launches["close"].astype(float),
            )[[
                "ts_code", "launch_date", "close_now",
            ]]
        else:
            LOG.debug("[process_one] %s no surge points", code)
            launches = None

        snap = collect_features(df)
        if snap is not None and not snap.empty:
            LOG.debug("[process_one] %s snapshots rows=%d", code, len(snap))
            snap["ts_code"] = code
        else:
            snap = None
        return snap, launches
    except Exception as e:
        # 避免单只股票异常阻断整体任务
        LOG.exception("[process_one] fail %s: %s", code, e)
        return None, None

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

def _run_single(ts_code: str, out_dir: Path):
    LOG.info("单股特征挖掘开始: %s", ts_code)
    df = read_df(ts_code)
    if df is None or df.empty:
        LOG.warning("无数据: %s", ts_code)
        return

    df = label_surge_dynamic(df, P=SURGE_P, N=SURGE_N, Z_THR=SURGE_Z_THR, cooldown=SURGE_COOLDOWN)
    LOG.info("[_run_single] surge kept=%d", int(df["is_surge_dyn"].sum()))
    
    stats_list = []
    for W in W_LIST:
        stats = aggregate_window_features(df, surge_flag_col="is_surge_dyn", W=W)
        if not stats.empty:
            stats["ts_code"] = ts_code
            stats_list.append(stats)
            LOG.info("[_run_single] W=%d stats rows=%d", W, len(stats))

    if stats_list:
        out = pd.concat(stats_list, ignore_index=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / f"surge_window_stats_{ts_code.replace('.','_')}.parquet", index=False)
        LOG.info("窗口画像导出 rows=%d", len(out))
    else:
        LOG.info("没有命中的启动点: %s", ts_code)

    LOG.info("单股特征挖掘结束: %s", ts_code)

def main():
    global MAX_WORKERS
    # —— 不使用 CLI：用 USER_PARAMS 构造“args” ——
    args = SimpleNamespace(
        out=OUT_DIR,
        log_file=LOG_FILE,
        log_level=LOG_LEVEL,
        max_workers=MAX_WORKERS,
        codes=list(USER_PARAMS.get("CODES", [])),
    )
    _setup_logging(args.log_level, args.log_file)
    buy_info = _check_buy_rules(BUY_RULES_PATH)
    LOG.info("BUY 规则文件检测: path=%s exists=%s readable=%s non_comment_lines=%s sha1=%s",
             buy_info["path"], buy_info["exists"], buy_info["readable"],
             buy_info["non_comment_lines"], buy_info["sha1"])

    LOG.info("buy_signal 函数: %s",
             ("已加载(backtest_core.buy_signal)" if callable(buy_signal) else "未加载(占位 False)"))

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)

    # 线程数动态化
    MAX_WORKERS = max(1, int(args.max_workers))

    start_ts = time()

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
    user_codes = [ _normalize(c) for c in (args.codes or []) if c ]
    if user_codes:
        for ts_code in user_codes:
            _run_single(ts_code, out_dir)
        LOG.info("总耗时 %.2fs", time() - start_ts)
        return

    # ===== 批量模式 =====
    LOG.info("批量特征回测开始")
    try:
        codes = load_all_codes()
        LOG.info("加载股票代码数=%d", len(codes))
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
                LOG.debug("[main] add snapshots %s rows=%d", code, len(snap))
            if launches is not None and not launches.empty:
                all_launch.append(launches)
                LOG.debug("[main] add launches %s rows=%d", code, len(launches))
    
    # for code in codes:
    #     df = read_df(code)
    #     if df is None or df.empty: continue
    #     df = label_launch(df)
        
    #     # ——导出启动日清单(逐只股票)——
    #     if "ts_code" not in df.columns:
    #         df["ts_code"] = code
    #     launches = df[df["is_surge_dyn"] == 1].copy()
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
        LOG.warning("[main] no features collected; errors=%d", errors)
        print("no data"); return
        
    ts_tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    feats = pd.concat(all_feat, ignore_index=True)
    feats.to_parquet(out_dir / f"prelaunch_features_{ts_tag}.parquet", index=False)
    LOG.info("[main] features total rows=%d", len(feats))
    # 简单统计输出
    g = feats.groupby("rel_day").agg(
    count=("rel_day", "size")
    # 这里还可以继续加: 其他列名=("列名", "聚合函数")
)
    g.to_csv(out_dir / f"prelaunch_summary_{ts_tag}.csv", encoding="utf-8-sig")
    LOG.info("[main] summary by rel_day rows=%d", len(g))
    if all_launch:
            launch_df = pd.concat(all_launch, ignore_index=True)
            # 按日期、代码排序，查阅更方便
            launch_df = launch_df.sort_values(["launch_date","ts_code"])
            launch_df.to_csv(out_dir / f"launch_dates_{ts_tag}.csv", index=False, encoding="utf-8-sig")
            # 如需 Parquet 版本：
            launch_df.to_parquet(out_dir / f"launch_dates_{ts_tag}.parquet", index=False)
            LOG.info("[main] launch list rows=%d", len(launch_df))

    LOG.info("批量完成 codes=%d errors=%d 耗时=%.2fs", len(codes), errors, time() - start_ts)

if __name__ == "__main__":
    main()
