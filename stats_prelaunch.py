import glob
import math
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os
import re
from datetime import datetime

# ================= 配置（可改） =================
GLOB = "./output/feature_mining/prelaunch_features_*.parquet"
REL_DAYS = [-1, -2, -3, -4, -5]
COVERAGE = 0.75     # 覆盖占比（75%）
EXCLUDE_COLS = {"ts_code", "ts", "trade_date", "rel_day"}

# —— 新增：baseline（普通数据）GLob：指向日度“带指标”的分区 —— 
# 例：./data/stock/daily/daily_qfq_indicators/trade_date=*/part-*.parquet
BASELINE_GLOB = None

# 对哪些列做 log1p 再标准化（VR 强右偏，建议先 log1p）
LOG1P_COLS = {"vr"}

# 分箱默认边界（标准化后在 z 空间），若未用 --bins/--common-bins
DEFAULT_STD_BINS = [-3, -2, -1, 0, 1, 2, 3]

# 每次最多抽样的 baseline 行数/文件数（避免一次性读爆）
BASE_MAX_ROWS = 800_000
BASE_MAX_FILES = 300


# 为所有指标统一使用同一组分箱边界：
# 例：COMMON_BINS = [-2, -1, 0, 1, 2]
COMMON_BINS = None
# 为某些指标设置各自的边界（字典形式）
# 例：BINS_PER_FEATURE = {"z_score": [-2,-1,0,1,2], "vr":[0,1,2,3]}
BINS_PER_FEATURE = {"z_score": [-10,-2,-1,0,1,2,3,4,5,6,7,99], "vr": [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2, 3]}

# 也可通过命令行传入分箱字符串：
#   --bins "z_score:-2,-1,0,1:0,1,2,3"
# 或统一： --common-bins "-2,-1,0,1,2"
# ==============================================


_TS_RE = re.compile(r"prelaunch_features_(\d{8}_\d{6})\.parquet$")

def _sort_parquets(paths: list[str]) -> list[str]:
    """
    先按文件名中的时间戳 YYYYMMDD_HHMMSS 排序，解析失败就回退按 mtime 排序。
    返回按时间新->旧的列表（倒序，方便切片取最新 N 个）。
    """
    def parse_ts(p: str):
        m = _TS_RE.search(os.path.basename(p))
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp()
            except Exception:
                pass
        # 回退：文件修改时间
        try:
            return os.path.getmtime(p)
        except Exception:
            return 0.0
    return sorted(paths, key=parse_ts, reverse=True)


def _read_some_parquets(glob_pat: str, cols: list[str], max_files: int = None, max_rows: int = None) -> pd.DataFrame:
    """从分区目录里抽一些 parquet，仅读需要的列。"""
    if not glob_pat:
        return pd.DataFrame(columns=cols)
    files = glob.glob(glob_pat)
    if not files:
        return pd.DataFrame(columns=cols)
    # 尽量按文件名排序（trade_date=* 目录一般已按日期组织）
    files = sorted(files, reverse=True)
    parts, rows = [], 0
    used = 0
    for p in files:
        try:
            sub = pd.read_parquet(p, columns=[c for c in cols if c != "rel_day"])  # baseline 没有 rel_day
        except Exception:
            continue
        if sub is None or sub.empty:
            continue
        parts.append(sub)
        rows += len(sub)
        used += 1
        if (max_files and used >= max_files) or (max_rows and rows >= max_rows):
            break
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def compute_baseline_params(base_df: pd.DataFrame, feats: list[str], log1p_cols: set[str]) -> dict[str, tuple[float,float]]:
    """返回 {feature: (mu, sigma)}，在 baseline 上估计。"""
    params = {}
    for f in feats:
        s = pd.to_numeric(base_df.get(f), errors="coerce")
        if f in log1p_cols:
            s = np.log1p(s)
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            params[f] = (0.0, 1.0)
            continue
        mu = float(s.mean())
        sd = float(s.std(ddof=0)) or 1.0
        params[f] = (mu, sd)
    return params


def apply_standardize(df: pd.DataFrame, feats: list[str], params: dict[str, tuple[float,float]], log1p_cols: set[str], suffix: str="_std") -> pd.DataFrame:
    """把 df 的列按 baseline 的 (mu, sigma) 标准化，输出新列 f+suffix。"""
    out = df.copy()
    for f in feats:
        x = pd.to_numeric(out.get(f), errors="coerce")
        if f in log1p_cols:
            x = np.log1p(x)
        mu, sd = params.get(f, (0.0, 1.0))
        out[f + suffix] = (x - mu) / (sd + 1e-9)
    return out


def load_all(glob_pat: str, latest_only: bool = False, limit: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    paths = glob.glob(glob_pat)
    if not paths:
        raise FileNotFoundError(f"未找到匹配文件：{glob_pat}")
    paths = _sort_parquets(paths)  # 新->旧
    if latest_only:
        paths = paths[:1]
    elif isinstance(limit, int) and limit > 0:
        paths = paths[:limit]
    else:
        pass
    # 读取并纵向拼接
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    if "rel_day" not in df.columns:
        raise KeyError("数据缺少 rel_day 列")
    return df, paths


def pick_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in EXCLUDE_COLS]


def min_width_interval(values: pd.Series, coverage: float) -> tuple[float, float, float, int]:
    """
    经验最小区间（HDI）：给定 coverage（如 0.75），在排序后的数据上
    找到包含 k=ceil(n*coverage) 个点且宽度最小的区间 [low, high]。
    返回 (low, high, width, k)。若样本为空，返回 (nan, nan, nan, 0)。
    """
    s = pd.to_numeric(values, errors="coerce").dropna().sort_values().to_numpy()
    n = s.size
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    k = max(1, int(math.ceil(n * coverage)))
    if k == 1:
        return (float(s[0]), float(s[0]), 0.0, 1)

    best_i = 0
    best_width = float("inf")
    # 滑动窗口：i..i+k-1
    for i in range(0, n - k + 1):
        width = s[i + k - 1] - s[i]
        if width < best_width:
            best_width = width
            best_i = i
    low = float(s[best_i])
    high = float(s[best_i + k - 1])
    return (low, high, float(best_width), k)


def summarize_hdi(df: pd.DataFrame, rel_days: list[int], coverage: float) -> pd.DataFrame:
    feats = pick_numeric_feature_cols(df)
    need = df[df["rel_day"].isin(rel_days)][["rel_day"] + feats].copy()

    rows = []
    for d in rel_days:
        sub = need[need["rel_day"] == d]
        n_all = len(sub)
        for f in feats:
            Z_HDI_CLIP = 20.0  # zscore绝对值裁剪阈，按需调整
            s = pd.to_numeric(sub[f], errors="coerce") \
                    .replace([np.inf, -np.inf], np.nan) \
                    .dropna()
            if f.lower() == "z_score":
                s = s.clip(lower=-Z_HDI_CLIP, upper=Z_HDI_CLIP)

            n = s.size
            if n == 0:
                rows.append(dict(
                    feature=f, rel_day=d, count=0,
                    mean=np.nan, std=np.nan, median=np.nan,
                    hdi_low=np.nan, hdi_high=np.nan, hdi_width=np.nan, hdi_k=0,
                ))
                continue
            hlow, hhigh, hwidth, hk = min_width_interval(s, coverage)
            rows.append(dict(
                feature=f, rel_day=d, count=int(n),
                mean=float(s.mean()),
                std=float(s.std(ddof=1)) if n > 1 else 0.0,
                median=float(s.median()),
                hdi_low=hlow, hdi_high=hhigh, hdi_width=hwidth, hdi_k=int(hk),
            ))
    out = pd.DataFrame(rows).sort_values(["feature", "rel_day"])
    return out


def parse_bins_str(bins_str: str) -> dict:
    """
    解析形如 "z_score:-2,-1,0,1:0,1,2,3" 的字符串为字典。
    """
    res = {}
    if not bins_str:
        return res
    for part in bins_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        name, edges = part.split(":", 1)
        name = name.strip()
        edges_list = []
        for t in edges.split(","):
            t = t.strip()
            if t:
                edges_list.append(float(t))
        if len(edges_list) >= 2:
            res[name] = edges_list
    return res


def bins_probability(df: pd.DataFrame,
                     rel_days: list[int],
                     bins_common: list[float] | None,
                     bins_per_feature: dict[str, list[float]] | None) -> pd.DataFrame:
    """
    统计各 rel_day × 指标 × 分箱 的占比。
    优先使用 bins_per_feature[feature]；否则落到 bins_common（若给）。
    """
    feats = pick_numeric_feature_cols(df)
    need = df[df["rel_day"].isin(rel_days)].copy()
    rows = []
    for f in feats:
        # 决定该指标使用的边界
        edges = None
        if bins_per_feature and f in bins_per_feature:
            edges = bins_per_feature[f]
        elif bins_common:
            edges = bins_common
        else:
            continue  # 没配置边界就跳过

        col = pd.to_numeric(need[f], errors="coerce")
        edges = sorted(edges)
        cats = pd.IntervalIndex.from_breaks(edges, closed="right")
        for d in rel_days:
            s = col[need["rel_day"] == d].dropna()
            n = int(s.size)
            if n == 0:
                # 也输出空计数以便表格齐整
                for i, iv in enumerate(cats):
                    rows.append(dict(
                        feature=f, rel_day=d, bin=str(iv),
                        bin_left=float(iv.left), bin_right=float(iv.right), bin_order=i,
                        count=0, prob=np.nan
                    ))
                continue
            
            cut = pd.cut(s, bins=cats, right=True, include_lowest=True)
            counts = cut.value_counts().reindex(cats, fill_value=0)  # 用分类顺序
            for i, iv in enumerate(cats):
                cnt = int(counts.loc[iv])
                rows.append(dict(
                    feature=f, rel_day=d, bin=str(iv),
                    bin_left=float(iv.left), bin_right=float(iv.right), bin_order=i,
                    count=cnt, prob=float(cnt / n)
                ))
    out = (pd.DataFrame(rows)
           .sort_values(["feature", "rel_day", "bin_order"])
           .reset_index(drop=True))
    return out


def bins_probability_std_lift(
    event_df: pd.DataFrame,
    base_df: pd.DataFrame,
    rel_days: list[int],
    bins_common: list[float] | None,
    bins_per_feature: dict[str, list[float]] | None,
    alpha: float = 0.5,
    use_std_cols: bool = True,   # True 时用 *_std 列
) -> pd.DataFrame:
    feats = pick_numeric_feature_cols(event_df)
    rows = []
    for f in feats:
        col = f + "_std" if use_std_cols and (f + "_std") in event_df.columns else f
        colb = f + "_std" if use_std_cols and (f + "_std") in base_df.columns else f

        # 选边界（若没配，标准化后落默认 z-bins）
        if bins_per_feature and f in bins_per_feature:
            edges = sorted(bins_per_feature[f])
        elif bins_common:
            edges = sorted(bins_common)
        else:
            edges = sorted(DEFAULT_STD_BINS)
        cats = pd.IntervalIndex.from_breaks(edges, closed="right")

        sb = pd.to_numeric(base_df[colb], errors="coerce").dropna()
        if sb.empty:
            # baseline 缺列时，跳过该指标
            continue
        base_cut = pd.cut(sb, bins=cats, right=True, include_lowest=True)
        K = len(cats)
        base_counts = base_cut.value_counts().reindex(cats, fill_value=0)
        base_total = int(base_counts.sum())
        p_base = (base_counts + alpha) / (base_total + alpha * K)

        for d in rel_days:
            se = pd.to_numeric(event_df.loc[event_df["rel_day"] == d, col], errors="coerce").dropna()
            if se.empty:
                for i, iv in enumerate(cats):
                    rows.append(dict(
                        feature=f, rel_day=d, bin=str(iv),
                        bin_left=float(iv.left), bin_right=float(iv.right), bin_order=i,
                        count_event=0, count_base=int(base_counts.iloc[i]),
                        p_event=np.nan, p_base=float(p_base.iloc[i]),
                        lift=np.nan, woe=np.nan,
                    ))
                continue
            evt_cut = pd.cut(se, bins=cats, right=True, include_lowest=True)
            evt_counts = evt_cut.value_counts().reindex(cats, fill_value=0)
            evt_total = int(evt_counts.sum())
            p_evt = (evt_counts + alpha) / (evt_total + alpha * K)

            for i, iv in enumerate(cats):
                pe = float(p_evt.iloc[i])
                pb = float(p_base.iloc[i])
                lf = pe / pb if pb > 0 else np.nan
                rows.append(dict(
                    feature=f, rel_day=d, bin=str(iv),
                    bin_left=float(iv.left), bin_right=float(iv.right), bin_order=i,
                    count_event=int(evt_counts.iloc[i]),
                    count_base=int(base_counts.iloc[i]),
                    p_event=pe, p_base=pb,
                    lift=lf,
                    woe=(np.log(lf) if (lf > 0 and np.isfinite(lf)) else np.nan),
                ))
    out = (pd.DataFrame(rows)
           .sort_values(["feature", "rel_day", "bin_order"])
           .reset_index(drop=True))
    return out


def main():
    parser = argparse.ArgumentParser(description="统计最小宽度 75% 区间 + 自定义分箱概率")
    parser.add_argument("--glob", default=GLOB, help="prelaunch_features parquet 通配路径")
    parser.add_argument("--rel", default=",".join(map(str, REL_DAYS)), help="要统计的 rel_day 列表，用逗号分隔，如 -1,-2,-3,-4,-5")
    parser.add_argument("--coverage", type=float, default=COVERAGE, help="覆盖占比（默认 0.75）")
    parser.add_argument("--common-bins", default=None, help="统一分箱边界字符串，如 \"-2,-1,0,1,2\"")
    parser.add_argument("--bins", default=None, help="每指标边界，如 \"z_score:-2,-1,0,1:0,1,2,3\"")
    # parser.add_argument("--latest", action="store_true", help="只读取最新的一个 parquet 文件")
    parser.add_argument("--all", action="store_true", help="读取所有匹配的 parquet 文件")
    parser.add_argument("--limit", type=int, default=None, help="只读取最新的 N 个 parquet 文件（与 --latest 互斥，--latest 优先）")
    args = parser.parse_args()

    rel_days = [int(x.strip()) for x in args.rel.split(",") if x.strip()]
    df, used_paths = load_all(args.glob, latest_only=not args.all, limit=args.limit)
    out_dir = Path("./output/stats_prelaunch")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 记录来源文件 ---
    sources_txt = "\n".join(used_paths)
    (out_dir / "sources.txt").write_text(sources_txt, encoding="utf-8")
    print(f"已导出：sources.txt")
    
    # --- HDI 结果 ---
    hdi = summarize_hdi(df, rel_days, args.coverage)
    hdi.to_csv(out_dir / "stats_hdi.csv", index=False, encoding="utf-8-sig")
    print("已导出：stats_hdi.csv")

    # --- 分箱概率 ---
    bins_common = None
    if args.common_bins:
        bins_common = [float(x.strip()) for x in args.common_bins.split(",") if x.strip()]

    bins_per = parse_bins_str(args.bins) if args.bins else dict(BINS_PER_FEATURE)  # 允许用默认字典
    if COMMON_BINS and not bins_common:
        bins_common = COMMON_BINS

    prob_df = bins_probability(df, rel_days, bins_common, bins_per)
    if not prob_df.empty:
        prob_df.to_csv(out_dir / "prob_bins.csv", index=False, encoding="utf-8-sig")
        print("已导出：prob_bins.csv")
    else:
        print("未提供任何分箱边界，跳过分箱概率统计。")


if __name__ == "__main__":
    main()
