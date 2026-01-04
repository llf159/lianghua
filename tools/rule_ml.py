# -*- coding: utf-8 -*-
"""
rule_ml.py — 用机器学习评估规则重要度（独立脚本，零侵入）

功能：
- 从 details.db 提取规则命中（ok=True），与 stock_data.db 对齐收盘价
- 生成规则特征矩阵（规则名 → 特征列，取 add/points 作为强度，未命中为 0）
- 以未来 N 日收益为标签，使用 XGBoost 训练回归模型
- 自动尝试 GPU（gpu_hist），不可用则回落 CPU（hist）
- 输出特征重要度与验证集指标

示例：
  ./venv/bin/python tools/rule_ml.py \
    --horizons 5 10 20 \
    --target-horizon 5 \
    --min-hits 50 \
    --start-date 20240101 --end-date 20251231

输出目录（默认）：output/rule_reports
- rule_ml_importance.csv：特征重要度（gain）
- rule_ml_summary.md：训练/验证指标与 Top 规则
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import duckdb
import numpy as np
import pandas as pd


def _abs_norm(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def load_config_paths() -> Tuple[str, str]:
    """从 config.py 读取数据库路径，失败时使用默认路径。"""
    try:
        from config import DATA_ROOT, UNIFIED_DB_PATH, SC_OUTPUT_DIR, SC_DETAIL_DB_PATH
    except Exception:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_root = os.path.join(base_dir, "stock_data")
        unified_db = os.path.join(data_root, "stock_data.db")
        details_db = os.path.join(base_dir, "output", "score", "details", "details.db")
        return _abs_norm(unified_db), _abs_norm(details_db)

    stock_db = _abs_norm(os.path.join(DATA_ROOT, UNIFIED_DB_PATH))
    details_db = _abs_norm(os.path.join(SC_OUTPUT_DIR, SC_DETAIL_DB_PATH))
    return stock_db, details_db


def parse_args() -> argparse.Namespace:
    stock_db, details_db = load_config_paths()
    parser = argparse.ArgumentParser(description="用 ML 评估规则重要度（独立脚本，不改现有代码）")
    parser.add_argument("--stock-db", default=stock_db, help="统一行情数据库路径（含 stock_data 表）")
    parser.add_argument("--details-db", default=details_db, help="评分明细数据库路径（含 stock_details 表）")
    parser.add_argument(
        "--log-return",
        action="store_true",
        help="标签使用对数收益（默认用简单收益）",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        help="未来收益窗口天数，默认 5 10 20",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=5,
        help="训练标签使用的 horizon（需在 horizons 内）",
    )
    parser.add_argument("--start-date", type=str, default=None, help="ref_date 下限 YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=None, help="ref_date 上限 YYYYMMDD")
    parser.add_argument(
        "--min-hits",
        type=int,
        default=0,
        help="规则最少命中次数，低于则丢弃特征（0 表示不限）",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="按时间拆分的验证集比例，默认 0.2",
    )
    parser.add_argument(
        "--output-dir",
        default=_abs_norm(os.path.join(os.path.dirname(__file__), "..", "output", "rule_reports")),
        help="输出目录",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help="时间滚动验证折数（>1 启用），默认 1 表示只做单次时间拆分",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=400,
        help="XGBoost 训练轮数（num_boost_round），默认 400",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=50,
        help="早停轮数，<=0 表示关闭早停，默认 50",
    )
    return parser.parse_args()


def load_rule_hits(details_db: str, start: str | None, end: str | None) -> pd.DataFrame:
    """解析 details.db 的规则命中记录（仅 ok=True）。"""
    if not os.path.exists(details_db):
        raise FileNotFoundError(f"details 数据库不存在: {details_db}")

    con = duckdb.connect(details_db, read_only=True)
    try:
        sql = "SELECT ts_code, ref_date, score, rules FROM stock_details"
        params: List = []
        where = []
        if start:
            where.append("ref_date >= ?")
            params.append(start)
        if end:
            where.append("ref_date <= ?")
            params.append(end)
        if where:
            sql += " WHERE " + " AND ".join(where)
        df = con.execute(sql, params).df()
    finally:
        con.close()

    records: List[Dict] = []
    for row in df.itertuples(index=False):
        rules_raw = row.rules
        try:
            rules = json.loads(rules_raw) if isinstance(rules_raw, str) else []
        except json.JSONDecodeError:
            continue
        for r in rules:
            if not r or not r.get("ok", False):
                continue
            records.append(
                {
                    "rule": r.get("name", "unknown"),
                    "ts_code": row.ts_code,
                    "ref_date": str(row.ref_date),
                    "add": float(r.get("add", r.get("points", 0.0) or 0.0)),
                }
            )
    hits = pd.DataFrame(records)
    if hits.empty:
        raise RuntimeError("未解析到规则命中记录，检查 details 数据或日期范围")
    return hits


def load_price_panel(
    stock_db: str, codes: Iterable[str], min_date: str, max_date: str, horizons: List[int], log_return: bool
) -> pd.DataFrame:
    """读取价格并计算未来收益。"""
    if not os.path.exists(stock_db):
        raise FileNotFoundError(f"stock_data 数据库不存在: {stock_db}")
    codes = list(sorted(set(codes)))
    codes_df = pd.DataFrame({"ts_code": codes})

    con = duckdb.connect(stock_db, read_only=True)
    try:
        con.register("codes_df", codes_df)
        price_df = con.execute(
            """
            SELECT s.ts_code, s.trade_date, s.close
            FROM stock_data s
            JOIN codes_df c ON s.ts_code = c.ts_code
            WHERE s.adj_type = 'qfq'
              AND s.trade_date BETWEEN ? AND ?
            """,
            [min_date, max_date],
        ).df()
    finally:
        con.close()

    if price_df.empty:
        raise RuntimeError("未读取到价格数据，请检查 stock_data 表或日期范围")

    price_df["trade_date"] = price_df["trade_date"].astype(str)
    price_df = price_df.sort_values(["ts_code", "trade_date"])
    price_df = price_df.groupby("ts_code", group_keys=False).apply(
        lambda g: _add_forward_returns(g, horizons, log_return=log_return)
    )
    return price_df


def _add_forward_returns(df: pd.DataFrame, horizons: List[int], log_return: bool) -> pd.DataFrame:
    df = df.copy()
    df["close"] = df["close"].astype(float)
    for h in horizons:
        if log_return:
            df[f"ret_{h}d"] = np.log(df["close"].shift(-h) / df["close"])
        else:
            df[f"ret_{h}d"] = df["close"].shift(-h) / df["close"] - 1.0
    return df


def build_dataset(
    hits: pd.DataFrame, prices: pd.DataFrame, horizons: List[int], target_h: int, min_hits: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """构建特征矩阵 X 和标签 y。"""
    merged = hits.merge(
        prices,
        left_on=["ts_code", "ref_date"],
        right_on=["ts_code", "trade_date"],
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("规则命中与价格数据无交集")

    # 过滤低频规则
    rule_counts = merged["rule"].value_counts()
    keep_rules = set(rule_counts[rule_counts >= min_hits].index)
    merged = merged[merged["rule"].isin(keep_rules)].copy()
    if merged.empty:
        raise RuntimeError("过滤低频后无数据，请降低 --min-hits")

    # 构建特征：rule -> add（未命中补 0）
    feat = merged.pivot_table(
        index=["ts_code", "ref_date"],
        columns="rule",
        values="add",
        aggfunc="sum",
        fill_value=0.0,
    )
    feat = feat.sort_index()

    target_col = f"ret_{target_h}d"
    if target_col not in merged:
        raise RuntimeError(f"未找到标签列 {target_col}")
    target = (
        merged.drop_duplicates(subset=["ts_code", "ref_date"])
        .set_index(["ts_code", "ref_date"])
        [target_col]
    )
    # 对齐
    common_idx = feat.index.intersection(target.index)
    feat = feat.loc[common_idx]
    target = target.loc[common_idx]
    # 去除缺失标签
    mask = target.notna()
    feat = feat[mask]
    target = target[mask]
    if len(feat) < 100:
        raise RuntimeError("样本不足（<100），无法训练")
    return feat, target


def split_time(feat: pd.DataFrame, target: pd.Series, test_ratio: float = 0.2):
    """按 ref_date 时间拆分训练/验证。"""
    df = feat.copy()
    df["target"] = target
    df = df.reset_index()
    df["ref_date"] = df["ref_date"].astype(int)
    df = df.sort_values("ref_date")
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    X_train = train_df.drop(columns=["ts_code", "ref_date", "target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["ts_code", "ref_date", "target"])
    y_test = test_df["target"]
    return X_train, y_train, X_test, y_test


def split_time_folds(
    feat: pd.DataFrame, target: pd.Series, test_ratio: float, folds: int
) -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """时间滚动拆分，用于简单的时序交叉验证。"""
    df = feat.copy()
    df["target"] = target
    df = df.reset_index()
    df["ref_date"] = df["ref_date"].astype(int)
    df = df.sort_values("ref_date")
    test_size = max(1, int(len(df) * test_ratio))
    results: List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]] = []
    for i in range(max(1, folds)):
        test_end = len(df) - (max(1, folds) - 1 - i) * test_size
        test_start = test_end - test_size
        if test_start <= 0 or test_end <= test_start:
            continue
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:test_end]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        X_train = train_df.drop(columns=["ts_code", "ref_date", "target"])
        y_train = train_df["target"]
        X_test = test_df.drop(columns=["ts_code", "ref_date", "target"])
        y_test = test_df["target"]
        results.append((X_train, y_train, X_test, y_test))
    return results


def detect_tree_method() -> str:
    """优先使用 gpu_hist，失败时回落 hist。"""
    try:
        import xgboost  # noqa: F401
    except Exception:
        return "hist"
    # 简单检查 CUDA，可按需扩展
    if os.environ.get("XGBOOST_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
        return "hist"
    return "gpu_hist"


def train_xgb(
    X_train,
    y_train,
    X_test,
    y_test,
    tree_method: str,
    num_rounds: int,
    early_stopping: int,
):
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "tree_method": tree_method,
        "eval_metric": "rmse",
    }
    train_kwargs = {
        "params": params,
        "dtrain": dtrain,
        "num_boost_round": num_rounds,
        "evals": [(dtrain, "train"), (dtest, "test")],
        "verbose_eval": False,
    }
    if early_stopping and early_stopping > 0:
        train_kwargs["early_stopping_rounds"] = early_stopping
    try:
        booster = xgb.train(**train_kwargs)
    except xgb.core.XGBoostError:
        if tree_method == "gpu_hist":
            params["tree_method"] = "hist"
            booster = xgb.train(**train_kwargs)
        else:
            raise

    def _predict(bst, dmat):
        if getattr(bst, "best_iteration", None) is not None:
            return bst.predict(dmat, iteration_range=(0, bst.best_iteration + 1))
        return bst.predict(dmat)

    train_pred = _predict(booster, dtrain)
    test_pred = _predict(booster, dtest)
    metrics = {
        "train_rmse": float(np.sqrt(((train_pred - y_train) ** 2).mean())),
        "test_rmse": float(np.sqrt(((test_pred - y_test) ** 2).mean())),
        "train_mae": float(np.abs(train_pred - y_train).mean()),
        "test_mae": float(np.abs(test_pred - y_test).mean()),
        "train_mean": float(train_pred.mean()),
        "test_mean": float(test_pred.mean()),
        "test_size": int(len(y_test)),
        "best_iteration": int(getattr(booster, "best_iteration", num_rounds - 1)),
    }
    importance = booster.get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame(
            {"feature": list(importance.keys()), "gain": list(importance.values())}
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    return booster, metrics, imp_df


def save_outputs(
    output_dir: str,
    imp_df: pd.DataFrame,
    metrics: Dict[str, float],
    target_h: int,
    log_return: bool,
    cv_metrics: List[Dict[str, float]] | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    imp_path = os.path.join(output_dir, "rule_ml_importance.csv")
    md_path = os.path.join(output_dir, "rule_ml_summary.md")
    imp_df.to_csv(imp_path, index=False)

    md_lines = [
        "# 规则 ML 重要度报告",
        "",
        f"- 标签：未来 {target_h} 日收益（{'对数' if log_return else '简单'}收益）",
        f"- 训练/验证样本：{metrics.get('test_size', 0)} 条验证样本",
        f"- 训练 RMSE：{metrics.get('train_rmse'):.6f}",
        f"- 验证 RMSE：{metrics.get('test_rmse'):.6f}",
        f"- 验证 MAE：{metrics.get('test_mae'):.6f}",
        f"- 最佳迭代：{metrics.get('best_iteration', 0)}",
        "",
        "## Top 30 规则（按 gain）",
    ]
    md_lines.append(imp_df.head(30).to_markdown(index=False))

    if cv_metrics:
        test_rmses = [m["test_rmse"] for m in cv_metrics]
        test_maes = [m["test_mae"] for m in cv_metrics]
        md_lines.extend(
            [
                "",
                f"## 时间滚动验证（folds={len(cv_metrics)}）",
                f"- 验证 RMSE 均值：{np.mean(test_rmses):.6f}，标准差：{np.std(test_rmses):.6f}",
                f"- 验证 MAE 均值：{np.mean(test_maes):.6f}，标准差：{np.std(test_maes):.6f}",
            ]
        )
        for i, m in enumerate(cv_metrics):
            md_lines.append(
                f"  - Fold {i+1}: RMSE={m['test_rmse']:.6f}, MAE={m['test_mae']:.6f}, best_iter={m.get('best_iteration', 0)}"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main():
    args = parse_args()
    horizons = sorted(set([h for h in args.horizons if h > 0]))
    target_h = args.target_horizon
    if target_h not in horizons:
        raise ValueError("--target-horizon 必须包含在 --horizons 内")

    hits = load_rule_hits(args.details_db, args.start_date, args.end_date)
    min_ref = hits["ref_date"].min()
    max_ref = hits["ref_date"].max()
    max_h = max(horizons)

    price_df = load_price_panel(
        args.stock_db,
        codes=hits["ts_code"].unique(),
        min_date=min_ref,
        max_date=str(int(max_ref) + 1000),  # 粗略向后取一年，保证未来窗口
        horizons=horizons,
        log_return=args.log_return,
    )

    X, y = build_dataset(hits, price_df, horizons, target_h, args.min_hits)
    X_train, y_train, X_test, y_test = split_time(X, y, test_ratio=args.test_ratio)

    tree_method = detect_tree_method()
    cv_metrics: List[Dict[str, float]] = []
    try:
        model, metrics, imp_df = train_xgb(
            X_train,
            y_train,
            X_test,
            y_test,
            tree_method,
            num_rounds=args.num_rounds,
            early_stopping=args.early_stopping,
        )
    except ModuleNotFoundError:
        raise RuntimeError("未安装 xgboost，请先在当前环境安装：pip install xgboost")

    if args.cv_folds and args.cv_folds > 1:
        folds = split_time_folds(X, y, test_ratio=args.test_ratio, folds=args.cv_folds)
        for Xtr, ytr, Xte, yte in folds:
            _, m, _ = train_xgb(
                Xtr,
                ytr,
                Xte,
                yte,
                tree_method,
                num_rounds=args.num_rounds,
                early_stopping=args.early_stopping,
            )
            cv_metrics.append(m)

    save_outputs(args.output_dir, imp_df, metrics, target_h, args.log_return, cv_metrics)

    print(f"训练完成（tree_method={tree_method}）")
    print(f"验证集大小: {metrics.get('test_size', 0)}")
    print(f"验证 RMSE: {metrics.get('test_rmse'):.6f}, MAE: {metrics.get('test_mae'):.6f}")
    print(f"Top 10 规则按 gain：")
    print(imp_df.head(10).to_string(index=False))
    if cv_metrics:
        mean_rmse = np.mean([m["test_rmse"] for m in cv_metrics])
        mean_mae = np.mean([m["test_mae"] for m in cv_metrics])
        print(f"时间滚动验证 folds={len(cv_metrics)}: RMSE 均值={mean_rmse:.6f}, MAE 均值={mean_mae:.6f}")
    print(f"结果已写入: {args.output_dir}")


if __name__ == "__main__":
    main()
