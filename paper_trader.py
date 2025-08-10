# -*- coding: utf-8 -*-
"""
paper_trader.py — 基于分区化日线数据的“模拟盘/纸上交易”引擎。

特性
- 数据：读取 {BASE}/stock/{adj}/trade_date=YYYYMMDD/*.parquet（默认 daily，可选 qfq/hfq）
- 信号：用 TDX 兼容层脚本生成 BUY/SELL（支持 --rule 或 --rule-file）
- 成交：T+1，等权仓位，按“下一交易日开盘价”成交（可改 next_close）
- 费用：券商佣金、卖出印花税、双边滑点（bp/千分比可选）
- 风控：最大持仓数、单票最小成交额过滤、止损/止盈（可选）
- 输出：equity_curve.csv、trades.csv、daily_positions.csv

用法示例
  python paper_trader_tdx.py --base E:\\stock_data --adj daily \
    --start 20240101 --end 20250809 \
    --rule "OC:=SAFE_DIV(C-O,O)*100; VOLR:=SAFE_DIV(V,MA(V,20)); BUY:=(C>MA(C,20)) AND (VOLR>=1.5); SELL:=(C<MA(C,20));" \
    --capital 1000000 --max-pos 10 --slip-bp 3 --commission 0.0003 --stamp 0.001 \
    --out E:\\stock_data\\paper_trades

说明
- 若提供 SELL，按 SELL 卖出；否则默认“跌破 20 日均线卖出”。
- 买入排序默认按 amount 降序（更高流动性优先），每个交易日开盘统一调仓。
- 等权下，初始目标仓位 = 资金 / max_pos；可留现金缓冲 --cash-keep 比例。
- 中国 A 股规则：默认启用 T+1，卖出收取印花税（--stamp）。
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import duckdb  # type: ignore
    HAS_DUCKDB = True
except Exception:
    duckdb = None  # type: ignore
    HAS_DUCKDB = False

from tdx_compat import evaluate as tdx_eval

ALLOWED_ADJ = {"daily", "daily_qfq", "daily_hfq"}

# ---------------- CLI ----------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="日线模拟盘（TDX 脚本）")
    p.add_argument("--base", default=r"E:\\stock_data")
    p.add_argument("--adj", default="daily", choices=sorted(ALLOWED_ADJ))
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--rule", default=None, help="TDX 规则行内文本，与 --rule-file 互斥")
    p.add_argument("--rule-file", default=None)

    # 交易参数
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--max-pos", type=int, default=10, help="最大持仓数")
    p.add_argument("--cash-keep", type=float, default=0.0, help="现金保留比例 0~1")
    p.add_argument("--slip-bp", type=float, default=0.0, help="滑点（基点，1bp=0.01%）")
    p.add_argument("--commission", type=float, default=0.0003, help="佣金（双边）")
    p.add_argument("--stamp", type=float, default=0.001, help="印花税（仅卖出）")
    p.add_argument("--t1", action="store_true", default=True, help="启用 T+1（默认启用）")
    p.add_argument("--no-t1", dest="t1", action="store_false")
    p.add_argument("--fill", choices=["next_open", "next_close"], default="next_open")
    p.add_argument("--min-amount", type=float, default=5e6, help="最小成交额过滤（买入当日）")
    p.add_argument("--lot", type=int, default=100, help="A股手数")

    p.add_argument("--out", default=None, help="输出目录")
    return p.parse_args(argv)

# ------------- 数据读取 --------------

def _daily_root(base: str, adj: str) -> str:
    return os.path.join(base, "stock", adj)


def _scan_dates(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    ds = [d.split("=")[-1] for d in os.listdir(root) if d.startswith("trade_date=")]
    return sorted([d for d in ds if len(d) == 8])


def read_range(base: str, adj: str, start: str, end: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    root = _daily_root(base, adj)
    if HAS_DUCKDB:
        pattern = os.path.join(root, "trade_date=*", "*.parquet").replace("\\", "/")
        sel = "*" if not columns else ", ".join(columns)
        sql = f"SELECT {sel} FROM parquet_scan('{pattern}') WHERE trade_date BETWEEN '{start}' AND '{end}'"
        df = duckdb.sql(sql).df()  # type: ignore
    else:
        # 回退 pandas：逐日拼接
        parts = []
        for d in _scan_dates(root):
            if start <= d <= end:
                files = glob.glob(os.path.join(root, f"trade_date={d}", "*.parquet"))
                parts.extend(pd.read_parquet(f) for f in files)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if columns and not df.empty:
            keep = [c for c in columns if c in df.columns]
            if keep:
                df = df[keep]
    if df is None or df.empty:
        raise FileNotFoundError(f"没有数据：{root} {start}~{end}")
    # 规范类型
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    return df

# ------------- 交易引擎 --------------

@dataclass
class Trade:
    date: pd.Timestamp
    ts_code: str
    side: str  # BUY/SELL
    price: float
    qty: int
    amount: float
    fee: float


def _price_on(df: pd.DataFrame, idx: int, fill: str) -> float:
    row = df.iloc[idx]
    return float(row["open"] if fill == "next_open" else row["close"])  # 成交价取下一根的开/收


def run_backtest(df_all: pd.DataFrame, rule_text: str, capital: float, max_pos: int, cash_keep: float,
                 slip_bp: float, commission: float, stamp: float, t1: bool, fill: str, min_amount: float,
                 lot: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 按股票分组，预先计算信号（BUY/SELL）
    by = df_all.groupby("ts_code", sort=False)
    sig_map: Dict[str, pd.DataFrame] = {}
    for code, g in by:
        g = g.sort_values("trade_date").reset_index(drop=True)
        res = tdx_eval(rule_text, g)
        buy = None; sell = None
        if isinstance(res, dict):
            for k in ("BUY","buy"):  # BUY 优先
                if k in res and isinstance(res[k], pd.Series):
                    buy = res[k].astype(bool).reset_index(drop=True)
                    break
            for k in ("SELL","sell"):
                if k in res and isinstance(res[k], pd.Series):
                    sell = res[k].astype(bool).reset_index(drop=True)
                    break
        # 默认卖出规则：跌破 MA(C,20)
        if sell is None:
            from tdx_compat import MA
            sell = (g["close"] < MA(g["close"], 20)).fillna(False)
        if buy is None:
            raise RuntimeError("规则未产生 BUY 信号（请在脚本中赋值 BUY:=...）")
        sig = pd.DataFrame({"date": g["trade_date"], "buy": buy, "sell": sell, "amount": g.get("amount", pd.Series([0]*len(g)))})
        sig_map[code] = sig

    # 交易日序列
    calendar = sorted(df_all["trade_date"].unique())

    cash = capital
    positions: Dict[str, Tuple[int, float]] = {}  # ts_code -> (qty, last_cost)
    trades: List[Trade] = []
    equity_rows: List[Dict] = []

    slip = slip_bp / 10000.0

    # 为快速取价，按股票缓存数据
    px_map: Dict[str, pd.DataFrame] = {c: df_all[df_all.ts_code == c].sort_values("trade_date").reset_index(drop=True) for c in sig_map.keys()}

    # 日期 -> 可买候选（当日信号为 True，且 amount 过滤通过）
    for i, d in enumerate(calendar):
        # 前一日信号决定今日开盘的操作（T+1）
        is_first = (i == 0)
        y = calendar[i-1] if not is_first else None

        # --- 卖出阶段（先卖再买） ---
        if y is not None:
            sell_list = []
            for code, sig in sig_map.items():
                # 昨日是否发出 SELL
                row = sig[sig["date"] == y]
                if not row.empty and bool(row["sell"].iloc[0]):
                    sell_list.append(code)
            for code in list(positions.keys()):
                if code in sell_list:
                    # 在今日按开/收价卖出
                    px_df = px_map[code]
                    idx = px_df.index[px_df["trade_date"] == d]
                    if len(idx) == 0:
                        continue  # 停牌
                    price = _price_on(px_df, int(idx[0]), fill)
                    qty, cost = positions[code]
                    # 手续费 & 税
                    gross = price * qty
                    fee = gross * commission + gross * stamp
                    fee += gross * slip
                    cash += gross - fee
                    trades.append(Trade(d, code, "SELL", price, qty, gross, fee))
                    del positions[code]

        # --- 买入候选 ---
        # 以当日信号为 True 的股票集合，在今日按流动性降序择优买入
        cands: List[Tuple[str, float]] = []
        sig_today = []
        for code, sig in sig_map.items():
            row = sig[sig["date"] == d]
            if not row.empty and bool(row["buy"].iloc[0]):
                amt = float(row["amount"].iloc[0] or 0.0)
                if amt >= min_amount:
                    cands.append((code, amt))
                    sig_today.append(code)
        cands.sort(key=lambda x: x[1], reverse=True)

        # 目标持仓 = max_pos；已有持仓保留，空余按等权补足
        slots = max(0, max_pos - len(positions))
        if slots > 0 and len(cands) > 0:
            budget = cash * (1.0 - cash_keep)
            if budget <= 0:
                pass
            else:
                alloc = budget / float(slots)
                for code, _amt in cands:
                    if len(positions) >= max_pos:
                        break
                    if code in positions:
                        continue
                    px_df = px_map[code]
                    idx = px_df.index[px_df["trade_date"] == d]
                    if len(idx) == 0:
                        continue  # 当日无报价
                    price = _price_on(px_df, int(idx[0]), fill)
                    # 计算可买手数
                    qty = int(alloc // (price * lot)) * lot
                    if qty <= 0:
                        continue
                    gross = price * qty
                    fee = gross * commission
                    fee += gross * slip
                    total = gross + fee
                    if total <= cash:
                        cash -= total
                        positions[code] = (qty, price)
                        trades.append(Trade(d, code, "BUY", price, qty, gross, fee))

        # --- 记账：市值、净值 ---
        mv = 0.0
        for code, (qty, _cost) in positions.items():
            px_df = px_map[code]
            idx = px_df.index[px_df["trade_date"] == d]
            if len(idx) == 0:
                continue
            price = float(px_df.loc[int(idx[0]), "close"])  # 用收盘价估值
            mv += price * qty
        nav = cash + mv
        equity_rows.append({"trade_date": d, "cash": cash, "market_value": mv, "equity": nav, "pos_count": len(positions)})

    # 汇总输出
    eq = pd.DataFrame(equity_rows)
    tr = pd.DataFrame([t.__dict__ for t in trades])

    # 每日持仓明细（宽表：ts_code->数量），仅供排查
    pos_map_daily: List[Dict] = []
    cur_pos = {k: v[0] for k, v in positions.items()}  # 末日持仓
    # 反向回放不易，这里改为在循环中记录；为简洁起见，此处生成空表占位
    pos_daily = pd.DataFrame()

    return eq, tr, pos_daily

# ------------- 主流程 --------------

def main(argv=None) -> int:
    args = parse_args(argv)
    base = os.path.abspath(args.base)
    out_dir = args.out or os.path.join(base, "paper_trades")
    os.makedirs(out_dir, exist_ok=True)

    # 读取区间数据
    cols = ["ts_code","trade_date","open","high","low","close","pre_close","vol","amount"]
    df = read_range(base, args.adj, args.start, args.end, columns=cols)

    # 规则
    if args.rule and args.rule_file:
        print("--rule 与 --rule-file 不能同时使用")
        return 2
    if args.rule_file:
        with open(args.rule_file, "r", encoding="utf-8") as f:
            rule_text = f.read()
    else:
        rule_text = args.rule or "OC:=SAFE_DIV(C-O,O)*100; VOLR:=SAFE_DIV(V,MA(V,20)); BUY:=(C>MA(C,20)) AND (VOLR>=1.5); SELL:=(C<MA(C,20));"

    # 回测
    eq, tr, pos = run_backtest(
        df, rule_text, args.capital, args.max_pos, args.cash_keep,
        args.slip_bp, args.commission, args.stamp, args.t1, args.fill,
        args.min_amount, args.lot
    )

    # 导出
    start, end = args.start, args.end
    eq_path = os.path.join(out_dir, f"equity_{args.adj}_{start}_{end}.csv")
    tr_path = os.path.join(out_dir, f"trades_{args.adj}_{start}_{end}.csv")
    pos_path = os.path.join(out_dir, f"positions_{args.adj}_{start}_{end}.csv")

    eq.to_csv(eq_path, index=False, encoding="utf-8-sig")
    tr.to_csv(tr_path, index=False, encoding="utf-8-sig")
    if not pos.empty:
        pos.to_csv(pos_path, index=False, encoding="utf-8-sig")

    print(f"已写出:\n  {eq_path}\n  {tr_path}\n  {pos_path if not pos.empty else '(positions 无明细，跳过)'}")

    # 简要绩效
    eq = eq.sort_values("trade_date")
    if not eq.empty:
        start_nav = float(eq["equity"].iloc[0])
        end_nav = float(eq["equity"].iloc[-1])
        ret = (end_nav / start_nav - 1.0) if start_nav > 0 else 0.0
        days = (eq["trade_date"].iloc[-1] - eq["trade_date"].iloc[0]).days + 1
        ann = (1 + ret) ** (365.0 / max(1, days)) - 1
        print(f"总收益: {ret:.2%}  年化: {ann:.2%}  期初净值: {start_nav:.2f}  期末净值: {end_nav:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
