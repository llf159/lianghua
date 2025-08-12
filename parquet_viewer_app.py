
# -*- coding: utf-8 -*-
"""
注意：
- 本应用基于你提供的 parquet_viewer.py 封装，因此请把 parquet_viewer_app.py 和 parquet_viewer.py 放在同一目录下。
- DuckDB 可选但强烈推荐（更快）。未安装时自动退回 pandas+pyarrow。
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
from typing import Optional, List, Tuple
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent  # 当前脚本所在的根目录
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parquet_viewer as pv

# 配置文件路径：放在当前用户目录下
CONFIG_PATH = Path.home() / ".parquet_viewer_app.json"
# 默认 base（也允许用环境变量覆盖）
DEFAULT_BASE = os.getenv("PARQUET_BASE", "E:\\stock_data")


# -------------------- 业务包装函数 --------------------
def normalize_ts(ts_input: str, asset: str) -> str:
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

def load_base(default_base: str) -> str:
    """读取上次保存的 base；失败或无配置时返回 default_base。"""
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            base = data.get("base")
            if isinstance(base, str) and base.strip():
                return base
    except Exception:
        pass
    return default_base

def save_base(base: str):
    """保存本次的 base 到配置文件；出错时静默忽略。"""
    try:
        CONFIG_PATH.write_text(
            json.dumps({"base": base}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass

def get_info(base: str) -> str:
    base = base or "DEFAULT_BASE"
    entries = [
        ("index/daily", pv.asset_root(base, "index", "daily")),
        ("stock/daily", pv.asset_root(base, "stock", "daily")),
        ("stock/daily_qfq", pv.asset_root(base, "stock", "daily_qfq")),
        ("stock/daily_hfq", pv.asset_root(base, "stock", "daily_hfq")),
    ]
    lines = []
    lines.append(f"数据目录：{os.path.abspath(base)}")
    lines.append(f"DuckDB：{'可用' if pv.HAS_DUCKDB else '不可用（建议 pip install duckdb）'}")
    for name, root in entries:
        dates = pv.list_trade_dates(root)
        if not dates:
            lines.append(f"- {name:<16}（无数据） 路径: {root}")
            continue
        lines.append(f"- {name:<16} {dates[0]} ~ {dates[-1]}  共 {len(dates)} 天  路径: {root}")
    return "\n".join(lines)

def get_dates(base: str, asset: str, adj: str) -> str:
    root = pv.asset_root(base, asset, adj if asset == "stock" else "daily")
    dates = pv.list_trade_dates(root)
    if not dates:
        return "（无分区）"
    return f"{asset} / {('daily' if asset=='index' else adj)}: {len(dates)} 天\n" + ", ".join(dates)

def run_day(base: str, asset: str, adj: str, date: str, limit: Optional[int]) -> pd.DataFrame:
    df = pv.read_day(base, asset, adj, date, limit)
    return df

def run_show(base: str, asset: str, adj: str, ts: str, start: str, end: str,
             columns: Optional[str], limit: Optional[int]) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    ts_norm = normalize_ts(ts, asset)
    cols = [c.strip() for c in columns.split(",")] if columns else None
    adj2 = adj if asset == "stock" else "daily"
    df = pv.read_range(base, asset, adj2, ts_norm, start, end, cols, limit)

    fig = None
    # 如果包含 trade_date 和 close，则画一张简单的收盘价曲线
    if isinstance(df, pd.DataFrame) and not df.empty:
        if ("trade_date" in df.columns) and (("close" in df.columns) or ("close" in (cols or []))):
            # 转换日期
            dfx = df.copy()
            try:
                dfx["trade_date"] = pd.to_datetime(dfx["trade_date"], format="%Y%m%d")
            except Exception:
                pass
            if "close" in dfx.columns:
                fig = plt.figure()
                plt.plot(dfx["trade_date"], dfx["close"])
                plt.title(f"{ts_norm} close price from {start} to {end}")
                plt.xlabel("Date")
                plt.ylabel("Close")
                plt.tight_layout()
    return df, fig

def run_schema(file_path: str) -> str:
    if not os.path.isfile(file_path):
        return f"文件不存在：{file_path}"
    try:
        if pv.HAS_PYARROW:
            import pyarrow.parquet as pq  # type: ignore
            pf = pq.ParquetFile(file_path)
            md = pf.metadata
            lines = [f"File: {file_path}",
                     f"Row Groups: {md.num_row_groups}, Rows: {md.num_rows}, Columns: {md.num_columns}",
                     "Schema:"]
            for i in range(md.num_columns):
                col = md.schema.column(i)
                lines.append(f"  - {col.name}: {col.physical_type}")
            return "\n".join(lines)
        else:
            df = pd.read_parquet(file_path)
            return f"File: {file_path}\nRows: {len(df)}, Columns: {len(df.columns)}\nColumns: {', '.join(df.columns)}"
    except Exception as e:
        return f"读取失败：{e}"

def run_export(base: str, asset: str, adj: str, ts: str, start: str, end: str,
               columns: Optional[str], out_dir: str) -> Tuple[str, Optional[str]]:
    ts_norm = normalize_ts(ts, asset)
    cols = [c.strip() for c in columns.split(",")] if columns else None
    adj2 = adj if asset == "stock" else "daily"
    df = pv.read_range(base, asset, adj2, ts_norm, start, end, cols, None)
    if df is None or df.empty:
        return "（空结果，未生成 CSV）", None
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, f"{ts_norm}_{start}_{end}.csv"))
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return f"已导出：{out_path}  行数={len(df)}", out_path

# -------------------- Gradio 界面 --------------------
def build_ui(base_default: str = DEFAULT_BASE):
    base_default = load_base(base_default)
    with gr.Blocks(title="Parquet 数据浏览器--基于Gradio") as demo:
        gr.Markdown("## Parquet 数据浏览器  \n基于 `parquet_viewer.py` 封装。")

        with gr.Tab("概览 info"):
            base_info = gr.Textbox(label="数据根目录 base", value=base_default)
            btn_info = gr.Button("刷新概览")
            out_info = gr.Textbox(label="概览", lines=10)
            btn_info.click(get_info, inputs=[base_info], outputs=[out_info])

        with gr.Tab("分区日期 dates"):
            base_dates = gr.Textbox(label="数据根目录 base", value=base_default)
            asset_dates = gr.Radio(choices=["index", "stock"], value="index", label="资产 asset")
            adj_dates = gr.Dropdown(choices=["daily", "daily_qfq", "daily_hfq"], value="daily", label="复权 adj（index 固定为 daily）")
            btn_dates = gr.Button("列出日期")
            out_dates = gr.Textbox(label="分区日期", lines=10)
            btn_dates.click(get_dates, inputs=[base_dates, asset_dates, adj_dates], outputs=[out_dates])

        with gr.Tab("某天 day"):
            base_day = gr.Textbox(label="数据根目录 base", value=base_default)
            asset_day = gr.Radio(choices=["index", "stock"], value="index", label="资产 asset")
            adj_day = gr.Dropdown(choices=["daily", "daily_qfq", "daily_hfq"], value="daily", label="复权 adj（index 固定为 daily）")
            date_day = gr.Textbox(label="交易日 YYYYMMDD")
            limit_day = gr.Number(label="最多显示行数（可空）", precision=10)
            btn_day = gr.Button("查询")
            out_day = gr.Dataframe(label="结果（day）")
            btn_day.click(run_day, inputs=[base_day, asset_day, adj_day, date_day, limit_day], outputs=[out_day])

        with gr.Tab("区间/代码 show"):
            base_show = gr.Textbox(label="数据根目录 base", value=base_default)
            asset_show = gr.Radio(choices=["index", "stock"], value="index", label="资产 asset")
            adj_show = gr.Dropdown(choices=["daily", "daily_qfq", "daily_hfq"], value="daily", label="复权 adj（index 固定为 daily）")
            ts_show = gr.Textbox(label="ts_code（如 000001）")
            start_show = gr.Textbox(label="起始日期 YYYYMMDD")
            end_show = gr.Textbox(label="结束日期 YYYYMMDD")
            cols_show = gr.Textbox(label="仅显示这些列（逗号分隔，可空）")
            limit_show = gr.Number(label="最多显示行数（可空）", precision=10)
            btn_show = gr.Button("查询")
            out_show_df = gr.Dataframe(label="结果（show）")
            out_show_fig = gr.Plot(label="（可选）收盘价曲线：若存在 trade_date + close 列自动绘图")
            btn_show.click(run_show,
                           inputs=[base_show, asset_show, adj_show, ts_show, start_show, end_show, cols_show, limit_show],
                           outputs=[out_show_df, out_show_fig])

        with gr.Tab("Schema / 统计"):
            file_schema = gr.Textbox(label="Parquet 文件路径")
            btn_schema = gr.Button("查看 schema")
            out_schema = gr.Textbox(label="Schema / 统计", lines=12)
            btn_schema.click(run_schema, inputs=[file_schema], outputs=[out_schema])

        with gr.Tab("导出 CSV"):
            base_exp = gr.Textbox(label="数据根目录 base", value=base_default)
            asset_exp = gr.Radio(choices=["index", "stock"], value="index", label="资产 asset")
            adj_exp = gr.Dropdown(choices=["daily", "daily_qfq", "daily_hfq"], value="daily", label="复权 adj（index 固定为 daily）")
            ts_exp = gr.Textbox(label="ts_code（如 000001）")
            start_exp = gr.Textbox(label="起始日期 YYYYMMDD")
            end_exp = gr.Textbox(label="结束日期 YYYYMMDD")
            cols_exp = gr.Textbox(label="仅导出这些列（逗号分隔，可空）")
            out_dir = gr.Textbox(label="导出目录（将创建，不含文件名）", value="exports")
            btn_export = gr.Button("导出 CSV")
            out_msg = gr.Textbox(label="导出结果")
            out_file = gr.File(label="下载 CSV", interactive=False)
            btn_export.click(run_export,
                             inputs=[base_exp, asset_exp, adj_exp, ts_exp, start_exp, end_exp, cols_exp, out_dir],
                             outputs=[out_msg, out_file])

        gr.Markdown("> 小贴士：安装了duckdb查询速度会更快。未安装时会自动退回 pandas+pyarrow。")

    return demo

def _enable_queue_compat(demo):
    """
    适配不同 Gradio 版本的 queue() 参数：
    - Gradio 3.x: queue(concurrency_count=...)
    - Gradio 4.x+: queue(default_concurrency_limit=...) 或 queue()
    """
    import inspect
    try:
        sig = inspect.signature(demo.queue)
        if "concurrency_count" in sig.parameters:
            return demo.queue(concurrency_count=20)
        if "default_concurrency_limit" in sig.parameters:
            return demo.queue(default_concurrency_limit=20)
        # 参数名都不支持时，直接启用队列
        return demo.queue()
    except Exception:
        # 若出现异常，退化为不传参数
        return demo.queue()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="开启公网/局域网临时可访问链接")
    args = parser.parse_args()

    demo = build_ui()
    _enable_queue_compat(demo)
    try:
        import gradio as gr
        print(f"[Info] Gradio version: {getattr(gr, '__version__', 'unknown')}")
    except Exception:
        pass
    demo.launch(share=args.share)

if __name__ == "__main__":
    main()
