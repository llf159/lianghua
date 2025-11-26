
# -*- coding: utf-8 -*-
"""
数据查看器应用
- 基于parquet_viewer.py封装
- 支持DuckDB和pandas+pyarrow两种数据访问方式
- 提供数据概览和完整性检查功能
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from pathlib import Path
import json
import datetime as dt
import re
from log_system import get_logger

# 初始化日志记录器
logger = get_logger("app_pv")

# 网络配置
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

ROOT = Path(__file__).resolve().parent  # 当前脚本所在的根目录
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import normalize_ts
from database_manager import get_database_manager

# ========== 数据库适配器 ==========
class DatabaseAdapter:
    """数据库适配器，提供与 parquet_viewer.py 兼容的接口"""
    
    def __init__(self):
        # 使用统一的数据库管理器
        logger.info("[数据库连接] 开始获取数据库管理器实例 (初始化数据库适配器)")
        self.db_manager = get_database_manager()
        # 添加数据库路径用于直接查询
        from config import DATA_ROOT, UNIFIED_DB_PATH
        self.db_path = os.path.abspath(os.path.join(DATA_ROOT, UNIFIED_DB_PATH))
    
    def list_trade_dates(self, root: str) -> List[str]:
        """获取交易日期列表（兼容 parquet_viewer.list_trade_dates）"""
        try:
            # 使用数据库管理器获取交易日期
            return self.db_manager.get_trade_dates(self.db_path)
        except Exception as e:
            logger.error(f"获取交易日期失败: {e}")
            return []
    
    def _extract_adj_type_from_path(self, root: str) -> str:
        """从路径中提取复权类型"""
        if "raw" in root:
            return "raw"
        elif "qfq" in root:
            return "qfq"
        elif "hfq" in root:
            return "hfq"
        elif "ind" in root:
            return "ind"
        else:
            return "qfq"  # 默认使用qfq
    
    def get_connection(self):
        """获取数据库连接的上下文管理器 - 使用统一的数据库管理器"""
        return self.db_manager.get_connection(self.db_path)
    
    def _apply_precision_and_limit(self, df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
        """应用精度控制和限制"""
        if df.empty:
            return df
        
        # 应用精度控制
        price_cols = ["open", "high", "low", "close", "vol", "amount"]
        for col in price_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(float).round(2)
        
        # 对指标列应用精度控制
        indicator_cols = ["kdj_k", "kdj_d", "kdj_j", "rsi", "bbi", "ma5", "ma10", "ma20", "ma60"]
        for col in indicator_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(float).round(2)
        
        # 应用限制
        if limit is not None:
            if limit < 0:
                df = df.tail(-limit)
            else:
                df = df.head(limit)
        
        return df
    
    def read_day(self, base: str, asset: str, adj: str, day: str, limit: Optional[int] = None) -> pd.DataFrame:
        """读取某一天的数据（兼容 parquet_viewer.read_day）"""
        try:
            # 转换复权类型
            adj_type = adj if adj != "daily" else "raw"
            
            # 使用数据库管理器查询单日数据
            df = self.db_manager.query_stock_data(
                db_path=self.db_path,
                start_date=day,
                end_date=day,
                adj_type=adj_type
            )
            
            return self._apply_precision_and_limit(df, limit)
        except Exception as e:
            logger.error(f"查询单日数据失败: {e}")
            return pd.DataFrame()
    
    def read_range(self, base: str, asset: str, adj: str, ts_code: str, start: str, end: str,
                   columns: Optional[List[str]] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """读取日期范围的数据（兼容 parquet_viewer.read_range）"""
        try:
            # 转换复权类型
            adj_type = adj if adj != "daily" else "raw"
            
            # 使用数据库管理器查询范围数据
            df = self.db_manager.query_stock_data(
                db_path=self.db_path,
                ts_code=ts_code,
                start_date=start,
                end_date=end,
                columns=columns,
                adj_type=adj_type
            )
            
            # 应用精度控制和限制
            df = self._apply_precision_and_limit(df, limit)
            
            return df
        except Exception as e:
            logger.error(f"查询范围数据失败: {e}")
            return pd.DataFrame()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            # 使用数据库管理器获取数据库信息
            return self.db_manager.get_database_info(self.db_path)
        except Exception as e:
            logger.error(f"获取数据库统计信息失败: {e}")
            return {}

# 全局数据库适配器实例
_db_adapter = None

def get_db_adapter() -> DatabaseAdapter:
    """获取数据库适配器实例（单例模式）"""
    global _db_adapter
    if _db_adapter is None:
        _db_adapter = DatabaseAdapter()
    return _db_adapter

# 配置文件路径：放在当前用户目录下
CONFIG_PATH = Path.home() / ".parquet_viewer_app.json"
# 默认 base（也允许用环境变量覆盖）
DEFAULT_BASE = os.getenv("PARQUET_BASE", "E:\\stock_data")


# -------------------- 小工具 --------------------
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


def _date_list(root: str) -> List[str]:
    # 使用数据库适配器
    try:
        db_adapter = get_db_adapter()
        return db_adapter.list_trade_dates(root)
    except Exception as e:
        logger.error(f"获取交易日期失败: {e}")
        return []


def _date_gaps(dates: List[str]) -> List[str]:
    """粗略检查日期不连续（忽略周末/节假日，设置 >21 天为可疑缺口）。"""
    gaps: List[str] = []
    for i in range(len(dates)-1):
        d0, d1 = dates[i], dates[i+1]
        try:
            x0 = dt.datetime.strptime(d0, "%Y%m%d")
            x1 = dt.datetime.strptime(d1, "%Y%m%d")
            delta = (x1 - x0).days
            if delta > 21:
                gaps.append(f"{d0}->{d1}({delta}d)")
        except Exception:
            continue
    return gaps


def _distinct_count_latest_k(root: str, k: int = 3) -> Dict[str, int]:
    """
    统计最近 k 个有分区的 trade_date 的行数（或去重 ts_code 数）。
    返回 {trade_date: rows}
    """
    try:
        # 使用数据库适配器
        db_adapter = get_db_adapter()
        dates = db_adapter.list_trade_dates(root)
        if not dates:
            return {}
        
        pick = dates[-k:]
        out: Dict[str, int] = {}
        
        # 使用数据库管理器查询每天的行数
        for date in pick:
            try:
                df = db_adapter.db_manager.query_stock_data(
                    db_path=db_adapter.db_path,
                    start_date=date,
                    end_date=date
                )
                out[date] = len(df)
            except Exception as e:
                logger.warning(f"查询日期 {date} 数据失败: {e}")
                out[date] = 0
        
        return out
    except Exception as e:
        logger.error(f"数据库查询行数失败: {e}")
        return {}


def _stock_universe_size(base: str) -> Optional[int]:
    """尝试读取 {base}/stock_list.csv 作为对照的股票基数。"""
    f = os.path.join(base, "stock_list.csv")
    if os.path.isfile(f):
        try:
            df = pd.read_csv(f, dtype=str)
            return int(len(df))
        except Exception:
            return None
    return None


def _indicators_alignment(base: str) -> List[Dict[str, Any]]:
    """比较 daily_{adj} 与 daily_{adj}_indicators 的覆盖对齐情况。"""
    rows: List[Dict[str, Any]] = []
    try:
        # 使用数据库适配器获取统计信息
        db_adapter = get_db_adapter()
        stats = db_adapter.get_database_stats()
        
        for kind in ["raw", "qfq", "hfq"]:
            try:
                # 从数据库统计信息中获取数据
                if kind in stats and isinstance(stats[kind], dict):
                    base_last = stats[kind].get('latest_date')
                    base_exists = stats[kind].get('count', 0) > 0
                    
                    # 对于数据库版本，基础数据和指标数据都在同一个表中
                    # 所以基础数据和指标数据是同步的
                    ind_exists = base_exists
                    ind_last = base_last
                    lag = 0  # 数据库版本中基础数据和指标数据是同步的
                else:
                    base_exists = False
                    ind_exists = False
                    base_last = None
                    ind_last = None
                    lag = None
                
                rows.append(dict(adj=kind, base_exists=base_exists, ind_exists=ind_exists,
                                 base_last=base_last, ind_last=ind_last, lag=lag))
            except Exception:
                # 某个 kind 不存在时安静跳过
                continue
    except Exception:
        # 如果数据库查询失败，返回空结果
        logger.warning("数据库查询失败，无法获取指标对齐信息")
        for kind in ["raw", "qfq", "hfq"]:
            rows.append(dict(adj=kind, base_exists=False, ind_exists=False,
                             base_last=None, ind_last=None, lag=None))
    return rows


def _health_score(entry: Dict[str, Any]) -> Tuple[str, str]:
    """
    根据规则给出健康状态：(status, reason)
    - 日期缺口>0 或 最新行数明显小于基数*0.6 → WARN/CRIT
    - 指标目录存在且落后>0 → WARN
    """
    gaps = entry.get("gap_cnt", 0) or 0
    latest_rows = entry.get("latest_rows", 0) or 0
    expected = entry.get("expected_rows")
    ind_lag = entry.get("ind_lag")

    # 规则
    if gaps > 0:
        return "WARN", "存在日期缺口"
    if expected is not None and latest_rows > 0 and latest_rows < max(100, expected * 0.6):
        return "CRIT", "最新样本日行数显著偏低"
    if isinstance(ind_lag, int) and ind_lag < 0:
        return "WARN", "指标目录落后基础目录"
    return "OK", ""


def _adj_dir_name(adj_kind: str) -> str:
    amap = {"daily":"daily", "raw":"daily_raw", "qfq":"daily_qfq", "hfq":"daily_hfq"}
    k = str(adj_kind).lower()
    return amap.get(k, "daily_qfq")


# -------------------- 底层函数 --------------------
def overview_df(base: str, adj_kind: str='qfq') -> pd.DataFrame:
    base = base or "DEFAULT_BASE"
    adj_dir = _adj_dir_name(adj_kind)
    
    # 使用数据库适配器获取数据
    try:
        db_adapter = get_db_adapter()
        dates = db_adapter.list_trade_dates(base)
        db_info = db_adapter.get_database_stats()
        
        # 构建概览信息
        rows: List[Dict[str, Any]] = []
        
        if dates:
            # 数据库有数据
            gaps = _date_gaps(dates)
            latest_counts = _distinct_count_latest_k(base, k=1)
            latest_date = list(latest_counts.keys())[-1] if latest_counts else dates[-1]
            latest_rows = list(latest_counts.values())[-1] if latest_counts else 0
            
            # 获取股票基数
            stock_base = _stock_universe_size(base)
            
            # 创建概览条目
            item = {
                "name": f"stock/{adj_dir}",
                "path": base,
                "first": dates[0],
                "last": dates[-1],
                "days": len(dates),
                "gap_cnt": len(gaps),
                "gap_samples": "; ".join(gaps[:3]) if gaps else "",
                "latest_date": latest_date,
                "latest_rows": latest_rows,
                "expected_rows": stock_base,
                "files_latest": 0,  # 数据库模式不统计文件数
                "ind_lag": 0,  # 数据库模式中基础数据和指标数据是同步的
            }
            
            status, reason = _health_score(item)
            item.update(status=status, reason=reason)
            rows.append(item)
        else:
            # 数据库无数据
            item = {
                "name": f"stock/{adj_dir}",
                "path": base,
                "first": None,
                "last": None,
                "days": 0,
                "gap_cnt": None,
                "gap_samples": None,
                "latest_date": None,
                "latest_rows": None,
                "expected_rows": None,
                "files_latest": None,
                "ind_lag": None,
                "status": "EMPTY",
                "reason": "无数据"
            }
            rows.append(item)
            
    except Exception as e:
        logger.error(f"生成概览失败: {e}")
        # 返回错误信息
        item = {
            "name": f"stock/{adj_dir}",
            "path": base,
            "first": None,
            "last": None,
            "days": 0,
            "gap_cnt": None,
            "gap_samples": None,
            "latest_date": None,
            "latest_rows": None,
            "expected_rows": None,
            "files_latest": None,
            "ind_lag": None,
            "status": "ERROR",
            "reason": f"数据库错误: {e}"
        }
        rows = [item]

    df = pd.DataFrame(rows, columns=[
        "name","status","reason","first","last","days","gap_cnt","gap_samples",
        "latest_date","latest_rows","expected_rows","files_latest","ind_lag","path"
    ])
    return df


def get_info(base: str, adj_kind: str='qfq') -> str:
    """保留一个精简文本版（方便复制），详细请看"概览表格"与"诊断建议"。"""
    base = base or "DEFAULT_BASE"
    lines = []
    lines.append(f"数据目录：{os.path.abspath(base)}")
    
    # 检查数据库状态
    try:
        db_adapter = get_db_adapter()
        db_info = db_adapter.get_database_stats()
        if db_info.get('exists', False):
            lines.append(f"数据库：可用 (大小: {db_info.get('size', 0)} 字节)")
        else:
            lines.append("数据库：不可用")
    except Exception as e:
        lines.append(f"数据库状态检查失败: {e}")
    
    try:
        df = overview_df(base, adj_kind)
        for _, r in df.iterrows():
            lines.append(
                f"- {r['name']:<16} {str(r['first'])} ~ {str(r['last'])} 共{int(r['days'])}天 "
                f"最新日({r['latest_date']})行数={int(r['latest_rows'])} "
                f"缺口={int(r['gap_cnt']) if pd.notna(r['gap_cnt']) else 0} "
                f"状态={r['status']} 路径:{r['path']}"
            )
    except Exception as e:
        lines.append(f"[概览生成失败] {e}")
    return "\n".join(lines)


def overview_table(base: str, adj_kind: str='qfq') -> pd.DataFrame:
    """用于 UI 显示的完整性表格。"""
    return overview_df(base, adj_kind)


def overview_advice(base: str, adj_kind: str='qfq') -> str:
    """根据表格给出动作建议。"""
    try:
        df = overview_df(base, adj_kind)
        tips = []
        for _, r in df.iterrows():
            name = r["name"]
            st = r["status"]
            reason = r["reason"]
            if st == "EMPTY":
                tips.append(f"【{name}】目录为空 → 检查下载/合并是否执行；或确认 base 路径是否正确。")
            elif st in ("WARN","CRIT"):
                if reason == "存在日期缺口":
                    tips.append(f"【{name}】存在日期缺口: {r['gap_samples']} → 建议对缺口区间重跑增量同步。")
                if reason == "最新样本日行数显著偏低":
                    exp = r['expected_rows']
                    tips.append(f"【{name}】{r['latest_date']} 行数={r['latest_rows']}，期望≈{exp} → 可能分区漏写或合并失败，检查日志。")
                if pd.notna(r.get("ind_lag", None)) and int(r["ind_lag"]) < 0:
                    tips.append(f"【{name}】指标目录落后基础目录 → 触发一次指标合并(duckdb_merge_symbol_products_to_daily)。")
        if not tips:
            return "✅ 初步检查：一切正常。"
        return "\n".join(f"- {t}" for t in tips)
    except Exception as e:
        return f"[生成建议失败] {e}"


# -------------------- 业务包装函数（保留原有功能） --------------------
def get_dates(base: str, asset: str, adj: str) -> str:
    # 根据资产类型和复权类型确定实际使用的复权类型
    if asset == "index":
        actual_adj = "ind"  # 指数使用ind复权类型
    else:
        actual_adj = adj  # 股票使用选择的复权类型
    
    # 使用数据库适配器
    try:
        db_adapter = get_db_adapter()
        dates = db_adapter.list_trade_dates(base)
    except Exception as e:
        logger.error(f"获取交易日期失败: {e}")
        return f"获取交易日期失败: {e}"
    
    if not dates:
        return "（无分区）"
    return f"{asset} / {actual_adj}: {len(dates)} 天\n" + ", ".join(dates)


def run_day(base: str, asset: str, adj: str, date: str, limit: Optional[int]) -> pd.DataFrame:
    limit_clean = (limit or "").strip()
    is_int = limit_clean.lstrip("-").isdigit()
    limit = int(limit_clean) if is_int else None
    
    # 使用数据库适配器
    try:
        db_adapter = get_db_adapter()
        df = db_adapter.read_day(base, asset, adj, date, limit)
    except Exception as e:
        logger.error(f"查询单日数据失败: {e}")
        return pd.DataFrame()
    
    return df


def run_show(base: str, asset: str, adj: str, ts: str, start: str, end: str,
             columns: Optional[str], limit: Optional[int]) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    ts_norm = normalize_ts(ts, asset)
    cols = [c.strip() for c in re.split(r"[，,;；\s]+", columns) if c.strip()] if columns else None
    # 根据资产类型确定复权类型
    if asset == "index":
        adj2 = "ind"  # 指数使用ind复权类型
    else:
        adj2 = adj  # 股票使用选择的复权类型
    limit_clean = (limit or "").strip()
    is_int = limit_clean.lstrip("-").isdigit()
    limit = int(limit_clean) if is_int else None
    
    # 使用数据库适配器
    try:
        db_adapter = get_db_adapter()
        df = db_adapter.read_range(base, asset, adj2, ts_norm, start, end, cols, limit)
    except Exception as e:
        logger.error(f"查询范围数据失败: {e}")
        df = pd.DataFrame()

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


def run_schema(file_path: str = "", file_obj=None) -> str:
    if not os.path.isfile(file_path):
        if file_obj:
            file_path = file_obj.name
        else:
            return f"文件不存在：{file_path}"
    try:
        # 尝试使用pyarrow读取parquet文件
        try:
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
        except ImportError:
            # 回退到pandas
            df = pd.read_parquet(file_path)
            return f"File: {file_path}\nRows: {len(df)}, Columns: {len(df.columns)}\nColumns: {', '.join(df.columns)}"
    except Exception as e:
        return f"读取失败：{e}"


def run_export(base: str, asset: str, adj: str, ts: str, start: str, end: str,
               columns: Optional[str], out_dir: str) -> Tuple[str, Optional[str]]:
    ts_norm = normalize_ts(ts, asset)
    cols = [c.strip() for c in columns.split(",")] if columns else None
    # 根据资产类型确定复权类型
    if asset == "index":
        adj2 = "ind"  # 指数使用ind复权类型
    else:
        adj2 = adj  # 股票使用选择的复权类型
    
    # 使用数据库适配器
    try:
        db_adapter = get_db_adapter()
        df = db_adapter.read_range(base, asset, adj2, ts_norm, start, end, cols, None)
    except Exception as e:
        logger.error(f"查询导出数据失败: {e}")
        return f"查询数据失败: {e}", None
    
    if df is None or df.empty:
        return "（空结果，未生成 CSV）", None
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, f"{ts_norm}_{start}_{end}.csv"))
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return f"已导出：{out_path}  行数={len(df)}", out_path


def run_custom_parquet(file_path: str, columns: Optional[str], limit: Optional[int]) -> pd.DataFrame:
    """自定义读取指定 Parquet 并按列/行数限制显示"""
    if not os.path.isfile(file_path):
        return pd.DataFrame([{"error": f"文件不存在: {file_path}"}])
    try:
        df = pd.read_parquet(file_path)
        if columns:
            col_list = [c.strip() for c in re.split(r"[，,;；\s]+", columns) if c.strip() in df.columns]
            if col_list:
                df = df[col_list]
        if isinstance(limit, int):
            if limit < 0:
                df = df.tail(-limit)
            elif limit > 0:
                df = df.head(limit)
        return df
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])


def run_custom_parquet_ui(text_path: Optional[str], file_obj, columns: Optional[str], limit: Optional[int]) -> pd.DataFrame:
    """
    自定义 Parquet 显示（UI 版）：同时兼容 文本路径 和 拖拽上传的文件。
    - 若 file_obj 不为空，优先使用 file_obj.name（Gradio 会给出本机临时路径）
    - 否则回退到手输的 text_path
    """
    picked_path = (getattr(file_obj, "name", None) or text_path or "").strip()
    limit_clean = (limit or "").strip()
    is_int = limit_clean.lstrip("-").isdigit()
    limit_val = int(limit_clean) if is_int else None
    return run_custom_parquet(picked_path, columns, limit_val)

# -------------------- Gradio 界面 --------------------
def build_ui(base_default: str = DEFAULT_BASE):
    base_default = load_base(base_default)
    with gr.Blocks(title="Parquet 数据浏览器--基于Gradio") as demo:
        gr.Markdown("## Parquet 数据浏览器  \n基于 `parquet_viewer.py` 封装。")

        with gr.Tab("概览 & 检查"):
            base_info = gr.Textbox(label="数据根目录 base", value=base_default)
            adj_pick = gr.Dropdown(choices=["qfq","hfq","raw","daily"], value="qfq", label="复权类型")
            with gr.Row():
                btn_refresh = gr.Button("刷新概览", variant="primary")
                btn_save = gr.Button("保存 base")
            out_info = gr.Textbox(label="概览（文本）", lines=8)
            out_df = gr.Dataframe(label="完整性表格（可筛选排序）")
            out_advice = gr.Markdown(label="诊断建议")

            btn_refresh.click(get_info, inputs=[base_info, adj_pick], outputs=[out_info])
            btn_refresh.click(overview_table, inputs=[base_info, adj_pick], outputs=[out_df])
            btn_refresh.click(overview_advice, inputs=[base_info, adj_pick], outputs=[out_advice])
            btn_save.click(lambda b: save_base(b) or "已保存", inputs=[base_info], outputs=[out_info])

        with gr.Tab("分区日期 dates"):
            base_dates = gr.Textbox(label="数据根目录 base", value=base_default)
            asset_dates = gr.Radio(choices=["index", "stock"], value="index", label="资产 asset")
            adj_dates = gr.Dropdown(choices=["raw", "qfq", "hfq", "ind"], value="qfq", label="复权 adj（index 使用 ind）")
            btn_dates = gr.Button("列出日期")
            out_dates = gr.Textbox(label="分区日期", lines=10)
            btn_dates.click(get_dates, inputs=[base_dates, asset_dates, adj_dates], outputs=[out_dates])

        with gr.Tab("某天 day"):
            base_day = gr.Textbox(label="数据根目录 base", value=base_default)
            asset_day = gr.Radio(choices=["index", "stock"], value="index", label="资产 asset")
            adj_day = gr.Dropdown(choices=["raw", "qfq", "hfq", "ind"], value="qfq", label="复权 adj（index 使用 ind）")
            date_day = gr.Textbox(label="交易日 YYYYMMDD")
            limit_day = gr.Textbox(label="最多显示行数", placeholder="可空，默认为全部，负数为倒数")
            btn_day = gr.Button("查询")
            out_day = gr.Dataframe(label="结果（day）")
            btn_day.click(run_day, inputs=[base_day, asset_day, adj_day, date_day, limit_day], outputs=[out_day])

        with gr.Tab("区间/代码 show"):
            base_show = gr.Textbox(label="数据根目录 base", value=base_default)
            asset_show = gr.Radio(choices=["index", "stock"], value="index", label="资产 asset")
            adj_show = gr.Dropdown(choices=["raw", "qfq", "hfq", "ind"], value="qfq", label="复权 adj（index 使用 ind）")
            ts_show = gr.Textbox(label="ts_code（如 000001）")
            start_show = gr.Textbox(label="起始日期 YYYYMMDD")
            end_show = gr.Textbox(label="结束日期 YYYYMMDD")
            cols_show = gr.Textbox(label="仅显示这些列", placeholder="支持半/全角的逗号分号以及空格分隔，可空")
            limit_show = gr.Textbox(label="最多显示行数", placeholder="可空，默认为全部，负数为倒数")
            btn_show = gr.Button("查询")
            out_show_df = gr.Dataframe(label="结果（show）")
            out_show_fig = gr.Plot(label="（可选）收盘价曲线：若存在 trade_date + close 列自动绘图")
            btn_show.click(run_show,
                           inputs=[base_show, asset_show, adj_show, ts_show, start_show, end_show, cols_show, limit_show],
                           outputs=[out_show_df, out_show_fig])

        with gr.Tab("Schema / 统计"):
            file_schema = gr.Textbox(label="Parquet 文件路径")
            file_custom_file = gr.File(label="直接选择/拖拽 Parquet 文件", file_types=[".parquet"])
            btn_schema = gr.Button("查看 schema")
            out_schema = gr.Textbox(label="Schema / 统计", lines=12)
            btn_schema.click(run_schema, inputs=[file_schema, file_custom_file], outputs=[out_schema])

        with gr.Tab("导出 CSV"):
            base_exp = gr.Textbox(label="数据根目录 base", value=base_default)
            asset_exp = gr.Radio(choices=["index", "stock"], value="index", label="资产 asset")
            adj_exp = gr.Dropdown(choices=["raw", "qfq", "hfq", "ind"], value="qfq", label="复权 adj（index 使用 ind）")
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
            
        with gr.Tab("自定义 Parquet 显示"):
            file_custom_path = gr.Textbox(label="Parquet 文件路径（可手输）")
            file_custom_file = gr.File(label="或直接选择/拖拽 Parquet 文件", file_types=[".parquet"])
            cols_custom = gr.Textbox(label="仅显示这些列", placeholder="支持半/全角的逗号分号以及空格分隔，可空")
            limit_custom = gr.Textbox(label="最多显示行数", placeholder="可空，默认为全部，负数为倒数")
            btn_custom = gr.Button("显示")
            out_custom_df = gr.Dataframe(label="结果（自定义）")
            btn_custom.click(run_custom_parquet_ui,
                            inputs=[file_custom_path, file_custom_file, cols_custom, limit_custom],
                            outputs=[out_custom_df])


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
    parser.add_argument("--share", action="store_true", help="开启 Gradio 公开分享")
    parser.add_argument("--base", type=str, default=DEFAULT_BASE, help="数据根目录")
    args = parser.parse_args()

    demo = build_ui(args.base)
    demo = _enable_queue_compat(demo)
    demo.launch(
            share=False,                 # 强制本地，不去申请外网隧道
            server_name="127.0.0.1",     # 只监听回环地址
            server_port=None,            # 端口冲突时自动换
            inbrowser=True,              # 自动在浏览器打开
            show_error=True
        )


if __name__ == "__main__":
    main()
