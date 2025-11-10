# -*- coding: utf-8 -*-
from __future__ import annotations

import os, io, json, re
import warnings
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict
import threading
from log_system import get_logger
import pandas as pd
import numpy as np
import streamlit as st

# 忽略tushare的FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="tushare.pro.data_pro",
    message=".*fillna.*method.*deprecated.*"
)

# 初始化日志记录器
logger = get_logger("score_ui")
def ui_cleanup_database_connections():
    """强制清理所有数据库连接 - 统一使用 data_reader 管理"""
    try:
        # 延迟导入 data_reader，避免启动时立即初始化数据库连接
        try:
            from database_manager import clear_connections_only
        except ImportError as e:
            st.error(f"无法导入 database_manager 模块: {e}")
            return False
        
        # 清理数据库连接（轻量级清理，不关闭工作线程）
        clear_connections_only()
        
        # 数据库连接已通过 database_manager 清理
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        st.success("✅ 数据库连接清理完成")
        return True
        
    except Exception as e:
        st.error(f"数据库连接清理失败: {e}")
        return False

def check_database_status():
    """检查数据库状态（使用状态文件）"""
    try:
        from database_manager import check_database_status as check_status
        from database_manager import get_database_manager
        
        # 使用状态文件检查数据库状态
        result = check_status()
        
        if not result.get('status_file_exists', False):
            st.warning("状态文件不存在，建议生成状态文件")
            # 回退到原来的检查方式
            try:
                from database_manager import get_database_info
                db_info = get_database_info()
                manager = get_database_manager()
                enhanced_stats = manager.get_stats()
                st.info(f"数据库管理器: {enhanced_stats}")
                st.info(f"数据库信息: {db_info}")
            except Exception as e:
                st.error(f"检查数据库状态失败: {e}")
            return False
        
        # 显示状态文件信息
        status_file_generated_at = result.get('status_file_generated_at', '未知')
        st.info(f"状态文件生成时间: {status_file_generated_at}")
        
        # 显示当前状态
        current_status = result.get('current_status', {})
        stock_data = current_status.get('stock_data', {})
        details_data = current_status.get('details_data', {})
        
        # 显示股票数据状态
        if stock_data.get('database_exists', False):
            st.success(f"股票数据库: 存在")
            st.info(f"  - 总记录数: {stock_data.get('total_records', 0)}")
            st.info(f"  - 股票数量: {stock_data.get('total_stocks', 0)}")
            adj_types = stock_data.get('adj_types', {})
            for adj_type, adj_status in adj_types.items():
                st.info(f"  - {adj_type}: {adj_status.get('min_date', 'N/A')} ~ {adj_status.get('max_date', 'N/A')}")
        else:
            st.warning("股票数据库: 不存在")
        
        # 显示细节数据状态
        if details_data.get('database_exists', False):
            st.success(f"细节数据库: 存在")
            st.info(f"  - 总记录数: {details_data.get('total_records', 0)}")
            st.info(f"  - 股票数量: {details_data.get('stock_count', 0)}")
            if details_data.get('min_date') and details_data.get('max_date'):
                st.info(f"  - 日期范围: {details_data.get('min_date')} ~ {details_data.get('max_date')}")
        else:
            st.warning("细节数据库: 不存在")
        
        # 显示差异
        differences = result.get('differences', {})
        if differences:
            st.warning("发现状态差异，建议更新状态文件")
            st.json(differences)
        else:
            st.success("状态文件是最新的")
        
        return True
    except Exception as e:
        st.error(f"检查数据库状态失败: {e}")
        return False

# 进程控制功能已移除，相关问题在database_manager中统一处理

import streamlit.components.v1 as components
from contextlib import contextmanager
import shutil
import uuid
import time
import queue
import traceback

# 延迟导入，避免启动时立即初始化数据库连接
# import download as dl
import scoring_core as se
import config as cfg
import stats_core as stats
from utils import normalize_ts, ensure_datetime_index, normalize_trade_date, market_label
# 使用 database_manager 替代 data_reader
from database_manager import (
    get_database_manager, query_stock_data, get_trade_dates, 
    get_stock_list, get_latest_trade_date, get_smart_end_date,
    get_database_info, get_data_source_status, close_all_connections,
    clear_connections_only,
    is_details_db_reading_enabled, get_details_db_path_with_fallback, is_details_db_available
)

def _lazy_import_download():
    """延迟导入 download 模块的函数"""
    try:
        import download as dl
        return dl
    except ImportError as e:
        logger = get_logger("score_ui")
        logger.error(f"导入 download 失败: {e}")
        return None

# 直接使用 database_manager 函数，不再需要包装器
import os
from config import DATA_ROOT, API_ADJ, SC_DETAIL_STORAGE, SC_USE_DB_STORAGE, SC_DB_FALLBACK_TO_JSON
import tdx_compat as tdx
from stats_core import _pick_trade_dates, _prev_trade_date
import indicators as ind
import predict_core as pr
from rule_editor import render_rule_editor
from predict_core import (
    PredictionInput, PositionCheckInput,
    run_prediction, run_position_checks,
    load_prediction_rules, load_position_policies, load_opportunity_policies,
    Scenario
)

# ---- Streamlit context guard & cache alias (auto-injected) ----
def _in_streamlit():
    try:
        import streamlit as st
        exists = getattr(getattr(st, "runtime", None), "exists", None)
        if callable(exists):
            return bool(exists())
    except Exception:
        pass
    # 回落到 get_script_run_ctx（并配合方案1静音）
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    return get_script_run_ctx() is not None


def _noop_cache_data(*dargs, **dkw):
    def deco(fn): 
        return fn
    return deco

cache_data = (st.cache_data if _in_streamlit() else _noop_cache_data)
# --------------------------------------------------------------

def _safe_path_hash(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except (OSError, FileNotFoundError):
        return hash(str(p))

def _is_valid_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y%m%d")
        return True
    except ValueError:
        return False

def _safe_int(x, default: int = 60) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)

def _init_session_state():
    """统一初始化 Streamlit session_state 的关键字段，避免重复判断散落各处。"""
    try:
        import streamlit as st  # local import so this can run outside Streamlit too
        if not _in_streamlit():
            return
        defaults = {
            "cur_pid": None,
            "cur_pf": None,
            "rules_obj": {
                "prescreen": getattr(se, "SC_PRESCREEN_RULES", []),
                "rules": getattr(se, "SC_RULES", []),
            },
            "export_pref": {"style": "space", "with_suffix": True},
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
        
        # 添加表达式选股时的数据库连接管理
        if "expression_screening_active" not in st.session_state:
            st.session_state["expression_screening_active"] = False
        
        # 添加details数据库读取控制标记，默认不读取数据库避免写入冲突
        if "details_db_reading_enabled" not in st.session_state:
            st.session_state["details_db_reading_enabled"] = False
        
        # 添加数据查看页面的数据库查询控制标记，默认不查询数据库避免写入冲突
        if "data_view_db_enabled" not in st.session_state:
            st.session_state["data_view_db_enabled"] = False
    except Exception:
        pass

if _in_streamlit():
    st.set_page_config(page_title="ScoreApp", layout="wide")
    _init_session_state()
# ===== 常量路径 =====
SC_OUTPUT_DIR = Path(getattr(cfg, "SC_OUTPUT_DIR", "output/score"))
TOP_DIR  = SC_OUTPUT_DIR / "top"
ALL_DIR  = SC_OUTPUT_DIR / "all"
DET_DIR  = SC_OUTPUT_DIR / "details"
ATTN_DIR = SC_OUTPUT_DIR / "attention"
LOG_DIR  = Path("./log")


for p in [TOP_DIR, ALL_DIR, DET_DIR, ATTN_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def _apply_overrides(
    base: str,
    assets: List[str],
    start: str,
    end: str,
    api_adj: str,
    fast_threads: int,
    inc_threads: int,
    inc_ind_workers: int | None,
):
    """把 UI 输入同步到 download.py 的全局，以便其函数读取。"""
    # 延迟导入 download 模块
    dl = _lazy_import_download()
    if dl is None:
        raise ImportError("无法导入 download 模块")
    
    # download.py 内部多数直接使用模块级常量，这里原地覆写它们
    dl.DATA_ROOT = base
    dl.ASSETS = [a.lower() for a in assets]
    dl.START_DATE = start
    dl.END_DATE = end
    dl.API_ADJ = api_adj.lower()
    dl.FAST_INIT_THREADS = int(max(1, fast_threads))
    dl.STOCK_INC_THREADS = int(max(1, inc_threads))
    if inc_ind_workers is not None and int(inc_ind_workers) > 0:
        dl.INC_RECALC_WORKERS = int(inc_ind_workers)

    # 同步到 config，以便其他模块（如 parquet_viewer）看到一致的 base/adj
    try:
        cfg.DATA_ROOT = base
        cfg.API_ADJ = api_adj.lower() if api_adj.lower() in {"raw","qfq","hfq"} else getattr(cfg, "API_ADJ", "qfq")
    except Exception:
        pass

@cache_data(show_spinner=False, ttl=300)
def _latest_trade_date(base: str, adj: str) -> str | None:
    try:
        # 使用 database_manager 获取最新交易日
        latest_date = get_latest_trade_date()
        return latest_date
    except Exception:
        return None

# -------------------- 执行动作（封装 download.py） --------------------
def _run_fast_init(end_use: str):
    # 延迟导入 download 模块
    dl = _lazy_import_download()
    if dl is None:
        raise ImportError("无法导入 download 模块")
    
    dl.fast_init_download(end_use)                       # 首次全量（单股缓存）
    # 数据库操作已迁移到 data_reader.py，合并操作已集成到下载过程中


def _run_increment(start_use: str, end_use: str, do_stock: bool, do_index: bool, do_indicators: bool):
    # 延迟导入 download 模块
    dl = _lazy_import_download()
    if dl is None:
        raise ImportError("无法导入 download 模块")
    
    # 若 fast_init 的缓存存在，先合并一次（与 main() 逻辑一致）
    try:
        if any(
            os.path.isdir(os.path.join(dl.FAST_INIT_STOCK_DIR, d))
            and any(f.endswith(".parquet") for f in os.listdir(os.path.join(dl.FAST_INIT_STOCK_DIR, d)))
            for d in ("raw","qfq","hfq")
        ):
            # 数据库操作已迁移到 data_reader.py
            pass
    except Exception:
        pass

    if do_stock and ("stock" in set(dl.ASSETS)):
        dl.sync_stock_daily_fast(start_use, end_use, threads=dl.STOCK_INC_THREADS)
    if do_index and ("index" in set(dl.ASSETS)):
        dl.sync_index_daily_fast(start_use, end_use, dl.INDEX_WHITELIST)
    if do_indicators:
        workers = getattr(dl, "INC_RECALC_WORKERS", None) or ((os.cpu_count() or 4) * 2)
        dl.recalc_symbol_products_for_increment(start_use, end_use, threads=workers)

# ===== 小工具 =====
def _path_top(ref: str) -> Path: return TOP_DIR / f"score_top_{ref}.csv"
def _path_all(ref: str) -> Path: return ALL_DIR / f"score_all_{ref}.csv"
def _path_detail(ref: str, ts: str) -> Path: return DET_DIR / ref / f"{normalize_ts(ts)}_{ref}.json"
def _today_str() -> str:
    return date.today().strftime("%Y%m%d")

@cache_data(show_spinner=False, hash_funcs={Path: _safe_path_hash}, ttl=60)
def _read_df(path: Path, usecols=None, dtype=None, encoding: str = "utf-8-sig") -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, encoding=encoding, engine="c")
    except Exception:
        return pd.DataFrame()

@cache_data(show_spinner=False, ttl=600)
def _cached_trade_dates(base: str, adj: str):
    # 使用 database_manager 获取交易日列表
    return get_trade_dates() or []

# ==== 进度转发到主线程：仅子线程/子进程入队，主线程消费并渲染 ====
@contextmanager
def se_progress_to_streamlit():
    if not _in_streamlit():
        # bare/子线程下：挂空回调，啥也不画，避免任何 st.* 调用
        def _noop(*a, **k): 
            pass
        # 使用新的日志系统替代废弃的 set_progress_handler
        from log_system import get_logger
        logger = get_logger("scoring_core")
        logger.info("使用新的日志系统进行进度跟踪")
        try:
            yield None, None, None
        finally:
            pass
        return
    status = st.status("准备中…", expanded=True)
    bar = st.progress(0, text="就绪")
    info = st.empty()

    import queue as _q
    _evq = _q.Queue()
    
    # 后台线程只入队，不直接碰 st.*
    def _enqueue_handler(phase, current=None, total=None, message=None, **kw):
        try:
            _evq.put_nowait((phase, current, total, message))
        except Exception:
            pass

    def _render_event(phase, current=None, total=None, message=None):
        txt = {
            "select_ref_date": "选择参考日", "compute_read_window": "计算读取区间",
            "build_universe_done": "构建评分清单", "score_start": "并行评分启动",
            "score_progress": "评分进行中", "screen_start": "筛选启动",
            "screen_progress": "筛选进行中", "screen_done": "筛选完成",
            "write_cache_lists": "写入黑白名单", "write_top_all_start": "写出 Top/All",
            "write_top_all_done": "Top/All 完成", "hooks_start": "统计/回看",
            "hooks_done": "统计完成",
        }.get(phase, phase)
        if total and current is not None:
            pct = int(current * 100 / max(total, 1))
            # 显示进度详情：评分和筛选都显示数量
            if phase in ("score_progress", "screen_progress"):
                bar.progress(pct, text=f"{txt} · {current}/{total}")
            else:
                bar.progress(pct, text=txt)
        else:
            # 使用message作为主要显示内容，如果没有则使用txt
            display_text = message if message else txt
            info.write(display_text)

    # 主线程消费：供 run_se_run_for_date_in_bg 循环调用
    def _drain():
        try:
            while True:
                ev = _evq.get_nowait()
                _render_event(*ev)
        except _q.Empty:
            pass

    # 使用新的日志系统并设置进度处理器
    from log_system import get_logger
    logger = get_logger("scoring_core")
    logger.info("使用新的日志系统进行进度跟踪")
    
    # 关键：设置进度处理器，使评分系统能够发送进度事件
    _orig_drain = getattr(se, "drain_progress_events", None)
    se.set_progress_handler(_enqueue_handler)
    se.drain_progress_events = _drain  # 将"抽干"替换成主线程渲染
    
    try:
        yield status, bar, info
    finally:
        # 还原 drain（保持模块整洁）
        if callable(_orig_drain):
            se.drain_progress_events = _orig_drain
        else:
            se.drain_progress_events = lambda: None

@cache_data(show_spinner=False)
def _read_md_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except Exception:
        # 兜底提示，避免页面报错
        return "⚠️ 未找到帮助文档：" + path


def run_se_run_for_date_in_bg(arg):
    """在后台线程运行 se.run_for_date(arg)，并在主线程渲染进度"""
    with se_progress_to_streamlit() as (status, bar, info):
        done = threading.Event()
        result = {"path": None, "err": None}

        def _worker():
            try:
                try:
                    # prefer local UI cleanup if present
                    if 'ui_cleanup_database_connections' in globals():
                        ui_cleanup_database_connections()
                    else:
                        # 使用轻量级清理函数，避免关闭工作线程
                        from database_manager import clear_connections_only
                        clear_connections_only()
                except Exception:
                    pass
                
                # 在子线程中运行评分，但确保数据库连接正确初始化
                from database_manager import get_database_manager
                logger.info("[数据库连接] 开始获取数据库管理器实例 (评分线程)")
                manager = get_database_manager()
                
                # 确保数据库管理器已正确初始化，避免连接问题
                try:
                    # 测试数据库连接是否正常
                    test_date = manager.get_latest_trade_date()
                    if test_date:
                        logger.info(f"[评分] 数据库连接正常，最新交易日: {test_date}")
                    else:
                        logger.warning("[评分] 数据库连接正常但无最新交易日数据")
                except Exception as e:
                    logger.warning(f"[评分] 数据库连接测试失败: {e}")
                
                result["path"] = se.run_for_date(arg)
            except Exception as e:
                result["err"] = e
            finally:
                done.set()
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # 主线程循环抽取进度事件并刷新 UI
        while not done.is_set():
            se.drain_progress_events()
            time.sleep(0.05)
        # 抽干剩余事件
        se.drain_progress_events()
        if status is not None:
            status.update(label="已完成", state="complete")

        if result["err"]:
            raise result["err"]
        return result["path"]


def run_se_screen_in_bg(*, when_expr, ref_date, timeframe, window, scope, universe, write_white, write_black_rest, return_df=True):
    """在后台线程运行 se.tdx_screen(...)，并在主线程渲染进度（用于“普通选股”）"""
    with se_progress_to_streamlit() as (status, bar, info):
        import threading, time
        done = threading.Event()
        result = {"df": None, "err": None}

        def _worker():
            try:
                try:
                    # prefer local UI cleanup if present
                    if 'ui_cleanup_database_connections' in globals():
                        ui_cleanup_database_connections()
                    else:
                        # 使用轻量级清理函数，避免关闭工作线程
                        from database_manager import clear_connections_only
                        clear_connections_only()
                except Exception:
                    pass
                
                st.session_state["expression_screening_active"] = True
                
                try:
                    result["df"] = se.tdx_screen(
                        when_expr,
                        ref_date=ref_date,
                        timeframe=timeframe,
                        window=_safe_int(window, 30),
                        scope=scope,
                        universe=universe,
                        write_white=write_white,
                        write_black_rest=write_black_rest,
                        return_df=return_df
                    )
                finally:
                    st.session_state["expression_screening_active"] = False
                    
            except Exception as e:
                result["err"] = e
                st.session_state["expression_screening_active"] = False
            finally:
                done.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # 主线程循环抽取进度事件并刷新 UI
        while not done.is_set():
            se.drain_progress_events()
            time.sleep(0.05)
        # 抽干剩余事件
        se.drain_progress_events()
        if status is not None:
            status.update(label="已完成", state="complete")

        if result["err"]:
            raise result["err"]
        return result["df"]


def _get_latest_date_from_files() -> Optional[str]:
    """从评分结果文件名中提取最新日期"""
    files = sorted(TOP_DIR.glob("score_top_*.csv"))
    dates = []
    for p in files:
        m = re.search(r"(\d{8})", p.name)
        if m: dates.append(m.group(1))
    return max(dates) if dates else None


def _get_latest_date_from_database() -> Optional[str]:
    """从数据库获取最新交易日"""
    try:
        from database_manager import get_latest_trade_date
        latest = get_latest_trade_date()
        if latest:
            logger.info(f"从数据库获取最新交易日: {latest}")
            return latest
    except Exception as e:
        logger.warning(f"从数据库获取最新交易日失败: {e}")
    return None


def _get_latest_date_from_daily_partition() -> Optional[str]:
    """从daily分区获取最新交易日"""
    try:
        from database_manager import get_trade_dates
        dates = get_trade_dates()
        if dates:
            latest = dates[-1]
            logger.info(f"从daily分区获取最新交易日: {latest}")
            return latest
    except Exception as e:
        logger.warning(f"从daily分区获取最新交易日失败: {e}")
    return None


def _pick_smart_ref_date() -> Optional[str]:
    """智能获取参考日期，按优先级尝试多种方式"""
    # 1. 优先从数据库获取
    latest = _get_latest_date_from_database()
    if latest:
        return latest
    
    # 2. 从daily分区获取
    latest = _get_latest_date_from_daily_partition()
    if latest:
        return latest
    
    # 3. 最后从评分结果文件获取
    latest = _get_latest_date_from_files()
    if latest:
        logger.warning(f"回退到评分结果文件中的最新日期: {latest}")
    else:
        logger.error("无法获取任何参考日期")
    
    return latest


def _prev_ref_date(cur: str) -> Optional[str]:
    files = sorted(TOP_DIR.glob("score_top_*.csv"))
    dates = []
    for p in files:
        m = re.search(r"(\d{8})", p.name)
        if m and m.group(1) < cur:
            dates.append(m.group(1))
    return dates[-1] if dates else None


def _from_last_hints(days: list[int] | None = None,
                     base: str = DATA_ROOT, adj: str = API_ADJ,
                     last: str | None = None):
    """
    基于“最新交易日 last（缺省=本地数据的最后一天）”，返回：
      - 文本提示串（含星期），用于展示；
      - 映射 dict: {n: d8}，n 个交易日前对应的 yyyymmdd 字符串。
    """
    try:
        ds = get_trade_dates() or []
        if not ds:
            return "", {}
        last = last or ds[-1]
        if last not in ds:
            return "", {}

        idx = ds.index(last)
        days = sorted({int(x) for x in (days or []) if int(x) >= 1})

        from datetime import date as _d
        def _fmt(d8: str) -> str:
            y, m, d = int(d8[:4]), int(d8[4:6]), int(d8[6:])
            wk = "一二三四五六日"[_d(y, m, d).weekday()]
            return f"{y:04d}-{m:02d}-{d:02d}(周{wk})"

        parts = [f"最新={_fmt(last)}"]
        mapping = {}
        for n in days:
            j = idx - n
            if j >= 0:
                mapping[n] = ds[j]
                parts.append(f"{n}个交易日前={_fmt(ds[j])}")
            else:
                parts.append(f"{n}个交易日前=--（数据不足）")
        return " · ".join(parts), mapping
    except Exception:
        return "", {}


def _rule_to_screen_args(rule: dict):
    """返回 (when_expr, timeframe, window, scope)"""
    if rule.get("clauses"):
        tfs = {str(c.get("timeframe","D")).upper() for c in rule["clauses"]}
        wins = {int(c.get("window", 60)) for c in rule["clauses"]}
        scopes = {str(c.get("scope","ANY")).upper() for c in rule["clauses"]}
        whens = [f"({c.get('when','').strip()})" for c in rule["clauses"] if c.get("when","").strip()]
        if not whens:
            raise ValueError("复合规则缺少 when")
        # 目前仅支持“相同 tf/window/scope”的复合规则；否则就无法一次性屏全市场
        if len(tfs)==len(wins)==len(scopes)==1:
            return " AND ".join(whens), list(tfs)[0], list(wins)[0], list(scopes)[0]
        else:
            raise ValueError("全市场跑目前仅支持各子句 tf/window/scope 完全一致的复合规则")
    else:
        when = (rule.get("when") or "").strip()
        if not when:
            raise ValueError("when 不能为空")
        tf = str(rule.get("timeframe","D")).upper()
        win = int(rule.get("window", 60))
        scope = str(rule.get("scope","ANY")).upper()
        # --- substitute placeholders (K/M/N) for scope ---
        try:
            import re
            k = int(rule.get("k", rule.get("n", 0)) or 0)
            m = int(rule.get("m", 0) or 0)
            # COUNT>=K -> COUNT>=<k or 3>
            if "COUNT" in scope and re.search(r"\bK\b", scope):
                scope = re.sub(r"\bK\b", str(k or 3), scope)
            # CONSEC>=M -> CONSEC>=<m or 3>
            if "CONSEC" in scope and re.search(r"\bM\b", scope):
                scope = re.sub(r"\bM\b", str(m or 3), scope)
            # ANY_N / ALL_N -> ANY_<k or 3> / ALL_<k or 3>
            scope = scope.replace("ANY_N", f"ANY_{k or 3}").replace("ALL_N", f"ALL_{k or 3}")
        except Exception:
            pass
        return when, tf, win, scope


def _load_detail_json(ref: str, ts: str) -> Optional[Dict]:
    """
    加载个股详情，优先从数据库读取，失败时回退到JSON文件
    注意：只有当 details_db_reading_enabled 为 True 时才会读取数据库，避免与写入操作冲突
    """
    # 检查是否允许读取数据库（使用统一的函数）
    db_reading_enabled = is_details_db_reading_enabled()
    
    # 1. 优先从数据库读取（只有当db_reading_enabled为True且数据库可用时才读取）
    if db_reading_enabled and is_details_db_available():
        db_success = False
        try:
            # 使用统一的函数获取details数据库路径（包含回退逻辑）
            details_db_path = get_details_db_path_with_fallback()
            if not details_db_path:
                # 数据库文件不存在，回退到JSON
                logger.debug(f"数据库文件不存在，回退到JSON: {ts}_{ref}")
                db_success = False
                raise FileNotFoundError("Details数据库文件不存在")
            
            # 查询股票详情表
            logger.info(f"[数据库连接] 开始获取数据库管理器实例 (查询股票详情: {ts}, {ref})")
            manager = get_database_manager()
            sql = "SELECT * FROM stock_details WHERE ts_code = ? AND ref_date = ?"
            df = manager.execute_sync_query(details_db_path, sql, [ts, ref], timeout=30.0)
            
            if not df.empty:
                row = df.iloc[0]
                
                # 解析 rules 字段：优先 json.loads，失败则 ast.literal_eval，最后保证是 list[dict]
                rules_raw = row.get('rules')
                rules = []
                if rules_raw:
                    if isinstance(rules_raw, str):
                        try:
                            rules = json.loads(rules_raw)
                        except Exception:
                            try:
                                import ast
                                rules = ast.literal_eval(rules_raw)
                            except Exception:
                                rules = []
                    elif isinstance(rules_raw, list):
                        rules = rules_raw
                
                # 确保 rules 是 list[dict] 格式
                if not isinstance(rules, list):
                    rules = []
                
                # 解析 highlights/drawbacks/opportunities 字段为 list[str]
                def parse_string_list(field_value):
                    if not field_value:
                        return []
                    if isinstance(field_value, str):
                        try:
                            parsed = json.loads(field_value)
                            return parsed if isinstance(parsed, list) else []
                        except Exception:
                            try:
                                import ast
                                parsed = ast.literal_eval(field_value)
                                return parsed if isinstance(parsed, list) else []
                            except Exception:
                                return []
                    elif isinstance(field_value, list):
                        return field_value
                    return []
                
                highlights = parse_string_list(row.get('highlights'))
                drawbacks = parse_string_list(row.get('drawbacks'))
                opportunities = parse_string_list(row.get('opportunities'))
                
                # 获取 rank 和 total 值
                rank_val = row.get('rank')
                total_val = row.get('total')
                
                # 组装 summary，包含 rank 和 total
                summary = {
                    'score': row.get('score'),
                    'tiebreak': row.get('tiebreak'),
                    'highlights': highlights,
                    'drawbacks': drawbacks,
                    'opportunities': opportunities,
                    'rank': int(rank_val) if pd.notna(rank_val) else None,
                    'total': int(total_val) if pd.notna(total_val) else None,
                }
                
                # 组装成与 JSON 文件完全一致的结构，保持兼容性
                result = {
                    'ts_code': row.get('ts_code'),
                    'ref_date': row.get('ref_date'),
                    'summary': summary,
                    'rules': rules,
                    'rank': summary['rank'],   # 兼容旧调用
                    'total': summary['total'],
                }
                db_success = True
                return result
            else:
                # 数据库查询成功但无数据，回退到JSON
                logger.debug(f"数据库查询为空，回退到JSON: {ts}_{ref}")
                db_success = False
        except Exception as e:
            # 数据库读取失败，回退到JSON
            logger.debug(f"数据库读取失败，回退到JSON {ts}_{ref}: {e}")
            db_success = False
        
        # 如果数据库读取失败（异常或查询为空），回退到JSON
        if not db_success:
            # 继续下面的JSON回退逻辑
            pass
        else:
            # 数据库读取成功，已返回结果，不应该到达这里
            pass
    
    # 2. 如果数据库失败且配置了回退，或者配置了JSON存储，或者未启用数据库读取，则使用JSON文件
    if (not db_reading_enabled) or (SC_DB_FALLBACK_TO_JSON) or SC_DETAIL_STORAGE in ["json", "both"]:
        try:
            p = _path_detail(ref, ts)
            if not p.exists(): 
                # JSON文件不存在，正常返回None，不报错
                return None
            try:
                data = json.loads(p.read_text(encoding="utf-8-sig"))
                return data
            except json.JSONDecodeError as e:
                # JSON解析失败，记录日志但不报错
                logger.debug(f"JSON文件解析失败 {ts}_{ref}: {e}")
                return None
            except Exception as e:
                # 其他读取异常，记录日志但不报错
                logger.debug(f"JSON文件读取失败 {ts}_{ref}: {e}")
                return None
        except Exception as e:
            # 路径构建或其他异常，记录日志但不报错
            logger.debug(f"JSON文件路径处理失败 {ts}_{ref}: {e}")
            return None
    
    # 兜底：所有路径都失败，返回None
    return None


def _codes_to_txt(codes: List[str], style: str="space", with_suffix: bool=True) -> str:
    def fmt(c):
        c = normalize_ts(c)
        return c if with_suffix else c.split(".")[0]
    arr = [fmt(c) for c in codes]
    return (" ".join(arr)) if style == "space" else ("\n".join(arr))


def _download_txt(label: str, text: str, filename: str, key: Optional[str]=None):
    st.download_button(label, data=text.encode("utf-8-sig"),
                       file_name=filename, mime="text/plain",
                       width='stretch', key=key)


def copy_txt_button(text: str, label: str = "一键复制（TXT）", key: str = "copy0"):
    st.code(text or "", language="text")
    components.html(f"""
    <button id="{key}" style="padding:6px 10px;border:1px solid #444;border-radius:8px;cursor:pointer">{label}</button>
    <script>
      const btn = document.getElementById("{key}");
      const payload = {json.dumps(text or "")};
      btn.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(payload);
          btn.innerText = "已复制";
        }} catch (e) {{
          btn.innerText = "复制失败（请手动 Ctrl+C）";
        }}
      }});
    </script>
    """, height=50)


def _tail(path: Path, n: int=400) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""


def _fmt_retcols_percent(df):
    df = df.copy()
    cols = [c for c in df.columns if str(c).startswith("ret_fwd_")]
    if not cols:
        return df
    for c in cols:
        # 转成数值
        s = pd.to_numeric(df[c], errors="coerce")
        finite = s[np.isfinite(s)]
        if finite.shape[0] == 0:
            continue
        q95 = finite.abs().quantile(0.95)
        # 小于等于 0.5 说明是小数（例如 0.034），需要×100
        if pd.notna(q95) and q95 <= 0.5:
            s = s * 100.0
        # 统一两位小数 + 百分号
        df[c] = s.map(lambda x: (f"{x:.2f}%" if pd.notna(x) else None))
    return df


def _apply_runtime_overrides(rules_obj: dict,
                             topk: int, tie_break: str, max_workers: int,
                             attn_on: bool, universe: str|List[str]):
    # 规则覆盖配置
    if rules_obj:
        pres = rules_obj.get("prescreen")
        rules = rules_obj.get("rules")
        if pres is not None: setattr(se, "SC_PRESCREEN_RULES", pres)
        if rules is not None: setattr(se, "SC_RULES", rules)
    setattr(se, "SC_TOP_K", int(topk))
    setattr(se, "SC_TIE_BREAK", str(tie_break))
    setattr(se, "SC_MAX_WORKERS", int(max_workers))
    # setattr(se, "SC_ATTENTION_ENABLE", bool(attn_on))
    setattr(se, "SC_UNIVERSE", universe)


def _humanize_error(err) -> tuple[str, list[str], list[str], str]:
    s = str(err) if not isinstance(err, dict) else str(err.get("error", ""))
    causes, fixes = [], []
    title = "运行出错"

    # 结构化判断
    if "JSONDecodeError" in s or "Expecting value" in s or "Invalid control character" in s:
        title = "JSON 格式错误"
        causes = ["JSON 不合法（逗号/引号/花括号/结尾逗号等）"]
        fixes = ["用 JSON 校验工具检查；字段名一律双引号；最后一项不要加逗号"]
    elif "表达式错误" in s or "evaluate_bool" in s:
        title = "策略表达式语法错误"
        causes = ["括号不配对 / 参数缺失 / 不支持的函数或列名"]
        fixes = ["检查括号与逗号；确认列名存在；必要时简化表达式逐段排查"]
    elif "timeframe" in s or "resample" in s:
        title = "不支持的周期 (timeframe)"
        causes = ["传入了未实现的周期"]
        fixes = ["改为项目支持的 D/W/M/60MIN 等"]
    elif "empty-window" in s or "empty window" in s or "无可用标的" in s:
        title = "数据窗口无数据"
        causes = ["窗口区间过短或参考日无交易数据", "标的退市/长期停牌导致无数据"]
        fixes = ["拉长 window；更换参考日；调整股票池/市场范围"]
    elif "KeyError" in s or "missing" in s or "列" in s:
        title = "缺少列/指标"
        causes = ["表达式引用了数据中不存在的列"]
        fixes = ["在数据侧补列，或使用内置兜底（如 J/VR）"]
    elif "database is locked" in s or "file is locked" in s or "database is busy" in s or "file is being used" in s or "另一个程序正在使用此文件" in s:
        title = "数据库被占用"
        causes = ["多个进程同时访问数据库文件", "数据库文件被其他程序锁定", "系统资源不足"]
        fixes = ["等待其他操作完成", "重启应用程序", "检查是否有其他程序在使用数据库文件", "使用内存数据库模式"]

    return title, causes, fixes, s


def show_database_diagnosis():
    """显示数据库诊断信息"""
    try:
        # 诊断功能需要重新实现
        # 使用 database_manager 获取诊断信息
        logger.info("[数据库连接] 开始获取数据库管理器实例 (数据库诊断)")
        manager = get_database_manager()
        stats = manager.get_stats()
        diagnosis = {
            "database_status": "connected" if stats else "disconnected",
            "queue_size": stats.get("queue_size", 0),
            "worker_count": stats.get("worker_count", 0)
        }
        
        st.subheader("数据库诊断信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("数据库文件存在", "是" if diagnosis.get("database_exists") else "否")
            st.metric("数据库文件被锁定", "是" if diagnosis.get("database_locked") else "否")
            if diagnosis.get("file_size"):
                st.metric("文件大小", f"{diagnosis['file_size'] / (1024*1024):.1f} MB")
        
        with col2:
            if diagnosis.get("file_permissions"):
                st.metric("文件权限", diagnosis["file_permissions"])
            if diagnosis.get("last_modified"):
                import datetime
                last_mod = datetime.datetime.fromtimestamp(diagnosis["last_modified"])
                st.metric("最后修改", last_mod.strftime("%Y-%m-%d %H:%M:%S"))
        
        # 显示进程占用信息
        processes = diagnosis.get("processes_using_db", [])
        if processes:
            st.warning(f"⚠️ 发现 {len(processes)} 个进程正在使用数据库文件:")
            for proc in processes:
                st.write(f"- PID: {proc['pid']}, 进程名: {proc['name']}")
        else:
            st.success("✅ 没有发现其他进程占用数据库文件")
        
        if diagnosis.get("database_locked"):
            st.error("数据库文件被锁定，这可能导致表达式选股失败")
            st.info("建议：检查是否有其他应用在使用数据库文件，或重启相关进程")
        
        if st.button("重新诊断"):
            st.rerun()
            
    except Exception as e:
        st.error(f"诊断数据库失败: {e}")


def show_database_status():
    """显示数据库连接状态"""
    try:
        # get_data_source_status 已从 database_manager 导入
        status = get_data_source_status()
        
        st.subheader("数据库连接状态")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("数据库文件存在", "是" if status.get("database_file_exists") else "否")
            st.metric("数据库文件被锁定", "是" if status.get("database_file_locked") else "否")
            st.metric("使用统一数据库", "是" if status.get("use_unified_db") else "否")
        
        with col2:
            dispatcher_stats = status.get("dispatcher_stats", {})
            st.metric("工作线程数", dispatcher_stats.get("worker_threads", 0))
            st.metric("缓存大小", dispatcher_stats.get("cache_size", 0))
            st.metric("队列大小", dispatcher_stats.get("queue_size", 0))
        
        if status.get("database_file_locked"):
            st.error("⚠️ 数据库文件被锁定，这可能导致表达式选股失败")
            st.info("建议：等待其他操作完成或重启应用程序")
        
        if st.button("刷新状态"):
            st.rerun()
            
    except Exception as e:
        st.error(f"检查数据库状态失败: {e}")


def _indicator_options(tag: str | None = "product"):
    try:
        # 只列出有 py_func 的（能在本地算）的指标名
        names = [k for k, m in getattr(ind, "REGISTRY", {}).items() if getattr(m, "py_func", None)]
        if tag and hasattr(ind, "names_by_tag"):
            tagged = set(ind.names_by_tag(tag))  # 只要打了 product 标签的
            names = [n for n in names if n in tagged] or names
        return sorted(set(names))
    except Exception:
        # 兜底：保持现在的三个
        return ["kdj", "ma", "macd"]


@cache_data(show_spinner=False, ttl=300)
def _get_rule_names() -> list[str]:
    """获取规则名称列表，带缓存"""
    try:
        rule_names = [str(r.get("name") or f"RULE_{i}") for i, r in enumerate(getattr(se, "SC_RULES", []) or [])]
        return sorted(list(dict.fromkeys(rule_names)))
    except Exception:
        return []

@cache_data(show_spinner=False, ttl=300)
def _cached_load_prediction_rules() -> list[dict]:
    """缓存版本的 load_prediction_rules 函数"""
    try:
        return load_prediction_rules()
    except Exception:
        return []

def _apply_tiebreak_sorting(df: pd.DataFrame, tiebreak_mode: str = "none") -> pd.DataFrame:
    """
    对股票表格应用排序
    
    Args:
        df: 包含ts_code和score列的DataFrame
        tiebreak_mode: 排序模式 ("none", "kdj_j_asc")
    
    Returns:
        排序后的DataFrame
    """
    if df.empty or "ts_code" not in df.columns or "score" not in df.columns:
        return df
    
    # 创建副本避免修改原数据
    df_sorted = df.copy()
    
    if tiebreak_mode == "kdj_j_asc" and "tiebreak_j" in df_sorted.columns:
        # 按得分降序，同分时按J值升序，再同分时按代码升序（兜底）
        df_sorted = df_sorted.sort_values(["score", "tiebreak_j", "ts_code"], ascending=[False, True, True]).reset_index(drop=True)
    else:
        # 默认：只按得分降序排序，同分时按代码升序（兜底）
        df_sorted = df_sorted.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    
    return df_sorted

@cache_data(show_spinner=False, ttl=120)
def _resolve_pred_universe(label: str, ref: str) -> list[str]:
    """
    将 UI 的范围标签展开为 ts_code 列表：
    - all：读 output/score/all/score_all_<ref>.csv（若无，则退回 top）
    - white/black：读 scoring_core 的缓存名单
    - attention：读“特别关注榜”（若找不到则尝试按文件名匹配）
    """
    label = (label or "").strip().lower()
    codes: list[str] = []

    if label == "all":
        p_all = _path_all(ref)  # 这个工具在现有文件里已定义
        if p_all.exists() and p_all.stat().st_size > 0:
            df = _read_df(p_all, dtype={"ts_code": str})
            if df is not None and not df.empty and "ts_code" in df.columns:
                codes = df["ts_code"].astype(str).tolist()
        if not codes:
            # 兜底用 Top（至少不会是空）
            p_top = _path_top(ref)
            if p_top.exists() and p_top.stat().st_size > 0:
                df = _read_df(p_top, dtype={"ts_code": str})
                if df is not None and not df.empty and "ts_code" in df.columns:
                    df = df.sort_values(df.columns[0])  # 任意稳定顺序
                    codes = df["ts_code"].astype(str).tolist()

    elif label in {"white", "black"}:
        try:
            kind = "whitelist" if label == "white" else "blacklist"
            codes = se._read_cache_list_codes(ref, kind) or []
        except Exception:
            codes = []

    elif label == "attention":
        try:
            codes = se._load_attention_codes(ref) or []
        except Exception:
            # 退回按文件名找 “attention*<ref>.csv”
            p = _find_attn_file_by_date(ref)  # 这个工具已在文件内定义
            if p:
                df = _read_df(p, dtype={"ts_code": str})
                if df is not None and not df.empty:
                    # 智能识别列名
                    for cand in ["ts_code", "code", "ts", "symbol"]:
                        if cand in df.columns:
                            codes = df[cand].astype(str).tolist()
                            break

    # 规范化、去重、排序
    try:
        codes = [normalize_ts(c) for c in codes if c]
    except Exception:
        codes = [str(c).strip() for c in codes if c]
    return sorted(set(codes))

# ==== 强度榜文件定位====
def _pick_latest_attn_date() -> Optional[str]:
    """
    扫描 attention 目录所有 CSV。
    规则：把“文件名里最后一次出现的 8 位数字”视为该文件的结束日；
         若同一结束日有多份，则按文件修改时间(mtime)取最新那份的结束日。
    """
    best_key = None
    best_ref = None
    for p in ATTN_DIR.glob("*.csv"):
        ms = re.findall(r"(\d{8})", p.name)
        if not ms:
            continue
        end = ms[-1]  # 视为结束日（最后 8 位）
        key = (end, p.stat().st_mtime)  # 先比日期，再比修改时间
        if best_key is None or key > best_key:
            best_key, best_ref = key, end
    return best_ref


def _find_attn_file_by_date(ref: str) -> Optional[Path]:
    """
    根据结束日 ref 定位文件：
    - 先收集所有包含该 ref 的 attention CSV；
    - 优先“规范化命名”的文件（以 attention_ 开头且包含 _win/_topM/_topN 这些关键字）；
    - 其余则按修改时间倒序作为次序。
    """
    cands = list(ATTN_DIR.glob(f"*attention*{ref}*.csv"))
    if not cands:
        return None

    def _score(p: Path):
        nm = p.name
        normalized = nm.startswith("attention_") and ("_win" in nm or nm == f"attention_{ref}.csv")
        return (1 if normalized else 0, p.stat().st_mtime)

    return sorted(cands, key=_score, reverse=True)[0]

# ---- 读取 config 的稳健工具 ----
def cfg_int(name: str, default: int) -> int:
    val = getattr(cfg, name, default)
    try:
        # 过滤 None / "" 等异常取值
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return int(default)
        return int(val)
    except Exception:
        return int(default)


def cfg_str(name: str, default: str) -> str:
    val = getattr(cfg, name, default)
    if val is None:
        return str(default)
    s = str(val).strip()
    return s if s else str(default)


def cfg_bool(name: str, default: bool) -> bool:
    val = getattr(cfg, name, default)
    if isinstance(val, bool):
        return val
    if val is None:
        return bool(default)
    s = str(val).strip().lower()
    if s in {"1","true","yes","y","on","t"}:
        return True
    if s in {"0","false","no","n","off","f"}:
        return False
    return bool(default)

# 通用多阶段进度器（统一管理单次任务的进度条+状态日志）
class Stepper:
    """
    用法：
        steps = ["准备环境", "下载源数据", "合并增量", "写出导出", "自动排名"]
        sp = Stepper("下载/同步", steps, key_prefix="dl_sync")  # 每次点击都会生成独立 run_id
        sp.start()

        sp.step("准备环境")     # 做准备...
        sp.tick(0.3, "校验目标目录")
        sp.tick(1.0)           # 本步骤收尾

        sp.step("下载源数据")   # 具体下载...
        sp.step("合并增量")
        sp.step("写出导出")
        sp.step("自动排名", visible=auto_rank)  # 支持按条件显示/跳过
        sp.finish(success=True, note="全部完成")
    """
    def __init__(self, title, steps, key_prefix="stepper"):
        self.title = title
        self.steps_all = steps[:]  # 文案列表（可含 None）
        self.steps = [s for s in steps if s]  # 实际参与统计的步骤
        self.total = len(self.steps)
        self.key = f"{key_prefix}_{uuid.uuid4().hex[:8]}"
        self._init_state()

    def _init_state(self):
        st.session_state[self.key] = {
            "idx": 0,        # 已完成到第几个（从 0 开始）
            "run_id": self.key,
        }

    def start(self):
        self.status = st.status(
            label=f"{self.title}：开始（0/{self.total}）",
            state="running",
        )
        self.prog = st.progress(0, text="准备中…")
        with self.status:
            st.write("🟡 开始任务…")

    def _update_prog(self, idx, label):
        pct = 0 if self.total == 0 else int(idx / self.total * 100)
        self.prog.progress(pct, text=f"{idx}/{self.total}：{label}")

    def step(self, label, visible=True, info=None):
        """进入下一主步骤；visible=False 时，仅记录日志，不纳入进度比例"""
        if not visible:
            # 仅追加日志提示
            with self.status:
                st.write(f"⏭️ 跳过：{label}")
            return

        state = st.session_state[self.key]
        state["idx"] += 1
        idx = min(state["idx"], self.total)
        text = label if not info else f"{label}｜{info}"

        with self.status:
            st.write(f"▶️ {text}")
        self._update_prog(idx, text)

    def tick(self, delta_ratio, info=None):
        """在当前步骤中显示细粒度推进（例如循环/分批处理）"""
        state = st.session_state[self.key]
        # 按当前主步骤位置 + 细分比例 合成一个更平滑的百分比展示
        base = min(state["idx"], self.total - 1)
        now = min(1.0, max(0.0, float(delta_ratio)))
        overall = int(((base + now) / self.total) * 100) if self.total else 0
        self.prog.progress(overall, text=info or "处理中…")
        # 在日志里也可打点
        if info:
            with self.status:
                st.write(f"… {info}")

    def finish(self, success=True, note=None):
        if success:
            self.status.update(
                label=f"{self.title}：完成（{self.total}/{self.total}）",
                state="complete",
            )
            self.prog.progress(100, text=note or "完成")
        else:
            self.status.update(
                label=f"{self.title}：失败",
                state="error",
            )


@contextmanager
def pred_progress_to_streamlit():
    if not _in_streamlit():
        # 非Streamlit环境回调
        def _noop(*a, **k): pass
        # 使用新的日志系统替代废弃的 set_progress_handler
        from log_system import get_logger
        logger = get_logger("predict_core")
        logger.info("使用新的日志系统进行进度跟踪")
        try:
            yield None, None, None
        finally:
            pass
        return

    status = st.status("准备中…", expanded=True)
    bar = st.progress(0, text="就绪")
    info = st.empty()

    import queue as _q
    _evq = _q.Queue()

    # 后台线程只入队，不直接碰 st.*
    def _enqueue_handler(phase, current=None, total=None, message=None, **kw):
        try:
            _evq.put_nowait((phase, current, total, message))
        except Exception:
            pass

    def _render_event(phase, current=None, total=None, message=None):
        txt = {
            "pred_select_ref_date": "选择参考日",
            "pred_build_universe_done": "构建模拟清单",
            "pred_start": "模拟开始",
            "pred_progress": "模拟进行中",
            "pred_done": "模拟完成",
        }.get(phase, phase)
        
        # 使用message作为主要显示内容，如果没有则使用txt
        display_text = message if message else txt
        
        if total and current is not None:
            pct = int(current * 100 / max(total, 1))
            bar.progress(pct, text=f"{display_text} ({current}/{total})")
        else:
            bar.progress(0, text=display_text)

    # 主线程消费：供 run_prediction_in_bg 循环调用
    def _drain():
        try:
            while True:
                ev = _evq.get_nowait()
                _render_event(*ev)
        except _q.Empty:
            pass

    # 使用新的日志系统替代废弃的 set_progress_handler
    from log_system import get_logger
    logger = get_logger("predict_core")
    logger.info("使用新的日志系统进行进度跟踪")
    _orig_drain = getattr(pr, "drain_progress_events", None)
    pr.drain_progress_events = _drain  # 关键：把"抽干"替换成主线程渲染

    try:
        yield status, bar, info
    finally:
        # 还原 drain（保持模块整洁）
        if callable(_orig_drain):
            pr.drain_progress_events = _orig_drain
        else:
            pr.drain_progress_events = lambda: None


def run_prediction_in_bg(inp):
    with pred_progress_to_streamlit() as (status, bar, info):
        done = threading.Event()
        result = {"df": None, "err": None}
        def _worker():
            try:
                # 使用安全的数据库操作
                from predict_core import run_prediction
                result["df"] = run_prediction(inp)
            except Exception as e:
                result["err"] = e
            finally:
                done.set()
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while not done.is_set():
            # 消费 predict_core 的进度事件（如果内部使用事件队列的话）
            try:
                pr.drain_progress_events()
            except Exception:
                pass
            time.sleep(0.05)
        # 抽干剩余事件
        try:
            pr.drain_progress_events()
        except Exception:
            pass
        if status is not None:
            status.update(label="已完成", state="complete")
        if result["err"]:
            raise result["err"]
        return result["df"]

# ===== 主ui部分 =====
if _in_streamlit():
    # ===== 页眉 =====
    st.title("ScoreApp")
    

    # === Global portfolio state (added by assistant) ===
    if "cur_pid" not in st.session_state:
        st.session_state["cur_pid"] = None
    if "cur_pf" not in st.session_state:
        st.session_state["cur_pf"] = None

    # Local aliases for convenience across tabs
    cur_pid = st.session_state.get("cur_pid")
    cur_pf  = st.session_state.get("cur_pf")
    # === End global portfolio state ===

    if "rules_obj" not in st.session_state:
        st.session_state["rules_obj"] = {
            # "prescreen": getattr(cfg, "SC_PRESCREEN_RULES", []),
            # "rules": getattr(cfg, "SC_RULES", []),
            "prescreen": getattr(se, "SC_PRESCREEN_RULES", []),
            "rules": getattr(se, "SC_RULES", []),
        }
    if "export_pref" not in st.session_state:
        st.session_state["export_pref"] = {"style": "space", "with_suffix": True}

    # ===== 顶层页签 =====
    tab_rank, tab_detail, tab_position, tab_predict, tab_rules, tab_attn, tab_custom, tab_screen, tab_tools, tab_port, tab_stats, tab_data_view, tab_logs, = st.tabs(
        ["排名", "个股详情", "持仓建议", "明日模拟", "规则编辑", "强度榜", "自选榜", "选股", "工具箱", "组合模拟/持仓", "统计", "数据管理", "日志"])

    # ================== 排名 ==================
    with tab_rank:
        st.subheader("排名")
        with st.expander("参数设置（运行前请确认）", expanded=True):
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            with c1:
                ref_inp = st.text_input("参考日（YYYYMMDD；留空=自动取最新）", value="", key="rank_ref_input")
                topk = st.number_input("Top-K", min_value=1, max_value=2000, value=cfg_int("SC_TOP_K", 50))
            with c2:
                tie_default = cfg_str("SC_TIE_BREAK", "none").lower()
                tie = st.selectbox("同分排序（Tie-break）", ["none", "kdj_j_asc"], index=0 if tie_default=="none" else 1)
                maxw = st.number_input("最大并行数", min_value=1, max_value=64, value=cfg_int("SC_MAX_WORKERS", 8))
            with c3:
                universe = st.selectbox("评分范围", ["全市场","仅白名单","仅黑名单"], index=0)
                style = st.selectbox("TXT 导出格式", ["空格分隔", "一行一个"], index=0)
            with c4:
                attn_on = False
                with_suffix = st.checkbox("导出带交易所后缀（.SZ/.SH）", value=False)
            st.session_state["export_pref"] = {"style": "space" if style=="空格分隔" else "lines",
                                            "with_suffix": with_suffix}
            run_col1, run_col2 = st.columns([1,1])
            with run_col1:
                run_btn = st.button("🚀 运行评分（写入 Top/All/Details）", width='stretch')
            with run_col2:
                latest_btn = st.button("📅 读取最近一次结果（不重新计算）", width='stretch')

        # 运行
        ref_to_use = ref_inp.strip() or _pick_smart_ref_date()
        if run_btn:
            # 禁用details数据库读取，避免写入冲突
            st.session_state["details_db_reading_enabled"] = False
            logger.info(f"用户点击运行评分按钮: 参考日={ref_to_use}, TopK={topk}, 并行数={maxw}, 范围={universe}，已禁用details数据库读取")
            
            # 关闭details数据库的连接池，断开所有连接
            try:
                manager = get_database_manager()
                if manager:
                    details_db_path = get_details_db_path_with_fallback()
                    if details_db_path:
                        manager.close_db_pools(details_db_path)
                        logger.info(f"已关闭details数据库连接池: {details_db_path}")
            except Exception as e:
                logger.warning(f"关闭details数据库连接池时出错: {e}")
            
            _apply_runtime_overrides(st.session_state["rules_obj"], topk, tie, maxw, attn_on,
                                    {"全市场":"all","仅白名单":"white","仅黑名单":"black","仅特别关注榜":"attention"}[universe])
            try:
                top_path = run_se_run_for_date_in_bg(ref_inp.strip() or None)
                st.success(f"评分完成：{top_path}")
                # 解析参考日
                m = re.search(r"(\d{8})", str(top_path))
                if m:
                    ref_to_use = m.group(1)
                    if latest_btn and not ref_to_use:
                        ref_to_use = _pick_smart_ref_date()
            except Exception as e:
                st.error(f"评分失败：{e}")
                ref_to_use = None

        # "读取最近一次结果"按钮：仅读取，不计算
        if latest_btn and not run_btn:
            # 优先使用用户输入的参考日，如果没有输入则使用最新的文件日期
            ref_to_use = ref_inp.strip() or _get_latest_date_from_files()

        # ---- 统一的 Top 预览区块（无论 run 或 读取最近一次） ----
        if ref_to_use:
            # 获取最新排名文件日期和数据库最新日期用于对比
            latest_rank_date = _get_latest_date_from_files()
            db_latest_date = _get_latest_date_from_database()
            
            # 显示三个日期的对比
            col1, col2, col3 = st.columns(3)
            with col1:
                if latest_rank_date:
                    st.markdown(f"**最新排名文件：{latest_rank_date}**")
                else:
                    st.markdown("**最新排名文件：未知**")
            with col2:
                st.markdown(f"**当前显示排名：{ref_to_use}**")
            with col3:
                if db_latest_date:
                    st.markdown(f"**数据库最新日期：{db_latest_date}**")
                else:
                    st.markdown("**数据库最新日期：未知**")
            
            # 如果有日期差异，给出提示
            if latest_rank_date and latest_rank_date != ref_to_use:
                st.info(f"当前显示的是 {ref_to_use} 的排名，最新排名文件是 {latest_rank_date}")
            
            if db_latest_date and db_latest_date != ref_to_use:
                if db_latest_date > ref_to_use:
                    st.warning(f"排名数据日期（{ref_to_use}）早于数据库最新日期（{db_latest_date}），建议重新运行评分获取最新排名")
                else:
                    st.info(f"排名数据日期（{ref_to_use}）晚于数据库最新日期（{db_latest_date}），排名数据基于较新的数据")
            
            df_all = _read_df(_path_all(ref_to_use))
        else:
            st.info("未找到任何 Top 文件，请先运行评分或检查输出目录。")

        st.divider()

        with st.container(border=True):
            st.markdown("**Top-K 预览**")
            show_mode = st.radio("展示方式", ["限制条数", "显示全部"], horizontal=True, key="topk_show_mode")
            rows_to_show = None
            if show_mode == "限制条数":
                rows_to_show = st.number_input("Top-K 显示行数", min_value=5, max_value=1000, value=cfg_int("SC_TOPK_ROWS", 30), key="topk_rows_cfg")
            if ref_to_use and df_all is not None and not df_all.empty:
                if show_mode == "显示全部":
                    rows_eff = len(df_all)
                    st.caption(f"已选择显示全部（共 {rows_eff} 行）。")
                else:
                    rows_eff = int(rows_to_show)
                st.dataframe(df_all.head(rows_eff), width='stretch', height=420)
                if "ts_code" in df_all.columns:
                    codes = df_all["ts_code"].astype(str).head(rows_eff).tolist()
                    txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"], st.session_state["export_pref"]["with_suffix"])
                    copy_txt_button(txt, label="📋 复制以上（按当前预览）", key=f"copy_top_{ref_to_use}")
            else:
                st.caption("暂无 Top-K 数据")

    # ================== 个股详情 ==================
    with tab_detail:
        st.subheader("个股详情")

        # —— 数据库读取控制按钮 ——
        db_reading_enabled = is_details_db_reading_enabled()
        col_db_ctrl1, col_db_ctrl2 = st.columns([3, 1])
        with col_db_ctrl1:
            if db_reading_enabled:
                st.success("✅ 数据库读取已启用（数据将从数据库读取）")
            else:
                st.info("ℹ️ 数据库读取未启用（避免与写入操作冲突）")
        with col_db_ctrl2:
            if not db_reading_enabled:
                if st.button("🔓 启用数据库读取", key="enable_db_reading"):
                    st.session_state["details_db_reading_enabled"] = True
                    st.rerun()
            else:
                # 一旦启用就不再显示按钮，保持启用状态
                pass
        
        st.divider()

        # —— 选择参考日 + 代码（支持从 Top-K 下拉选择） ——
        c0, c1 = st.columns([1,2])
        with c0:
            ref_d = st.text_input("参考日（留空=自动最新）", value="", key="detail_ref_input")
        ref_real = (ref_d or "").strip() or _get_latest_date_from_files() or ""
        # 读取该参考日 Top 文件以便下拉选择
        try:
            # 刷新缓存
            if ref_real:
                top_path = _path_top(ref_real)
                if top_path.exists():
                    # 清除可能的缓存
                    if hasattr(_read_df, 'clear'):
                        _read_df.clear()
                    df_top_ref = _read_df(top_path)
                else:
                    df_top_ref = pd.DataFrame()
            else:
                df_top_ref = pd.DataFrame()
                
            options_codes = df_top_ref["ts_code"].astype(str).tolist() if ("ts_code" in df_top_ref.columns and not df_top_ref.empty) else []
            st.caption(f"调试: 参考日={ref_real}, TopK文件行数={len(df_top_ref)}, 可选股票数={len(options_codes)}")
        except Exception as e:
            options_codes = []
            st.caption(f"调试: 读取TopK文件失败: {e}")
        with c1:
            # 确保options_codes不为空，且index有效
            if options_codes:
                code_from_list = st.selectbox("从 Top-K 选择（可选）", options=options_codes,
                                            index=0, placeholder="也可手动输入 ↓", key="detail_code_from_top")
            else:
                # 当没有TopK数据时，提供一个默认选项但不禁用
                code_from_list = st.selectbox("从 Top-K 选择（可选）", options=[""],
                                            index=0, placeholder="暂无Top-K数据，请手动输入 ↓", 
                                            key="detail_code_from_top")

        # 初始化session_state
        if 'detail_last_code' not in st.session_state:
            st.session_state.detail_last_code = ""
        
        # 确定默认显示的代码
        default_code = ""
        if st.session_state.detail_last_code:
            # 如果有历史记录，使用历史记录
            default_code = st.session_state.detail_last_code
        elif options_codes:
            # 如果没有历史记录但有Top-K数据，使用第一名
            default_code = options_codes[0]
        
        # 始终显示手动输入框（平级输入方式）
        code_typed = st.text_input("或手动输入股票代码", 
                                 value=default_code,
                                 key="detail_code_input")

        # —— 平级合并逻辑：谁变化用谁 ——
        if 'detail_prev_select' not in st.session_state:
            st.session_state.detail_prev_select = ""
        if 'detail_prev_input' not in st.session_state:
            st.session_state.detail_prev_input = ""

        cur_select = (code_from_list or "").strip()
        cur_input  = (code_typed or "")
        changed_select = bool(cur_select) and (cur_select != st.session_state.detail_prev_select)
        changed_input  = (cur_input != st.session_state.detail_prev_input)

        if changed_select:
            effective_code = cur_select
        elif changed_input:
            effective_code = cur_input
        else:
            # 二者都未变化时，取当前非空输入；再兜底默认
            effective_code = cur_input or cur_select or default_code

        # 记录当前值，供下一次对比
        st.session_state.detail_prev_select = cur_select
        st.session_state.detail_prev_input = cur_input

        # 更新历史记录
        if effective_code and effective_code.strip() != "":
            st.session_state.detail_last_code = effective_code
        
        code_norm = normalize_ts(effective_code) if effective_code else ""

        # —— 渲染详情（含 old 版功能） ——
        if code_norm and ref_real:
            obj = _load_detail_json(ref_real, code_norm)
            if not obj:
                st.warning("未找到该票的详情数据(请检查数据库是否解锁以及是否写入)。")
            else:
                data = obj
                # 兼容数据库格式和JSON格式
                if "summary" in data:
                    # 统一格式：{ts_code, ref_date, summary: {...}, rules}
                    summary = data.get("summary", {}) or {}
                    ts = data.get("ts_code", code_norm)
                else:
                    # 兼容旧格式：{ts_code, ref_date, score, highlights, drawbacks, opportunities, rules, ...}
                    summary = {
                        "score": data.get("score", 0.0),
                        "tiebreak": data.get("tiebreak"),
                        "highlights": data.get("highlights", []),
                        "drawbacks": data.get("drawbacks", []),
                        "opportunities": data.get("opportunities", []),
                        "rank": data.get("rank"),
                        "total": data.get("total")
                    }
                    ts = data.get("ts_code", code_norm)
                
                # 显示数据来源信息（只有在允许读取数据库时才查询数据库状态）
                db_reading_enabled = is_details_db_reading_enabled()
                if db_reading_enabled:
                    try:
                        # 使用统一的函数获取details数据库路径（包含回退逻辑）
                        details_db_path = get_details_db_path_with_fallback()
                        if details_db_path:
                            # 查询股票详情表
                            logger.info(f"[数据库连接] 开始获取数据库管理器实例 (查询股票详情: {code_norm}, {ref_real})")
                            manager = get_database_manager()
                            sql = "SELECT * FROM stock_details WHERE ts_code = ? AND ref_date = ?"
                            df = manager.execute_sync_query(details_db_path, sql, [code_norm, ref_real], timeout=30.0)
                            
                            if not df.empty:
                                st.info("数据来源：数据库")
                            else:
                                st.info("数据来源：JSON文件")
                        else:
                            st.info("数据来源：JSON文件")
                    except:
                        st.info("数据来源：JSON文件")
                else:
                    st.info("数据来源：JSON文件（数据库读取未启用）")
                
                try:
                    score = float(summary.get("score", 0))
                    if not np.isfinite(score):
                        score = 0.0
                except Exception:
                    score = 0.0
                # 计算当日排名（优先 JSON → 全量CSV → Top-K 回退）
                rank_display = "—"
                r_json = summary.get("rank")
                t_json = summary.get("total")
                if isinstance(r_json, (int, float)) and int(r_json) > 0:
                    rank_display = f"{int(r_json)}" + (f" / {int(t_json)}" if isinstance(t_json, (int, float)) and int(t_json) > 0 else "")
                else:
                    all_path = _path_all(ref_real)
                    if all_path.exists():
                        try:
                            df_allx = _read_df(all_path, dtype={"ts_code": str}, encoding="utf-8-sig")
                            if not df_allx.empty:
                                row = df_allx.loc[df_allx["ts_code"].astype(str) == str(ts)]
                                if not row.empty and "rank" in row.columns:
                                    rank_display = f"{int(row['rank'].iloc[0])} / {len(df_allx)}"
                        except Exception:
                            pass
                    # 2) 若全量无果，回退到 Top 文件：按行号近似名次
                    if rank_display == "—":
                        top_path = _path_top(ref_real)
                        if top_path.exists():
                            try:
                                df_topx = _read_df(top_path, dtype={"ts_code": str}, encoding="utf-8-sig")
                                if not df_topx.empty:
                                    if "rank" not in df_topx.columns:
                                        df_topx = df_topx.reset_index(drop=True)
                                        df_topx["rank"] = np.arange(1, len(df_topx) + 1)
                                    row = df_topx.loc[df_topx["ts_code"].astype(str) == str(ts)]
                                    if not row.empty and "rank" in row.columns:
                                        rank_display = f"{int(row['rank'].iloc[0])} / {len(df_topx)}"
                            except Exception:
                                pass
                            try:
                                df_topx = pd.read_csv(top_path, dtype={"ts_code": str}, encoding="utf-8-sig")
                                pos = df_topx.index[df_topx["ts_code"].astype(str) == str(ts)]
                                if len(pos) > 0:
                                    rank_display = str(int(pos[0]) + 1)
                            except Exception:
                                pass

                # 总览 + 高亮/缺点（美化显示）
                colA, colB = st.columns([1,1])
                with colA:
                    st.markdown("**总览**")
                    # 美化显示summary内容
                    with st.container(border=True):
                        # 基本信息
                        st.metric("代码", ts)
                        st.metric("市场", market_label(ts))
                        st.metric("参考日", ref_real)
                        st.divider()
                        # 评分信息
                        if "score" in summary:
                            st.metric("分数", f"{summary.get('score', 0.0):.2f}")
                        if "tiebreak" in summary and summary.get("tiebreak") is not None:
                            st.metric("KDJ-J", f"{summary.get('tiebreak', 0.0):.2f}")
                        if "rank" in summary and summary.get("rank") is not None:
                            total = summary.get("total", 0)
                            rank_val = summary.get("rank", 0)
                            if total > 0:
                                st.metric("排名", f"{rank_val} / {total}")
                            else:
                                st.metric("排名", str(rank_val))
                        # 显示其他summary字段
                        other_fields = {k: v for k, v in summary.items() 
                                      if k not in ["score", "tiebreak", "rank", "total", "highlights", "drawbacks", "opportunities"]}
                        if other_fields:
                            with st.expander("其他信息", expanded=False):
                                for key, value in other_fields.items():
                                    st.text(f"{key}: {value}")
                with colB:
                    st.markdown("**高亮 / 缺点**")
                    # 美化显示highlights和drawbacks
                    with st.container(border=True):
                        highlights = summary.get("highlights", [])
                        drawbacks = summary.get("drawbacks", [])
                        
                        if highlights:
                            st.markdown("**✅ 高亮**")
                            for h in highlights:
                                if h:
                                    st.success(f"• {h}")
                        
                        if drawbacks:
                            st.markdown("**⚠️ 缺点**")
                            for d in drawbacks:
                                if d:
                                    st.error(f"• {d}")
                        
                        if not highlights and not drawbacks:
                            st.caption("暂无")

                # 交易性机会
                ops = (summary.get("opportunities") or [])
                with st.expander("交易性机会", expanded=True):
                    if ops:
                        for t in ops:
                            st.write("• " + str(t))
                    else:
                        st.caption("暂无")

                # 逐规则明细（可选显示 when）
                # rules字段已经通过_load_detail_json统一解析为list[dict]格式
                rules_list = data.get("rules", [])
                if not isinstance(rules_list, list):
                    rules_list = []
                rules = pd.DataFrame(rules_list)
                name_to_when = {}
                
                from datetime import datetime
                import re

                if not rules.empty:
                    
                    def _days_from_ref(d: str | None) -> int | None:
                        if isinstance(d, str) and re.fullmatch(r"\d{8}", d):
                            return (datetime.strptime(ref_real, "%Y%m%d") - datetime.strptime(d, "%Y%m%d")).days
                        return None

                    if not rules.empty:
                        def _pick_last_hit_days(row):
                            # 先看 lag 是否有值（仅 RECENT/DIST/NEAR）
                            lag = row.get("lag")
                            if pd.notna(lag):
                                try:
                                    return int(lag)
                                except Exception:
                                    pass
                            # 否则回落到 hit_date → 天数
                            return _days_from_ref(row.get("hit_date"))

                        rules["last_hit_days"] = rules.apply(_pick_last_hit_days, axis=1)
                        # 可选：显示更干净（支持空值）
                        rules["last_hit_days"] = rules["last_hit_days"].astype("Int64")
                try:
                    for r in (getattr(se, "SC_RULES", []) or []):
                        if "clauses" in r and r["clauses"]:
                            ws = [c.get("when","") for c in r["clauses"] if c.get("when")]
                            name_to_when[str(r.get("name","<unnamed>"))] = " AND ".join(ws)
                        else:
                            name_to_when[str(r.get("name","<unnamed>"))] = str(r.get("when",""))
                except Exception:
                    name_to_when = {}
                
                # 创建name到explain的映射（从策略仓库中获取）
                name_to_explain = {}
                try:
                    for r in (getattr(se, "SC_RULES", []) or []):
                        rule_name = str(r.get("name", ""))
                        if rule_name:
                            explain_val = r.get("explain")
                            if explain_val:
                                name_to_explain[rule_name] = str(explain_val)
                except Exception:
                    name_to_explain = {}
                show_when = st.checkbox("显示规则 when 表达式", value=False, key="detail_show_when")
                if not rules.empty:
                    if show_when:
                        rules["when"] = rules["name"].map(name_to_when).fillna("")
                    st.markdown("**规则明细**")
                    
                    # 创建用于显示的DataFrame副本
                    rules_display = rules.copy()
                    
                    # 从策略仓库中获取explain（如果数据库中没有）
                    if "name" in rules_display.columns:
                        # 添加explain列（从策略仓库中获取）
                        if "explain" not in rules_display.columns:
                            rules_display["explain"] = rules_display["name"].map(lambda n: name_to_explain.get(str(n), ""))
                        else:
                            # 如果数据库中有explain列，但可能为空，则从策略仓库补充
                            rules_display["explain"] = rules_display.apply(
                                lambda row: row.get("explain") if pd.notna(row.get("explain")) and str(row.get("explain")).strip() 
                                else name_to_explain.get(str(row.get("name", "")), ""), axis=1
                            )
                    
                    # 使用streamlit dataframe显示，保留原生交互功能（排序、筛选等）
                    # 确保列顺序：name在最前，explain在最后（如果有）
                    col_order = ["name"]
                    for col in rules_display.columns:
                        if col not in ["name", "explain"]:
                            col_order.append(col)
                    # explain列放在最后（如果有）
                    if "explain" in rules_display.columns:
                        col_order.append("explain")
                    col_order = [c for c in col_order if c in rules_display.columns]
                    rules_display = rules_display[col_order]
                    
                    # 配置列的显示方式（如果explain存在，为name列添加help提示）
                    column_config = None
                    if "explain" in rules_display.columns and "name" in rules_display.columns:
                        try:
                            # 尝试使用column_config配置（streamlit >= 1.23.0支持）
                            column_config = {}
                            # name列的配置，提示用户可以查看explain列
                            column_config["name"] = st.column_config.TextColumn(
                                "策略名称",
                                help="策略的简短名称，详细说明见右侧explain列"
                            )
                            # explain列的配置，说明这是详细说明
                            column_config["explain"] = st.column_config.TextColumn(
                                "详细说明",
                                help="策略的详细说明（鼠标悬浮在此列标题上可查看提示）",
                                width="medium"
                            )
                        except Exception:
                            # 如果column_config不支持，使用默认方式
                            column_config = None
                    
                    # 显示streamlit dataframe，保留所有原生交互功能
                    st.dataframe(
                        rules_display,
                        width='stretch',
                        height=420,
                        hide_index=True,
                        column_config=column_config
                    )
                else:
                    st.info("无规则明细。")
                
                # 显示未触发的规则
                try:
                    # 获取所有规则列表
                    all_rules = getattr(se, "SC_RULES", []) or []
                    # 获取已触发规则的名称集合
                    triggered_rule_names = set()
                    if not rules.empty and "name" in rules.columns:
                        triggered_rule_names = set(rules["name"].astype(str).unique())
                    
                    # 找出未触发的规则
                    untriggered_rules = []
                    for r in all_rules:
                        rule_name = str(r.get("name", ""))
                        if rule_name and rule_name not in triggered_rule_names:
                            # 获取时间周期
                            tf = str(r.get("timeframe", "D")).upper()
                            # 获取命中口径
                            scope = str(r.get("scope", "ANY")).upper().strip()
                            # 对于 scope: LAST，不显示 score_windows
                            if scope == "LAST":
                                win = None  # scope: LAST 不需要 score_windows
                            else:
                                # 获取回看窗口（优先使用 score_windows，如果没有则使用 window，最后使用默认值）
                                win = int(r.get("score_windows") or r.get("window", 60))
                            # 获取分数
                            points = float(r.get("points", 0))
                            # 获取前置门槛
                            gate = r.get("gate")
                            gate_str = ""
                            if gate:
                                if isinstance(gate, dict):
                                    gate_when = gate.get("when", "")
                                    if gate_when:
                                        gate_str = f"gate: {gate_when}"
                                elif isinstance(gate, str):
                                    gate_str = f"gate: {gate}"
                            
                            # 处理子句信息
                            clauses_info = ""
                            if "clauses" in r and r.get("clauses"):
                                clauses = r.get("clauses", [])
                                clause_parts = []
                                for c in clauses:
                                    c_tf = str(c.get("timeframe", "D")).upper()
                                    c_scope = str(c.get("scope", "ANY")).upper().strip()
                                    # 对于 scope: LAST，不显示窗口值
                                    if c_scope == "LAST":
                                        clause_parts.append(f"{c_tf}/-/LAST")
                                    else:
                                        # 优先使用 score_windows，如果没有则使用 window，最后使用默认值
                                        c_win = int(c.get("score_windows") or c.get("window", 60))
                                        clause_parts.append(f"{c_tf}/{c_win}/{c_scope}")
                                if clause_parts:
                                    clauses_info = f"子句: {len(clauses)}个 ({', '.join(clause_parts)})"
                            
                            rule_data = {
                                "name": rule_name,
                                "timeframe": tf,
                                "scope": scope,
                                "points": points,
                                "gate": gate_str if gate_str else "",
                                "clauses": clauses_info if clauses_info else "",
                                "when": name_to_when.get(rule_name, ""),
                                "explain": str(r.get("explain", ""))
                            }
                            # 只有非 LAST scope 才添加 score_windows
                            if scope != "LAST":
                                rule_data["score_windows"] = win
                            # scope: LAST 不添加 score_windows 字段
            
                            untriggered_rules.append(rule_data)
                    
                    if untriggered_rules:
                        st.markdown("**未触发的规则**")
                        untriggered_df = pd.DataFrame(untriggered_rules)
                        
                        # 创建用于显示的DataFrame副本
                        untriggered_display = untriggered_df.copy()
                        
                        # 如果show_when为False，移除when列
                        if not show_when and "when" in untriggered_display.columns:
                            untriggered_display = untriggered_display.drop(columns=["when"])
                        
                        # 确保列顺序：name在最前，然后是timeframe, score_windows(仅非LAST), scope, points, gate, clauses, when(可选), explain在最后
                        col_order = ["name"]
                        # 添加主要字段
                        for col in ["timeframe", "scope", "points"]:
                            if col in untriggered_display.columns:
                                col_order.append(col)
                        # score_windows 只在非 LAST scope 时显示
                        if "score_windows" in untriggered_display.columns:
                            # 检查是否有非 LAST 的规则
                            if not untriggered_display.empty and "scope" in untriggered_display.columns:
                                has_non_last = (untriggered_display["scope"].astype(str).str.upper() != "LAST").any()
                                if has_non_last:
                                    col_order.append("score_windows")
                            else:
                                col_order.append("score_windows")
                        # 添加可选字段（总是显示，即使为空）
                        for col in ["gate", "clauses"]:
                            if col in untriggered_display.columns:
                                col_order.append(col)
                        # 添加when列（如果show_when为True）
                        if show_when and "when" in untriggered_display.columns:
                            col_order.append("when")
                        # explain列放在最后
                        if "explain" in untriggered_display.columns:
                            col_order.append("explain")
                        # 确保只包含存在的列
                        col_order = [c for c in col_order if c in untriggered_display.columns]
                        untriggered_display = untriggered_display[col_order]
                        
                        # 配置列的显示方式
                        untriggered_column_config = None
                        try:
                            untriggered_column_config = {}
                            if "name" in untriggered_display.columns:
                                untriggered_column_config["name"] = st.column_config.TextColumn(
                                    "策略名称",
                                    help="策略的简短名称"
                                )
                            if "timeframe" in untriggered_display.columns:
                                untriggered_column_config["timeframe"] = st.column_config.TextColumn(
                                    "时间周期",
                                    help="D(日线)/W(周线)/M(月线)",
                                    width="small"
                                )
                            if "score_windows" in untriggered_display.columns:
                                # 对于 score_windows 列，使用 TextColumn 以便显示空值或 "-"
                                # 因为 scope: LAST 的规则没有此字段
                                untriggered_column_config["score_windows"] = st.column_config.TextColumn(
                                    "计分窗口",
                                    help="计分窗口条数（用于计分判断的时间窗口）。scope: LAST 的规则不需要此字段，显示为空",
                                    width="small"
                                )
                            if "scope" in untriggered_display.columns:
                                untriggered_column_config["scope"] = st.column_config.TextColumn(
                                    "命中口径",
                                    help="ANY/EACH/PERBAR等",
                                    width="small"
                                )
                            if "points" in untriggered_display.columns:
                                untriggered_column_config["points"] = st.column_config.NumberColumn(
                                    "分数",
                                    help="命中时加/减分",
                                    width="small"
                                )
                            if "gate" in untriggered_display.columns:
                                untriggered_column_config["gate"] = st.column_config.TextColumn(
                                    "前置门槛",
                                    help="前置条件表达式",
                                    width="medium"
                                )
                            if "clauses" in untriggered_display.columns:
                                untriggered_column_config["clauses"] = st.column_config.TextColumn(
                                    "子句信息",
                                    help="多子句组合信息",
                                    width="medium"
                                )
                            if "when" in untriggered_display.columns:
                                untriggered_column_config["when"] = st.column_config.TextColumn(
                                    "条件表达式",
                                    help="TDX风格表达式",
                                    width="large"
                                )
                            if "explain" in untriggered_display.columns:
                                untriggered_column_config["explain"] = st.column_config.TextColumn(
                                    "详细说明",
                                    help="策略的详细说明",
                                    width="medium"
                                )
                        except Exception:
                            untriggered_column_config = None
                        
                        # 显示未触发的规则
                        st.dataframe(
                            untriggered_display,
                            width='stretch',
                            height=420,
                            hide_index=True,
                            column_config=untriggered_column_config
                        )
                    else:
                        st.info("所有规则均已触发。")
                except Exception as e:
                    logger.debug(f"显示未触发规则失败: {e}")
                    # 静默失败，不影响主流程
                
                # st.markdown('<div id="rank_rule_anchor"></div>', unsafe_allow_html=True)
                st.markdown('<div id="detail_rule_anchor_detail"></div>', unsafe_allow_html=True)

    # ================== 持仓建议 ==================
    with tab_position:
        st.subheader("持仓建议（个股）")
        with st.expander("输入", expanded=True):
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                pos_ref = st.text_input("参考日（YYYYMMDD；留空=自动取最新）", value="", key="pos_ref_input")
                price_mode = st.selectbox("买点来源", ["按日期取价", "策略取价（可选）", "手工输入"], index=0)
            with c2:
                raw_code = st.text_input("股票代码（支持多种写法）", value="", key="pos_code_input")
                price_field = st.selectbox("价格口径", ["开盘价(open)", "收盘价(close)", "最高价(high)", "最低价(low)"], index=0)
            with c3:
                # recompute_opts = st.multiselect("仅重算需要的指标", ["kdj","ma","macd"], default=["kdj"], key="pos_recompute_indicators")
                recalc_mode_pos = st.radio("指标重算", ["自选", "全部(all)", "不重算(none)"],
                                        index=0, horizontal=True, key="pos_recalc_mode")
                if recalc_mode_pos == "自选":
                    recompute_opts = st.multiselect("仅重算需要的指标",
                                                    _indicator_options(),
                                                    default=["kdj"],
                                                    key="pos_recompute_pick")
                    recompute_to_pass = tuple(recompute_opts) if recompute_opts else ("kdj",)
                elif recalc_mode_pos == "全部(all)":
                    recompute_to_pass = "all"
                else:
                    recompute_to_pass = "none"

                use_virtual = st.checkbox("基于“明日虚拟日”检查（勾选后按下方场景）", value=False)

            # 场景参数（仅当 use_virtual）
            scen = Scenario()
            if use_virtual:
                # with st.expander("明日情景参数", expanded=False):
                with st.container(border=True):
                    st.markdown("**明日情景参数**")
                    cc1, cc2, cc3 = st.columns([1,1,1])
                    with cc1:
                        scen_mode = st.selectbox("价格模式", ["close_pct","open_pct","gap_then_close_pct","flat","limit_up","limit_down"], index=0)
                        pct = st.number_input("涨跌幅 pct（%）", value=2.0, step=0.5, format="%.2f")
                        gap_pct = st.number_input("跳空 gap_pct（%）", value=0.0, step=0.5, format="%.2f")
                    with cc2:
                        vol_mode = st.selectbox("量能模式", ["same","pct","mult"], index=2)
                        vol_arg = st.number_input("量能参数（% 或 倍数）", value=1.2, step=0.1, format="%.2f")
                        hl_mode = st.selectbox("高低生成", ["follow","atr_like","range_pct"], index=0)
                    with cc3:
                        range_pct = st.number_input("range_pct（%）", value=2.0, step=0.5, format="%.2f")
                        atr_mult = st.number_input("atr_mult", value=1.0, step=0.1, format="%.2f")
                        lock_hi_open = st.checkbox("锁定收盘高于开盘", value=False)
                    scen = Scenario(mode=scen_mode, pct=pct, gap_pct=gap_pct, vol_mode=vol_mode, vol_arg=vol_arg,
                                    hl_mode=hl_mode, range_pct=range_pct, atr_mult=atr_mult,
                                    lock_higher_than_open=lock_hi_open)

            # 买点来源
            entry_price = None
            # 统一参考日
            try:
                trade_dates = get_trade_dates()
                latest_ref = trade_dates[-1] if trade_dates else ""
            except Exception:
                latest_ref = ""
            ref_use = pos_ref.strip() or latest_ref

            code_norm = normalize_ts(raw_code.strip()) if raw_code.strip() else ""
            if price_mode == "按日期取价":
                sel_date = st.text_input("买点日期（YYYYMMDD）", value=ref_use, key="pos_entry_date")
                if st.button("取价", width='stretch'):
                    if code_norm and sel_date:
                        try:
                            # 读取该日的价格
                            try:
                                from config import DATA_ROOT, UNIFIED_DB_PATH
                                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                                df = query_stock_data(
                                    db_path=db_path,
                                    ts_code=code_norm,
                                    start_date=sel_date,
                                    end_date=sel_date,
                                    adj_type="qfq"
                                )
                            except:
                                # 回退到直接查询
                                from config import DATA_ROOT, UNIFIED_DB_PATH
                                db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                                logger.info(f"[数据库连接] 开始获取数据库管理器实例 (回退查询单日数据: {code_norm}, {sel_date})")
                                manager = get_database_manager()
                                sql = "SELECT * FROM stock_data WHERE ts_code = ? AND trade_date = ?"
                                df = manager.execute_sync_query(db_path, sql, [code_norm, sel_date], timeout=30.0)
                            if not df.empty:
                                row = df.sort_values("trade_date").iloc[-1]
                                fld = {"开盘价(open)":"open","收盘价(close)":"close","最高价(high)":"high","最低价(low)":"low"}[price_field]
                                entry_price = float(row[fld])
                                st.success(f"买点={entry_price:.4f}")
                            else:
                                st.warning("该日无数据")
                        except Exception as e:
                            st.error(f"取价失败：{e}")
            elif price_mode == "手工输入":
                entry_price = st.number_input("手工输入买点", value=0.0, step=0.01, format="%.4f")
            else:
                # 策略取价（可选）
                opps = load_opportunity_policies()
                names = [r.get("name","") for r in opps]
                if not names:
                    st.info("暂无“买点策略（个股）”可用，请在 strategies_repo.py 填写 OPPORTUNITY_POLICIES。")
                opp_name = st.selectbox("选择买点策略", names, index=0 if names else None)
                lookback_days = st.number_input("回看天数", min_value=30, max_value=1000, value=180)
                if st.button("按策略取最近一次触发日并定价", width='stretch', disabled=not (code_norm and names)):
                    try:
                        start = (datetime.strptime(ref_use, "%Y%m%d") - timedelta(days=int(lookback_days))).strftime("%Y%m%d")
                        try:
                            from config import DATA_ROOT, UNIFIED_DB_PATH
                            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                            df = query_stock_data(
                                db_path=db_path,
                                ts_code=code_norm,
                                start_date=start,
                                end_date=ref_use,
                                adj_type="qfq"
                            )
                        except:
                            # 回退到直接查询
                            from config import DATA_ROOT, UNIFIED_DB_PATH
                            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                            logger.info(f"[数据库连接] 开始获取数据库管理器实例 (回退查询日期范围数据: {code_norm}, {start}~{ref_use})")
                            manager = get_database_manager()
                            sql = "SELECT * FROM stock_data WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?"
                            df = manager.execute_sync_query(db_path, sql, [code_norm, start, ref_use], timeout=30.0)
                        df = df.sort_values("trade_date")
                        if df.empty:
                            st.warning("无数据")
                        else:
                            # 表达式运行
                            expr = next((r.get("when") or r.get("check") or "" for r in opps if r.get("name")==opp_name), "")
                            if not expr:
                                st.warning("策略没有 when/check 表达式")
                            else:
                                # 计算指标（扩展数据）
                                try:
                                    # 确保数据格式正确
                                    df['open'] = pd.to_numeric(df['open'], errors='coerce')
                                    df['high'] = pd.to_numeric(df['high'], errors='coerce')
                                    df['low'] = pd.to_numeric(df['low'], errors='coerce')
                                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                                    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
                                    
                                    # 计算KDJ指标
                                    if 'j' in expr.lower():
                                        df['j'] = ind.kdj(df)
                                    
                                    # 扩展ctx，包含指标
                                    ctx = {}
                                    for col in df.columns:
                                        if col.lower() not in ['trade_date', 'ts_code', 'adj_factor']:
                                            try:
                                                ctx[col.upper()] = pd.to_numeric(df[col], errors='coerce').values
                                            except:
                                                pass
                                    
                                    # 如果ctx为空，使用基础OHLCV
                                    if not ctx:
                                        ctx = {
                                            "OPEN": df["open"].astype(float).values,
                                            "HIGH": df["high"].astype(float).values,
                                            "LOW": df["low"].astype(float).values,
                                            "CLOSE": df["close"].astype(float).values,
                                            "V": df["vol"].astype(float).values,
                                        }
                                    
                                    # 设置EXTRA_CONTEXT以便GET_LAST_CONDITION_PRICE等函数使用
                                    original_ctx_df = None
                                    try:
                                        from tdx_compat import EXTRA_CONTEXT
                                        original_ctx_df = EXTRA_CONTEXT.get("DF")
                                        # 创建包含所有列的DataFrame
                                        ctx_df = pd.DataFrame(ctx)
                                        # 确保列名小写（tdx_compat需要）
                                        ctx_df.columns = ctx_df.columns.str.lower()
                                        EXTRA_CONTEXT["DF"] = ctx_df
                                    except:
                                        pass
                                    
                                    try:
                                        # 使用evaluate_bool评估表达式
                                        sig = tdx.evaluate_bool(expr, pd.DataFrame(ctx))
                                    finally:
                                        # 恢复原始EXTRA_CONTEXT
                                        if original_ctx_df is not None:
                                            try:
                                                from tdx_compat import EXTRA_CONTEXT
                                                EXTRA_CONTEXT["DF"] = original_ctx_df
                                            except:
                                                pass
                                    idx = [i for i, v in enumerate(sig) if bool(v)]
                                    if not idx:
                                        st.info("回看期内无触发")
                                    else:
                                        last_i = idx[-1]
                                        row = df.iloc[last_i]
                                        fld = {"开盘价(open)":"open","收盘价(close)":"close","最高价(high)":"high","最低价(low)":"low"}[price_field]
                                        entry_price = float(row[fld])
                                        st.success(f"触发日 {row['trade_date']}，买点={entry_price:.4f}")
                                except Exception as e2:
                                    st.error(f"计算失败：{e2}")
                                    import traceback
                                    st.code(traceback.format_exc())
                    except Exception as e:
                        st.error(f"策略取价失败：{e}")

        # 选择“持仓检查策略（个股）”
        pos_rules = load_position_policies()
        pos_names = [r.get("name","") for r in pos_rules]
        selected = st.multiselect("选择要检查的策略", pos_names, default=pos_names)
        selected_rules = [r for r in pos_rules if r.get("name") in set(selected)]

        if st.button("执行检查", width='stretch', disabled=not (code_norm and ref_use and selected_rules)):
            try:
                # 决定 entry_price
                ep = float(entry_price) if entry_price else None
                if ep is None:
                    st.warning("请先设置买点（上面的【取价】或手工输入）。")
                else:
                    pci = PositionCheckInput(
                        ref_date=ref_use,
                        ts_code=code_norm,
                        rules=selected_rules,
                        entry_price=ep,
                        use_scenario=bool(use_virtual),
                        scenario=scen if use_virtual else None,
                        # recompute_indicators=tuple(recompute_opts) if recompute_opts else ("kdj",),
                        recompute_indicators=recompute_to_pass,
                        extra_vars=None
                    )
                    tbl = run_position_checks(pci)
                    st.dataframe(tbl, width='stretch')
                    # 导出
                    csv = tbl.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("导出 CSV", data=csv, file_name=f"position_checks_{code_norm}_{ref_use}.csv", mime="text/csv", width='stretch')
            except Exception as e:
                st.error(f"执行失败：{e}")

    # ================== 明日模拟 ==================
    with tab_predict:
        st.subheader("明日模拟")
        
        # 明日模拟
        # 使用 st.form 防止参数变化时立即刷新UI
        with st.form("prediction_form"):
            with st.expander("输入参数", expanded=True):
                c1, c2 = st.columns([1,1])
                with c1:
                    pred_ref = st.text_input("参考日（YYYYMMDD；留空=自动取最新交易日）", value="", key="pred_ref_input")
                    if not pred_ref.strip():
                        # 显示当前会自动使用的参考日
                        auto_ref = _pick_smart_ref_date()
                        if auto_ref:
                            st.caption(f"💡 将自动使用最新交易日: {auto_ref}")
                        else:
                            st.caption("⚠️ 无法自动获取最新交易日，请手动输入")
                    use_rule_scen = st.checkbox("使用规则内置场景（若规则提供）", value=False)
                    expr_text = st.text_input("临时检查表达式（可留空）", value="")
                    # recompute_opts = st.multiselect("仅重算需要的指标", ["kdj","ma","macd"], default=["kdj"], key="pred_recompute_indicators")
                    recalc_mode_pred = st.radio("指标重算", ["自选", "全部(all)", "不重算(none)"],
                                                index=0, horizontal=True, key="pred_recalc_mode")
                    if recalc_mode_pred == "自选":
                        recompute_opts = st.multiselect("仅重算需要的指标",
                                                        _indicator_options(),
                                                        default=["kdj"],
                                                        key="pred_recompute_pick")
                        recompute_to_pass = tuple(recompute_opts) if recompute_opts else ("kdj",)
                    elif recalc_mode_pred == "全部(all)":
                        recompute_to_pass = "all"
                    else:
                        recompute_to_pass = "none"
                with c2:
                    uni_choice_pred = st.selectbox(
                        "选股范围",
                        ["自定义（下方文本）","全市场","仅白名单","仅黑名单","仅特别关注榜"],
                        index=0, key="pred_uni_choice")
                    # 文本框仅在"自定义"时使用
                    pasted = st.text_area("选股范围（支持多种分隔符：空格、换行、逗号、分号、竖线等；可混合 ts_code / 简写）", height=120, placeholder="例：\n000001.SZ 600000.SH 000001\n或：\n000001.SZ,600000.SH;000001|300001", disabled=(not uni_choice_pred.startswith("自定义")) )
            # with st.expander("全局场景（若未使用规则内置场景则生效）", expanded=False):
            with st.container(border=True):
                st.markdown("**全局场景（若未使用规则内置场景则生效）**")
                cc1, cc2, cc3 = st.columns([1,1,1])
                with cc1:
                    scen_mode = st.selectbox("价格模式", ["close_pct","open_pct","gap_then_close_pct","flat","limit_up","limit_down","reverse_indicator"], index=0)
                    pct = st.number_input("涨跌幅 pct（%）", value=2.0, step=0.5, format="%.2f")
                    gap_pct = st.number_input("跳空 gap_pct（%）", value=0.0, step=0.5, format="%.2f")
                with cc2:
                    vol_mode = st.selectbox("量能模式", ["same","pct","mult"], index=2)
                    vol_arg = st.number_input("量能参数（% 或 倍数）", value=1.2, step=0.1, format="%.2f")
                    hl_mode = st.selectbox("高低生成", ["follow","atr_like","range_pct"], index=0)
                with cc3:
                    range_pct = st.number_input("range_pct（%）", value=2.0, step=0.5, format="%.2f")
                    atr_mult = st.number_input("atr_mult", value=1.0, step=0.1, format="%.2f")
                    lock_hi_open = st.checkbox("锁定收盘高于开盘", value=False)
            
            # 反推模式参数配置
            if scen_mode == "reverse_indicator":
                with st.container(border=True):
                    st.markdown("**反推模式参数**")
                    rc1, rc2, rc3 = st.columns([1,1,1])
                    with rc1:
                        reverse_indicator = st.selectbox("指标名称", ["j", "rsi", "ma", "macd", "diff"], index=0)
                        reverse_target_value = st.number_input("目标指标值", value=10.0, step=0.1, format="%.2f")
                    with rc2:
                        reverse_method = st.selectbox("求解方法", ["optimize", "binary_search", "grid_search"], index=0)
                        reverse_tolerance = st.number_input("求解精度", value=1e-6, step=1e-7, format="%.2e")
                    with rc3:
                        reverse_max_iterations = st.number_input("最大迭代次数", value=1000, step=100, min_value=100, max_value=10000)
                        st.caption("反推模式说明：根据目标指标值反推价格数据")
            
            # 规则选择（使用缓存）
            rules = _cached_load_prediction_rules()
            names = [r.get("name","") for r in rules]
            chosen = st.multiselect("选择模拟策略（可留空）", names, default=[])
            chosen_rules = [r for r in rules if r.get("name") in set(chosen)]
            
            # Tie-break排序选择
            tiebreak_pred = st.selectbox("同分排序", ["none", "kdj_j_asc"], index=1, key="pred_tiebreak")

            # 提交按钮
            submitted = st.form_submit_button("运行明日模拟", width='stretch')
        
        # 只有在表单提交时才执行计算
        if submitted:
            # 参考日与代码集 - 使用智能获取函数
            ref_use = pred_ref.strip() or _pick_smart_ref_date() or ""

            # 解析粘贴的文本范围 - 支持空格和各种分隔符的兼容版本
            def _parse_codes(txt: str):
                out = []
                if not txt:
                    return out
                
                # 支持多种分隔符：换行、空格、制表符、逗号、分号、竖线
                import re
                # 使用正则表达式分割，支持多种分隔符
                separators = r'[\s\n\r\t,;|]+'
                codes = re.split(separators, txt)
                
                for code in codes:
                    s = code.strip()
                    if not s:
                        continue
                    try:
                        out.append(normalize_ts(s))
                    except Exception:
                        continue
                # 去重
                return sorted(set([x for x in out if x]))
            uni = _parse_codes(pasted)

            # 创建Scenario对象，根据模式包含不同参数
            if scen_mode == "reverse_indicator":
                scen = Scenario(
                    mode=scen_mode, 
                    pct=pct, 
                    gap_pct=gap_pct, 
                    vol_mode=vol_mode, 
                    vol_arg=vol_arg,
                    hl_mode=hl_mode, 
                    range_pct=range_pct, 
                    atr_mult=atr_mult,
                    lock_higher_than_open=lock_hi_open,
                    # 反推模式参数
                    reverse_indicator=reverse_indicator,
                    reverse_target_value=reverse_target_value,
                    reverse_method=reverse_method,
                    reverse_tolerance=reverse_tolerance,
                    reverse_max_iterations=reverse_max_iterations
                )
            else:
                scen = Scenario(
                    mode=scen_mode, 
                    pct=pct, 
                    gap_pct=gap_pct, 
                    vol_mode=vol_mode, 
                    vol_arg=vol_arg,
                    hl_mode=hl_mode, 
                    range_pct=range_pct, 
                    atr_mult=atr_mult,
                    lock_higher_than_open=lock_hi_open
                )

            _uni_map = {"全市场": "all", "仅白名单": "white", "仅黑名单": "black", "仅特别关注榜": "attention"}
            use_codes = uni_choice_pred.startswith("自定义")
            if use_codes:
                uni_arg = uni  # 粘贴的自定义列表，前面已 normalize 去重
            else:
                uni_label = _uni_map.get(uni_choice_pred, "all")
                uni_arg = _resolve_pred_universe(uni_label, ref_use)

            # 只有当 ref 有效且范围"非空"时才允许运行
            can_run = bool(ref_use) and bool(uni_arg)

            # 可选：为空时给个提示
            if not use_codes and not uni_arg:
                st.info(f"【{uni_choice_pred}】在 {ref_use} 无可用代码源，请先在\"排名\"页签生成当日 all/top 文件或检查名单缓存。")
            
            if use_codes:
                if uni_arg:
                    st.success(f"✅ 自定义名单解析成功：共 {len(uni_arg)} 只股票")
                    # 显示前几只股票作为预览
                    preview_codes = uni_arg[:5]
                    st.caption(f"预览：{', '.join(preview_codes)}{'...' if len(uni_arg) > 5 else ''}")
                else:
                    st.warning("⚠️ 自定义名单为空，请检查输入的股票代码格式")
                    st.caption("支持的格式：000001、000001.SZ、SZ000001、600000.SH 等；支持分隔符：空格、换行、逗号、分号、竖线")

            if can_run:
                try:
                    inp = PredictionInput(
                        ref_date=ref_use,
                        universe=uni_arg,
                        scenario=scen,
                        rules=chosen_rules if chosen_rules else None,
                        expr=(expr_text or None),
                        use_rule_scenario=bool(use_rule_scen),
                        # recompute_indicators=tuple(recompute_opts) if recompute_opts else ("kdj",),
                        recompute_indicators=recompute_to_pass,
                        cache_dir="cache/sim_pred"
                    )
                    # df = run_prediction(inp)
                    df = run_prediction_in_bg(inp)
                    # 应用Tie-break排序
                    df_sorted = _apply_tiebreak_sorting(df, tiebreak_pred)
                    
                    # 显示结果信息
                    if not df_sorted.empty:
                        st.caption(f"命中 {len(df_sorted)} 只；参考日：{ref_use}")
                        if 'score' in df_sorted.columns and df_sorted['score'].notna().any():
                            st.caption("已按得分排序（降序），同分时按J值升序")
                        else:
                            st.caption("未找到得分数据，按默认排序")
                    
                    st.dataframe(df_sorted, width='stretch')
                    
                    # 复制代码功能（与选股页面保持一致）
                    if not df_sorted.empty and "ts_code" in df_sorted.columns:
                        codes = df_sorted["ts_code"].astype(str).tolist()
                        txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"], 
                                          st.session_state["export_pref"]["with_suffix"])
                        copy_txt_button(txt, label="📋 复制命中代码", key=f"copy_prediction_{ref_use}")
                    
                    # 下载
                    csv = df_sorted.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("导出 CSV", data=csv, file_name=f"prediction_hits_{ref_use}.csv", mime="text/csv", width='stretch')
                    # 仅导出代码 TXT
                    if not df_sorted.empty:
                        codes_txt = "\n".join(df_sorted["ts_code"].astype(str).tolist())
                        st.download_button("导出代码TXT（仅命中集）", data=codes_txt, file_name=f"prediction_hits_{ref_use}.txt", mime="text/plain", width='stretch')
                except Exception as e:
                    st.error(f"运行失败：{e}")
                    with st.expander("调试信息", expanded=False):
                        st.write(f"""
**错误详情：**
- 参考日：{ref_use}
- 股票数量：{len(uni_arg) if uni_arg else 0}
- 股票列表：{uni_arg[:10] if uni_arg else '无'}
- 场景模式：{scen_mode}
- 规则数量：{len(chosen_rules) if chosen_rules else 0}
- 表达式：{expr_text or '无'}

**可能的原因：**
1. 股票代码格式不正确
2. 参考日无交易数据
3. 股票在参考日停牌或退市
4. 规则表达式有语法错误
5. 数据文件缺失或损坏

**建议：**
1. 检查股票代码格式（如：000001.SZ）
2. 尝试使用其他参考日
3. 检查规则表达式语法
4. 确认数据文件完整性
                        """)
            else:
                st.warning("请检查参数设置，确保参考日和选股范围都有效")

    # ================== 规则编辑辅助模块 ==================
    with tab_rules:
        render_rule_editor()
        
        st.markdown("---")
        
        # ================== 策略语法检查 ==================
        st.subheader("策略语法检查器")
        st.info("自动检查本地策略文件的语法错误、必填字段、表达式有效性等")
        
        with st.expander("使用方法 / 字段说明", expanded=False):
            st.markdown("""
            **策略语法检查器功能：**
            
            1. **自动文件定位** - 自动扫描并定位策略文件
            2. **语法验证** - 验证策略规则的语法和字段有效性
            3. **表达式检查** - 检查TDX表达式的正确性
            4. **字段验证** - 检查必填字段和字段类型
            5. **指标检查** - 验证指标依赖关系
            
            **支持的策略类型：**
            - 排名策略 (ranking)
            - 筛选策略 (filter)  
            - 模拟策略 (prediction)
            - 持仓策略 (position)
            - 买点策略 (opportunity)
            
            **检查内容：**
            - ✅ Python文件语法正确性
            - ✅ 策略列表结构正确性
            - ✅ 每个规则的字段和表达式
            - ✅ 必填字段完整性
            - ✅ 字段类型正确性
            - ✅ 表达式语法正确性
            - ✅ 支持的函数和变量
            - ✅ 缺失的数据列和指标
            """)

        # 导入验证器
        try:
            from strategies_repo import validate_strategy_file
            validation_available = True
        except ImportError:
            st.error("策略验证器模块未找到，请确保 strategy_validator.py 文件存在")
            validation_available = False
        
        if validation_available:
            # 自动定位策略文件
            strategy_files = []
            import glob
            import os
            
            # 按优先级搜索策略文件
            search_paths = [
                "strategies_repo.py",  # 当前目录
                "strategies/strategies_repo.py",  # strategies目录
                "**/strategies_repo.py",  # 递归搜索
            ]
            
            for pattern in search_paths:
                files = glob.glob(pattern, recursive=True)
                for file in files:
                    if os.path.isfile(file):
                        # 转换为绝对路径避免重复
                        abs_path = os.path.abspath(file)
                        if abs_path not in strategy_files:
                            strategy_files.append(abs_path)
            
            # 去重并排序
            strategy_files = sorted(strategy_files)
            
            if strategy_files:
                # 自动选择第一个文件（通常是主要的策略文件）
                default_file = strategy_files[0]
                
                if len(strategy_files) > 1:
                    selected_file = st.selectbox(
                        "选择策略文件",
                        strategy_files,
                        index=0,
                        help=f"自动定位到 {len(strategy_files)} 个策略文件，默认选择: {default_file}"
                    )
                else:
                    selected_file = default_file
                    st.info(f"自动定位到策略文件: {selected_file}")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    check_btn = st.button("🔍 检查语法", width='stretch')
                with col2:
                    if st.button("📄 查看文件内容", width='stretch'):
                        try:
                            with open(selected_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            st.code(content, language='python')
                        except Exception as e:
                            st.error(f"读取文件失败: {e}")
                
                if check_btn:
                    with st.spinner("正在检查策略文件..."):
                        try:
                            result = validate_strategy_file(selected_file)
                            
                            # 显示验证结果
                            if result.is_valid:
                                st.success("✅ 策略文件验证通过！")
                            else:
                                st.error("❌ 策略文件验证失败")
                            
                            # 显示错误信息
                            if result.errors:
                                st.markdown("#### 🚨 错误")
                                for error in result.errors:
                                    field_info = f" (字段: {error['field']})" if error.get('field') else ""
                                    st.error(f"• {error['message']}{field_info}")
                            
                            if result.warnings:
                                st.markdown("#### ⚠️ 警告")
                                for warning in result.warnings:
                                    field_info = f" (字段: {warning['field']})" if warning.get('field') else ""
                                    st.warning(f"• {warning['message']}{field_info}")
                            
                            # 显示建议
                            if result.suggestions:
                                st.markdown("#### 💡 建议")
                                for suggestion in result.suggestions:
                                    field_info = f" (字段: {suggestion['field']})" if suggestion.get('field') else ""
                                    st.info(f"• {suggestion['message']}{field_info}")
                            
                            # 显示缺失的列和指标
                            if result.missing_columns:
                                st.markdown("#### 📊 缺失的数据列")
                                st.warning(f"以下列在数据中不存在: {', '.join(result.missing_columns)}")
                            
                            if result.missing_indicators:
                                st.markdown("#### 🔧 缺失的指标")
                                st.warning(f"以下指标未注册: {', '.join(result.missing_indicators)}")
                            
                            if result.syntax_issues:
                                st.markdown("#### 🔍 语法问题")
                                for issue in result.syntax_issues:
                                    st.warning(f"• {issue}")
                            
                        except Exception as e:
                            st.error(f"文件验证出错: {e}")
            else:
                st.warning("未找到策略文件，请确保 strategies_repo.py 文件存在")

    # ================== 强度榜 ==================
    with tab_attn:
        st.subheader("强度榜")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            src = st.selectbox("来源", ["top","white","black","attention"], index=0)
            method = st.selectbox("方法", ["强度（带权）","次数（不带权）"], index=0)
        with c2:
            win_n = st.number_input("窗口天数 N", min_value=1, max_value=365, value=60)
            top_m = st.number_input("Top-M 过滤（仅统计前 M 名）", min_value=1, max_value=5000, value=3000)
        with c3:
            weight = st.selectbox("时间权重", ["不加权","指数半衰","线性最小值"], index=1)
            out_n = st.number_input("输出 Top-N", min_value=1, max_value=1000, value=100)
        with c4:
            # date_end = st.text_input("结束日（YYYYMMDD；留空=自动最新）", value="")
            date_end = st.text_input("结束日（YYYYMMDD；留空=自动最新）", value="", key="attn_end_date")
            gen_btn = st.button("生成并预览", width='stretch')

        if gen_btn:
            try:
                # 1) 计算 start/end（按交易日）
                days = _cached_trade_dates(DATA_ROOT, API_ADJ)
                end = (date_end or (days[-1] if days else None))
                if not end:
                    st.error("未能确定结束日"); st.stop()
                if end in days:
                    j = days.index(end)
                    start = days[max(0, j - int(win_n))]
                else:
                    start = days[-int(win_n)] if days else end

                # 2) 参数映射
                mode_map = {"强度（带权）": "strength", "次数（不带权）": "hits"}
                w_map    = {"不加权": "none", "指数半衰": "exp", "线性最小值": "linear"}

                # 3) 正确调用 scoring_core 接口
                csv_path = se.build_attention_rank(
                    start=start, end=end, source=src,
                    min_hits=None, topN=int(out_n), write=True,
                    mode=mode_map[method], weight_mode=w_map[weight],
                    topM=int(top_m)
                )
                st.success(f"强度榜已生成：{csv_path}")
                df_a = pd.read_csv(csv_path)
                st.dataframe(df_a, width='stretch', height=480)
                try:
                    if df_a is not None and not df_a.empty:
                        # 识别代码列（优先 ts_code）
                        code_col = None
                        for cand in ["ts_code", "code", "ts", "symbol"]:
                            if cand in df_a.columns:
                                code_col = cand
                                break

                        if code_col:
                            codes = df_a[code_col].astype(str).tolist()
                            txt = _codes_to_txt(
                                codes,
                                st.session_state["export_pref"]["style"],
                                st.session_state["export_pref"]["with_suffix"]
                            )
                            # 复制按钮（使用已有的 copy_txt_button）
                            copy_txt_button(
                                txt,
                                label="📋 复制强度榜（按当前输出）",
                                key=f"copy_attn_{end}_{src}"
                            )
                            # TXT 导出（文件名含参数，便于追溯）
                            _download_txt(
                                "导出强度榜 TXT",
                                txt,
                                f"attention_{src}_{mode_map[method]}_{w_map[weight]}_{start}_{end}.txt",
                                key="dl_attention_txt"
                            )
                        else:
                            st.caption("未找到代码列（期望列名：ts_code）。")
                except Exception as e:
                    st.warning(f"导出/复制失败：{e}")
                    
                # —— 以下为“强度榜落盘（CSV/TXT，含清晰文件名）” 
                save_extra = cfg_bool("SC_ATTENTION_SAVE_EXTRA", False)
                if save_extra:
                    try:
                        fname_base = f"attention_{src}_{mode_map[method]}_{w_map[weight]}_win{int(win_n)}_topM{int(top_m)}_{start}_{end}_topN{int(out_n)}"
                        dest_csv = ATTN_DIR / f"{fname_base}.csv"
                        dest_txt = ATTN_DIR / f"{fname_base}.txt"

                        # 1) 复制 CSV（若名字不同）
                        try:
                            if str(csv_path) != str(dest_csv):
                                shutil.copyfile(csv_path, dest_csv)
                        except Exception as _e:
                            st.warning(f"CSV 落盘失败（不影响页面预览）：{_e}")

                        # 2) 写 TXT（只有前面生成过 txt 才写）
                        if 'txt' in locals():
                            try:
                                dest_txt.write_text(txt, encoding="utf-8-sig")
                            except Exception as _e:
                                st.warning(f"TXT 落盘失败（不影响页面预览）：{_e}")

                        st.caption(f"已落盘：{dest_csv.name} / {dest_txt.name}（目录：{ATTN_DIR}）")
                    except Exception as _e:
                        st.warning(f"强度榜落盘出现异常：{_e}")

            except Exception as e:
                st.error(f"生成失败：{e}")
                
        st.subheader("本地读取")

        c1, c2 = st.columns([1,1])
        with c1:
            ref_inp_attn = st.text_input("参考日（YYYYMMDD；留空=自动取最新）", value="", key="attn_ref_input")
        with c2:
            sort_key = st.selectbox("排序依据", ["score ↓", "rank ↑", "保持原文件顺序"], index=0, key="attn_sort_key")
        topn_attn = st.number_input("Top-K 显示行数", min_value=5, max_value=1000, value=cfg_int("SC_ATTENTION_TOP_K", 50), key="attn_topn")
        # 决定参考日与文件路径
        ref_attn = (ref_inp_attn.strip() or _pick_latest_attn_date())
        if not ref_attn:
            st.info("未在 attention 目录发现任何 CSV，请先产出强度榜或检查输出路径。")
        else:
            attn_path = _find_attn_file_by_date(ref_attn)
            st.caption(f"参考日：{ref_attn}")
            if not attn_path or (not attn_path.exists()):
                st.warning("未找到该日的强度榜文件。")
            else:
                # 读取强度榜
                df_attn = _read_df(attn_path)
                if df_attn is None or df_attn.empty:
                    st.warning("强度榜文件为空或无法读取。")

        # 只有在有有效数据时才进行排序和显示
        if 'df_attn' in locals() and df_attn is not None and not df_attn.empty:
            # 统一/容错排序：默认优先按 score 降序，同分时按J值升序；没有 score 则按 rank 升序；否则保持原顺序
            def _auto_sort(df: pd.DataFrame) -> pd.DataFrame:
                if "score" in df.columns:
                    if "tiebreak_j" in df.columns:
                        return df.sort_values(["score", "tiebreak_j"], ascending=[False, True])
                    else:
                        return df.sort_values(["score"], ascending=[False])
                if "rank" in df.columns:
                    return df.sort_values(["rank"], ascending=[True])
                return df

            if sort_key == "score ↓" and "score" in df_attn.columns:
                if "tiebreak_j" in df_attn.columns:
                    df_attn = df_attn.sort_values(["score", "tiebreak_j"], ascending=[False, True])
                else:
                    df_attn = df_attn.sort_values(["score"], ascending=[False])
            elif sort_key == "rank ↑" and "rank" in df_attn.columns:
                df_attn = df_attn.sort_values(["rank"], ascending=[True])
            # "保持原文件顺序" 就不动

            # 预览 + 导出/复制，行为与"排名"页尽量一致
            st.divider()
            with st.container(border=True):
                rows_eff = int(topn_attn)
                st.markdown("**强度榜 Top-N 预览**")
                st.dataframe(df_attn.head(rows_eff), width='stretch', height=420)

                # TXT 复制（按你的导出偏好）
                if "ts_code" in df_attn.columns:
                    codes = df_attn["ts_code"].astype(str).head(rows_eff).tolist()
                    txt = _codes_to_txt(
                        codes,
                        st.session_state["export_pref"]["style"],
                        st.session_state["export_pref"]["with_suffix"]
                    )
                    copy_txt_button(txt, label="复制以上", key=f"copy_attn_{ref_attn}")

        # --- 轻量：前几日 Top-K 扫描（只看 Top，不算强度） ---
        with st.expander("前几日 Top-K 扫描（轻量）", expanded=True):
            # —— 参数区 ——
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                end_use = st.text_input("观察日（YYYYMMDD）", value=_get_latest_date_from_database() or "", key="lite_end")
            with c2:
                lookback_days = st.number_input("回看天数 D（不含今天）", min_value=1, max_value=60, value=3, key="lite_D")
            with c3:
                k_min = st.number_input("K 最小（含）", min_value=1, max_value=10000, value=1, key="lite_kmin")
            with c4:
                k_max = st.number_input("K 最大（含）", min_value=1, max_value=10000, value=cfg_int("SC_TOP_K", 50), key="lite_kmax")

            c5, c6, c7 = st.columns(3)
            with c5:
                hit_mode = st.selectbox(
                    "命中口径",
                    ["与今天Top交集", "累计上榜次数≥N", "连续上榜天数≥N"],
                    index=0, key="lite_mode"
                )
            with c6:
                n_th = st.number_input("N（阈值）", min_value=1, max_value=60, value=2, key="lite_N",
                                    disabled=(hit_mode == "与今天Top交集"))
            with c7:
                today_topk = st.number_input("今天对比 Top-K", min_value=1, max_value=5000,
                                            value=cfg_int("SC_TOP_K", 50), key="lite_todayK",
                                            disabled=(hit_mode != "与今天Top交集"))

            limit = st.number_input("输出条数上限", min_value=5, max_value=2000, value=200, key="lite_limit")

            go = st.button("计算（轻量 Top-K）", width='stretch', key="btn_lite_calc")

            if go:
                try:
                    days = _cached_trade_dates(DATA_ROOT, API_ADJ) or []
                    if not days:
                        st.warning("无法获取交易日历。"); st.stop()
                    # 观察日处理：若手填不在交易日里，取最近一个交易日
                    if not end_use or end_use not in days:
                        end_idx = len(days) - 1
                        end = days[end_idx]
                        if end_use and end_use not in days:
                            st.caption(f"观察日不在交易日历内，已改用最近交易日：{end}")
                    else:
                        end_idx = days.index(end_use)
                        end = end_use
                    if end_idx <= 0:
                        st.info("观察日前没有更早交易日可统计。"); st.stop()

                    # 回看窗口（不含今天 end）
                    D = int(lookback_days)
                    start_idx = max(0, end_idx - D)
                    win_days = days[start_idx:end_idx]  # t-D ~ t-1

                    # K 范围校正
                    k1, k2 = int(min(k_min, k_max)), int(max(k_min, k_max))

                    # —— 汇总前 D 日 Top-K（K∈[k1,k2]）——
                    occ = {}             # 累计命中次数
                    best_rank = {}       # 窗口内最好名次（越小越好）
                    last_seen = {}       # 最近出现日
                    day_index = {d:i for i,d in enumerate(win_days)}   # 便于算连续
                    appear_idx = {}      # ts_code -> 出现的日序号列表

                    for d in win_days:
                        p = _path_top(d)
                        if not p.exists() or p.stat().st_size == 0:
                            continue
                        df = _read_df(p, dtype={"ts_code": str})
                        if df is None or df.empty:
                            continue
                        if "rank" not in df.columns:
                            df = df.reset_index(drop=True)
                            df["rank"] = np.arange(1, len(df) + 1)
                        # 只取 K 范围
                        df = df[(df["rank"] >= k1) & (df["rank"] <= k2)]
                        for ts, rk in zip(df["ts_code"].astype(str), df["rank"].astype(int)):
                            occ[ts] = occ.get(ts, 0) + 1
                            best_rank[ts] = min(best_rank.get(ts, 10**9), rk)
                            last_seen[ts] = d if (ts not in last_seen or d > last_seen[ts]) else last_seen[ts]
                            appear_idx.setdefault(ts, []).append(day_index[d])

                    # 连续天数（窗口内的最大连续段）
                    max_streak = {}
                    for ts, idxs in appear_idx.items():
                        idxs = sorted(set(idxs))
                        if not idxs:
                            max_streak[ts] = 0
                            continue
                        best = cur = 1
                        for a, b in zip(idxs, idxs[1:]):
                            if b == a + 1:
                                cur += 1
                                best = max(best, cur)
                            else:
                                cur = 1
                        max_streak[ts] = best

                    # 汇总 DataFrame（只含在窗口内出现过的）
                    rows = []
                    for ts in sorted(occ.keys()):
                        rows.append({
                            "ts_code": ts,
                            "prev_hits": int(occ.get(ts, 0)),
                            "max_streak": int(max_streak.get(ts, 0)),
                            "best_rank_prev": int(best_rank.get(ts, 10**9)),
                            "last": last_seen.get(ts, None),
                        })
                    df_prev = pd.DataFrame(rows).sort_values(
                        ["prev_hits","best_rank_prev","ts_code"], ascending=[False, True, True]
                    ).reset_index(drop=True)

                    # —— 命中计算 —— 
                    if hit_mode == "与今天Top交集":
                        p_today = _path_top(end)
                        if not p_today.exists() or p_today.stat().st_size == 0:
                            st.info(f"{end} 的 Top 文件不存在或为空。"); st.stop()
                        df_today = _read_df(p_today, dtype={"ts_code": str})
                        if df_today is None or df_today.empty:
                            st.info(f"{end} 的 Top 文件读取为空。"); st.stop()
                        if "rank" not in df_today.columns:
                            df_today = df_today.reset_index(drop=True)
                            df_today["rank"] = np.arange(1, len(df_today) + 1)
                        df_today = df_today.sort_values("rank").head(int(today_topk))
                        today_set = set(df_today["ts_code"].astype(str))
                        hit = df_prev[df_prev["ts_code"].astype(str).isin(today_set)].copy()
                        hit = hit.merge(
                            df_today[["ts_code","rank"]].rename(columns={"rank":"today_rank"}),
                            on="ts_code", how="left"
                        ).sort_values(
                            ["prev_hits","today_rank","best_rank_prev","ts_code"],
                            ascending=[False, True, True, True]
                        ).head(int(limit))
                        st.markdown(
                            f"**窗口：{win_days[0] if win_days else '—'} ~ {win_days[-1] if win_days else '—'}（不含今天 {end}）｜K∈[{k1},{k2}]，今天对比 Top-K={int(today_topk)}**"
                        )
                        st.markdown("**命中：与今天 Top 交集（延续/再上榜）**")
                        if hit.empty:
                            st.caption("无命中。")
                        else:
                            st.dataframe(hit, width='stretch', height=360)
                            # 复制代码
                            codes = hit["ts_code"].astype(str).tolist()
                            txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"],
                                                st.session_state["export_pref"]["with_suffix"])
                            copy_txt_button(txt, label="📋 复制命中代码", key=f"copy_lite_inter_{end}")

                    elif hit_mode == "累计上榜次数≥N":
                        hit = df_prev[df_prev["prev_hits"] >= int(n_th)].copy()
                        hit = hit.sort_values(
                            ["prev_hits","best_rank_prev","ts_code"],
                            ascending=[False, True, True]
                        ).head(int(limit))
                        st.markdown(
                            f"**窗口：{win_days[0] if win_days else '—'} ~ {win_days[-1] if win_days else '—'}（不含今天 {end}）｜K∈[{k1},{k2}]**"
                        )
                        st.markdown(f"**命中：累计上榜次数 ≥ {int(n_th)}**")
                        if hit.empty:
                            st.caption("无命中。")
                        else:
                            st.dataframe(hit, width='stretch', height=360)
                            codes = hit["ts_code"].astype(str).tolist()
                            txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"],
                                                st.session_state["export_pref"]["with_suffix"])
                            copy_txt_button(txt, label="📋 复制命中代码", key=f"copy_lite_cnt_{end}")

                    else:  # 连续上榜天数≥N
                        hit = df_prev[df_prev["max_streak"] >= int(n_th)].copy()
                        hit = hit.sort_values(
                            ["max_streak","best_rank_prev","ts_code"],
                            ascending=[False, True, True]
                        ).head(int(limit))
                        st.markdown(
                            f"**窗口：{win_days[0] if win_days else '—'} ~ {win_days[-1] if win_days else '—'}（不含今天 {end}）｜K∈[{k1},{k2}]**"
                        )
                        st.markdown(f"**命中：连续上榜天数 ≥ {int(n_th)}**")
                        if hit.empty:
                            st.caption("无命中。")
                        else:
                            st.dataframe(hit, width='stretch', height=360)
                            codes = hit["ts_code"].astype(str).tolist()
                            txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"],
                                                st.session_state["export_pref"]["with_suffix"])
                            copy_txt_button(txt, label="📋 复制命中代码", key=f"copy_lite_streak_{end}")

                except Exception as e:
                    st.error(f"计算失败：{e}")

            # # CSV 下载（Top-N）
            # st.download_button(
            #     "⬇️ 导出 Top-N（CSV）",
            #     data=df_attn.head(rows_eff).to_csv(index=False).encode("utf-8-sig"),
            #     file_name=f"attention_top{rows_eff}_{ref_attn}.csv",
            #     width='stretch',
            #     key=f"dl_attn_{ref_attn}"
            # )

    # ================== 自选榜 ==================
    with tab_custom:
        st.subheader("自选榜（策略组合）")
        
        # 从 strategies_repo.py 加载策略组合
        def load_custom_combos():
            """从 strategies_repo.py 加载策略组合配置，转换为字典格式（name -> combo_data）"""
            try:
                from strategies_repo import CUSTOM_COMBOS
                if isinstance(CUSTOM_COMBOS, list):
                    # 列表格式转换为字典格式
                    return {combo.get('name', ''): combo for combo in CUSTOM_COMBOS if combo.get('name')}
                elif isinstance(CUSTOM_COMBOS, dict):
                    # 兼容旧格式（字典）
                    return dict(CUSTOM_COMBOS)
                return {}
            except Exception as e:
                logger.warning(f"加载策略组合失败: {e}")
                return {}
        
        def save_custom_combos(combos: dict):
            """保存策略组合配置到 strategies_repo.py"""
            try:
                import re
                from pathlib import Path
                
                repo_path = Path("strategies_repo.py")
                if not repo_path.exists():
                    st.error("找不到 strategies_repo.py 文件")
                    return False
                
                # 读取文件内容
                content = repo_path.read_text(encoding="utf-8-sig")
                
                # 构建新的 CUSTOM_COMBOS 内容（列表格式）
                new_combos_str = "# 自选榜策略组合配置\n"
                new_combos_str += "# 格式：列表，每个元素包含 name（组合名称）、rules（策略名列表）、agg_mode（聚合方法：OR/AND）、output_name（落盘名称）、explain（说明）等字段\n"
                new_combos_str += "CUSTOM_COMBOS = [\n"
                for combo_name, combo_data in combos.items():
                    rules = combo_data.get("rules", [])
                    agg_mode = combo_data.get("agg_mode", "OR")
                    output_name = combo_data.get("output_name", combo_name)
                    explain = combo_data.get("explain", "")
                    new_combos_str += "    {\n"
                    new_combos_str += f"        'name': '{combo_name}',\n"
                    new_combos_str += f"        'rules': {repr(rules)},\n"
                    new_combos_str += f"        'agg_mode': '{agg_mode}',\n"
                    new_combos_str += f"        'output_name': '{output_name}',\n"
                    if explain:
                        new_combos_str += f"        'explain': '{explain}',\n"
                    new_combos_str += "    },\n"
                new_combos_str += "]\n"
                
                # 使用正则表达式替换 CUSTOM_COMBOS 部分（包括注释）
                # 匹配从注释开始到 CUSTOM_COMBOS 列表/字典结束的整个块
                # 使用平衡括号匹配来正确匹配嵌套的结构
                pattern = r"# 自选榜策略组合配置.*?CUSTOM_COMBOS\s*=\s*[\[{]"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    # 找到列表/字典开始位置
                    start_pos = match.end() - 1  # 回退到 [ 或 { 的位置
                    # 从开始位置找到匹配的结束括号
                    brace_count = 1
                    end_pos = start_pos + 1
                    open_char = content[start_pos]
                    close_char = ']' if open_char == '[' else '}'
                    while brace_count > 0 and end_pos < len(content):
                        if content[end_pos] == open_char:
                            brace_count += 1
                        elif content[end_pos] == close_char:
                            brace_count -= 1
                        end_pos += 1
                    # 替换整个块（包括注释）
                    content = content[:match.start()] + new_combos_str.rstrip() + content[end_pos:]
                else:
                    # 如果不存在，在 POSITION_POLICIES 后添加
                    # 找到 POSITION_POLICIES 的结束位置（包括空行）
                    pos_pattern = r"(POSITION_POLICIES\s*=\s*\[[^\]]*\]\s*\n)"
                    if re.search(pos_pattern, content, re.DOTALL):
                        # 在 POSITION_POLICIES 后添加
                        content = re.sub(
                            pos_pattern,
                            r"\1\n" + new_combos_str,
                            content,
                            flags=re.DOTALL
                        )
                    else:
                        # 在文件末尾添加
                        content += "\n\n" + new_combos_str
                
                # 写回文件
                repo_path.write_text(content, encoding="utf-8-sig")
                
                # 重新加载模块
                import importlib
                import strategies_repo
                importlib.reload(strategies_repo)
                
                return True
            except Exception as e:
                logger.error(f"保存策略组合失败: {e}")
                st.error(f"保存失败: {e}")
                return False
        
        # 获取所有规则名
        rule_names = _get_rule_names()
        
        # 策略组合管理
        with st.expander("策略组合管理", expanded=False):
            combos = load_custom_combos()
            
            # 显示现有组合
            if combos:
                st.markdown("**现有策略组合：**")
                for combo_name, combo_data in combos.items():
                    with st.container(border=True):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{combo_name}**")
                            rules_list = combo_data.get("rules", [])
                            agg_mode = combo_data.get("agg_mode", "OR")
                            explain = combo_data.get("explain", "")
                            st.caption(f"策略组: {', '.join(rules_list[:3])}{'...' if len(rules_list) > 3 else ''} | 聚合方法: {agg_mode}")
                            if explain:
                                st.caption(f"说明: {explain}")
                        with col2:
                            if st.button("删除", key=f"del_{combo_name}"):
                                del combos[combo_name]
                                if save_custom_combos(combos):
                                    st.success(f"已删除策略组合: {combo_name}")
                                    st.rerun()
                        with col3:
                            if st.button("使用", key=f"use_{combo_name}"):
                                st.session_state["selected_combo"] = combo_name
                                st.session_state["selected_rules"] = rules_list
                                st.session_state["selected_agg_mode"] = agg_mode
                                st.rerun()
            
            st.divider()
            
            # 新建/编辑策略组合
            st.markdown("**新建/编辑策略组合：**")
            combo_name_input = st.text_input("组合名称（name）", value="", key="combo_name_input", placeholder="例如：突破组合")
            selected_rules_input = st.multiselect("策略组（rules）", options=rule_names, default=[], key="combo_rules_input")
            agg_mode_input = st.radio("聚合方法（agg_mode）", ["OR（任一命中）", "AND（全部命中）"], index=0, horizontal=True, key="combo_agg_mode")
            combo_output_name_input = st.text_input("落盘名称（output_name）", value="", key="combo_output_name_input", placeholder="例如：突破组合（用于生成文件名）")
            combo_explain_input = st.text_input("说明（explain）", value="", key="combo_explain_input", placeholder="例如：突破相关策略组合")
            
            col_save1, col_save2 = st.columns([1, 1])
            with col_save1:
                if st.button("保存策略组合", key="save_combo"):
                    if not combo_name_input.strip():
                        st.warning("请输入组合名称")
                    elif not selected_rules_input:
                        st.warning("请至少选择一个策略")
                    else:
                        # 落盘名称默认为组合名称
                        output_name = combo_output_name_input.strip() if combo_output_name_input.strip() else combo_name_input.strip()
                        combos[combo_name_input.strip()] = {
                            "rules": selected_rules_input,
                            "agg_mode": "OR" if agg_mode_input.startswith("OR") else "AND",
                            "output_name": output_name,
                            "explain": combo_explain_input.strip() if combo_explain_input.strip() else ""
                        }
                        if save_custom_combos(combos):
                            st.success(f"已保存策略组合: {combo_name_input.strip()}")
                            st.rerun()
            with col_save2:
                if st.button("清空输入", key="clear_combo"):
                    st.session_state["combo_name_input"] = ""
                    st.session_state["combo_rules_input"] = []
                    st.session_state["combo_output_name_input"] = ""
                    st.session_state["combo_explain_input"] = ""
                    st.rerun()
        
        st.divider()
        
        # 生成自选榜
        st.markdown("### 生成自选榜")
        
        # 加载repo中的策略组合
        combos_repo = load_custom_combos()
        
        # 选择策略组合或手动选择
        if combos_repo:
            # 有repo中的策略组合，提供选择
            combo_options = ["手动选择策略"] + list(combos_repo.keys())
            selected_combo_key = st.selectbox("选择策略组合", options=combo_options, index=0, key="select_combo_from_repo")
            
            if selected_combo_key == "手动选择策略":
                # 手动选择策略
                selected_rules = st.multiselect("选择策略（可多选）", options=rule_names, default=[], key="manual_rules")
                selected_agg_mode = st.radio("聚合模式", ["OR（任一命中）", "AND（全部命中）"], index=0, horizontal=True, key="manual_agg_mode")
                selected_agg_mode = "OR" if selected_agg_mode.startswith("OR") else "AND"
                selected_combo_name = None
                selected_output_name = None
                selected_combo_data = None
            else:
                # 使用repo中的策略组合
                selected_combo_data = combos_repo.get(selected_combo_key)
                if selected_combo_data:
                    selected_rules = selected_combo_data.get("rules", [])
                    selected_agg_mode = selected_combo_data.get("agg_mode", "OR")
                    selected_combo_name = selected_combo_key
                    selected_output_name = selected_combo_data.get("output_name", selected_combo_name)
                    st.info(f"**策略组合：{selected_combo_name}** | 策略组: {', '.join(selected_rules[:5])}{'...' if len(selected_rules) > 5 else ''} | 聚合方法: {selected_agg_mode}")
                    st.caption(f"落盘名称: {selected_output_name}")
                    explain = selected_combo_data.get("explain", "")
                    if explain:
                        st.caption(f"说明: {explain}")
                else:
                    selected_rules = []
                    selected_agg_mode = "OR"
                    selected_combo_name = None
                    selected_output_name = None
                    selected_combo_data = None
        else:
            # 没有repo中的策略组合，只能手动选择
            selected_rules = st.multiselect("选择策略（可多选）", options=rule_names, default=[], key="manual_rules")
            selected_agg_mode = st.radio("聚合模式", ["OR（任一命中）", "AND（全部命中）"], index=0, horizontal=True, key="manual_agg_mode")
            selected_agg_mode = "OR" if selected_agg_mode.startswith("OR") else "AND"
            selected_combo_name = None
            selected_output_name = None
            selected_combo_data = None
        
        # 参数设置
        with st.form("custom_rank_form"):
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                ref_date_custom = st.text_input("参考日（YYYYMMDD；留空=自动最新）", value="", key="custom_ref_date")
            with c2:
                universe_custom = st.selectbox("选股范围", ["全市场", "仅白名单", "仅黑名单", "仅强度榜"], index=0, key="custom_universe")
            with c3:
                tiebreak_custom = st.selectbox("同分排序", ["none", "kdj_j_asc"], index=1, key="custom_tiebreak")
            with c4:
                topN_custom = st.number_input("输出 Top-N（留空=不限制）", min_value=0, max_value=5000, value=200, step=10, key="custom_topN")
            
            # 榜单名称：如果使用了repo中的策略组合，使用其落盘名称；手动选择时才需要填写
            if selected_output_name:
                # 使用repo中的策略组合，显示落盘名称（只读）
                st.text_input("落盘名称（用于文件名）", value=selected_output_name, key="custom_combo_name", disabled=True)
                combo_name_output = selected_output_name
            else:
                # 手动选择策略，需要填写榜单名称
                combo_name_output = st.text_input("榜单名称（用于文件名）", value="custom", key="custom_combo_name", placeholder="例如：突破组合")
            
            gen_btn = st.form_submit_button("生成并落盘", width='stretch')
        
        if gen_btn:
            if not selected_rules:
                st.warning("请至少选择一个策略")
            elif not combo_name_output.strip():
                st.warning("请输入榜单名称")
            else:
                try:
                    # 自动启用数据库读取（和选股页面的按触发规则筛选一样）
                    if not is_details_db_reading_enabled():
                        st.session_state["details_db_reading_enabled"] = True
                    
                    ref_real = ref_date_custom.strip() or _get_latest_date_from_files() or ""
                    if not ref_real:
                        st.error("未能确定参考日")
                    else:
                        universe_map = {"全市场": "all", "仅白名单": "white", "仅黑名单": "black", "仅强度榜": "strength"}
                        universe = universe_map.get(universe_custom, "all")
                        
                        # 调用生成函数
                        from scoring_core import build_custom_rank
                        result = build_custom_rank(
                            combo_name=combo_name_output.strip(),
                            rule_names=selected_rules,
                            agg_mode=selected_agg_mode,
                            ref_date=ref_real,
                            universe=universe,
                            tiebreak=tiebreak_custom,
                            topN=topN_custom if topN_custom > 0 else None,
                            write=True
                        )
                        
                        if result:
                            st.success(f"自选榜已生成并落盘: {result}")
                            # 读取并显示结果
                            try:
                                df_result = pd.read_csv(result)
                                st.dataframe(df_result, width='stretch', height=480)
                                
                                # 导出 TXT
                                if "ts_code" in df_result.columns:
                                    codes = df_result["ts_code"].astype(str).tolist()
                                    txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"], st.session_state["export_pref"]["with_suffix"])
                                    copy_txt_button(txt, label="📋 复制代码", key=f"copy_custom_{ref_real}")
                            except Exception as e:
                                st.warning(f"读取结果失败: {e}")
                        else:
                            st.info("未筛到命中标的")
                except Exception as e:
                    st.error(f"生成失败: {e}")
        
        # 查看历史榜单
        st.divider()
        st.markdown("### 查看历史榜单")
        custom_dir = SC_OUTPUT_DIR / "custom"
        if custom_dir.exists():
            csv_files = sorted(custom_dir.glob("custom_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
            if csv_files:
                selected_file = st.selectbox("选择文件", options=[f.name for f in csv_files[:20]], key="custom_file_select")
                if selected_file:
                    try:
                        df_view = pd.read_csv(custom_dir / selected_file)
                        st.dataframe(df_view, width='stretch', height=420)
                        if "ts_code" in df_view.columns:
                            codes = df_view["ts_code"].astype(str).tolist()
                            txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"], st.session_state["export_pref"]["with_suffix"])
                            copy_txt_button(txt, label="📋 复制代码", key=f"copy_custom_file_{selected_file}")
                    except Exception as e:
                        st.error(f"读取文件失败: {e}")
            else:
                st.info("暂无历史榜单文件")
        else:
            st.info("自定义榜单目录不存在")

    # ================== 选股 ==================
    with tab_screen:
        st.subheader("选股")

        # === 统一参考日 & 范围 ===
        c_top1, c_top2 = st.columns([1,1])
        with c_top1:
            refD_unified = st.text_input("参考日（YYYYMMDD，留空=自动最新）", value=st.session_state.get("screen_refD",""), key="screen_refD")
        with c_top2:
            uni_choice = st.selectbox("选股范围", ["全市场","仅白名单","仅黑名单","仅特别关注榜"], index=0, key="screen_uni_choice")
        _uni_map = {"全市场":"all", "仅白名单":"white", "仅黑名单":"black", "仅特别关注榜":"attention"}

        # ========== 1) 表达式筛选 ==========
        with st.form("expression_screening_form"):
            st.markdown("### 表达式筛选")
            exp = st.text_input("表达式（示例：CLOSE>MA(CLOSE,20) AND VOL>MA(VOL,5)）", value=st.session_state.get("screen_expr",""), key="screen_expr")
            c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])
            with c1:
                level = st.selectbox("时间级别", ["D","W","M"], index=0, key="screen_level")
            with c2:
                window = st.number_input("窗口长度", min_value=1, max_value=500, value=30, key="screen_window")
            with c3:
                scope_logic = st.selectbox("命中范围(scope)", ["LAST","ANY","ALL","COUNT>=k","CONSEC>=m","ANY_n","ALL_n"], index=0, key="screen_scope_logic")
            with c4:
                n_k_m = st.number_input("k/m/n(特定选择才生效)", min_value=1, max_value=500, value=3, key="screen_nkm")
            with c5:
                tiebreak_expr = st.selectbox("同分排序", ["none", "kdj_j_asc"], index=1, key="screen_tiebreak_expr")
            with c6:
                run_btn = st.form_submit_button("运行筛选", width='stretch')

        if run_btn:
            logger.info(f"用户点击运行筛选: 表达式={exp[:50]}..., 级别={level}, 窗口={window}, 范围={scope_logic}")
            try:
                if not exp.strip():
                    st.warning("请先输入表达式。")
                else:
                    # 组装 scope
                    scope = scope_logic
                    if scope_logic.startswith("COUNT"):
                        scope = f"COUNT>={int(n_k_m)}"
                    elif scope_logic.startswith("CONSEC"):
                        scope = f"CONSEC>={int(n_k_m)}"
                    elif scope_logic == "ANY_n":
                        scope = f"ANY_{int(n_k_m)}"
                    elif scope_logic == "ALL_n":
                        scope = f"ALL_{int(n_k_m)}"

                    df_sel = run_se_screen_in_bg(
                        when_expr=exp.strip(),
                        ref_date=(refD_unified.strip() or None),
                        timeframe=level,
                        window=_safe_int(window, 30),
                        scope=scope,
                        universe=_uni_map.get(uni_choice,"all"),
                        write_white=False,
                        write_black_rest=False,
                        return_df=True,
                    )
                    if df_sel is None or df_sel.empty:
                        st.info("无命中。")
                    else:
                        # 结果已经按得分排序，直接显示
                        st.caption(f"命中 {len(df_sel)} 只；参考日：{(df_sel['ref_date'].iloc[0] if 'ref_date' in df_sel.columns and len(df_sel)>0 else (refD_unified or '自动'))}")
                        if 'score' in df_sel.columns:
                            st.caption("已按得分排序（降序），同分时按J值升序")
                        st.dataframe(df_sel, width='stretch', height=480)
                        # 导出 TXT（代码）
                        if "ts_code" in df_sel.columns:
                            txt = _codes_to_txt(df_sel["ts_code"].astype(str).tolist(),
                                                st.session_state["export_pref"]["style"],
                                                st.session_state["export_pref"]["with_suffix"])
                            copy_txt_button(txt, label="📋 复制以上（按当前预览）", key=f"copy_screen_expr_{refD_unified or 'auto'}")
            except Exception as e:
                st.error(f"筛选失败：{e}")

        st.divider()

        # ========== 2) 按触发规则筛选（当日全市场，多选） ==========
        with st.form("rule_screening_form"):
            st.markdown("### 按触发规则筛选（当日全市场，多选）")
            st.caption("说明：读取当日 details 数据；按所选规则名判断：只要策略触发（ok=True）就视为命中；支持\"任一/全部\"聚合。")
            # 规则名来自 se.SC_RULES（使用缓存）
            rule_names = _get_rule_names()
            picked = st.multiselect("规则名（可多选）", options=rule_names, default=[], key="detail_multi_rules")
            agg_mode = st.radio("命中逻辑", ["任一命中（OR）","全部命中（AND）"], index=0, horizontal=True, key="detail_hit_mode")
            cA, cB, cC = st.columns([1,1,1])
            with cA:
                limit_n = st.number_input("最多显示/导出 N 条", min_value=10, max_value=5000, value=200, step=10, key="detail_limit_n")
            with cB:
                tiebreak_rule = st.selectbox("同分排序", ["none", "kdj_j_asc"], index=1, key="screen_tiebreak_rule")
            with cC:
                run_detail = st.form_submit_button("筛选当日命中标的", width='stretch')

        if run_detail:
            # 自动启用数据库读取（和个股详情里的解锁逻辑一样）
            if not is_details_db_reading_enabled():
                st.session_state["details_db_reading_enabled"] = True
            
            ref_real = refD_unified.strip() or _get_latest_date_from_files() or ""
            if not ref_real:
                st.error("未能确定参考日。")
            elif not picked:
                st.warning("请先选择至少一个规则名。")
            else:
                rows = []
                try:
                    # 检查是否允许读取数据库（使用统一的函数）
                    db_reading_enabled = is_details_db_reading_enabled()
                    
                    # 优先使用数据库查询（只有当db_reading_enabled为True且数据库可用时才读取数据库）
                    if db_reading_enabled and is_details_db_available():
                        # 使用 database_manager 查询详情
                        logger.info("[数据库连接] 开始获取数据库管理器实例 (查询股票详情用于UI显示)")
                        manager = get_database_manager()
                        if manager:
                            # 使用统一的函数获取details数据库路径（包含回退逻辑）
                            details_db_path = get_details_db_path_with_fallback()
                            if details_db_path:
                                sql = "SELECT * FROM stock_details WHERE ref_date = ?"
                                df_all = manager.execute_sync_query(details_db_path, sql, [ref_real], timeout=30.0)
                            else:
                                df_all = pd.DataFrame()
                        else:
                            df_all = pd.DataFrame()
                        
                        if not df_all.empty:
                            for _, row in df_all.iterrows():
                                ts2 = str(row.get("ts_code", "")).strip()
                                if not ts2:
                                    continue
                                
                                # 使用统一的 _load_detail_json 函数获取数据
                                data = _load_detail_json(str(ref_real), ts2)
                                if not data:
                                    continue
                                
                                # 从统一格式中提取数据
                                summary = data.get("summary", {})
                                sc = float(summary.get("score", 0.0))
                                rules = data.get("rules", [])
                                
                                names_today = set()
                                hit_dates_map = {}  # 策略名 -> 触发日期列表
                                for rr in rules:
                                    # 只要策略触发（ok=True），就视为命中，无需检查add字段
                                    # 或者add>0也视为命中（兼容原有逻辑）
                                    ok_val = rr.get("ok")
                                    add_val = rr.get("add")
                                    if bool(ok_val) or (add_val is not None and float(add_val) > 0.0):
                                        n = rr.get("name")
                                        if n: 
                                            names_today.add(str(n))
                                            # 收集触发日期列表
                                            hit_date = rr.get("hit_date")
                                            hit_dates = rr.get("hit_dates", [])
                                            # 合并hit_date和hit_dates
                                            all_dates = []
                                            if hit_date:
                                                all_dates.append(str(hit_date))
                                            if hit_dates:
                                                all_dates.extend([str(d) for d in hit_dates if d])
                                            # 去重并排序
                                            all_dates = sorted(set(all_dates))
                                            if all_dates:
                                                hit_dates_map[str(n)] = all_dates
                                
                                if names_today:
                                    if agg_mode.startswith("任一"):
                                        hit = any((n in names_today) for n in picked)
                                    else:
                                        hit = all((n in names_today) for n in picked)
                                    if hit:
                                        # 收集所有选中策略的触发日期列表
                                        trigger_dates_list = []
                                        for rule_name in picked:
                                            if rule_name in hit_dates_map:
                                                trigger_dates_list.extend(hit_dates_map[rule_name])
                                        # 去重并排序
                                        trigger_dates_list = sorted(set(trigger_dates_list))
                                        rows.append({
                                            "ts_code": ts2, 
                                            "score": sc,
                                            "trigger_dates": trigger_dates_list if trigger_dates_list else []
                                        })
                    
                    # 回退到JSON文件查询
                    else:
                        ddir = DET_DIR / str(ref_real)
                        allow_set = None
                        if ddir.exists():
                            for p in ddir.glob("*.json"):
                                try:
                                    j = json.loads(p.read_text(encoding="utf-8-sig"))
                                except Exception:
                                    continue
                                ts2 = str(j.get("ts_code","")).strip()
                                if not ts2:
                                    continue
                                if (allow_set is not None) and (ts2 not in allow_set):
                                    continue
                                sm = j.get("summary") or {}
                                sc = float(sm.get("score", 0.0))
                                names_today = set()
                                hit_dates_map = {}  # 策略名 -> 触发日期列表
                                for rr in (j.get("rules") or []):
                                    # 只要策略触发（ok=True），就视为命中，无需检查add字段
                                    # 或者add>0也视为命中（兼容原有逻辑）
                                    ok_val = rr.get("ok")
                                    add_val = rr.get("add")
                                    if bool(ok_val) or (add_val is not None and float(add_val) > 0.0):
                                        n = rr.get("name")
                                        if n: 
                                            names_today.add(str(n))
                                            # 收集触发日期列表
                                            hit_date = rr.get("hit_date")
                                            hit_dates = rr.get("hit_dates", [])
                                            # 合并hit_date和hit_dates
                                            all_dates = []
                                            if hit_date:
                                                all_dates.append(str(hit_date))
                                            if hit_dates:
                                                all_dates.extend([str(d) for d in hit_dates if d])
                                            # 去重并排序
                                            all_dates = sorted(set(all_dates))
                                            if all_dates:
                                                hit_dates_map[str(n)] = all_dates
                                if names_today:
                                    if agg_mode.startswith("任一"):
                                        hit = any((n in names_today) for n in picked)
                                    else:
                                        hit = all((n in names_today) for n in picked)
                                    if hit:
                                        # 收集所有选中策略的触发日期列表
                                        trigger_dates_list = []
                                        for rule_name in picked:
                                            if rule_name in hit_dates_map:
                                                trigger_dates_list.extend(hit_dates_map[rule_name])
                                        # 去重并排序
                                        trigger_dates_list = sorted(set(trigger_dates_list))
                                        rows.append({
                                            "ts_code": ts2, 
                                            "score": sc,
                                            "trigger_dates": trigger_dates_list if trigger_dates_list else []
                                        })
                    
                    df_hit = pd.DataFrame(rows)
                    if df_hit.empty:
                        st.info("未筛到命中标的。")
                    else:
                        # 应用Tie-break排序
                        df_hit_sorted = _apply_tiebreak_sorting(df_hit, tiebreak_rule)
                        n = int(limit_n)
                        df_show = df_hit_sorted.head(n)
                        # 调整列顺序：ts_code, score, trigger_dates
                        if "trigger_dates" in df_show.columns:
                            col_order = ["ts_code", "score", "trigger_dates"]
                            # 添加其他列（如果有）
                            for col in df_show.columns:
                                if col not in col_order:
                                    col_order.append(col)
                            df_show = df_show[[c for c in col_order if c in df_show.columns]]
                        st.caption(f"命中 {len(df_hit_sorted)} 只；显示前 {len(df_show)} 只；参考日：{ref_real}")
                        st.dataframe(df_show, width='stretch', height=420)
                        # 导出 TXT
                        if "ts_code" in df_show.columns:
                            txt = _codes_to_txt(df_show["ts_code"].astype(str).tolist(),
                                                st.session_state["export_pref"]["style"],
                                                st.session_state["export_pref"]["with_suffix"])
                            copy_txt_button(txt, label="📋 复制以上（按当前预览）", key=f"copy_screen_rule_{ref_real}")
                except Exception as e:
                    st.error(f"读取明细失败：{e}")

    # ================== 工具箱 ==================
    with tab_tools:
        st.subheader("工具箱")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**自动补算最近 N 个交易日**")
            n_back = st.number_input("天数 N", min_value=1, max_value=100, value=20)
            inc_today = st.checkbox("包含参考日当天", value=True,
                                    help="勾选后窗口包含参考日（例如 N=5 → [ref-(N-1), ref]；未勾选则 [ref-N, ref-1]）")
            do_force = st.checkbox("强制重建（覆盖已有）", value=False,
                                help="若之前失败留下了 0 字节文件或想重算，勾选此项。")

            go_fill = st.button("执行自动补算", width='stretch')
            if go_fill:
                try:
                    if hasattr(se, "backfill_prev_n_days"):
                        out = se.backfill_prev_n_days(n=int(n_back), include_today=bool(inc_today), force=bool(do_force))
                        st.success(f"已处理：{out}")
                    else:
                        st.warning("未检测到 backfill_prev_n_days。")
                except Exception as e:
                    st.error(f"补算失败：{e}")

        with colB:
            st.markdown("**补齐缺失的 All 排名文件**")
            # start = st.text_input("起始日 YYYYMMDD", value="")
            start = st.text_input("起始日 YYYYMMDD", value="", key="tools_fix_start")
            end = st.text_input("结束日 YYYYMMDD", value="", key="tools_fix_end")
            do_force_fix = st.checkbox("强制重建（覆盖已有）", value=False)
            go_fix = st.button("补齐缺失", width='stretch')
            if go_fix and start and end:
                try:
                    if hasattr(se, "backfill_missing_ranks"):                   
                        out = se.backfill_missing_ranks(start, end, force=bool(do_force_fix))
                        st.success(f"已补齐：{out}")
                    else:
                        st.warning("未检测到 backfill_missing_ranks。")
                except Exception as e:
                    st.error(f"处理失败：{e}")
        st.markdown("---")
        with st.expander("查看已有数据（Top / All / Details / 日历）", expanded=True):
            if "scan_inventory_loaded" not in st.session_state:
                st.session_state["scan_inventory_loaded"] = False
            col0, col1 = st.columns([1,3])
            with col0:
                do_scan = st.button("加载/刷新列表", key="btn_scan_inventory", width='stretch')
                if do_scan:
                    st.session_state["scan_inventory_loaded"] = True
            if not st.session_state["scan_inventory_loaded"]:
                st.info("（首次进入不扫描磁盘，点击上方 **加载/刷新列表** 才读取文件清单。）")
            if st.session_state["scan_inventory_loaded"]:
                try:
                    all_files = sorted(ALL_DIR.glob("score_all_*.csv"))
                    top_files = sorted(TOP_DIR.glob("score_top_*.csv"))
                    det_dirs  = sorted([p for p in DET_DIR.glob("*") if p.is_dir()])

                    all_dates = [p.stem.replace("score_all_", "") for p in all_files]
                    top_dates = [p.stem.replace("score_top_", "") for p in top_files]
                    det_dates = [p.name for p in det_dirs]

                    zero_all = [p.name for p in all_files if p.stat().st_size == 0]
                    zero_top = [p.name for p in top_files if p.stat().st_size == 0]

                    cov_min = min(all_dates) if all_dates else ""
                    cov_max = max(all_dates) if all_dates else ""

                    # 交易日日历（若存在则用于对比缺失）
                    missing: list[str] = []
                    try:
                        trade_dates = get_trade_dates() or []
                        if trade_dates and cov_min and cov_max:
                            rng = [d for d in trade_dates if cov_min <= d <= cov_max]
                            aset = set(all_dates)
                            missing = [d for d in rng if d not in aset]
                    except Exception:
                        trade_dates = []

                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("All 文件数", len(all_files))
                    with col2: st.metric("Top 文件数", len(top_files))
                    with col3: st.metric("Details 日期目录", len(det_dirs))
                    with col4: st.metric("0 字节文件", len(zero_all) + len(zero_top))

                    if cov_min:
                        st.caption(f"All 覆盖区间：{cov_min} ~ {cov_max}（缺失 {len(missing)} 天）")
                    else:
                        st.caption("All 目录为空。")

                    if zero_all or zero_top:
                        names = zero_all[:8] + zero_top[:8]
                        st.warning("检测到 0 字节文件（可用“强制重建”覆盖）：\n" + "，".join(names) + (" ……" if len(zero_all)+len(zero_top) > len(names) else ""))
                    colL, colR = st.columns([1, 2])
                    with colL:
                        kind = st.radio("数据类型", ["All 排名", "Top 排名", "Details"], horizontal=True, key="view_kind")
                        if kind == "All 排名":
                            cand = all_dates
                        elif kind == "Top 排名":
                            cand = top_dates
                        else:
                            cand = det_dates
                        sel_date = st.selectbox("选择日期（倒序）", cand[::-1] if cand else [], key="view_date") if cand else None
                        show_missing = st.checkbox("显示缺失日期（基于交易日历）", value=False, disabled=not missing)
                    with colR:
                        if sel_date:
                            if kind == "All 排名":
                                p = _path_all(sel_date)
                                if p.exists() and p.stat().st_size > 0:
                                    st.dataframe(_read_df(p).head(200), width='stretch', height=360)
                                else:
                                    st.info("该日 All 文件不存在或为空。")
                            elif kind == "Top 排名":
                                p = _path_top(sel_date)
                                if p.exists() and p.stat().st_size > 0:
                                    st.dataframe(_read_df(p).head(200), width='stretch', height=360)
                                else:
                                    st.info("该日 Top 文件不存在或为空。")
                            else:
                                pdir = DET_DIR / sel_date
                                if pdir.exists():
                                    st.info(f"{sel_date} 共有 {len(list(pdir.glob('*.json')))} 个详情文件。")
                                else:
                                    st.info("该日没有 Details 目录。")

                    if show_missing and missing:
                        st.markdown("**缺失日期（相对 All 覆盖区间）**")
                        txt = " ".join(missing[:200]) + (" ..." if len(missing) > 200 else "")
                        st.code(txt)
                except Exception as e:
                    st.error(f"扫描失败：{e}")

    # ================== 组合模拟 / 持仓 ==================
    with tab_port:
        st.subheader("组合模拟 / 持仓")
        from stats_core import PortfolioManager
        pm = PortfolioManager()

        # —— 全局配置（用于新建组合的默认值） ——
        with st.expander("全局配置（默认用于新建组合；来自 config.PF_*）", expanded=True):
            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.text_input("账本名称", value=cfg_str("PF_LEDGER_NAME", "账本1"), key="pf_ledger")
                st.number_input("初始资金（总额）", min_value=0.0, value=float(getattr(cfg, "PF_INIT_CASH", 1_000_000.0)), key="pf_init_cash")
            with colB:
                st.number_input("初始可用资金", min_value=0.0, value=float(getattr(cfg, "PF_INIT_AVAILABLE", getattr(cfg, "PF_INIT_CASH", 1_000_000.0))), key="pf_init_avail")
                st.selectbox("成交价口径", ["next_open","close"], index=(0 if cfg_str("PF_TRADE_PRICE_MODE","next_open")=="next_open" else 1), key="pf_pxmode")
            with colC:
                st.number_input("买入费率（bp）", min_value=0.0, value=float(getattr(cfg, "PF_FEE_BPS_BUY", 15.0)), key="pf_fee_buy")
                st.number_input("卖出费率（bp）", min_value=0.0, value=float(getattr(cfg, "PF_FEE_BPS_SELL", 15.0)), key="pf_fee_sell")
            with colD:
                st.number_input("最低费用（元）", min_value=0.0, value=float(getattr(cfg, "PF_MIN_FEE", 0.0)), key="pf_min_fee")
            st.caption("以上为默认值；新建组合时会带入（每个组合可覆盖）。")

        # —— 新建/选择组合 ——
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown("**新建组合**")
            new_name = st.text_input("名称", value=st.session_state.get("pf_ledger","default"))
            if st.button("创建组合", width='stretch'):
                pid = pm.create_portfolio(
                    name=new_name,
                    init_cash=float(st.session_state["pf_init_cash"]),
                    init_available=float(st.session_state["pf_init_avail"]),
                    trade_price_mode=str(st.session_state["pf_pxmode"]),
                    fee_bps_buy=float(st.session_state["pf_fee_buy"]),
                    fee_bps_sell=float(st.session_state["pf_fee_sell"]),
                    min_fee=float(st.session_state["pf_min_fee"]),
                )
                st.success(f"已创建：{new_name}（id={pid}）")

        with col2:
            st.markdown("**当前组合**")
            ports = pm.list_portfolios()
                # st.stop()
            # 以 name 排序
            ports_items = sorted(list(ports.items()), key=lambda kv: kv[1].name) if ports else []
            if not ports_items:
                st.info("暂无组合，请先创建。")
                cur_pid, cur_pf = None, None
                st.session_state['cur_pid'] = None
                st.session_state['cur_pf'] = None
                st.session_state['cur_pid'] = cur_pid
                st.session_state['cur_pf'] = cur_pf
            else:
                names = [f"{p.name} ({pid[:6]})" for pid, p in ports_items]
                sel = st.selectbox("选择组合", options=list(range(len(ports_items))), format_func=lambda i: names[i], index=0)
                cur_pid, cur_pf = ports_items[sel]
                st.session_state['cur_pid'] = cur_pid
                st.session_state['cur_pf'] = cur_pf

        st.divider()

        # —— 录入成交（价格参考区间） ——
        st.markdown("**录入成交（带参考价区间）**")
        colx, coly, colz, colw = st.columns([1.2, 1.2, 1.2, 2])
        with colx:
            side = st.selectbox("方向", ["BUY","SELL"], index=0)
        with coly:
            d_exec = st.text_input("成交日（YYYYMMDD）", value=_get_latest_date_from_database() or "")
        with colz:
            ts = st.text_input("代码", value="")
        # 读取当日 O/H/L/C 作为参考
        ref_low = ref_high = px_open = px_close = None
        try:
            ts_norm = normalize_ts(ts) if ts else ""
            if ts_norm and d_exec:
                try:
                    from config import DATA_ROOT, UNIFIED_DB_PATH
                    db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                    df_one = query_stock_data(
                        db_path=db_path,
                        ts_code=ts_norm,
                        start_date=d_exec,
                        end_date=d_exec,
                        adj_type="qfq"
                    )
                except:
                    # 回退到直接查询
                    from config import DATA_ROOT, UNIFIED_DB_PATH
                    db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
                    logger.info(f"[数据库连接] 开始获取数据库管理器实例 (回退查询K线数据: {ts_norm}, {d_exec})")
                    manager = get_database_manager()
                    sql = "SELECT open,high,low,close FROM stock_data WHERE ts_code = ? AND trade_date = ?"
                    df_one = manager.execute_sync_query(db_path, sql, [ts_norm, d_exec], timeout=30.0)
                if df_one is not None and not df_one.empty:
                    row = df_one.iloc[-1]
                    px_open = float(row.get("open", float("nan")))
                    px_close = float(row.get("close", float("nan")))
                    ref_low  = float(row.get("low", float("nan")))
                    ref_high = float(row.get("high", float("nan")))
        except Exception:
            pass
        with colw:
            st.write({"open": px_open, "close": px_close, "low": ref_low, "high": ref_high})

        colq, colp = st.columns([1.2, 1.8])
        with colq:
            qty = st.number_input("数量（股）", min_value=0, value=0, step=100)
        with colp:
            price_mode = st.radio("成交价来源", ["按口径自动","自定义价格"], index=0, horizontal=True)
            if price_mode == "自定义价格":
                price = st.number_input("成交价（留空则用口径价）", min_value=0.0, value=float(px_close or px_open or 0.0), step=0.01)
            else:
                price = None

        if cur_pf and st.button("记录成交", width='stretch', key="btn_rec_trade"):
            try:
                pm.record_trade(pid=cur_pid, date=str(d_exec), ts_code=str(ts_norm), side=str(side), qty=int(qty),
                                price_mode=(None if price is not None else cur_pf.trade_price_mode),
                                price=(None if price is None else float(price)), note="manual")
                st.success("已记录")
            except Exception as e:
                st.error(f"记录失败：{e}")

        st.divider()

        # —— 观察日估值 / 净值 ——
        st.markdown("**观察日收益与持仓估值**")
        obs = st.text_input("观察日（YYYYMMDD；默认=最新交易日）", value=_get_latest_date_from_database() or "")
        if obs and cur_pf:
            # 回放估值（从组合创建日至观察日）
            # 我们用 read_nav() 读取结果
            try:
                # 执行估值
                # pm.reprice_and_nav(cur_pid, date_start="19000101", date_end=str(obs), benchmarks=())
                tr = pm.read_trades(cur_pid)
                if tr is not None and not tr.empty:
                    # 组合首笔成交日
                    first_trade = str(pd.to_datetime(tr["date"].astype(str), errors="coerce").dt.strftime("%Y%m%d").min())
                    # 起点 = 首笔成交日前一个“交易日”
                    date_start_use = _prev_trade_date(first_trade, 1)
                else:
                    # 没有成交记录就从观察日开始（避免从远古起算）
                    date_start_use = str(obs)

                pm.reprice_and_nav(
                    cur_pid,
                    date_start=str(date_start_use),
                    date_end=str(obs),
                    benchmarks=(),
                )
                nav_df = pm.read_nav(cur_pid)
                pos_df = pm.read_positions(cur_pid)
            except Exception as e:
                st.error(f"估值失败：{e}")
                nav_df, pos_df = pd.DataFrame(), pd.DataFrame()
            if not nav_df.empty:
                row = nav_df.iloc[-1]
                if not pos_df.empty and "date" in pos_df.columns:
                    cur_pos = pos_df[pos_df["date"] == str(obs)].copy()
                    if not cur_pos.empty:
                        st.markdown("**当前持仓**")
                        show_cols = [c for c in ["ts_code","qty","cost","mkt_price","mkt_value","unreal_pnl"] if c in cur_pos.columns]
                        cur_pos = cur_pos[show_cols].sort_values("mkt_value", ascending=False)
                        st.dataframe(cur_pos, width='stretch', height=300)
                    else:
                        st.caption("观察日无持仓记录。")
                st.metric("组合市值", f"{(row.get('nav',1.0) * float(cur_pf.init_cash)):.0f}")
                st.metric("区间收益率", f"{(row.get('nav',1.0) - 1.0):.2%}")
                cols = [c for c in ["date","cash","position_mv","nav","ret_d","max_dd"] if c in nav_df.columns]
                st.dataframe(nav_df[cols].tail(5), width='stretch')
                st.markdown("**净值曲线（NAV）**")
                try:
                    st.line_chart(nav_df.set_index("date")["nav"])
                except Exception:
                    pass
            else:
                st.caption("暂无净值数据（可能还未有成交或行情数据缺失）")

    # ================== 统计（普通页签） ==================
    with tab_stats:
        st.subheader("统计")
        sub_tabs = st.tabs(["跟踪（Tracking）", "异动（Surge）", "共性（Commonality）"])

        # --- Tracking ---
        with sub_tabs[0]:
            refT = st.text_input("参考日", value="", key="ref_1")
            # 参考日/回看窗口的提示：告诉用户 t-n 是哪天
            _back_choices = [1, 3, 5, 10, 20]
            _hint_text, _n2d = _from_last_hints(_back_choices)
            if _hint_text:
                st.caption("按最新交易日回推： " + _hint_text)

            wins = st.text_input("未来收益窗口N（天，逗号分隔）", value="1,2,3,5,10,20")
            bench = st.text_input("对比指数基准代码（逗号，可留空）", value="")
            retrosT = st.text_input("附加回看天数", value="1,3,5")
            only_detail = st.checkbox("仅导出明细（不显示均值/标准差/胜率/分位数汇总）", value=True)
            gb_board = st.checkbox("分板块汇总", value=True)

            # === 跟踪增强：前日排行 / 名单 / 指标是否触发 / 后续涨幅 ===
            with st.expander("可选：选择要打勾的指标（来自打分规则；仅用于打勾，不影响样本）", expanded=True):
                import scoring_core as se
                # 规则名列表（去重）
                try:
                    rule_names = [str(r.get("name") or f"RULE_{i}") for i, r in enumerate(getattr(se, "SC_RULES", []) or [])]
                    rule_names = sorted(list(dict.fromkeys(rule_names)))
                except Exception:
                    rule_names = []
                track_rule_names = st.multiselect("指标（可多选）", options=rule_names, default=[])
                track_max_json = st.number_input("最多读取明细JSON（按当日排名排序）", min_value=50, max_value=5000, value=300, step=50, key="track_max_json")


            if st.button("生成跟踪表（含前日排行/名单/指标勾选/后续涨幅）", key="btn_run_tracking", width='stretch'):
                try:
                    from stats_core import run_tracking
                    import scoring_core as se
                    # 1) 基础 tracking 明细
                    wlist = [int(x) for x in wins.split(",") if x.strip().isdigit()]
                    blist = [s.strip() for s in bench.split(",") if s.strip()] or None
                    rlist = [int(x) for x in retrosT.split(",") if x.strip().isdigit()]
                    tr2 = run_tracking(
                        refT, wlist, benchmarks=blist, score_df=None,
                        group_by_board=gb_board, save=True,
                        retro_days=rlist, do_summary=(not only_detail)
                    )
                    detail = tr2.detail.copy()

                    # 2) 合并前日 rank
                    prev = _prev_ref_date(refT)
                    if prev:
                        df_prev = _read_df(_path_all(prev), usecols=["ts_code","rank"])
                        if df_prev is not None and len(df_prev) > 0:
                            df_prev = df_prev.rename(columns={"rank":"rank_tminus_1"})
                            detail = detail.merge(df_prev, on="ts_code", how="left")

                    # 3) 合并名单（白/黑/特别关注）
                    try:
                        whites = set(se._read_cache_list_codes(refT, "whitelist")) if hasattr(se, "_read_cache_list_codes") else set()
                        blacks = set(se._read_cache_list_codes(refT, "blacklist")) if hasattr(se, "_read_cache_list_codes") else set()
                        attns  = set(se._load_attention_codes(refT)) if hasattr(se, "_load_attention_codes") else set()
                        detail["in_whitelist"] = detail["ts_code"].astype(str).map(lambda c: c in whites)
                        detail["in_blacklist"] = detail["ts_code"].astype(str).map(lambda c: c in blacks)
                        detail["in_attention"] = detail["ts_code"].astype(str).map(lambda c: c in attns)
                    except Exception:
                        for c in ("in_whitelist","in_blacklist","in_attention"):
                            detail[c] = False

                    # 4) 指标是否触发（来自排名规则）
                    try:
                        import scoring_core as se
                        sel_rules = track_rule_names if isinstance(track_rule_names, list) else []
                        hit_cols = [f"hit:{name}" for name in sel_rules]
                        for col in hit_cols:
                            detail[col] = False
                        if sel_rules:
                            pick_n = int(track_max_json) if "track_max_json" in locals() else 300
                            head_codes = detail.sort_values(["rank"]).head(pick_n)["ts_code"].astype(str).tolist()
                            def _read_hits_one(ts):
                                obj = _load_detail_json(str(refT), str(ts))
                                res = {}
                                if not obj:
                                    return res
                                rules = obj.get("rules") or []
                                for rr in rules:
                                    nm = str(rr.get("name") or "")
                                    if nm in sel_rules:
                                        res[nm] = bool(rr.get("ok", False))
                                return res
                            for ts in head_codes:
                                hits_map = _read_hits_one(ts)
                                for nm in sel_rules:
                                    col = f"hit:{nm}"
                                    if nm in hits_map:
                                        detail.loc[detail["ts_code"].astype(str) == ts, col] = bool(hits_map[nm])
                    except Exception:
                        pass

                    # 5) 展示所需列
                    show_cols = [c for c in [
                        "ts_code","rank","rank_tminus_1",
                        "in_whitelist","in_blacklist","in_attention",
                        *[c for c in detail.columns if str(c).startswith("hit:")],
                        *[c for c in detail.columns if c.startswith("ret_fwd_")]
                    ] if c in detail.columns]
                    detail_fmt2 = _fmt_retcols_percent(detail)
                    st.dataframe(detail_fmt2[show_cols].sort_values(["rank"]).reset_index(drop=True),
                                width='stretch', height=460)
                    st.caption("ret_fwd_N = 未来 N 日涨幅（Tracking 已计算）；名单列来自 cache/attention；hit:<规则名> 为所选排名规则在参考日是否触发。")
                except Exception as e:
                    st.error(f"生成失败：{e}")

        # --- Surge ---
        with sub_tabs[1]:
            refS = st.text_input("参考日", value=_get_latest_date_from_files() or "", key="surge_ref")
            mode = st.selectbox("榜单口径", ["today","rolling"], index=1, key="surge_mode")
            rolling_days = st.number_input("rolling模式统计天数", min_value=2, max_value=20, value=5, key="surge_rolling")
            sel_type = st.selectbox("选样", ["top_n","top_pct"], index=0, key="surge_sel_type")
            sel_val = st.number_input("阈值（N或%）", min_value=1, max_value=1000, value=200, key="surge_sel_val")
            retros = st.text_input("回看天数集合（逗号）", value="1,2,3,4,5", key="surge_retros")
            split_label = st.selectbox("分组口径", ["600/000/科创北(3组)", "主vs其他", "各板块"], index=0, key="surge_split_label")
            split = {"600/000/科创北(3组)":"combo3", "主vs其他":"main_vs_others", "各板块":"per_board"}[split_label]

            with st.expander("可选：对当日样本按规则打勾（来自排名规则）", expanded=False):
                import scoring_core as se
                try:
                    rule_names = [str(r.get("name") or f"RULE_{i}") for i, r in enumerate(getattr(se, "SC_RULES", []) or [])]
                    rule_names = sorted(list(dict.fromkeys(rule_names)))
                except Exception:
                    rule_names = []
                surge_rule_names = st.multiselect("指标（可多选）", options=rule_names, default=[] if rule_names else [], key="surge_rule_names")
                surge_max_json = st.number_input("最多读取明细JSON（仅对样本内股票）", min_value=50, max_value=5000, value=100, step=50, key="surge_max_json")

            if st.button("运行 Surge", key="btn_run_surge", width='stretch'):
                with st.spinner("生成 Surge 榜单中…"):
                    try:
                        from stats_core import run_surge
                        rlist = [int(x) for x in (retros or "").split(",") if x.strip().isdigit()]
                        sr = run_surge(
                            ref_date=str(refS).strip(),
                            mode=mode,
                            rolling_days=int(rolling_days),
                            selection={"type": sel_type, "value": int(sel_val)},
                            retro_days=rlist,
                            split=split,
                            score_df=None,
                            save=True,
                        )
                        table = sr.table.copy()

                        # 命中打勾（可选）
                        if surge_rule_names:
                            codes2 = table["ts_code"].astype(str).unique().tolist()
                            if mode == "today":
                                obs_date = _prev_trade_date(str(refS), 1)  # t-1
                            else:
                                first_date = _pick_trade_dates(str(refS), int(rolling_days))[0]  # t-K
                                obs_date = _prev_trade_date(first_date, 1)                      # t-K-1
                            st.caption(f"命中口径：使用『{obs_date}』的 details 作为“启动前”判断。")

                            # 预创建列
                            for nm in surge_rule_names:
                                table[f"hit:{nm}"] = False

                            # 读取 JSON（限额）
                            for ts in codes2[:int(surge_max_json)]:
                                obj = _load_detail_json(str(obs_date), str(ts)) or {}
                                rules = obj.get("rules") or []
                                hits_map = {
                                    str(rr.get("name") or ""): (
                                        # 只要策略触发（ok=True），就视为命中，无需检查add字段
                                        # 或者add>0也视为命中（兼容原有逻辑）
                                        bool(rr.get("ok")) 
                                        or (rr.get("add") is not None and float(rr.get("add", 0.0)) > 0.0)
                                    )
                                    for rr in rules
                                }
                                for nm in surge_rule_names:
                                    col = f"hit:{nm}"
                                    if nm in hits_map:
                                        table.loc[table["ts_code"].astype(str) == ts, col] = bool(hits_map[nm])

                    except Exception as e:
                        st.error(f"Surge 失败：{e}")
                    else:
                        table_fmt = _fmt_retcols_percent(table)
                        st.dataframe(table_fmt, width='stretch', height=420)
                        st.caption("各分组文件已写入 output/surge_lists/<ref>/ 。")

        
        # --- Commonality ---
        with sub_tabs[2]:
            refC = st.text_input("参考日", value=_get_latest_date_from_files() or "", key="common_ref")
            retrosC = st.text_input("统计前 n 日集合（观察日前移 d，逗号）", value="1,3,5")
            modeC = st.selectbox("模式", ["rolling","today"], index=0, key="mode_2")
            rollingC = st.number_input("rolling 天数", min_value=2, max_value=20, value=5, key="rolling_2")
            selC = st.number_input("样本 Top-N", min_value=10, max_value=1000, value=200)
            splitC = st.selectbox("分组口径", ["main_vs_others","per_board"], index=0, key="split_2")
            bg = st.selectbox("背景集", ["all","same_group"], index=0)
            countStrat = st.checkbox("统计每个策略的触发次数（策略分析）", value=True)
            scopeC = st.selectbox("触发统计范围", ["仅样本(大涨)","同组全体","两者对比"], index=0, help="仅样本：只看大涨票；同组全体：样本+同组非样本；两者对比：同时输出两个口径")
            w_en = st.checkbox("对大涨样本加权（用于“同组全体/两者对比”口径）", value=False)
            w_pos = st.slider("样本权重", min_value=1.0, max_value=5.0, value=2.0, step=0.5, help="仅在“同组全体/两者对比”下生效")

            if st.button("运行 Commonality", width='stretch'):
                try:
                    from stats_core import run_commonality
                    rlist = [int(x) for x in retrosC.split(",") if x.strip().isdigit()]
                    cr = run_commonality(
                        ref_date=refC,
                        retro_day=(rlist[0] if rlist else 1),
                        retro_days=rlist,
                        mode=modeC,
                        rolling_days=int(rollingC),
                        selection={"type":"top_n","value":int(selC)},
                        split=splitC,
                        background=bg,
                        save=True,
                        count_strategy=countStrat,
                        count_strategy_scope=("pos" if scopeC=="仅样本(大涨)" else ("group" if scopeC=="同组全体" else "both")),
                        strategy_pos_weight=(float(w_pos) if w_en else 1.0),
                    )

                    if countStrat:
                        trig = None
                        if isinstance(cr.reports, dict):
                            trig = cr.reports.get("strategy_triggers")
                            if trig is None:
                                ks = [k for k in cr.reports if str(k).startswith("strategy_triggers__")]
                                if ks:
                                    trig = pd.concat([cr.reports[k] for k in ks if hasattr(cr.reports[k], "copy")], ignore_index=True, sort=False)
                        if isinstance(trig, pd.DataFrame) and not trig.empty:
                            if "trigger_count" not in trig.columns:
                                for alt in ("count","n","num"):
                                    if alt in trig.columns:
                                        trig = trig.rename(columns={alt: "trigger_count"})
                                        break
                                else:
                                    trig["trigger_count"] = 0
                            order_cols = [c for c in ["obs_date","scope","trigger_weighted","trigger_count","name"] if c in trig.columns]
                            if order_cols:
                                trig = trig.sort_values(order_cols, ascending=[True, True, False, False, True][:len(order_cols)])
                            st.dataframe(trig, width='stretch', height=420)

                            # —— 对比视图 ——
                            show_pivot = st.checkbox("按组/口径对比（透视表）", value=True)
                            if show_pivot:
                                
                                # 指标映射：改成「英文 -> 中文」更稳
                                _metric_map = {
                                    "trigger_count": "触发次数",
                                    "coverage": "覆盖率",
                                    "trigger_weighted": "加权触发次数",
                                    "coverage_weighted": "加权覆盖率",
                                }
                                options_en = [en for en in _metric_map if en in trig.columns]

                                # —— 持久化当前选择（按英文列名）——
                                _pref_key = "pivot_metric_en"
                                default_en = "coverage_weighted" if "coverage_weighted" in options_en else (options_en[0] if options_en else None)
                                if default_en is not None:
                                    if _pref_key not in st.session_state or st.session_state[_pref_key] not in options_en:
                                        st.session_state[_pref_key] = default_en

                                # 选择框：显示中文，值为英文
                                metric_en = st.selectbox(
                                    "选择对比指标",
                                    options_en,
                                    key=_pref_key,
                                    format_func=lambda en: _metric_map.get(en, en),
                                )

                                # 后续用 metric_en 直接做透视；若需要中文名可用：
                                pick_metric_cn = _metric_map.get(metric_en, metric_en)
                                pick_metric = metric_en

                                scopes_avail = sorted(trig["scope"].dropna().unique().tolist()) if "scope" in trig.columns else []
                                scope_pick = st.selectbox("选择口径", options=(scopes_avail or ["pos"]), index=0, key="pivot_scope")
                                dfp = trig.copy()
                                if "scope" in dfp.columns and scope_pick in scopes_avail:
                                    dfp = dfp[dfp["scope"]==scope_pick]
                                if "group" in dfp.columns:
                                    pv = dfp.pivot_table(index="name", columns="group", values=pick_metric, aggfunc="max")
                                    st.dataframe(pv, width='stretch', height=420)

                        # —— 每票命中条数分布 ——
                        ks_hist_single = [k for k in (cr.reports.keys() if isinstance(cr.reports, dict) else []) if str(k).startswith("hits_histogram_single__")]
                        ks_hist_each   = [k for k in (cr.reports.keys() if isinstance(cr.reports, dict) else []) if str(k).startswith("hits_histogram_each__")]
                        if ks_hist_single:
                            st.markdown("**单次型（ANY/LAST 等）命中条数分布**")
                            hist_single = pd.concat([cr.reports[k] for k in ks_hist_single], ignore_index=True, sort=False)
                            scopes_hist = sorted(hist_single["scope"].dropna().unique().tolist()) if "scope" in hist_single.columns else []
                            scope_show = st.selectbox("选择口径（单次型）", options=(scopes_hist or ["pos"]), index=0, key="hist_scope_single")
                            show = hist_single[hist_single["scope"]==scope_show] if scopes_hist else hist_single
                            if not show.empty:
                                pv2 = show.pivot_table(index="n_single_rules_hit", columns="group", values="ratio", aggfunc="max")
                                st.dataframe(pv2, width='stretch', height=280)
                        if ks_hist_each:
                            st.markdown("**多次型（EACH）命中条数分布**")
                            hist_each = pd.concat([cr.reports[k] for k in ks_hist_each], ignore_index=True, sort=False)
                            scopes_hist2 = sorted(hist_each["scope"].dropna().unique().tolist()) if "scope" in hist_each.columns else []
                            scope_show2 = st.selectbox("选择口径（多次型）", options=(scopes_hist2 or ["pos"]), index=0, key="hist_scope_each")
                            show2 = hist_each[hist_each["scope"]==scope_show2] if scopes_hist2 else hist_each
                            if not show2.empty:
                                pv3 = show2.pivot_table(index="n_each_rules_hit", columns="group", values="ratio", aggfunc="max")
                                st.dataframe(pv3, width='stretch', height=280)


                    st.caption("分析集/报告已写入 output/commonality/<ref>/ （包括 strategy_triggers__*.parquet, hits_by_stock__*.parquet, hits_histogram__*.parquet）。")

                except Exception as e:
                    st.error(f"Commonality 失败：{e}")

    # ================== 数据管理 ==================
    with tab_data_view:
        st.subheader("数据管理")
        
        # ===== 数据库基础检查和下载按钮 =====
        # 延迟导入 download 模块
        dl = _lazy_import_download()
        if dl is None:
            st.error("无法导入 download 模块")
        else:
            # 获取基础配置（优先使用config，如果没有则使用download模块的默认值）
            base = str(getattr(cfg, "DATA_ROOT", "./data"))
            api_adj = str(getattr(cfg, "API_ADJ", getattr(dl, "API_ADJ", "qfq"))).lower()
            
            # 下载配置项
            st.markdown("#### 数据下载配置")
            with st.expander("下载参数配置", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    # 从config获取默认值，如果没有则使用download模块的值
                    start_default = str(getattr(cfg, "START_DATE", getattr(dl, "START_DATE", "20200101")))
                    start_use = st.text_input("起始日 START_DATE (YYYYMMDD)", value=start_default, key="dl_start_date")
                    
                    end_default_cfg = getattr(cfg, "END_DATE", "today")
                    if str(end_default_cfg).strip().lower() == "today":
                        end_default = "today"
                    else:
                        end_default = str(end_default_cfg)
                    end_input = st.text_input("结束日 END_DATE ('today' 或 YYYYMMDD)", value=end_default, key="dl_end_date")
                    
                    assets_default = list(getattr(cfg, "ASSETS", getattr(dl, "ASSETS", ["stock", "index"]))) or ["stock", "index"]
                    assets = st.multiselect("资产 ASSETS", ["stock", "index"], default=assets_default, key="dl_assets")
                
                with c2:
                    api_adj_options = ["qfq", "hfq", "raw"]
                    api_adj_index = api_adj_options.index(api_adj) if api_adj in api_adj_options else 0
                    api_adj = st.selectbox("复权 API_ADJ", api_adj_options, index=api_adj_index, key="dl_api_adj").lower()
                    
                    fast_threads_default = int(getattr(cfg, "FAST_INIT_THREADS", getattr(dl, "FAST_INIT_THREADS", 16)))
                    fast_threads = st.number_input("FAST_INIT 并发", min_value=1, max_value=64, value=fast_threads_default, key="dl_fast_threads")
                    
                    inc_threads_default = int(getattr(cfg, "STOCK_INC_THREADS", getattr(dl, "STOCK_INC_THREADS", 16)))
                    inc_threads = st.number_input("增量下载线程", min_value=1, max_value=64, value=inc_threads_default, key="dl_inc_threads")
                
                with c3:
                    ind_workers_default = int(getattr(cfg, "INC_RECALC_WORKERS", getattr(dl, "INC_RECALC_WORKERS", 32)))
                    ind_workers = st.number_input("指标重算线程(可选)", min_value=0, max_value=128, value=ind_workers_default, key="dl_ind_workers")
                    
                    st.caption(f"数据根目录: {base}")
            
            # 处理结束日期
            end_use = _today_str() if str(end_input).strip().lower() == "today" else str(end_input).strip()
            start_use = str(start_use).strip()
            
            # 应用参数
            _apply_overrides(base, assets, start_use, end_use, api_adj, int(fast_threads), int(inc_threads), int(ind_workers) if ind_workers else None)
            
            # 显示当前状态
            latest = _latest_trade_date(base, api_adj)
            if latest:
                st.caption(f"当前 {api_adj} 最近交易日：{latest}")
            
            # 下载按钮
            run_download = st.button("🚀 运行下载", width='stretch', type="primary", key="run_download_btn")
            
            if run_download:
                logger.info(f"用户点击运行下载: 起始日期={start_use}, 结束日期={end_use}")
                try:
                    # 直接调用download模块，让它自己判断是否为增量
                    steps = [
                        "准备环境",
                        "数据下载（自动判断首次/增量）",
                        "清理与校验",
                    ]
                    sp = Stepper("数据下载", steps, key_prefix="dl_auto")
                    sp.start()
                    sp.step("准备环境")
                    sp.step("数据下载（自动判断首次/增量）")
                    
                    # 调用download模块的主函数，让它自己判断
                    dl = _lazy_import_download()
                    if dl is not None:
                        results = dl.download_data(
                            start_date=start_use,
                            end_date=end_use,
                            adj_type=api_adj,
                            assets=assets,
                            threads=int(inc_threads),
                            enable_warmup=True,
                            enable_adaptive_rate_limit=True
                        )
                        # 显示下载结果
                        for asset_type, stats in results.items():
                            st.success(f"{asset_type}: 成功={stats.success_count}, 空数据={stats.empty_count}, 失败={stats.error_count}")
                    
                    sp.step("清理与校验")
                    sp.finish(True, "下载完成")
                except Exception as e:
                    st.error(f"下载失败：{e}")
            
            st.divider()
        
        # ===== 数据浏览（保留原有功能） =====
        st.markdown("#### 数据浏览")
        st.info("用于可视化读取数据库原数据")
        
        # 数据源选择
        data_source = st.radio(
            "选择数据源",
            ["Details数据库", "股票原数据"],
            horizontal=True,
            help="Details: 存储评分详情数据 | 股票原数据: 存储股票行情和指标数据"
        )
        
        if data_source == "Details数据库":
            # Details数据库查看
            try:
                from database_manager import (
                    query_details_by_date,
                    query_details_by_stock,
                    query_details_top_stocks,
                    query_details_score_range,
                    query_details_recent_dates,
                    get_details_table_info,
                    get_details_db_path_with_fallback,
                    is_details_db_available
                )
                import os
                
                # 使用统一的函数获取details数据库路径（包含回退逻辑）
                db_path = get_details_db_path_with_fallback()
                
                if not db_path or not is_details_db_available():
                    st.warning(f"Details数据库不可用: {db_path if db_path else '路径获取失败'}")
                else:
                    # —— 数据库查询控制按钮 ——
                    data_view_db_enabled = st.session_state.get("data_view_db_enabled", False)
                    col_db_ctrl1, col_db_ctrl2 = st.columns([3, 1])
                    with col_db_ctrl1:
                        if data_view_db_enabled:
                            st.success("✅ 数据库查询已启用（可以查询数据库）")
                        else:
                            st.info("ℹ️ 数据库查询未启用（点击按钮后才会查询数据库，避免与写入操作冲突）")
                    with col_db_ctrl2:
                        if not data_view_db_enabled:
                            if st.button("🔓 启用数据库查询", key="enable_data_view_db"):
                                st.session_state["data_view_db_enabled"] = True
                                st.rerun()
                        else:
                            # 一旦启用就不再显示按钮，保持启用状态
                            pass
                    
                    # 只有在启用数据库查询后才执行查询操作
                    if not data_view_db_enabled:
                        st.warning("⚠️ 请先点击「启用数据库查询」按钮，然后才能查询数据库数据")
                        st.stop()
                    
                    # 以下是所有数据库查询操作，只有在启用后才会执行
                    # 查询类型选择
                    query_type = st.selectbox(
                        "查询类型",
                        ["按日期查看", "按股票代码查看", "Top-K股票", "分数范围查询"],
                        key="details_query_type"
                    )
                    
                    # 获取并缓存最新日期（延迟加载，避免UI启动时建立连接）
                    @st.cache_data
                    def get_latest_details_date():
                        try:
                            dates = query_details_recent_dates(1, db_path)
                            return dates[0] if dates else None
                        except:
                            return None
                    
                    # 只在需要显示时才调用，避免UI启动时建立连接
                    latest_date = None
                    if query_type == "按日期查看":
                        latest_date = get_latest_details_date()
                    
                    if query_type == "按日期查看":
                        # 如果还没有设置默认日期，使用最新日期
                        if "details_date_value" not in st.session_state:
                            st.session_state["details_date_value"] = latest_date if latest_date else ""
                        
                        limit_param = st.number_input("返回记录数", min_value=1, max_value=1000, value=200, key="details_limit")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            ref_date = st.text_input(
                                "参考日期（YYYYMMDD，留空=最新）",
                                value=st.session_state["details_date_value"],
                                key="details_date"
                            )
                        with col2:
                            st.write("")  # 占位
                            st.write("")
                            refresh_btn = st.button("刷新", key="details_refresh_btn")
                        
                        # 处理刷新或查询
                        date_to_use = ref_date.strip() if ref_date.strip() else latest_date
                        
                        if refresh_btn or date_to_use:
                            try:
                                df = query_details_by_date(date_to_use, limit=limit_param, db_path=db_path)
                                if not df.empty:
                                    st.dataframe(df, width='stretch')
                                    st.info(f"共找到 {len(df)} 条记录 | 查询日期: {date_to_use}")
                                else:
                                    st.warning("未找到数据")
                            except Exception as e:
                                st.error(f"查询失败: {e}")
                        
                        # 显示最近的日期列表
                        try:
                            recent_dates = query_details_recent_dates(7, db_path)
                            if recent_dates:
                                st.caption(f"最近的交易日: {', '.join(recent_dates)}")
                        except Exception as e:
                            pass
                    
                    elif query_type == "按股票代码查看":
                        ts_code = st.text_input("股票代码（如000001）", key="details_ts_code")
                        limit = st.number_input("返回记录数", min_value=1, max_value=100, value=10)
                        
                        if ts_code:
                            try:
                                df = query_details_by_stock(ts_code, limit, db_path)
                                if not df.empty:
                                    st.dataframe(df, width='stretch')
                                else:
                                    st.warning("未找到该股票的数据")
                            except Exception as e:
                                st.error(f"查询失败: {e}")
                    
                    elif query_type == "Top-K股票":
                        # 如果还没有设置默认日期，使用最新日期
                        if "details_topk_date_value" not in st.session_state:
                            st.session_state["details_topk_date_value"] = latest_date if latest_date else ""
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            ref_date = st.text_input("参考日期（YYYYMMDD）", value=st.session_state["details_topk_date_value"], key="details_topk_date")
                        with col2:
                            st.write("")  # 占位
                            st.write("")
                            refresh_topk_btn = st.button("刷新", key="details_topk_refresh_btn")
                        
                        top_k = st.number_input("Top-K", min_value=1, max_value=500, value=50)
                        
                        date_to_use = ref_date.strip() if ref_date.strip() else latest_date
                        
                        if refresh_topk_btn or date_to_use:
                            try:
                                df = query_details_top_stocks(date_to_use, top_k, db_path)
                                if not df.empty:
                                    st.dataframe(df, width='stretch')
                                    st.info(f"查询日期: {date_to_use}")
                                else:
                                    st.warning("未找到数据")
                            except Exception as e:
                                st.error(f"查询失败: {e}")
                    
                    elif query_type == "分数范围查询":
                        # 如果还没有设置默认日期，使用最新日期
                        if "details_score_date_value" not in st.session_state:
                            st.session_state["details_score_date_value"] = latest_date if latest_date else ""
                        
                        col1_date, col2_date = st.columns([3, 1])
                        with col1_date:
                            ref_date = st.text_input("参考日期（YYYYMMDD）", value=st.session_state["details_score_date_value"], key="details_score_date")
                        with col2_date:
                            st.write("")  # 占位
                            st.write("")
                            refresh_score_btn = st.button("刷新", key="details_score_refresh_btn")
                        
                        col1_score, col2_score = st.columns(2)
                        with col1_score:
                            min_score = st.number_input("最低分数", value=50.0, key="details_score_min")
                        with col2_score:
                            max_score = st.number_input("最高分数", value=100.0, key="details_score_max")
                        
                        date_to_use = ref_date.strip() if ref_date.strip() else latest_date
                        
                        if refresh_score_btn or date_to_use:
                            try:
                                df = query_details_score_range(date_to_use, min_score, max_score, db_path)
                                if not df.empty:
                                    st.dataframe(df, width='stretch')
                                    st.info(f"共找到 {len(df)} 条记录 | 查询日期: {date_to_use}")
                                else:
                                    st.warning("未找到数据")
                            except Exception as e:
                                st.error(f"查询失败: {e}")
                    
                    # 数据库信息（美化显示）
                    with st.expander("数据库信息"):
                        try:
                            info = get_details_table_info(db_path)
                            # 美化显示数据库信息
                            if isinstance(info, dict):
                                with st.container(border=True):
                                    for key, value in info.items():
                                        if isinstance(value, (int, float)):
                                            st.metric(key, value)
                                        elif isinstance(value, list):
                                            st.text(f"{key}: {len(value)} 项")
                                            if value and len(value) <= 10:
                                                for item in value:
                                                    st.text(f"  • {item}")
                                        elif isinstance(value, dict):
                                            st.text(f"{key}:")
                                            for sub_key, sub_value in value.items():
                                                st.text(f"  {sub_key}: {sub_value}")
                                        else:
                                            st.text(f"{key}: {value}")
                            else:
                                st.text(str(info))
                        except Exception as e:
                            st.error(f"获取信息失败: {e}")
            
            except Exception as e:
                st.error(f"初始化Details数据库读取器失败: {e}")
                import traceback
                st.text(traceback.format_exc())
        
        else:
            # 股票原数据查看
            from config import DATA_ROOT, UNIFIED_DB_PATH
            
            db_path = os.path.join(DATA_ROOT, UNIFIED_DB_PATH)
            
            if not os.path.exists(db_path):
                st.warning(f"股票数据数据库不存在: {db_path}")
            else:
                # 查询参数
                col1, col2 = st.columns(2)
                with col1:
                    asset_type = st.selectbox("资产类型", ["stock", "index"], index=0)
                    # 根据资产类型设置默认adj
                    default_adj = "ind" if asset_type == "index" else "qfq"
                with col2:
                    adj_type = st.selectbox(
                        "复权类型",
                        ["raw", "qfq", "hfq", "ind"],
                        index=["raw", "qfq", "hfq", "ind"].index(default_adj)
                    )
                
                view_mode = st.radio(
                    "查看模式",
                    ["单日查看", "区间查询", "单股历史"],
                    horizontal=True
                )
                
                @st.cache_data
                def get_trade_dates_list():
                    """获取交易日期列表"""
                    try:
                        dates = get_trade_dates(db_path)
                        return dates
                    except Exception as e:
                        st.error(f"获取交易日期失败: {e}")
                        return []
                
                # 延迟加载交易日期
                if 'trade_dates' not in st.session_state:
                    with st.spinner("正在加载交易日期..."):
                        st.session_state['trade_dates'] = get_trade_dates_list()
                
                if view_mode == "单日查看":
                    trade_dates = st.session_state['trade_dates']
                    if trade_dates:
                        selected_date = st.selectbox(
                            "选择日期",
                            trade_dates,
                            index=len(trade_dates)-1 if trade_dates else 0
                        )
                        
                        limit = st.number_input("显示行数（-1为全部）", value=100, min_value=-1, max_value=5000, step=50)
                        if limit == -1:
                            limit = None
                        
                        if st.button("查询", key="btn_query_day"):
                            try:
                                df = query_stock_data(
                                    db_path=db_path,
                                    start_date=selected_date,
                                    end_date=selected_date,
                                    adj_type=adj_type if asset_type != "index" else "ind",
                                    limit=limit
                                )
                                if not df.empty:
                                    st.dataframe(df, width='stretch')
                                    # 统计总行数（忽略limit）
                                    try:
                                        from database_manager import count_stock_data as _count_stock_data
                                        total_rows = _count_stock_data(
                                            db_path=db_path,
                                            ts_code=None,
                                            start_date=selected_date,
                                            end_date=selected_date,
                                            adj_type=adj_type if asset_type != "index" else "ind"
                                        )
                                    except Exception:
                                        total_rows = len(df)
                                    st.info(f"总行数: {total_rows} | 本次显示: {len(df)}")
                                else:
                                    st.warning("未找到数据")
                            except Exception as e:
                                st.error(f"查询失败: {e}")
                    else:
                        st.warning("无法获取交易日期列表")
                
                elif view_mode == "区间查询":
                    trade_dates = st.session_state['trade_dates']
                    if trade_dates:
                        col1, col2 = st.columns(2)
                        with col1:
                            start_date = st.selectbox("起始日期", trade_dates, index=len(trade_dates)-10 if len(trade_dates) >= 10 else 0)
                        with col2:
                            end_date = st.selectbox("结束日期", trade_dates, index=len(trade_dates)-1)
                        
                        ts_code = st.text_input("股票代码（留空=全市场，如000001.SZ）")
                        
                        columns_input = st.text_input("指定列（用逗号分隔，留空=所有列）", placeholder="如: trade_date,open,high,low,close,vol")
                        columns = [c.strip() for c in columns_input.split(",")] if columns_input else None
                        
                        limit = st.number_input("显示行数（-1为全部）", value=200, min_value=-1, max_value=10000, step=100)
                        if limit == -1:
                            limit = None
                        
                        if st.button("查询", key="btn_query_range"):
                            try:
                                df = query_stock_data(
                                    db_path=db_path,
                                    ts_code=ts_code if ts_code else None,
                                    start_date=start_date,
                                    end_date=end_date,
                                    columns=columns,
                                    adj_type=adj_type if asset_type != "index" else "ind",
                                    limit=limit
                                )
                                if not df.empty:
                                    st.dataframe(df, width='stretch')
                                    # 统计总行数（忽略limit）
                                    try:
                                        from database_manager import count_stock_data as _count_stock_data
                                        total_rows = _count_stock_data(
                                            db_path=db_path,
                                            ts_code=ts_code if ts_code else None,
                                            start_date=start_date,
                                            end_date=end_date,
                                            adj_type=adj_type if asset_type != "index" else "ind"
                                        )
                                    except Exception:
                                        total_rows = len(df)
                                    st.info(f"总行数: {total_rows} | 本次显示: {len(df)}")
                                else:
                                    st.warning("未找到数据")
                            except Exception as e:
                                st.error(f"查询失败: {e}")
                    
                    if not trade_dates:
                        st.warning("无法获取交易日期列表")
                
                elif view_mode == "单股历史":
                    ts_code = st.text_input("股票代码（如000001.SZ）", key="single_stock_code")
                    
                    trade_dates = st.session_state.get('trade_dates', [])
                    if not trade_dates:
                        st.warning("无法获取交易日期列表")
                    elif ts_code:
                        col1, col2 = st.columns(2)
                        with col1:
                            start_date = st.selectbox("起始日期", trade_dates, index=len(trade_dates)-60 if len(trade_dates) >= 60 else 0, key="single_start")
                        with col2:
                            end_date = st.selectbox("结束日期", trade_dates, index=len(trade_dates)-1, key="single_end")
                    
                    columns_input = st.text_input("指定列（用逗号分隔，留空=所有列）", placeholder="如: trade_date,open,high,low,close,vol,kdj_k,kdj_d,rsi", key="single_columns")
                    columns = [c.strip() for c in columns_input.split(",")] if columns_input else None
                    
                    limit = st.number_input("显示行数（-1为全部）", value=-1, min_value=-1, max_value=10000, step=100, key="single_limit")
                    if limit == -1:
                        limit = None
                    
                    if st.button("查询", key="btn_query_single"):
                        if not ts_code:
                            st.error("请输入股票代码")
                        else:
                            try:
                                df = query_stock_data(
                                    db_path=db_path,
                                    ts_code=ts_code,
                                    start_date=start_date,
                                    end_date=end_date,
                                    columns=columns,
                                    adj_type=adj_type if asset_type != "index" else "ind",
                                    limit=limit
                                )
                                if not df.empty:
                                    st.dataframe(df, width='stretch')
                                    
                                    # 如果有收盘价数据，绘制图表
                                    if "close" in df.columns and "trade_date" in df.columns:
                                        try:
                                            df_chart = df.copy()
                                            df_chart["trade_date"] = pd.to_datetime(df_chart["trade_date"])
                                            st.line_chart(df_chart.set_index("trade_date")[["close"]])
                                        except Exception as e:
                                            st.warning(f"无法绘制图表: {e}")
                                    # 统计总行数（忽略limit）
                                    try:
                                        from database_manager import count_stock_data as _count_stock_data
                                        total_rows = _count_stock_data(
                                            db_path=db_path,
                                            ts_code=ts_code,
                                            start_date=start_date,
                                            end_date=end_date,
                                            adj_type=adj_type if asset_type != "index" else "ind"
                                        )
                                    except Exception:
                                        total_rows = len(df)
                                    st.info(f"总行数: {total_rows} | 本次显示: {len(df)}")
                                else:
                                    st.warning("未找到该股票的数据")
                            except Exception as e:
                                st.error(f"查询失败: {e}")
                
                # 数据库信息（美化显示）
                with st.expander("数据库信息"):
                    try:
                        info = get_database_info()
                        # 美化显示数据库信息
                        if isinstance(info, dict):
                            with st.container(border=True):
                                for key, value in info.items():
                                    if isinstance(value, (int, float)):
                                        st.metric(key, value)
                                    elif isinstance(value, list):
                                        st.text(f"{key}: {len(value)} 项")
                                        if value and len(value) <= 10:
                                            for item in value:
                                                st.text(f"  • {item}")
                                    elif isinstance(value, dict):
                                        st.text(f"{key}:")
                                        for sub_key, sub_value in value.items():
                                            st.text(f"  {sub_key}: {sub_value}")
                                    else:
                                        st.text(f"{key}: {value}")
                        else:
                            st.text(str(info))
                    except Exception as e:
                        st.error(f"获取信息失败: {e}")

    # ================== 日志 ==================
    with tab_logs:
        st.subheader("日志")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**score.log（尾部 400 行）**")
            st.code(_tail(LOG_DIR / "score.log", 400), language="bash")
        with col2:
            st.markdown("**score_ui.log（尾部 400 行）**")
            st.code(_tail(LOG_DIR / "score_ui.log", 400), language="bash")

    _anchor = st.session_state.pop("scroll_after_rerun", None)
    if _anchor:
        components.html(f"""
        <script>
        (function() {{
        const id = {_anchor!r};
        function go() {{
            const doc = parent.document || document;
            // 1) 激活“个股详情”页签（按钮 role="tab"，文本以“个股详情”开头）
            const tabs = doc.querySelectorAll('button[role="tab"]');
            for (const btn of tabs) {{
            if ((btn.innerText || '').trim().startsWith('个股详情')) {{ btn.click(); break; }}
            }}
            // 2) 滚动到锚点
            const el = doc.getElementById(id);
            if (el) {{
            el.scrollIntoView({{behavior:'instant', block:'start'}});
            }} else {{
            // 兜底：把 hash 设置为锚点
            parent.location.hash = id;
            }}
        }}
        // 多次尝试，等外层 DOM 稳定
        setTimeout(go, 0); setTimeout(go, 200); setTimeout(go, 600);
        }})();
        </script>
        """, height=0)
