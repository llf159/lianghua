# -*- coding: utf-8 -*-
"""
Score UI — 无侧栏版（中文）
- 参数在“排名（运行+浏览）”页签中编辑（与原版一致的交互思路）
- 统计作为普通页签（Tracking / Surge / Commonality / Portfolio统计）
- 个股详情含“命中信号日期查询”
- 导出统一支持 TXT（空格分隔/一行一个，是否带交易所后缀）
- 组合模拟/持仓：基于 Top 文件生成当期持仓与相对上期调仓列表（可导出）
"""

from __future__ import annotations
import os, io, json, re
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from contextlib import contextmanager

import download as dl
import app_pv as apv

try:
    import scoring_core as se
except Exception as e:
    st.error(f"无法导入 scoring_core：{e}")
    st.stop()

try:
    import config as cfg
except Exception as e:
    st.error(f"无法导入 config：{e}")
    st.stop()

# 统计模块可选
try:
    import stats_core as stats
except Exception:
    stats = None
    
from utils import normalize_ts, ensure_datetime_index, normalize_trade_date, market_label
from parquet_viewer import read_range, asset_root, list_trade_dates
from config import PARQUET_BASE, PARQUET_ADJ
st.set_page_config(page_title="ScoreApp", layout="wide")
import tdx_compat as tdx

# ===== 常量路径 =====
SC_OUTPUT_DIR = Path(getattr(cfg, "SC_OUTPUT_DIR", "output/score"))
TOP_DIR  = SC_OUTPUT_DIR / "top"
ALL_DIR  = SC_OUTPUT_DIR / "all"
DET_DIR  = SC_OUTPUT_DIR / "details"
ATTN_DIR = SC_OUTPUT_DIR / "attention"
LOG_DIR  = Path("./log")
for p in [TOP_DIR, ALL_DIR, DET_DIR, ATTN_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def _today_str() -> str:
    return datetime.date.today().strftime("%Y%m%d")

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
        cfg.PARQUET_BASE = base
        cfg.DATA_ROOT = base
        cfg.PARQUET_ADJ = api_adj.lower() if api_adj.lower() in {"daily","raw","qfq","hfq"} else getattr(cfg, "PARQUET_ADJ", "qfq")
    except Exception:
        pass

@st.cache_data(show_spinner=False, ttl=300)
def _latest_trade_date(base: str, adj: str) -> str | None:
    try:
        from parquet_viewer import asset_root, list_trade_dates
        root = asset_root(base, "stock", adj)
        ds = list_trade_dates(root)
        return ds[-1] if ds else None
    except Exception:
        return None

# -------------------- 执行动作（封装 download.py） --------------------
def _run_fast_init(end_use: str):
    dl.fast_init_download(end_use)                       # 首次全量（单股缓存）
    if getattr(dl, "DUCK_MERGE_DAY_LAG", 5) >= 0:
        dl.duckdb_partition_merge()                     # 合并到 daily_*
    if getattr(dl, "WRITE_SYMBOL_INDICATORS", True):
        dl.duckdb_merge_symbol_products_to_daily()      # 合并指标到 daily_*_indicators


def _run_increment(start_use: str, end_use: str, do_stock: bool, do_index: bool, do_indicators: bool):
    # 若 fast_init 的缓存存在，先合并一次（与 main() 逻辑一致）
    try:
        if any(
            os.path.isdir(os.path.join(dl.FAST_INIT_STOCK_DIR, d))
            and any(f.endswith(".parquet") for f in os.listdir(os.path.join(dl.FAST_INIT_STOCK_DIR, d)))
            for d in ("raw","qfq","hfq")
        ):
            dl.duckdb_partition_merge()
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

@st.cache_data(show_spinner=False, hash_funcs={Path: lambda p: p.stat().st_mtime_ns})
def _read_df(path: Path, usecols=None, dtype=None, encoding: str = "utf-8-sig") -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, encoding=encoding, engine="c")
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=600)
def _cached_trade_dates(base: str, adj: str):
    from parquet_viewer import asset_root, list_trade_dates
    root = asset_root(base, "stock", adj)
    return list_trade_dates(root) or []

@contextmanager
def se_progress_to_streamlit():
    status = st.status("准备中…", expanded=True)
    bar = st.progress(0, text="就绪")
    info = st.empty()
    total = 0
    current = 0

    def _cb(phase, current=None, total=None, message=None, **kw):
        nonlocal status, bar, info
        # 更新文字
        tag = {
            "select_ref_date": "选择参考日",
            "compute_read_window": "计算读取区间",
            "build_universe_done": "构建评分清单",
            "score_start": "并行评分启动",
            "score_progress": "评分进行中",
            "write_cache_lists": "写入黑白名单",
            "write_top_all_start": "写出 Top/All",
            "write_top_all_done": "Top/All 完成",
            "hooks_start": "统计/回看",
            "hooks_done": "统计完成",
        }.get(phase, phase)
        txt = f"{tag}"
        if message:
            txt += f" · {message}"

        # 进度条
        if total is not None and total > 0 and current is not None:
            pct = int(current * 100 / max(total, 1))
            bar.progress(pct, text=txt)
        else:
            info.write(txt)

    se.set_progress_handler(_cb)
    try:
        yield
        status.update(label="已完成", state="complete")
    finally:
        se.set_progress_handler(None)

def _pick_latest_ref_date() -> Optional[str]:
    files = sorted(TOP_DIR.glob("score_top_*.csv"))
    dates = []
    for p in files:
        m = re.search(r"(\d{8})", p.name)
        if m: dates.append(m.group(1))
    return max(dates) if dates else None

def _prev_ref_date(cur: str) -> Optional[str]:
    files = sorted(TOP_DIR.glob("score_top_*.csv"))
    dates = []
    for p in files:
        m = re.search(r"(\d{8})", p.name)
        if m and m.group(1) < cur:
            dates.append(m.group(1))
    return dates[-1] if dates else None

# —— 小工具：把规则 JSON 转成 tdx_screen 所需参数 ——
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
        return when, tf, win, scope


def _load_detail_json(ref: str, ts: str) -> Optional[Dict]:
    p = _path_detail(ref, ts)
    if not p.exists(): return None
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
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
                       use_container_width=True, key=key)

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

def _scan_date_range(start_yyyymmdd: str, end_yyyymmdd: str) -> List[str]:
    s = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    e = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    out = []
    while s <= e:
        out.append(s.strftime("%Y%m%d"))
        s += timedelta(days=1)
    return out

def _apply_runtime_overrides(rules_obj: dict,
                             topk: int, tie_break: str, max_workers: int,
                             attn_on: bool, universe: str|List[str]):
    # 规则临时覆盖（仅当前进程）
    if rules_obj:
        pres = rules_obj.get("prescreen")
        rules = rules_obj.get("rules")
        if pres is not None: setattr(se, "SC_PRESCREEN_RULES", pres)
        if rules is not None: setattr(se, "SC_RULES", rules)
    setattr(se, "SC_TOP_K", int(topk))
    setattr(se, "SC_TIE_BREAK", str(tie_break))
    setattr(se, "SC_MAX_WORKERS", int(max_workers))
    setattr(se, "SC_ATTENTION_ENABLE", bool(attn_on))
    setattr(se, "SC_UNIVERSE", universe)
    
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


# ===== 会话状态 =====
if "rules_obj" not in st.session_state:
    st.session_state["rules_obj"] = {
        "prescreen": getattr(cfg, "SC_PRESCREEN_RULES", []),
        "rules": getattr(cfg, "SC_RULES", []),
    }
if "export_pref" not in st.session_state:
    st.session_state["export_pref"] = {"style": "space", "with_suffix": True}

# ===== 页眉 =====
st.title("ScoreApp")

# ===== 顶层页签 =====
tab_rank, tab_detail, tab_rules, tab_attn, tab_data, tab_screen, tab_tools, tab_port, tab_stats, tab_logs = st.tabs(
    ["排名", "个股详情", "规则编辑", "强度榜", "数据下载/浏览", "普通选股", "工具箱", "组合模拟/持仓", "统计", "日志"]
)

# ================== 排名 ==================
with tab_rank:
    st.subheader("① 排名")
    with st.expander("参数设置（运行前请确认）", expanded=True):
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            # ref_inp = st.text_input("参考日（YYYYMMDD；留空=自动取最新）", value="")
            ref_inp = st.text_input("参考日（YYYYMMDD；留空=自动取最新）", value="", key="rank_ref_input")
            # topk = st.number_input("Top-K", min_value=1, max_value=2000, value=int(getattr(cfg, "SC_TOP_K", 50)))
            topk = st.number_input("Top-K", min_value=1, max_value=2000, value=cfg_int("SC_TOP_K", 50))
        with c2:
            # tie = st.selectbox("同分排序（Tie-break）", ["none", "kdj_j_asc"], index=0 if str(getattr(cfg,"SC_TIE_BREAK","none")).lower()=="none" else 1)
            tie_default = cfg_str("SC_TIE_BREAK", "none").lower()
            tie = st.selectbox("同分排序（Tie-break）", ["none", "kdj_j_asc"], index=0 if tie_default=="none" else 1)
            # maxw = st.number_input("最大并行数", min_value=1, max_value=64, value=int(getattr(cfg, "SC_MAX_WORKERS", 8)))
            maxw = st.number_input("最大并行数", min_value=1, max_value=64, value=cfg_int("SC_MAX_WORKERS", 8))
        with c3:
            universe = st.selectbox("评分范围", ["全市场","仅白名单","仅黑名单","仅特别关注榜"], index=0)
            style = st.selectbox("TXT 导出格式", ["空格分隔", "一行一个"], index=0)
        with c4:
            attn_on = st.checkbox("评分后生成关注榜", value=cfg_bool("SC_ATTENTION_ENABLE", False))
            with_suffix = st.checkbox("导出带交易所后缀（.SZ/.SH）", value=False)
        st.session_state["export_pref"] = {"style": "space" if style=="空格分隔" else "lines",
                                           "with_suffix": with_suffix}
        run_col1, run_col2 = st.columns([1,1])
        with run_col1:
            run_btn = st.button("🚀 运行评分（写入 Top/All/Details）", use_container_width=True)
        with run_col2:
            latest_btn = st.button("📅 读取最近一次结果（不重新计算）", use_container_width=True)

    # 运行
    ref_to_use = ref_inp.strip() or _pick_latest_ref_date()
    if run_btn:
        _apply_runtime_overrides(st.session_state["rules_obj"], topk, tie, maxw, attn_on,
                                 {"全市场":"all","仅白名单":"white","仅黑名单":"black","仅特别关注榜":"attention"}[universe])
        try:
            with se_progress_to_streamlit():
                top_path = se.run_for_date(ref_inp.strip() or None)
            st.success(f"评分完成：{top_path}")
        # 解析参考日
            m = re.search(r"(\d{8})", str(top_path))
            if m:
                ref_to_use = m.group(1)
                if latest_btn and not ref_to_use:
                    ref_to_use = _pick_latest_ref_date()
        except Exception as e:
            st.error(f"评分失败：{e}")
            ref_to_use = None

    # “读取最近一次结果”按钮：仅读取，不计算
    if latest_btn and not run_btn:
        ref_to_use = _pick_latest_ref_date()

    # ---- 统一的 Top 预览区块（无论 run 或 读取最近一次） ----
    if ref_to_use:
        st.markdown(f"**参考日：{ref_to_use}**")
        df_all = _read_df(_path_all(ref_to_use))
    else:
        st.info("未找到任何 Top 文件，请先运行评分或检查输出目录。")

    st.divider()
    with st.container(border=True):
        st.markdown("**Top-K 预览(较大的表可能渲染较慢。**")
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
            st.dataframe(df_all.head(rows_eff), use_container_width=True, height=420)
            if "ts_code" in df_all.columns:
                codes = df_all["ts_code"].astype(str).head(rows_eff).tolist()
                txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"], st.session_state["export_pref"]["with_suffix"])
                copy_txt_button(txt, label="📋 复制以上（按当前预览）", key=f"copy_top_{ref_to_use}")
        else:
            st.caption("暂无 Top-K 数据")


# ================== 个股详情 ==================
with tab_detail:
    st.subheader("② 个股详情")

    # —— 选择参考日 + 代码（支持从 Top-K 下拉选择） ——
    c0, c1 = st.columns([1,2])
    with c0:
        ref_d = st.text_input("参考日（留空=自动最新）", value="", key="detail_ref_input")
    ref_real = (ref_d or "").strip() or _pick_latest_ref_date() or ""
    # 读取该参考日 Top 文件以便下拉选择
    try:
        df_top_ref = _read_df(_path_top(ref_real)) if ref_real else pd.DataFrame()
        options_codes = df_top_ref["ts_code"].astype(str).tolist() if ("ts_code" in df_top_ref.columns and not df_top_ref.empty) else []
    except Exception:
        options_codes = []
    with c1:
        code_from_list = st.selectbox("从 Top-K 选择（可选）", options=options_codes or [], index=0 if options_codes else None, placeholder="也可手动输入 ↓")
    code_typed = st.text_input("或手动输入股票代码", value=(code_from_list or ""), key="detail_code_input")
    code_norm = normalize_ts(code_typed) if code_typed else ""

    # —— 渲染详情（含 old 版功能） ——
    if code_norm and ref_real:
        obj = _load_detail_json(ref_real, code_norm)
        if not obj:
            st.warning("未找到该票的详情 JSON（可能当日未在样本内或未产出 Details）。")
        else:
            data = obj
            summary = data.get("summary", {}) or {}
            ts = data.get("ts_code", code_norm)
            score = float(summary.get("score", 0))
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

            cols = st.columns(5)
            cols[0].metric("代码", ts)
            cols[1].metric("分数", f"{score:.2f}")
            cols[2].metric("排名", rank_display)
            cols[3].metric("市场", market_label(ts))
            cols[4].metric("参考日", ref_real)

            st.divider()

            # 总览 + 高亮/缺点
            colA, colB = st.columns([1,1])
            with colA:
                st.markdown("**总览**")
                st.json(summary)
            with colB:
                st.markdown("**高亮 / 缺点**")
                st.write({"highlights": summary.get("highlights", []), "drawbacks": summary.get("drawbacks", [])})

            # 交易性机会
            ops = (summary.get("opportunities") or [])
            with st.expander("交易性机会", expanded=True):
                if ops:
                    for t in ops:
                        st.write("• " + str(t))
                else:
                    st.caption("暂无")

            # 逐规则明细（可选显示 when）
            rules = pd.DataFrame(data.get("rules", []))
            name_to_when = {}
            
            from datetime import datetime
            import re as _re

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
            show_when = st.checkbox("显示规则 when 表达式", value=False, key="detail_show_when")
            if not rules.empty:
                if show_when:
                    rules["when"] = rules["name"].map(name_to_when).fillna("")
                st.markdown("**规则明细**")
                st.dataframe(rules, use_container_width=True, height=420)
            else:
                st.info("无规则明细。")

            with st.expander("按触发某规则筛选（当日全市场）", expanded=True):
                st.caption("说明：基于当日 details JSON，筛出‘命中该规则’的股票；按 Score 降序（得分相同则按代码升序）。")
                # 规则名下拉：默认列出当前票当日详情里的规则名，方便选择
                rule_names = [r.get("name") for r in (data.get("rules") or []) if r.get("name")]
                chosen_rule_name = st.selectbox("选择规则（指标）", options=sorted(set(rule_names)) if rule_names else [], 
                                                index=0 if rule_names else None, placeholder="请选择一个规则名")
                colL, colR = st.columns([1,1])
                with colL:
                    only_topk = st.checkbox("仅限当日 Top-K 范围", value=True)
                with colR:
                    run_filter = st.button("筛选当日命中标的", use_container_width=True)

                if run_filter:
                    if not ref_real:
                        st.error("未能确定参考日。")
                    elif not chosen_rule_name:
                        st.warning("请先选择一个规则名。")
                    else:
                        # 读取当日全市场 details，挑出命中 chosen_rule_name 的股票
                        rows = []
                        ddir = DET_DIR / ref_real
                        try:
                            # 可选：用 All 文件限制 universe（Top-K）
                            allow_set = None
                            if only_topk:
                                df_allx = _read_df(_path_all(ref_real), dtype={"ts_code": str}, encoding="utf-8-sig")
                                if not df_allx.empty:
                                    # 若有 rank 列，默认 All 即全市场；若你希望严格 Top-K，可在这里进一步 head(K)
                                    allow_set = set(df_allx["ts_code"].astype(str))
                            if ddir.exists():
                                for p in ddir.glob("*.json"):
                                    try:
                                        j = json.loads(p.read_text(encoding="utf-8-sig"))
                                    except Exception:
                                        continue
                                    ts2 = str(j.get("ts_code", "")).strip()
                                    if not ts2:
                                        continue
                                    if (allow_set is not None) and (ts2 not in allow_set):
                                        continue
                                    sm = j.get("summary") or {}
                                    sc = float(sm.get("score", 0.0))
                                    hit = False
                                    for rr in j.get("rules", []) or []:
                                        # 命中标准：该规则名相等，且当日加分>0 或 ok=True
                                        if rr.get("name") == chosen_rule_name and (float(rr.get("add", 0.0)) > 0.0 or bool(rr.get("ok"))):
                                            hit = True
                                            break
                                    if hit:
                                        rows.append({"ts_code": ts2, "score": sc})
                            df_hit = pd.DataFrame(rows)
                            if df_hit.empty:
                                st.info("当日无标的命中该规则。")
                            else:
                                # 排序：score 降序；分数相同按代码升序（作为最终 tiebreak）
                                df_hit = df_hit.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
                                st.dataframe(df_hit, use_container_width=True, height=420)
                                # 导出 & 复制
                                codes = df_hit["ts_code"].astype(str).tolist()
                                txt = _codes_to_txt(codes, st.session_state["export_pref"]["style"], st.session_state["export_pref"]["with_suffix"])
                                copy_txt_button(txt, label="📋 复制筛选结果（按当前预览）", key=f"copy_rulehits_{ref_real}_{chosen_rule_name}")
                                _download_txt("导出筛选结果 TXT", txt, f"rulehits_{normalize_ts(chosen_rule_name)}_{ref_real}.txt", key="dl_rule_hits")
                        except Exception as e:
                            st.error(f"筛选失败：{e}")


# ================== 规则编辑 ==================
with tab_rules:
    st.subheader("③ 规则编辑（仅当前进程临时生效，保存不会改 config.py）")
    colL, colR = st.columns([2,1])
    default_text = json.dumps(st.session_state["rules_obj"], ensure_ascii=False, indent=2)
    with colL:
        text = st.text_area("规则 JSON（含 prescreen / rules）", value=default_text, height=300)
    with colR:
        up = st.file_uploader("从文件载入 JSON", type=["json"])
        if up:
            try:
                st.session_state["rules_obj"] = json.loads(up.read().decode("utf-8-sig"))
                st.success("已载入至编辑器（未应用）。")
            except Exception as e:
                st.error(f"载入失败：{e}")
        if st.button("✅ 校验并应用（仅当前进程）", use_container_width=True):
            try:
                st.session_state["rules_obj"] = json.loads(text)
                st.success("校验通过，已应用。运行评分时将使用该临时规则。")
            except Exception as e:
                st.error(f"JSON 解析失败：{e}")
        RULES_JSON = Path("./rules.json")
        if st.button("💾 导出到 rules.json", use_container_width=True):
            try:
                RULES_JSON.write_text(text, encoding="utf-8-sig")
                st.success(f"已写入 {RULES_JSON}")
            except Exception as e:
                st.error(f"写入失败：{e}")

        if st.button("📥 从 rules.json 载入", use_container_width=True):
            try:
                t = RULES_JSON.read_text(encoding="utf-8-sig")
                st.session_state["rules_obj"] = json.loads(t)
                st.success("已加载到编辑器（别忘了再点“校验并应用”）")
            except Exception as e:
                st.error(f"读取失败：{e}")

        if st.button("↩️ 还原为 config 默认", use_container_width=True):
            st.session_state["rules_obj"] = {
                "prescreen": getattr(cfg, "SC_PRESCREEN_RULES", []),
                "rules": getattr(cfg, "SC_RULES", []),
            }
            st.experimental_rerun()
    st.divider()
    with st.container(border=True):
        st.markdown("### 🧪 策略测试器（单条规则）")
        with st.expander("使用方法 / 字段说明", expanded=False):
            st.markdown("**必填字段**")
            st.markdown(
                "- `name`：规则名称\n"
                "- `when`：通达信风格布尔表达式\n"
                "- `timeframe`：`D`/`W`/`M`\n"
                "- `window`：回看窗口（整数）\n"
                "- `scope`：命中口径（如 `ANY`/`ALL`/`LAST`/`COUNT>=k`/`CONSEC>=m`/`ANY_5`/`ALL_3`/`EACH`/`RECENT`/`DIST`/`NEAR` 等)"
            )

            st.markdown("**可选字段**")
            st.markdown(
                "- `points`：命中加分（`EACH/PERBAR` 为“每K * points”）\n"
                "- `explain`：命中理由文案（默认展示；可用 `show_reason=false` 隐藏）\n"
                "- `dist_points`：`RECENT/DIST/NEAR` 的距离计分表，支持两种等价写法："
            )
            st.caption("列表写法")
            st.code('[[0,0,3], [1,2,2], [3,5,1]]', language="json")
            st.caption("字典写法")
            st.code('[{"min":0,"max":0,"points":3}, {"min":1,"max":2,"points":2}, {"min":3,"max":5,"points":1}]', language="json")
            st.markdown("- `clauses`：复合子句（与逻辑），子句里也可单独设置 `timeframe/window/scope/when`")

            st.markdown("**命中日期口径差异（很重要）**")
            st.markdown(
                "- `EACH/PERBAR`：逐K计数；`hit_date` 取窗口内最后一次为真；`hit_dates` 为窗口内所有为真。\n"
                "- `RECENT/DIST/NEAR`：先算最近一次命中的 `lag`，再由索引回推 `hit_date`；同时列出 `hit_dates`。\n"
                "- 其它（`ANY/ALL/LAST/COUNT/CONSEC/ANY_n/ALL_n` 等）：按布尔命中；`hit_dates` 为窗口内所有为真。"
            )
        # —— 临时指标编辑窗（默认给一条可用模板） ——
        default_rule = {
            "name": "测试：近3日放量并站上MA20",
            "timeframe": "D",
            "window": 60,
            "scope": "RECENT",
            "when": "VOL>MA(VOL,5) AND CLOSE>MA(CLOSE,20)",
            "points": 3,
            "dist_points": [
                {"min": 0, "max": 0, "points": 3},
                {"min": 1, "max": 2, "points": 2},
                {"min": 3, "max": 5, "points": 1}
            ],
            "explain": "近N日放量且站上MA20"
        }
        rule_raw = st.text_area("临时规则（JSON）", value=json.dumps(default_rule, ensure_ascii=False, indent=2),
                                height=260, key="tester_rule_json")

        colA, colB, colC = st.columns([1.2, 1.2, 1.2])
        with colA:
            ref_in = st.text_input("参考日（留空=自动最新）", value="")
        with colB:
            ts_in = st.text_input("个股代码", value="")
        with colC:
            uni_choice = st.selectbox("名单", ["全市场","仅白名单","仅黑名单","仅特别关注榜"], index=0, key="tester_uni")
        _uni_map = {"全市场":"all", "仅白名单":"white", "仅黑名单":"black", "仅特别关注榜":"attention"}
        
        colD, colE = st.columns([1,1])
        with colD:
            run_btn = st.button("在个股中运行", use_container_width=True)
        with colE:
            run_all_btn = st.button("在名单中运行", use_container_width=True, key="tester_run_all")

        refD_all = ref_in.strip() or _pick_latest_ref_date() or ""
        if run_all_btn:
            try:
                rule = json.loads(rule_raw)
                when_expr, tf, win, scope = _rule_to_screen_args(rule)

                # 2) 调 tdx_screen，注意不写黑白名单，仅预览
                df_sel = se.tdx_screen(
                    when_expr,
                    ref_date=refD_all.strip() or None,
                    timeframe=tf,
                    window=int(win),
                    scope=scope,                              # 支持 LAST/ANY/ALL/COUNT>=k/CONSEC>=m/ANY_n/ALL_n:contentReference[oaicite:7]{index=7}
                    universe=_uni_map.get(uni_choice,"all"),  # all/white/black/attention
                    write_white=False,
                    write_black_rest=False,
                    return_df=True
                )
                # 3) 友好列
                if not df_sel.empty:
                    df_sel["board"] = df_sel["ts_code"].map(market_label)
                    st.success(f"命中 {len(df_sel)} 只；参考日：{df_sel['ref_date'].iloc[0] if 'ref_date' in df_sel.columns and len(df_sel)>0 else (refD_all or '自动')}")
                    st.dataframe(df_sel, use_container_width=True, height=480)
                    # 导出
                    csv_bytes = df_sel.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("导出结果 CSV", data=csv_bytes, file_name="tester_screen_all.csv", mime="text/csv", use_container_width=True)
                else:
                    st.info("未命中。")

                # 4) （可选）点名看“单票明细” —— 和上面明细页同口径
                st.markdown("###### 查看某只股票的明细（与单票测试相同口径）")
                ts_pick_ori = st.text_input("输入 ts_code 查看（如 000001.SZ）", value="")
                ts_pick = normalize_ts((ts_pick_ori or "").strip())
                if ts_pick:
                    # 根据规则窗口估算最小读取区间，然后读单票数据
                    ref = (df_sel["ref_date"].iloc[0] if (not df_sel.empty and "ref_date" in df_sel.columns) else (refD_all or None)) or se._pick_ref_date()
                    start = se._compute_read_start(ref)  # 保证窗口足够:contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}
                    dfD = se._read_stock_df(ts_pick.strip(), start, ref, columns=["trade_date","open","high","low","close","vol","amount"])
                    per_rows = se._build_per_rule_detail(dfD, ref)  # 返回含 ok/add/cnt/lag/hit_date/hit_dates 等字段:contentReference[oaicite:11]{index=11}
                    df_detail = pd.DataFrame(per_rows)
                    # 只显示当前这条规则（按 name 或 when 关键字都可以）
                    name_key = str(rule.get("name","")).strip()
                    if name_key:
                        df_detail = df_detail[df_detail["name"] == name_key]
                    st.dataframe(df_detail, use_container_width=True)
            except Exception as e:
                st.error(f"运行失败：{e}")

        if run_btn:
            try:
                rule = json.loads(rule_raw or "{}")
            except Exception as e:
                st.error(f"规则 JSON 解析失败：{e}")
                st.stop()

            ts_code = normalize_ts((ts_in or "").strip())
            if not ts_code:
                st.error("请填写测试代码")
                st.stop()

            # 暂存并临时替换全局规则集，只跑这一条
            bak_rules = getattr(se, "SC_RULES", None)
            bak_pres  = getattr(se, "SC_PRESCREEN_RULES", None)
            try:
                setattr(se, "SC_RULES", [rule])
                setattr(se, "SC_PRESCREEN_RULES", [])

                # 参考日 & 读取窗口 / 列裁剪
                ref_use = (ref_in or "").strip() or (_pick_latest_ref_date() or "")
                if not ref_use:
                    st.error("未找到参考日：请先在“排名”页签跑一次或手填。")
                    st.stop()

                # 估算读取起点（按本条规则的 timeframe+window）
                start = se._start_for_tf_window(ref_use, str(rule.get("timeframe", "D")), int(rule.get("window", getattr(se, "SC_LOOKBACK_D", 60))))
                # 按需裁列（会扫描 when 里的列名，例如 j/vr）：
                columns = se._select_columns_for_rules()

                # 读取单票数据
                df = se._read_stock_df(ts_code, start, ref_use, columns)
                if df is None or df.empty:
                    st.warning("数据为空或读取失败。")
                    st.stop()

                # 保持与正式评分一致的表达式上下文
                try:
                    if tdx is not None:
                        tdx.EXTRA_CONTEXT.update(se.get_eval_env(ts_code, ref_use))
                except Exception:
                    pass

                # 直接用与“详情页”一致的构造函数得到逐规则明细
                rows = se._build_per_rule_detail(df, ref_use)
                if not rows:
                    st.info("未产生任何命中/细节。")
                    st.stop()

                import pandas as pd, numpy as np
                df_rules = pd.DataFrame(rows).copy()

                # —— “最后命中距今天数” 与详情页一致的口径：优先用 lag，其次 hit_date，再退 hit_dates 最末 —— 
                ref_dt = pd.to_datetime(ref_use)
                def _last_days(row: dict):
                    lag = row.get("lag")
                    if isinstance(lag, (int, float)) and not pd.isna(lag):
                        return int(lag)
                    hd = row.get("hit_date")
                    if isinstance(hd, str) and hd:
                        try: return int((ref_dt - pd.to_datetime(hd)).days)
                        except Exception: return None
                    hds = row.get("hit_dates") or []
                    if isinstance(hds, list) and hds:
                        try: return int((ref_dt - pd.to_datetime(hds[-1])).days)
                        except Exception: return None
                    return None

                df_rules["last_hit_days"] = [ _last_days(r) for r in df_rules.to_dict("records") ]

                # 展示列（存在才展示）
                show_cols = [c for c in [
                    "name","scope","timeframe","window","period","ok","points","add","cnt","lag",
                    "hit_date","hit_count","hit_dates","last_hit_days","gate_ok","gate_when","explain"
                ] if c in df_rules.columns]

                st.markdown(f"**测试代码：{ts_code} · 参考日：{ref_use}**")
                st.dataframe(df_rules[show_cols], use_container_width=True, height=420)

            except Exception as e:
                st.error(f"测试失败：{e}")
            finally:
                # 还原全局
                if bak_rules is not None: setattr(se, "SC_RULES", bak_rules)
                if bak_pres  is not None: setattr(se, "SC_PRESCREEN_RULES", bak_pres)
# ================== 强度榜 ==================
with tab_attn:
    st.subheader("④ 强度榜")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        src = st.selectbox("来源", ["top","white","black","attention"], index=0)
        method = st.selectbox("方法", ["强度（带权）","次数（不带权）"], index=0)
    with c2:
        win_n = st.number_input("窗口天数 N", min_value=1, max_value=365, value=60)
        top_m = st.number_input("Top-M 过滤（仅统计前 M 名）", min_value=1, max_value=2000, value=50)
    with c3:
        weight = st.selectbox("时间权重", ["不加权","指数半衰","线性最小值"], index=0)
        out_n = st.number_input("输出 Top-N", min_value=1, max_value=1000, value=200)
    with c4:
        # date_end = st.text_input("结束日（YYYYMMDD；留空=自动最新）", value="")
        date_end = st.text_input("结束日（YYYYMMDD；留空=自动最新）", value="", key="attn_end_date")
        gen_btn = st.button("生成并预览", use_container_width=True)

    if gen_btn:
        try:
            # 1) 计算 start/end（按交易日）
            from parquet_viewer import asset_root, list_trade_dates
            from config import PARQUET_BASE, PARQUET_ADJ
            root = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
            days = _cached_trade_dates(PARQUET_BASE, PARQUET_ADJ)
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
            st.success(f"关注榜已生成：{csv_path}")
            df_a = pd.read_csv(csv_path)
            st.dataframe(df_a, use_container_width=True, height=480)
        except Exception as e:
            st.error(f"生成失败：{e}")


with tab_data:
    st.subheader("数据下载 / 浏览检查")
    # —— 参数区 ——
    with st.expander("参数设置", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            base = st.text_input("数据根目录 DATA_ROOT", value=str(getattr(cfg, "DATA_ROOT", getattr(cfg, "PARQUET_BASE", "./data"))))
            assets = st.multiselect("资产 ASSETS", ["stock","index"], default=list(getattr(dl, "ASSETS", ["stock","index"])) or ["stock","index"])            
        with c2:
            start_in = st.text_input("起始日 START_DATE (YYYYMMDD)", value=str(getattr(dl, "START_DATE", "20200101")))
            end_default = str(getattr(dl, "END_DATE", "today"))
            end_in = st.text_input("结束日 END_DATE ('today' 或 YYYYMMDD)", value=end_default)
        with c3:
            api_adj = st.selectbox("复权 API_ADJ", ["qfq","hfq","raw"], index={"qfq":0,"hfq":1,"raw":2}.get(str(getattr(dl,"API_ADJ","qfq")).lower(),0))
            latest = _latest_trade_date(base, api_adj)
            do_plain = st.checkbox("写入单股(不带指标)", value=bool(getattr(dl, "WRITE_SYMBOL_PLAIN", True)))
            do_ind   = st.checkbox("写入单股(含指标)", value=bool(getattr(dl, "WRITE_SYMBOL_INDICATORS", True)))
        with c4:
            fast_threads = st.number_input("FAST_INIT 并发", min_value=1, max_value=64, value=int(getattr(dl,"FAST_INIT_THREADS",8)))
            inc_threads  = st.number_input("增量下载线程", min_value=1, max_value=64, value=int(getattr(dl,"STOCK_INC_THREADS",8)))
            ind_workers  = st.number_input("指标重算线程(可选)", min_value=0, max_value=128, value=int(getattr(dl,"INC_RECALC_WORKERS", 32)))
        if latest:
            st.caption(f"当前 {api_adj} 最近交易日：{latest}")
        # # 额外开关
        # colx1, colx2 = st.columns(2)
        # with colx1:
            
        # with colx2:
        
    # 将参数落到模块
    end_use = _today_str() if str(end_in).strip().lower() == "today" else str(end_in).strip()
    start_use = str(start_in).strip()
    _apply_overrides(base, assets, start_use, end_use, api_adj, int(fast_threads), int(inc_threads), int(ind_workers) if ind_workers else None)
    dl.WRITE_SYMBOL_PLAIN = bool(do_plain)
    dl.WRITE_SYMBOL_INDICATORS = bool(do_ind)

    # —— 按钮区 ——
    tab_dl, tab_view = st.tabs(["下载/同步", "浏览/检查(app_pv)"])

    # === 下载/同步 ===
    with tab_dl:
        mode = st.radio("运行模式", ["首次建库(FAST_INIT)", "日常增量(NORMAL)"], index=0 if not _latest_trade_date(base, api_adj) else 1, horizontal=True)
        st.markdown(
            """
            - **FAST_INIT**：按股票并发全历史抓取 → 合并到 `stock/daily/*` →（可选）合并指标目录 →（可选）指数。
            - **NORMAL**：先合并 fast_init 缓存 → 股票增量 → 指数增量 → 指标增量重算与合并。
            """
        )

        # 一键
        c1, c2 = st.columns(2)
        with c1:
            run_all = st.button("🚀 一键运行", use_container_width=True, type="primary")
        with c2:
            dry = st.checkbox("仅打印日志（不执行）", value=False, help="仅用于预览参数")

        # 单步按钮
        st.markdown("—— 或按步骤执行 ——")
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1: b_fast = st.button("① 首次建库")
        with s2: b_merge = st.button("② Fast→Daily 合并")
        with s3: b_stock = st.button("③ 股票增量")
        with s4: b_index = st.button("④ 指数增量")
        with s5: b_indic = st.button("⑤ 指标合并/重算")

        # 执行逻辑
        if run_all or b_fast or b_merge or b_stock or b_index or b_indic:
            if dry:
                st.info(f"[DRY-RUN] base={base} assets={assets} adj={api_adj} range={start_use}~{end_use} fast_threads={fast_threads} inc_threads={inc_threads}")
            else:
                with st.status("执行中…", expanded=True) as status:
                    try:
                        if run_all:
                            if mode.startswith("首次"):
                                status.update(label="FAST_INIT 全量…")
                                _run_fast_init(end_use)
                                if "index" in set(assets):
                                    status.update(label="指数全量/补齐…")
                                    dl.sync_index_daily_fast(start_use, end_use, dl.INDEX_WHITELIST)
                            else:
                                status.update(label="合并 FastInit 缓存…")
                                _run_increment(start_use, end_use, do_stock=True, do_index=True, do_indicators=True)
                            status.update(label="完成", state="complete")
                            st.success("✅ 完成")
                        else:
                            # 单步
                            if b_fast:
                                status.update(label="首次建库（FAST_INIT）…")
                                _run_fast_init(end_use)
                            if b_merge:
                                status.update(label="合并 Fast→Daily …")
                                dl.duckdb_partition_merge()
                            if b_stock:
                                status.update(label="股票增量…")
                                dl.sync_stock_daily_fast(start_use, end_use, threads=dl.STOCK_INC_THREADS)
                            if b_index:
                                status.update(label="指数增量…")
                                dl.sync_index_daily_fast(start_use, end_use, dl.INDEX_WHITELIST)
                            if b_indic:
                                status.update(label="指标重算并合并…")
                                workers = getattr(dl, "INC_RECALC_WORKERS", None) or ((os.cpu_count() or 4) * 2)
                                dl.recalc_symbol_products_for_increment(start_use, end_use, threads=workers)
                            status.update(label="完成", state="complete")
                            st.success("✅ 完成")
                    except Exception as e:
                        st.error(f"运行失败：{e}")

    # === 浏览/检查（集成 app_pv 的核心功能） ===
    with tab_view:
        st.markdown("#### 概览 & 诊断 (来自 app_pv)")
        c1, c2 = st.columns([2, 1])
        with c1:
            try:
                df = apv.overview_table(base, api_adj)
                st.dataframe(df, use_container_width=True, height=360)
            except Exception as e:
                st.error(f"概览失败：{e}")
        with c2:
            try:
                info = apv.get_info(base, api_adj)
                st.text_area("概览（文本）", value=str(info), height=180)
                adv = apv.overview_advice(base, api_adj)
                st.markdown(adv)
            except Exception as e:
                st.error(f"诊断失败：{e}")

        st.markdown("---")
        st.caption("数据浏览由 parquet_viewer 支持，可在其他页或命令行使用更丰富的功能。")

# ================== 普通选股（TDX表达式） ==================
with tab_screen:
    st.subheader("⑤ 普通选股（TDX 表达式）")
    # exp = st.text_input("表达式（示例：CLOSE>MA(CLOSE,20) AND VOL>MA(VOL,5)）", value="")
    exp = st.text_input("表达式（示例：CLOSE>MA(CLOSE,20) AND VOL>MA(VOL,5)）", value="", key="screen_expr")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        level = st.selectbox("时间级别", ["D","W","M"], index=0)
    with c2:
        window = st.number_input("窗口长度", min_value=1, max_value=500, value=60)
    with c3:
        scope_logic = st.selectbox("命中范围(scope)", ["LAST","ANY","ALL","COUNT>=k","CONSEC>=m"], index=0)
    with c4:
        refD = st.text_input("参考日（可选，YYYYMMDD）", value="")
    uni_choice = st.selectbox("选股范围", ["全市场","仅白名单","仅黑名单","仅特别关注榜"], index=0)
    _uni_map = {"全市场":"all", "仅白名单":"white", "仅黑名单":"black", "仅特别关注榜":"attention"}
    run_screen = st.button("运行筛选并预览", use_container_width=True)

    if run_screen:
        try:
            if hasattr(se, "tdx_screen"):
                df_sel = se.tdx_screen(
                    exp,
                    ref_date=refD.strip() or None,
                    timeframe=level,
                    window=int(window),
                    scope=scope_logic,
                    universe=_uni_map.get(uni_choice, "all"),
                    write_white=False,
                    write_black_rest=False,
                    return_df=True
                )
                st.dataframe(df_sel, use_container_width=True, height=480)
                st.caption(f"共 {len(df_sel)} 只股票")
                if not df_sel.empty and "ts_code" in df_sel.columns:
                    txt = _codes_to_txt(df_sel["ts_code"].astype(str).tolist(),
                                        st.session_state["export_pref"]["style"],
                                        st.session_state["export_pref"]["with_suffix"])
                    _download_txt("导出命中代码 TXT", txt, "select.txt", key="dl_select")
            else:
                st.warning("未检测到 tdx_screen，实现后此页自动可用。")
        except Exception as e:
            st.error(f"筛选失败：{e}")

# ================== 工具箱 ==================
with tab_tools:
    st.subheader("⑥ 工具箱")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**自动补算最近 N 个交易日**")
        n_back = st.number_input("天数 N", min_value=1, max_value=100, value=20)
        # inc_today = st.checkbox("包含参考日当天", value=False,
        #                         help="勾选后窗口包含参考日（例如 N=5 → [ref-(N-1), ref]；未勾选则 [ref-N, ref-1]）")
        inc_today = st.checkbox("包含参考日当天", value=True,
                                 help="勾选后窗口包含参考日（例如 N=5 → [ref-(N-1), ref]；未勾选则 [ref-N, ref-1]）")
        do_force = st.checkbox("强制重建（覆盖已有）", value=False,
                               help="若之前失败留下了 0 字节文件或想重算，勾选此项。")

        go_fill = st.button("执行自动补算", use_container_width=True)
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
        go_fix = st.button("补齐缺失", use_container_width=True)
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
            do_scan = st.button("加载/刷新列表", key="btn_scan_inventory", use_container_width=True)
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
                    cal_root = asset_root(PARQUET_BASE, "stock", PARQUET_ADJ)
                    trade_dates = list_trade_dates(cal_root) or []
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
                                st.dataframe(_read_df(p).head(200), use_container_width=True, height=360)
                            else:
                                st.info("该日 All 文件不存在或为空。")
                        elif kind == "Top 排名":
                            p = _path_top(sel_date)
                            if p.exists() and p.stat().st_size > 0:
                                st.dataframe(_read_df(p).head(200), use_container_width=True, height=360)
                            else:
                                st.info("该日 Top 文件不存在或为空。")
                        else:
                            pdir = DET_DIR / sel_date
                            if pdir.exists():
                                st.info(f"{sel_date} 共有 {len(list(pdir.glob('*.json')))} 个详情 JSON。")
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
    st.subheader("⑦ 组合模拟 / 持仓（重做版）")
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
        if st.button("创建组合", use_container_width=True):
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
        if not ports:
            st.info("暂无组合，请先创建。")
            st.stop()
        # 以 name 排序
        ports_items = sorted(list(ports.items()), key=lambda kv: kv[1].name)
        names = [f"{p.name} ({pid[:6]})" for pid, p in ports_items]
        sel = st.selectbox("选择组合", options=list(range(len(ports_items))), format_func=lambda i: names[i], index=0)
        cur_pid, cur_pf = ports_items[int(sel)][0], ports_items[int(sel)][1]

    st.divider()

    # —— 录入成交（价格参考区间） ——
    st.markdown("**录入成交（带参考价区间）**")
    colx, coly, colz, colw = st.columns([1.2, 1.2, 1.2, 2])
    with colx:
        side = st.selectbox("方向", ["BUY","SELL"], index=0)
    with coly:
        d_exec = st.text_input("成交日（YYYYMMDD）", value=_pick_latest_ref_date() or "")
    with colz:
        ts = st.text_input("代码", value="")
    # 读取当日 O/H/L/C 作为参考
    ref_low = ref_high = px_open = px_close = None
    try:
        ts_norm = normalize_ts(ts) if ts else ""
        if ts_norm and d_exec:
            df_one = read_range(PARQUET_BASE, "stock", PARQUET_ADJ, ts_norm, d_exec, d_exec, columns=["open","high","low","close"])
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

    if st.button("记录成交", use_container_width=True, key="btn_rec_trade"):
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
    obs = st.text_input("观察日（YYYYMMDD；默认=最新交易日）", value=_pick_latest_ref_date() or "")
    if obs:
        # 回放估值（从组合创建日至观察日）
        # 我们用 read_nav() 读取结果
        try:
            # 执行估值
            pm.reprice_and_nav(cur_pid, date_start="19000101", date_end=str(obs), benchmarks=())
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
                    st.dataframe(cur_pos, use_container_width=True, height=300)
                else:
                    st.caption("观察日无持仓记录。")
            st.metric("组合市值", f"{(row.get('nav',1.0) * float(cur_pf.init_cash)):.0f}")
            st.metric("区间收益率", f"{(row.get('nav',1.0) - 1.0):.2%}")
            cols = [c for c in ["date","cash","position_mv","nav","ret_d","max_dd"] if c in nav_df.columns]
            st.dataframe(nav_df[cols].tail(5), use_container_width=True)
            st.markdown("**净值曲线（NAV）**")
            try:
                st.line_chart(nav_df.set_index("date")["nav"])
            except Exception:
                pass
        else:
            st.caption("暂无净值数据（可能还未有成交或行情数据缺失）")
# ================== 统计（普通页签） ==================

with tab_stats:
    st.subheader("⑧ 统计")
    sub_tabs = st.tabs(["跟踪（Tracking）", "异动（Surge）", "共性（Commonality）"])

    # --- Tracking ---
    with sub_tabs[0]:
        refT = st.text_input("参考日", value=_pick_latest_ref_date() or "", key="ref_1")
        wins = st.text_input("窗口集合（逗号）", value="1,2,3,5,10,20")
        bench = st.text_input("基准代码（逗号，可留空）", value="")
        gb_board = st.checkbox("分板块汇总", value=True)
        if st.button("运行 Tracking", use_container_width=True):
            try:
                from stats_core import run_tracking
                wlist = [int(x) for x in wins.split(",") if x.strip().isdigit()]
                blist = [s.strip() for s in bench.split(",") if s.strip()] or None
                tr = run_tracking(refT, wlist, benchmarks=blist, score_df=None, group_by_board=gb_board, save=True)
                st.dataframe(tr.summary, use_container_width=True, height=420)
                st.caption("已落盘到 output/tracking/<ref>/ ，明细 detail 可据此深挖。")
            except Exception as e:
                st.error(f"Tracking 失败：{e}")
        # === 跟踪增强：前日排行 / 名单 / 指标是否触发 / 后续涨幅 ===
        with st.expander("可选：自选指标打勾（TDX 表达式，逗号分隔；仅用于打勾，不影响样本）", expanded=False):
            track_exprs = st.text_input("表达式", value="", key="track_exprs")
            track_tf = st.selectbox("时间框架", ["D","W","M"], index=0, key="track_tf")
            track_win = st.number_input("窗口长度", min_value=10, max_value=240, value=60, step=5, key="track_win")
            track_scope = st.selectbox("作用域", ["ANY","LAST"], index=1, key="track_scope")

        if st.button("生成跟踪表（含前日排行/名单/指标勾选/后续涨幅）", use_container_width=True):
            try:
                from stats_core import run_tracking
                import scoring_core as se
                # 1) 基础 tracking 明细
                wlist = [int(x) for x in wins.split(",") if x.strip().isdigit()]
                blist = [s.strip() for s in bench.split(",") if s.strip()] or None
                tr2 = run_tracking(refT, wlist, benchmarks=blist, score_df=None, group_by_board=gb_board, save=True)
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

                # 4) 自选指标是否触发（命中即 True）
                if track_exprs.strip() and hasattr(se, "tdx_screen"):
                    expr_list = [e.strip() for e in track_exprs.split(",") if e.strip()]
                    for i, e in enumerate(expr_list, start=1):
                        try:
                            hit_df = se.tdx_screen(e, ref_date=str(refT), timeframe=str(track_tf),
                                                   window=int(track_win), scope=str(track_scope),
                                                   return_df=True, write_white=False, write_black_rest=False)
                            hits = set(hit_df["ts_code"].astype(str)) if (hasattr(hit_df, "columns") and "ts_code" in hit_df.columns) else set()
                            detail[f"ind_hit_{i}"] = detail["ts_code"].astype(str).map(lambda c: c in hits)
                        except Exception:
                            detail[f"ind_hit_{i}"] = False

                # 5) 展示所需列
                show_cols = [c for c in [
                    "ts_code","rank","rank_tminus_1",
                    "in_whitelist","in_blacklist","in_attention",
                    *[c for c in detail.columns if c.startswith("ind_hit_")],
                    *[c for c in detail.columns if c.startswith("ret_fwd_")]
                ] if c in detail.columns]
                st.dataframe(detail[show_cols].sort_values(["rank"]).reset_index(drop=True),
                             use_container_width=True, height=460)
                st.caption("ret_fwd_N = 未来 N 日涨幅（Tracking 已计算）；名单列来自 cache/attention；ind_hit_* 为你输入的指标是否触发。")
            except Exception as e:
                st.error(f"生成失败：{e}")


    # --- Surge ---
    with sub_tabs[1]:
        refS = st.text_input("参考日", value=_pick_latest_ref_date() or "", key="surge_ref")
        mode = st.selectbox("模式", ["today","rolling"], index=1, key="mode_1")
        rolling_days = st.number_input("rolling 天数", min_value=2, max_value=20, value=5, key="rolling_1")
        sel_type = st.selectbox("选样", ["top_n","top_pct"], index=0)
        sel_val = st.number_input("阈值（N或%）", min_value=1, max_value=1000, value=200)
        retros = st.text_input("回看天数集合（逗号）", value="1,2,3,4,5")
        split_label = st.selectbox("分组口径", ["600/000/科创北(3组)", "主vs其他", "各板块"], index=0, key="split_1")
        split = {"600/000/科创北(3组)":"combo3", "主vs其他":"main_vs_others", "各板块":"per_board"}[split_label]
        if st.button("运行 Surge", use_container_width=True):
            try:
                from stats_core import run_surge
                rlist = [int(x) for x in retros.split(",") if x.strip().isdigit()]
                sr = run_surge(ref_date=refS, mode=mode, rolling_days=int(rolling_days),
                               selection={"type":sel_type,"value":int(sel_val)},
                               retro_days=rlist, split=split, score_df=None, save=True)
                
                # 可选：对当日样本做指标勾选
                with st.expander("可选：对当日样本打勾（TDX 表达式，逗号分隔）", expanded=False):
                    surge_exprs = st.text_input("表达式", value="", key="surge_exprs")
                    surge_tf = st.selectbox("时间框架", ["D","W","M"], index=0, key="surge_tf")
                    surge_win = st.number_input("窗口长度", min_value=10, max_value=240, value=60, step=5, key="surge_win")
                    surge_scope = st.selectbox("作用域", ["ANY","LAST"], index=1, key="surge_scope")

                table = sr.table.copy()
                if "ts_code" in table.columns and surge_exprs.strip():
                    import scoring_core as se
                    expr_list = [e.strip() for e in surge_exprs.split(",") if e.strip()]
                    for i, e in enumerate(expr_list, start=1):
                        try:
                            hit_df = se.tdx_screen(e, ref_date=str(refS), timeframe=str(surge_tf),
                                                   window=int(surge_win), scope=str(surge_scope),
                                                   return_df=True, write_white=False, write_black_rest=False)
                            hits = set(hit_df["ts_code"].astype(str)) if (hasattr(hit_df, "columns") and "ts_code" in hit_df.columns) else set()
                            table[f"ind_hit_{i}"] = table["ts_code"].astype(str).map(lambda c: c in hits)
                        except Exception:
                            table[f"ind_hit_{i}"] = False
                st.dataframe(table, use_container_width=True, height=420)

                st.caption("各分组文件已写入 output/surge_lists/<ref>/ 。")
            except Exception as e:
                st.error(f"Surge 失败：{e}")

    # --- Commonality ---
    with sub_tabs[2]:
        refC = st.text_input("参考日", value=_pick_latest_ref_date() or "", key="common_ref")
        retro_day = st.number_input("观察日前移 d（retro）", min_value=1, max_value=20, value=1)
        modeC = st.selectbox("模式", ["rolling","today"], index=0, key="mode_2")
        rollingC = st.number_input("rolling 天数", min_value=2, max_value=20, value=5, key="rolling_2")
        selC = st.number_input("样本 Top-N", min_value=10, max_value=1000, value=200)
        splitC = st.selectbox("分组口径", ["main_vs_others","per_board"], index=0, key="split_2")
        bg = st.selectbox("背景集", ["all","same_group"], index=0)
        if st.button("运行 Commonality", use_container_width=True):
            try:
                from stats_core import run_commonality
                cr = run_commonality(ref_date=refC, retro_day=int(retro_day), mode=modeC,
                                     rolling_days=int(rollingC), selection={"type":"top_n","value":int(selC)},
                                     split=splitC, background=bg, save=True)
                st.dataframe(cr.dataset.head(200), use_container_width=True, height=420)
                st.caption("分析集/报告已写入 output/commonality/<ref>/ ...")
            except Exception as e:
                st.error(f"Commonality 失败：{e}")
# ================== 日志 ==================
with tab_logs:
    st.subheader("⑨ 日志")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**score.log（尾部 400 行）**")
        st.code(_tail(LOG_DIR / "score.log", 400), language="bash")
    with col2:
        st.markdown("**score_ui.log（尾部 400 行）**")
        st.code(_tail(LOG_DIR / "score_ui.log", 400), language="bash")