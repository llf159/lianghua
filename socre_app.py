# streamlit_score_workbench.py
# -*- coding: utf-8 -*-
import os, json, io, time
from pathlib import Path
from collections import deque

import pandas as pd
import streamlit as st

import score_engine as se        # 用于运行评分 & 动态覆写配置
import config as cfg             # 读取默认配置（base、输出路径等）
from utils import normalize_ts
# ----------------- UI 日志（独立到 log/score_ui.log） -----------------
import logging
from logging.handlers import TimedRotatingFileHandler
os.makedirs("./log", exist_ok=True)
UI_LOGGER = logging.getLogger("score.ui")
if not UI_LOGGER.handlers:
    UI_LOGGER.setLevel(logging.DEBUG)
    fh = TimedRotatingFileHandler("./log/score_ui.log", when="midnight", interval=1, backupCount=7, encoding="utf-8-sig")
    fmt = logging.Formatter("%(asctime)s %(levelname)s [ui] %(message)s")
    fh.setFormatter(fmt)
    UI_LOGGER.addHandler(fh)

# ----------------- Helpers -----------------
OUT_DIR = getattr(cfg, "SC_OUTPUT_DIR", "./output/score")
DETAIL_DIR = os.path.join(OUT_DIR, "details")
TOP_DIR = os.path.join(OUT_DIR, "top")
ATTN_DIR = os.path.join(OUT_DIR, "attention")
RULES_JSON = Path("./rules.json")

def _tail(path: str, n: int = 400) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            dq = deque(f, maxlen=n)
        return "".join(dq)
    except Exception:
        return ""


def _load_top_latest() -> tuple[str|None, pd.DataFrame]:
    try:
        files = sorted(Path(TOP_DIR).glob("score_top_*.csv"))
        if not files:
            return None, pd.DataFrame()
        f = files[-1]
        return f.stem[-8:], pd.read_csv(f, encoding="utf-8-sig")
    except Exception:
        return None, pd.DataFrame()


def _find_detail(ts: str, ref: str) -> Path|None:
    p = Path(DETAIL_DIR)/f"{ts}_{ref}.json"
    return p if p.exists() else None


def _apply_runtime_overrides(rules_obj: dict|None, topk: int|None, tie: str|None, max_workers: int|None, attn_on: bool|None, universe: str | list[str] | None = None):
    # 动态把编辑后的规则/参数覆盖到引擎命名空间
    if rules_obj:
        pres = rules_obj.get("prescreen")
        rules = rules_obj.get("rules")
        if pres is not None:
            se.SC_PRESCREEN_RULES = pres
        if rules is not None:
            se.SC_RULES = rules
        UI_LOGGER.info("已应用编辑器中的规则到引擎: pres=%s, rules=%s",
                       (len(pres) if isinstance(pres, list) else None),
                       (len(rules) if isinstance(rules, list) else None))
    if topk is not None:
        se.SC_TOP_K = int(topk)
    if tie is not None:
        se.SC_TIE_BREAK = tie
    if max_workers is not None:
        se.SC_MAX_WORKERS = int(max_workers)
    if attn_on is not None:
        se.SC_ATTENTION_ENABLE = bool(attn_on)
    if universe is not None:
        se.SC_UNIVERSE = universe

# ----------------- UI -----------------
st.set_page_config(page_title="评分工作台", layout="wide")
st.title("评分工作台")

tab_run, tab_detail, tab_rules, tab_logs, tab_attn, tab_screen = st.tabs(
    ["运行评分", "单票详情", "规则编辑", "日志查看", "关注榜", "普通选股"]
)

# ---- 运行评分 ----
with tab_run:
    st.subheader("运行一次评分")
    c1, c2, c3, c4, c5 = st.columns([1.1,1,1,1,1])
    ref_inp = c1.text_input("参考日 YYYYMMDD（留空=自动推断最新）", "")
    topk_inp = c2.number_input("Top-K", min_value=1, max_value=5000, value=int(getattr(cfg, "SC_TOP_K", 300)))
    tie_inp  = c3.selectbox("Tie-break", options=["kdj_j_asc","none"], index=0 if getattr(cfg, "SC_TIE_BREAK","kdj_j_asc").lower()!="none" else 1)
    mw_inp   = c4.number_input("最大并行数", min_value=1, max_value=128, value=int(getattr(cfg,"SC_MAX_WORKERS",8) or 8))
    attn_inp = c5.checkbox("完成后生成关注榜", value=bool(getattr(cfg, "SC_ATTENTION_ENABLE", True)))
    uni_label = "评分范围"
    _uni_map  = {"全市场":"all", "仅白名单":"white", "仅黑名单":"black", "仅特别关注榜":"attention"}
    uni_inp   = st.selectbox(uni_label, list(_uni_map.keys()), index=0)

    st.markdown("— 如果你在“规则编辑”里改了规则，本页点击运行前会自动套用。")
    run_btn = st.button("🚀 运行评分", use_container_width=True)

    if run_btn:
        UI_LOGGER.info("用户触发评分: ref=%s topK=%s tie=%s workers=%s attn=%s",
                       ref_inp or "<auto>", topk_inp, tie_inp, mw_inp, attn_inp)
        # 尝试读取 session 中的编辑器规则
        rules_obj = st.session_state.get("rules_obj")
        _apply_runtime_overrides(rules_obj, topk_inp, tie_inp, mw_inp, attn_inp, _uni_map.get(uni_inp, "all"))

        with st.spinner("执行中..."):
            try:
                out_path = se.run_for_date(ref_inp.strip() or None)
                UI_LOGGER.info("评分完成: %s", out_path)
                st.success(f"评分完成：{out_path}")
                st.session_state["last_out_csv"] = out_path
                st.session_state["last_ref_date"] = Path(out_path).stem[-8:]
            except Exception as e:
                UI_LOGGER.exception("评分异常: %s", e)
                st.error(f"评分失败：{e}")

    # 展示结果（若存在最近一次）
    ref0, df0 = _load_top_latest()
    out_csv = st.session_state.get("last_out_csv")
    if out_csv and Path(out_csv).exists():
        df_show = pd.read_csv(out_csv, encoding="utf-8-sig")
        st.dataframe(df_show, use_container_width=True, height=480)
    elif not df0.empty:
        st.caption(f"最近一次结果：{ref0}")
        st.dataframe(df0, use_container_width=True, height=480)
    else:
        st.info("暂无 Top-K 输出。")
    # —— 新增：导出“前N名六位数字，空格分隔” —— 
    df_for_export = None
    ref_for_export = None
    if out_csv and Path(out_csv).exists():
        df_for_export = df_show
        ref_for_export = Path(out_csv).stem[-8:]
    elif not df0.empty:
        df_for_export = df0
        ref_for_export = ref0

    if df_for_export is not None and not df_for_export.empty:
        cexp1, cexp2 = st.columns([1,3])
        maxN = int(len(df_for_export))
        n_export = cexp1.number_input("导出前N（六位空格）", min_value=1, max_value=maxN, value=min(50, maxN))
        if cexp2.button("📤 导出前N（六位空格）", use_container_width=True):
            codes = (
                df_for_export["ts_code"].astype(str)
                .str.split(".").str[0].str.zfill(6)
                .head(int(n_export)).tolist()
            )
            text_out = " ".join(codes)
            os.makedirs(TOP_DIR, exist_ok=True)
            fname = os.path.join(TOP_DIR, f"codes_top{int(n_export)}_{ref_for_export}.txt")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(text_out + "\n")
            st.success(f"已导出：{fname}")
            st.code(text_out, language="text")
            st.download_button("下载该清单到本地", data=text_out, file_name=f"codes_top{int(n_export)}_{ref_for_export}.txt", mime="text/plain")


# ---- 单票详情 ----
with tab_detail:
    st.subheader("查看单票评分明细")
    # 选择参考日 + 代码
    # 先尝试使用最近一次 Top-K
    ref_date = st.session_state.get("last_ref_date")
    if not ref_date:
        ref_date, df_top = _load_top_latest()
    else:
        df_top = pd.read_csv(st.session_state["last_out_csv"], encoding="utf-8-sig") if st.session_state.get("last_out_csv") else pd.DataFrame()

    c1, c2 = st.columns([1,2])
    ref_sel = c1.text_input("参考日", value=(ref_date or ""))
    codes = df_top["ts_code"].tolist() if not df_top.empty else []
    code_sel = c2.selectbox("选择代码（首选来自 Top-K）", options=codes or [], index=0 if codes else None, placeholder="没有 Top-K？也可直接手输↓")
    code_typed_ori = st.text_input("或手动输入代码", value=(code_sel or ""))
    code_typed = normalize_ts(code_typed_ori)

    if ref_sel and code_typed:
        p = _find_detail(code_typed, ref_sel)
        if p:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
            colA, colB = st.columns([1,1])
            with colA:
                st.write("总览")
                st.json(data.get("summary", {}))
            with colB:
                st.write("高亮/缺点")
                summ = data.get("summary", {})
                st.write({"highlights": summ.get("highlights", []), "drawbacks": summ.get("drawbacks", [])})
            st.write("逐规则明细（可筛）")
            df_rules = pd.DataFrame(data.get("rules", []))
            st.dataframe(df_rules, use_container_width=True, height=500)
        else:
            st.warning("未找到明细 JSON（你已按上面的最小改动写入了吗？），先展示 Top-K 里的摘要：")
            if not df_top.empty:
                row = df_top[df_top["ts_code"]==code_typed].head(1)
                if not row.empty:
                    st.json(row.iloc[0].to_dict())
            st.info("可在“运行评分”页跑一次，或切到“规则编辑”页修改再跑。")

# ---- 规则编辑 ----
with tab_rules:
    st.subheader("规则编辑（JSON）")
    st.caption("编辑后点【应用到引擎】即可，不必改 config.py；也可导出/导入 rules.json。支持 prescreen & rules 两组。")

    # 初始填充
    default_obj = {
        "prescreen": getattr(cfg, "SC_PRESCREEN_RULES", []),
        "rules": getattr(cfg, "SC_RULES", [])
    }
    text = st.text_area("JSON 规则对象", value=json.dumps(default_obj, ensure_ascii=False, indent=2), height=420)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    if c1.button("✅ 校验并应用到引擎", use_container_width=True):
        try:
            obj = json.loads(text)
            st.session_state["rules_obj"] = obj
            _apply_runtime_overrides(obj, None, None, None, None)
            UI_LOGGER.info("规则已应用")
            st.success("规则已应用（仅对当前会话有效）")
        except Exception as e:
            st.error(f"解析失败：{e}")

    if c2.button("💾 导出到 rules.json", use_container_width=True):
        try:
            RULES_JSON.write_text(text, encoding="utf-8-sig")
            UI_LOGGER.info("规则已导出 rules.json")
            st.success(f"已写入 {RULES_JSON}")
        except Exception as e:
            st.error(f"写入失败：{e}")

    if c3.button("📥 从 rules.json 载入", use_container_width=True):
        try:
            t = RULES_JSON.read_text(encoding="utf-8-sig")
            st.session_state["rules_obj"] = json.loads(t)
            st.success("已加载到编辑器（别忘了再点“校验并应用”）")
        except Exception as e:
            st.error(f"读取失败：{e}")

    if c4.button("↩️ 还原为 config 默认", use_container_width=True):
        st.session_state.pop("rules_obj", None)
        st.experimental_rerun()

# ---- 日志查看 ----
with tab_logs:
    st.subheader("日志查看")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("引擎日志 ./log/score.log")
        st.text_area("score.log（末尾400行）", value=_tail("./log/score.log", 400), height=420)
    with col2:
        st.caption("UI 日志 ./log/score_ui.log")
        st.text_area("score_ui.log（末尾400行）", value=_tail("./log/score_ui.log", 400), height=420)
    if st.button("🔄 刷新日志", use_container_width=True):
        UI_LOGGER.debug("用户刷新日志")
        st.experimental_rerun()

# ---- 关注榜 ----
with tab_attn:
    st.subheader("特别关注榜")
    c1, c2, c3 = st.columns([1,1,2])
    src = c1.selectbox("统计来源", options=["top","white","black"], index=0)
    window = c2.number_input("窗口最近 N 个交易日", min_value=1, max_value=120, value=int(getattr(cfg,"SC_ATTENTION_WINDOW_D",20)))
    btn_build = c3.button("⚡ 生成当前参考日的关注榜（run_for_date 也会自动生成）")

    if btn_build:
        try:
            # 没有传 start/end 表示用 se 内部的 _trade_span 推断
            out = se.build_attention_rank(start=None, end=None, source=src, min_hits=int(getattr(cfg,"SC_ATTENTION_MIN_HITS",2)),
                                          topN=int(getattr(cfg,"SC_ATTENTION_TOP_K",200)), write=True)
            UI_LOGGER.info("手动生成关注榜: %s", out)
            st.success(f"已写入：{out}")
        except Exception as e:
            st.error(f"失败：{e}")

    # 预览最近生成的关注榜
    try:
        atn_files = sorted(Path(ATTN_DIR).glob("attention_*.csv"))
        if atn_files:
            st.caption(f"最近生成：{atn_files[-1].name}")
            st.dataframe(pd.read_csv(atn_files[-1], encoding="utf-8-sig"), use_container_width=True, height=420)
        else:
            st.info("尚未生成关注榜。")
    except Exception as e:
        st.warning(f"预览失败：{e}")


with tab_screen:
    st.subheader("普通选股（类通达信）")
    st.caption("示例：`C>O AND COUNT(C>MA(C,5),10)>=3` ；支持 `LAST/ANY/ALL/COUNT>=k/CONSEC>=m` 等 scope 写法")

    colA, colB, colC = st.columns([2,1,1])
    with colA:
        when = st.text_area("当日/窗口表达式（TDX风格）", value="C>O AND C>MA(C,5)")
    with colB:
        timeframe = st.selectbox("时间级别", ["D","W","M"], index=0)
        window = st.number_input("窗口长度", min_value=5, max_value=500, value=60, step=5)
    with colC:
        scope = st.text_input("命中范围(scope)", value="LAST", help="可填：LAST / ANY / ALL / COUNT>=k / CONSEC>=m")

    ref_hint = se._pick_ref_date()
    st.info(f"参考日（自动）：{ref_hint}；若需指定，请在表达式里用相对条件或去“运行评分”调整。")

    c1, c2, c3 = st.columns([1,1,2])
    # with c1:
    #     write_white = st.checkbox("写白名单", True)
    # with c2:
    #     write_black_rest = st.checkbox("未命中写黑名单(慎用)", False)
    run = st.button("运行选股", type="primary", use_container_width=True)

    # 普通选股：默认不写黑白名单，只“在名单里选”
    with c1:
        write_white = st.checkbox("写白名单(命中)", False, help="普通选股默认不写名单")
    with c2:
        write_black_rest = st.checkbox("未命中写黑名单(慎用)", False, help="普通选股默认不写名单")

    uni_label = "选股范围"
    uni_choice = st.selectbox(uni_label, ["全市场", "仅白名单", "仅黑名单", "仅特别关注榜"], index=0)
    _uni_map = {"全市场":"all", "仅白名单":"white", "仅黑名单":"black", "仅特别关注榜":"attention"}

    if run:
        try:
            df = se.tdx_screen(
                when,
                timeframe=timeframe,
                window=int(window),
                scope=scope.strip(),
                write_white=bool(write_white),
                write_black_rest=bool(write_black_rest),
                universe=_uni_map.get(uni_choice, "all"),
                return_df=True
            )

            # —— 新增：统计范围规模与跳过数量 —— 
            try:
                # 参考日优先用结果中的 ref_date，否则用自动提示日
                _ref_used = (df["ref_date"].iloc[0] if isinstance(df, pd.DataFrame) and not df.empty else ref_hint)
                _all_codes = se._list_codes_for_day(_ref_used)
                _cand, _src = se._apply_universe_filter(list(_all_codes), _ref_used, _uni_map.get(uni_choice, "all"))
                _total = len(_cand)
            except Exception:
                _total = None

            if isinstance(df, pd.DataFrame) and not df.empty:
                st.success(f"命中 {len(df)} 只；范围={uni_choice}；已写名单 白={int(write_white)} 黑={int(write_black_rest)}")
                st.dataframe(df, use_container_width=True, height=480)
                if _total is not None:
                    _skipped = max(_total - len(df), 0)
                    st.caption(f"范围内共 {_total} 只，命中 {len(df)} 只，**跳过 {_skipped} 只**。")

                # 另存一份结果 CSV
                out_dir = os.path.join(cfg.SC_OUTPUT_DIR, "select")
                os.makedirs(out_dir, exist_ok=True)
                fname = f"simple_select_{df['ref_date'].iloc[0]}.csv"
                outp = os.path.join(out_dir, fname)
                df.to_csv(outp, index=False, encoding="utf-8-sig")
                st.caption(f"结果已保存：{outp}")
                # ==== 导出六位代码（空格分隔，通达信友好） ====
                st.divider()
                st.write("导出命中清单（六位代码，空格分隔）")

                # 取命中 codes（只取连续 6 位数字），去重+升序
                codes_series = (
                    df["ts_code"].astype(str).str.extract(r"(\d{6})")[0].dropna().drop_duplicates()
                )
                codes_sorted = sorted(codes_series.tolist())
                if not codes_sorted:
                    st.info("结果里没有可导出的六位代码。")
                else:
                    # “导出前 N 名”：默认导出全部，可调整
                    n_export = st.number_input(
                        "导出前 N 名", min_value=1, max_value=int(len(codes_sorted)),
                        value=int(len(codes_sorted)), step=1
                    )
                    codes_out = codes_sorted[: int(n_export)]
                    text_out = " ".join(codes_out)

                    ref_for_export = str(df["ref_date"].iloc[0])
                    out_dir = os.path.join(cfg.SC_OUTPUT_DIR, "select")
                    os.makedirs(out_dir, exist_ok=True)
                    fname = os.path.join(out_dir, f"simple_select_codes_{ref_for_export}.txt")

                    # 写到本地 & 提供下载
                    try:
                        with open(fname, "w", encoding="utf-8") as f:
                            f.write(text_out + "\n")
                        st.success(f"已导出：{fname}")
                    except Exception as e:
                        st.warning(f"本地写文件失败（不影响下方下载）：{e}")

                    st.code(text_out, language="text")
                    st.download_button(
                        "⬇️ 下载到本地（TXT）",
                        data=text_out,
                        file_name=f"codes_simple_select_{ref_for_export}.txt",
                        mime="text/plain"
                    )

            else:
                st.warning("本次无命中。")
                if _total is not None:
                    st.caption(f"范围内共 {_total} 只，命中 0 只，**跳过 {_total} 只**。")
            UI_LOGGER.info("简单选股：tf=%s win=%s scope=%s 写白=%s 写黑=%s 表达式=%s",
                           timeframe, int(window), scope.strip(), write_white, write_black_rest, when)
        except Exception as e:
            UI_LOGGER.exception("普通选股失败：%s", e)
            st.error(f"普通选股失败：{e}")

    with st.expander("查看运行日志尾部（score.log）", expanded=False):
        st.code(_tail("./log/score.log", n=400), language="text")
