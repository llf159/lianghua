#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
抓取概念数据（东财 + 同花顺），分别落盘。
- 仅用于学习/测试，请自行评估使用合规性与频率限制。
- 输出目录：
    - 东财：stock_data/concepts/em/{concepts.csv, concept_members.csv, stock_concepts.csv}
    - 同花顺：stock_data/concepts/ths/{stock_concepts.csv}
- 运行：PYTHONPATH=. venv/bin/python scrape_concepts.py
"""

from __future__ import annotations

import logging
import os
import time
import shutil
import re
import sys
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm
from urllib3.exceptions import ProtocolError

# 禁用环境代理，防止请求被重定向到本地代理/socks
for _k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(_k, None)
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
# 禁用日志多进程队列（受限环境可能无法创建 SemLock）
os.environ.setdefault("LOG_DISABLE_MP_QUEUE", "1")

import config as cfg
from download import TokenBucketRateLimiter
from utils import normalize_em_ts_code

_em_session = requests.Session()
_em_adapter = HTTPAdapter(
    max_retries=Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=None,  # Retry on all methods including GET
        raise_on_status=False,
    )
)
_em_session.mount("http://", _em_adapter)
_em_session.mount("https://", _em_adapter)
# 明确隔离东财会话，避免意外带入同花顺 Cookie/头
_em_session.cookies.clear()
_em_session.headers.clear()

BASE_DIR = ROOT / "stock_data" / "concepts"
EM_DIR = BASE_DIR / "em"
THS_DIR = BASE_DIR / "ths"
STOCK_LIST_PATH = ROOT / "stock_data" / "stock_list.csv"
THS_OUTPUT_PATH = THS_DIR / "stock_concepts.csv"
EM_OUTPUT_PATH = EM_DIR / "stock_concepts.csv"
EM_CALLS_PER_MIN = 500  # 限频：东财每分钟最大调用次数
THS_CALLS_PER_MIN = 120  # 限频：同花顺每分钟最大调用次数（降低频率避免 403）
THS_RANDOM_SLEEP_MIN = 0.4  # 同花顺请求之间的随机等待下限
THS_RANDOM_SLEEP_MAX = 1.2  # 同花顺请求之间的随机等待上限
THS_RANDOM_SLEEP_MIN_WITH_COOKIE = 0.15  # 已有 Cookie 时的随机等待下限
THS_RANDOM_SLEEP_MAX_WITH_COOKIE = 0.6   # 已有 Cookie 时的随机等待上限
THS_TIMEOUT = 3       # 同花顺请求超时时间（秒）
EM_PROXIES = {"http": None, "https": None}
THS_PROXY_HOST = getattr(cfg, "THS_PROXY_HOST", "127.0.0.1") or "127.0.0.1"
THS_PROXY_PORT = int(getattr(cfg, "THS_PROXY_PORT", 12334) or 0)
THS_PROXY_OPTIONS = [{"label": "direct", "value": {"http": None, "https": None}}]
if THS_PROXY_PORT > 0:
    THS_PROXY_URL = f"http://{THS_PROXY_HOST}:{THS_PROXY_PORT}"
    THS_PROXY_OPTIONS.append({"label": f"proxy@{THS_PROXY_HOST}:{THS_PROXY_PORT}", "value": {"http": THS_PROXY_URL, "https": THS_PROXY_URL}})
_ths_proxy_idx = 0
THS_COOKIE_FILE = THS_DIR / "ths_cookie.txt"
THS_BACKOFF_DELAYS = [160, 80, 40, 20, 10]  # 失败重试等待（秒），倒序递减，成功后重置
EM_RETRY_MAX = 4
EM_RETRY_BASE_DELAY = 3.0
EM_RETRY_BACKOFF_FACTOR = 2.0
EM_MEMBER_RETRY_ROUND_MAX = 3  # 成分抓取失败后重新跑的轮次
EM_MEMBER_RETRY_SLEEP = 120     # 轮次之间的等待秒数

# 端口/协议容错：优先 http，失败时自动降级/切换到 https
EM_URL_FALLBACK_ENABLE = True
# 按优先级的东财域名（优先延迟/备站，最后才用主站）
EM_HOST_FALLBACKS = [
    "push2delay.eastmoney.com", # 302 指向的延迟域，部分网络更稳定
    "push2his.eastmoney.com",   # 备站
    "push2.eastmoney.com",      # 主站
]

# 东方财富接口参数（概念板块列表 + 成分）
EM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "http://quote.eastmoney.com/",
}

# 概念板块列表接口
PZ = 100  # 东财接口单页最大约100
URL_CONCEPT_LIST = (
    "http://push2.eastmoney.com/api/qt/clist/get"
    f"?pn={{pn}}&pz={PZ}&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
    "&fltt=2&invt=2&fid=f3&fs=m:90+t:3&fields=f12,f14"
)

# 概念成分接口（替换 {bk} 为板块代码，如 BK0816）
URL_CONCEPT_MEMBERS = (
    "http://push2.eastmoney.com/api/qt/clist/get"
    "?pn=1&pz=5000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
    "&fltt=2&invt=2&fid=f3&fs=b:{bk}&fields=f12,f14"
)

# 同花顺概念页（补抓缺失票/全量）
THS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://basic.10jqka.com.cn/",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# 轻量限频配置
PAUSE_SEC = 0.2  # 每个板块抓取间隔，避免过快
RETRY_FAIL_SLEEP = 10.0  # 单只失败后等待
RETRY_ROUND_SLEEP = 3.0  # 一轮失败后等待再重试
RETRY_ROUND_MAX = 3      # 最多重试轮次，防止无限循环
THS_COOKIE_ENV = "THS_COOKIE"  # 从环境变量读取同花顺 Cookie（可选）
THS_LOGIN_URL = "https://upass.10jqka.com.cn/login?redir=https://basic.10jqka.com.cn/"  # 自动获取 Cookie 时打开的登录页
THS_COOKIE_VERIFY_URL = "https://basic.10jqka.com.cn/"  # 登录后刷新以确保 Cookie 落到目标域


def _load_download_sources() -> List[str]:
    """解析配置中的概念抓取源，返回去重后的有序列表。"""
    default = ["em", "ths"]
    raw = getattr(cfg, "CONCEPT_DOWNLOAD_SOURCES", default)
    if isinstance(raw, str):
        raw_list = [raw]
    else:
        try:
            raw_list = list(raw)
        except Exception:
            raw_list = default

    normalized: List[str] = []
    for item in raw_list:
        key = str(item).strip().lower()
        if not key:
            continue
        if key in {"all", "both", "default"}:
            return default
        if key in {"em", "eastmoney", "dfcf", "东方财富"}:
            normalized.append("em")
        elif key in {"ths", "10jqka", "同花顺"}:
            normalized.append("ths")

    if not normalized:
        return default

    # 去重但保持顺序
    seen = set()
    ordered: List[str] = []
    for item in normalized:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _run_ths_with_restart(delay_seconds: int = 600, max_retries: int = 3) -> pd.DataFrame:
    """
    运行同花顺抓取，若出现无法自愈的网络问题则延迟后自动重启。
    判断标准：
    - run_ths_all_mode 抛出异常或 SystemExit（如连续 403）
    - 返回结果为空（len == 0）
    """
    attempt = 1
    while attempt <= max_retries:
        try:
            df = run_ths_all_mode()
            if df is not None and not df.empty:
                return df
            tqdm.write(f"同花顺抓取结果为空，判定为失败（尝试 {attempt}/{max_retries}）。")
        except SystemExit as e:
            tqdm.write(f"同花顺抓取触发 SystemExit({e.code})，可能为网络/403 问题。")
        except Exception as e:
            tqdm.write(f"同花顺抓取异常：{e}（尝试 {attempt}/{max_retries}）。")

        if attempt >= max_retries:
            break
        attempt += 1
        tqdm.write(f"{delay_seconds} 秒后自动重启同花顺抓取 ...")
        time.sleep(delay_seconds)

    tqdm.write("同花顺抓取多次失败，已停止自动重启。")
    return pd.DataFrame()


def _persist_ths_cookie(cookie: str) -> None:
    """持久化 Cookie 到本地文件，便于后续运行复用。"""
    cookie = (cookie or "").strip()
    if not cookie:
        return
    try:
        THS_DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(THS_COOKIE_FILE), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(cookie)
        try:
            THS_COOKIE_FILE.chmod(0o600)
        except Exception:
            pass
        logger.debug("已持久化同花顺 Cookie: %s", THS_COOKIE_FILE)
    except Exception as e:
        logger.warning("持久化同花顺 Cookie 失败: %s", e)


def _load_ths_cookie() -> str:
    """优先环境变量，其次本地文件，加载 THS Cookie。"""
    env_val = os.environ.get(THS_COOKIE_ENV, "").strip()
    if env_val:
        return env_val
    try:
        if THS_COOKIE_FILE.exists():
            text = THS_COOKIE_FILE.read_text(encoding="utf-8").strip()
            if text:
                os.environ[THS_COOKIE_ENV] = text
                logger.debug("已从持久缓存加载同花顺 Cookie: %s", THS_COOKIE_FILE)
                return text
    except Exception as e:
        logger.warning("读取同花顺 Cookie 持久缓存失败: %s", e)
    return ""


def _current_ths_proxy() -> Dict[str, Optional[str]]:
    """获取当前同花顺请求使用的代理设置。"""
    return THS_PROXY_OPTIONS[_ths_proxy_idx]["value"]


def _current_ths_proxy_label() -> str:
    return THS_PROXY_OPTIONS[_ths_proxy_idx]["label"]


def _switch_ths_proxy(reason: str = "") -> Dict[str, Optional[str]]:
    """在 403 等情况下切换代理/直连，返回新的代理配置。"""
    global _ths_proxy_idx
    _ths_proxy_idx = (_ths_proxy_idx + 1) % len(THS_PROXY_OPTIONS)
    label = _current_ths_proxy_label()
    msg = f"切换同花顺代理到 {label}"
    if reason:
        msg += f"（{reason}）"
    tqdm.write(msg)
    logger.info(msg)
    return _current_ths_proxy()


def _tqdm_ncols(default: int = 100) -> int:
    """根据终端宽度压缩进度条宽度，防止换行刷屏。"""
    try:
        env_override = os.environ.get("TQDM_COLS")
        if env_override:
            return max(40, int(env_override))
        width = shutil.get_terminal_size(fallback=(default, 20)).columns
        return max(50, min(width - 2, 140))
    except Exception:
        return default


def _tqdm_kwargs(**extra) -> Dict[str, object]:
    """统一 tqdm 参数，便于后续调整。"""
    disable_env = os.environ.get("TQDM_DISABLE", "").strip().lower()
    disable = disable_env in {"1", "true", "yes"}
    base = dict(
        ncols=_tqdm_ncols(),
        dynamic_ncols=True,
        mininterval=0.1,
        maxinterval=1.0,
        disable=disable,
        leave=True,
    )
    base.update(extra)
    return base


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("concept_scraper")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_dir = ROOT / "log"
    log_dir.mkdir(exist_ok=True)
    debug_file = log_dir / "concepts_em_debug.log"
    file_handler = logging.FileHandler(debug_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    return logger


logger = _build_logger()
em_limiter = TokenBucketRateLimiter(calls_per_min=EM_CALLS_PER_MIN, safe_calls_per_min=EM_CALLS_PER_MIN - 10)
ths_limiter = TokenBucketRateLimiter(calls_per_min=THS_CALLS_PER_MIN, safe_calls_per_min=THS_CALLS_PER_MIN - 10)
_ths_cookie = _load_ths_cookie()
if _ths_cookie:
    logger.debug("检测到同花顺 Cookie，已启用 Cookie 头。")
    if not THS_COOKIE_FILE.exists():
        _persist_ths_cookie(_ths_cookie)


def fetch_realtime_rank(
    src: str = "dc",
    fields: Optional[List[str]] = None,
    codes: Optional[List[str] | str] = None,
) -> pd.DataFrame:
    """
    调用 Tushare 爬虫接口 realtime_list（doc_id=317），获取实时涨跌幅排名。

    Args:
        src: 数据源，"dc"=东财（默认），"sina"=新浪。
        fields: 可选列白名单，留空返回全部字段。
        codes: 可选，指定股票代码列表/字符串（支持带或不带后缀，逗号/空格/换行分隔），仅返回这些股票。

    Returns:
        按涨跌幅降序排列的 DataFrame；接口异常或无数据时返回空 DataFrame。
    """
    try:
        import tushare as ts
    except ImportError as e:
        logger.error("未安装 tushare，请先 pip install tushare (%s)", e)
        raise

    token = getattr(cfg, "TOKEN", "") or os.environ.get("TUSHARE_TOKEN", "")
    if token:
        try:
            ts.set_token(token)
        except Exception as e:
            logger.debug("设置 Tushare token 失败：%s", e)

    try:
        df = ts.realtime_list(src=src or "dc")
    except Exception as e:
        logger.error("调用 ts.realtime_list 失败：%s", e)
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # 仅保留指定列
    if fields:
        keep = [c for c in fields if c in df.columns]
        if keep:
            df = df[keep]

    # 仅保留指定股票
    if codes:
        if isinstance(codes, str):
            parts = re.split(r"[\\s,;|]+", codes.strip())
            code_list = [p for p in parts if p]
        else:
            code_list = [str(c) for c in codes if str(c).strip()]
        code_set: set[str] = set()
        for c in code_list:
            cu = c.upper()
            code_set.add(cu)
            if "." in cu:
                code_set.add(cu.split(".")[0])
        code_col = None
        for cand in ("ts_code", "TS_CODE", "code", "CODE"):
            if cand in df.columns:
                code_col = cand
                break
        if code_col:
            df = df[df[code_col].astype(str).str.upper().map(
                lambda x: x if x not in code_set and "." not in x else (x if x in code_set else x.split(".")[0])
            ).isin(code_set)]

    # 按涨跌幅排序（兼容大小写列名）
    col_map = {c.lower(): c for c in df.columns}
    pct_col = col_map.get("pct_change")
    if pct_col:
        df = df.sort_values(pct_col, ascending=False).reset_index(drop=True)
    return df


def fetch_realtime_quote(
    ts_codes: Optional[List[str] | str] = None,
    src: str = "sina",
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    调用 Tushare 爬虫接口 realtime_quote（doc_id=315），获取实时盘口 TICK 快照。

    Args:
        ts_codes: 股票代码（字符串或列表，支持逗号/空格/换行/分号/竖线分隔）。sina 支持多只，最多 50；dc 只支持单只。
        src: 数据源，"sina"（默认，可多只）或 "dc"（仅单只）。
        fields: 可选列白名单，留空返回全部字段。

    Returns:
        DataFrame，包含买卖五档、成交、价格等；接口异常或无数据时返回空 DataFrame。
    """
    try:
        import tushare as ts
    except ImportError as e:
        logger.error("未安装 tushare，请先 pip install tushare (%s)", e)
        raise

    # 确保实时行情走直连而非系统代理
    for _k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(_k, None)
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,hq.sinajs.cn,finance.sina.com.cn,api.tushare.pro")
    os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])

    token = getattr(cfg, "TOKEN", "") or os.environ.get("TUSHARE_TOKEN", "")
    if token:
        try:
            ts.set_token(token)
        except Exception as e:
            logger.debug("设置 Tushare token 失败：%s", e)

    # 解析/限制代码数量
    code_list: List[str] = []
    if ts_codes:
        if isinstance(ts_codes, str):
            parts = re.split(r"[\\s,;|]+", ts_codes.strip())
            code_list = [p for p in parts if p]
        else:
            code_list = [str(c).strip() for c in ts_codes if str(c).strip()]

    if not code_list and src.lower() == "dc":
        raise ValueError("src=dc 需要提供单只 ts_code")

    if src.lower() == "dc" and len(code_list) > 1:
        code_list = code_list[:1]  # 东财只支持单只
    if src.lower() == "sina" and len(code_list) > 50:
        code_list = code_list[:50]  # 官方提示单次最多 50 只

    ts_code_arg = ",".join(code_list) if code_list else None

    try:
        df = ts.realtime_quote(ts_code=ts_code_arg, src=src or "sina")
    except Exception as e:
        logger.error("调用 ts.realtime_quote 失败：%s", e)
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if fields:
        keep = [c for c in fields if c in df.columns]
        if keep:
            df = df[keep]
    return df


def _get_json(url: str) -> Dict:
    """请求东财接口，带简单重试与指数退避，缓解偶发网络问题。"""
    def _candidate_urls(u: str) -> List[str]:
        if not EM_URL_FALLBACK_ENABLE:
            return [u]
        res: List[str] = []
        from urllib.parse import urlsplit, urlunsplit
        parsed = urlsplit(u)
        schemes = ["https", "http"]  # 优先 https，无法连通再降级 http
        hosts: List[str] = []
        # 先放备用/延迟域，再放原始域，避免始终卡在默认 host
        for h in EM_HOST_FALLBACKS:
            if h and h not in hosts:
                hosts.append(h)
        if parsed.hostname and parsed.hostname not in hosts:
            hosts.append(parsed.hostname)
        for sch in schemes:
            for host in hosts:
                netloc = host
                if parsed.port:
                    netloc = f"{host}:{parsed.port}"
                res.append(urlunsplit((sch, netloc, parsed.path, parsed.query, parsed.fragment)))
        # 去重保持顺序
        seen: set = set()
        uniq: List[str] = []
        for x in res:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        return uniq

    candidates = _candidate_urls(url)

    for idx, cur_url in enumerate(candidates, start=1):
        for attempt in range(1, EM_RETRY_MAX + 1):
            try:
                em_limiter.wait_if_needed()
                resp = _em_session.get(cur_url, headers=EM_HEADERS, timeout=10, proxies=EM_PROXIES)
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    # 可能返回了 HTML（如风控/验证码），记录片段便于判断
                    text = (resp.text or "")[:400]
                    raise ValueError(f"非JSON响应，状态{resp.status_code}，片段: {text!r}")
            except Exception as e:
                last_candidate = idx == len(candidates)
                # 对“远端关闭连接”类错误，直接换下一个候选域名以避免无效重试
                if isinstance(e, ProtocolError) or "RemoteDisconnected" in str(e):
                    if last_candidate and attempt >= EM_RETRY_MAX:
                        logger.warning("东财请求多次失败(第%s次)：%s | url=%s", attempt, e, cur_url)
                        raise
                    logger.debug("东财请求远端关闭连接，切换下一候选 host：%s", cur_url)
                    break
                if attempt >= EM_RETRY_MAX:
                    if last_candidate:
                        logger.warning("东财请求多次失败(第%s次)：%s | url=%s", attempt, e, cur_url)
                        raise
                    else:
                        logger.debug("东财请求已在 candidate %s/%s 用尽重试，切换下一个候选：%s", idx, len(candidates), cur_url)
                        break
                delay = EM_RETRY_BASE_DELAY * (EM_RETRY_BACKOFF_FACTOR ** (attempt - 1))
                logger.debug("东财请求失败(第%s次, candidate %s/%s)，%ss后重试：%s", attempt, idx, len(candidates), delay, e)
                time.sleep(delay)


def fetch_concept_list() -> pd.DataFrame:
    """分页抓取概念列表。"""
    rows = []
    total = None
    for pn in range(1, 100):  # 防御性上限
        data = _get_json(URL_CONCEPT_LIST.format(pn=pn))
        if total is None:
            total = data.get("data", {}).get("total", 0)
        items = data.get("data", {}).get("diff", []) or []
        if not items:
            break
        logger.debug("概念列表第 %s 页，返回 %s 条", pn, len(items))
        for it in items:
            code = it.get("f12")
            name = it.get("f14")
            if code and name:
                rows.append({"concept_code": code, "concept_name": name})
        if len(items) < PZ or (total and len(rows) >= total):
            break
        time.sleep(PAUSE_SEC)
    return pd.DataFrame(rows, dtype=str)


def fetch_members(bk_code: str) -> pd.DataFrame:
    url = URL_CONCEPT_MEMBERS.format(bk=bk_code)
    data = _get_json(url)
    items = data.get("data", {}).get("diff", []) or []
    rows = []
    for it in items:
        ts_code = normalize_em_ts_code(it.get("f12"))
        name = it.get("f14")
        if ts_code and name:
            rows.append(
                {
                    "concept_code": bk_code,
                    "concept_name": None,  # 事后填充
                    "ts_code": ts_code,
                    "name": name,
                }
            )
    return pd.DataFrame(rows, dtype=str)


def _ths_url(ts_code: str) -> str:
    """构造同花顺概念页 URL。"""
    code = str(ts_code).split(".")[0]
    return f"https://basic.10jqka.com.cn/{code}/concept.html"


def _sleep_ths_jitter() -> None:
    """同花顺请求间的随机等待，减少被限频概率。"""
    if _ths_cookie:
        delay = random.uniform(THS_RANDOM_SLEEP_MIN_WITH_COOKIE, THS_RANDOM_SLEEP_MAX_WITH_COOKIE)
    else:
        delay = random.uniform(THS_RANDOM_SLEEP_MIN, THS_RANDOM_SLEEP_MAX)
    time.sleep(delay)


def _build_ths_headers() -> Dict[str, str]:
    """构造同花顺请求头，按需注入 Cookie。"""
    headers = dict(THS_HEADERS)
    if _ths_cookie:
        headers["Cookie"] = _ths_cookie
    return headers


def _create_webdriver(driver_path: Optional[str]) -> Tuple[Optional[object], str]:
    """
    先尝试 Edge（msedgedriver），失败后回退 Chrome（chromedriver）。
    返回 (driver, name)。未成功则 driver 为 None。
    """
    def _log_driver_hint(browser: str, err: Exception) -> None:
        msg = f"{browser} 启动失败: {err}"
        tqdm.write(msg)
        logger.warning(msg)

    try:
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options as EdgeOptions  # type: ignore
        from selenium.webdriver.edge.service import Service as EdgeService  # type: ignore
        # 优先 Edge
        try:
            edge_options = EdgeOptions()
            edge_options.add_argument("--disable-blink-features=AutomationControlled")
            edge_options.add_argument("--no-sandbox")
            edge_options.add_argument("--disable-dev-shm-usage")
            edge_service = EdgeService(executable_path=driver_path) if driver_path else EdgeService()
            return webdriver.Edge(service=edge_service, options=edge_options), "Edge"
        except TypeError as edge_err:
            # 兼容旧 Selenium，回退旧式参数
            try:
                return webdriver.Edge(executable_path=driver_path, options=edge_options), "Edge"
            except Exception as edge_err2:  # pragma: no cover - 环境相关
                _log_driver_hint("Edge", edge_err2)
        except Exception as edge_err:
            _log_driver_hint("Edge", edge_err)
    except Exception as e:
        logger.debug("Edge 相关组件不可用: %s", e)

    # 回退 Chrome
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.chrome.service import Service as ChromeService

        chrome_options = ChromeOptions()
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_service = ChromeService(executable_path=driver_path) if driver_path else ChromeService()
        return webdriver.Chrome(service=chrome_service, options=chrome_options), "Chrome"
    except TypeError as chrome_err:
        # 兼容旧 Selenium，回退旧式参数
        try:
            return webdriver.Chrome(executable_path=driver_path, options=chrome_options), "Chrome"
        except Exception as chrome_err2:  # pragma: no cover
            _log_driver_hint("Chrome", chrome_err2)
    except Exception as chrome_err:
        _log_driver_hint("Chrome", chrome_err)
    return None, ""


def interactive_fetch_ths_cookie(
    login_url: str = THS_LOGIN_URL,
    driver_path: Optional[str] = None,
    wait_hint_sec: int = 120,
) -> str:
    """
    通过浏览器人工登录同花顺后，抓取 Cookie 串。
    使用方式：python tools/scrape_concepts.py --get-ths-cookie
    """
    def _prompt_manual_cookie(fail_reason: str = "") -> str:
        """手动粘贴 Cookie 的兜底方案。"""
        if fail_reason:
            tqdm.write(f"自动获取 Cookie 失败：{fail_reason}")
        if not sys.stdin.isatty():
            tqdm.write("当前非交互终端，无法手动输入 Cookie。")
            return ""
        try:
            cookie = input("请在浏览器登录后粘贴同花顺 Cookie（留空跳过）: ").strip()
            return cookie
        except KeyboardInterrupt:
            return ""
        except Exception as e:  # pragma: no cover
            logger.debug("手动输入 Cookie 失败: %s", e)
            return ""

    try:
        import selenium  # noqa: F401
    except Exception as e:  # pragma: no cover - 环境相关
        msg = "未安装 selenium，无法自动获取 Cookie。请先安装 `pip install selenium` 并确保浏览器驱动可用。"
        tqdm.write(msg)
        logger.warning("selenium 导入失败，无法获取 Cookie: %s", e)
        return _prompt_manual_cookie("缺少 selenium")

    driver, browser_name = _create_webdriver(driver_path)
    if driver is None:
        return _prompt_manual_cookie("启动浏览器失败，确认驱动在 PATH 或使用 --driver-path")

    driver.get(login_url)
    tqdm.write(
        f"已打开同花顺登录页：{login_url}\n"
        "如未自动跳转可手动访问 https://basic.10jqka.com.cn/ 完成登录，"
        f"请在 {wait_hint_sec} 秒内登录后回到终端按回车继续 ..."
    )
    try:
        input()
    except KeyboardInterrupt:
        driver.quit()
        return ""

    # 登录后强制访问目标域，确保 Cookie 生效
    try:
        driver.get(THS_COOKIE_VERIFY_URL)
        time.sleep(1.5)
    except Exception as e:  # pragma: no cover
        logger.debug("刷新目标域失败: %s", e)

    try:
        cookies = driver.get_cookies()
    except Exception as e:  # pragma: no cover - 环境相关
        driver.quit()
        return _prompt_manual_cookie(f"读取 Cookie 失败: {e}")
    driver.quit()
    pairs = []
    for c in cookies:
        if not c.get("name") or not c.get("value"):
            continue
        # 仅保留同花顺域名相关 Cookie
        domain = c.get("domain") or ""
        if "10jqka.com.cn" not in domain:
            continue
        pairs.append(f"{c['name']}={c['value']}")
    cookie_str = "; ".join(pairs)
    if cookie_str:
        tqdm.write("\n获取到同花顺 Cookie：\n")
        tqdm.write(cookie_str)
        tqdm.write(f"\n可临时设置环境变量使用：export {THS_COOKIE_ENV}='{cookie_str}'")
    else:
        tqdm.write("未获取到同花顺 Cookie，请确认已登录成功。")
    return cookie_str


def fetch_concepts_ths(ts_code: str) -> List[str]:
    """从同花顺概念页抓取单只股票的概念名称列表。"""
    url = _ths_url(ts_code)
    ths_limiter.wait_if_needed()
    _sleep_ths_jitter()
    session = requests.Session()
    session.trust_env = False  # 禁用环境代理
    proxies = _current_ths_proxy()
    session.proxies.update(proxies)
    resp = session.get(url, headers=_build_ths_headers(), timeout=THS_TIMEOUT, proxies=proxies)
    session.close()
    resp.raise_for_status()
    html = resp.content.decode("gbk", errors="ignore")
    concepts: List[str] = []
    table = re.search(r'gnContent">(.*?)</table>', html, re.S)
    if not table:
        return concepts
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table.group(1), re.S)
    for row in rows:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(tds) >= 2 and tds[0].strip().isdigit():
            name_html = tds[1]
            text = re.sub(r"<[^>]+>", "", name_html)
            text = text.replace("\t", "").replace("\n", "").strip()
            if text:
                concepts.append(text)
    return concepts


def _make_ths_session() -> requests.Session:
    """构建带连接池的 Session，减少文件句柄占用。"""
    session = requests.Session()
    session.trust_env = False  # 不读取系统代理
    session.proxies.update(_current_ths_proxy())  # 显式禁用代理/或使用当前代理
    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64, max_retries=Retry(total=2, backoff_factor=0.2))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _deduplicate_ths_output(df: pd.DataFrame) -> pd.DataFrame:
    """同花顺结果去重：优先保留非空概念，按 ts_code 唯一化。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code", "concepts_code", "concepts_name", "stock_name"])
    df = df.copy()
    df["concepts_name"] = df["concepts_name"].fillna("")
    df["_is_empty"] = df["concepts_name"].str.strip() == ""
    # 非空优先
    df = df.sort_values(by=["_is_empty"])
    df = df.drop_duplicates(subset=["ts_code"], keep="first")
    return df.drop(columns=["_is_empty"])


def build_all_from_ths(max_workers: Optional[int] = None) -> pd.DataFrame:
    """
    全量同花顺概念：对 stock_list 所有股票抓取概念，写出 stock_concepts。
    concepts_code 为空；concepts_name 为同花顺名称（去重）。
    """
    stock_list = _load_stock_list(STOCK_LIST_PATH)
    if stock_list.empty or "ts_code" not in stock_list.columns:
        tqdm.write("stock_list.csv 缺失或无 ts_code，无法抓取同花顺概念。")
        return pd.DataFrame(columns=["ts_code", "concepts_code", "concepts_name", "stock_name"])

    name_map = dict(zip(stock_list["ts_code"], stock_list.get("name", [""] * len(stock_list))))
    rows: List[Dict[str, str]] = []
    codes = stock_list["ts_code"].astype(str).tolist()
    existing = pd.DataFrame()
    existing_codes: set[str] = set()
    append_header = True
    empty_or_missing: set[str] = set()
    if THS_OUTPUT_PATH.exists():
        try:
            existing_raw = pd.read_csv(THS_OUTPUT_PATH, dtype=str)
            existing = _deduplicate_ths_output(existing_raw)
            if not existing_raw.equals(existing):
                try:
                    existing.to_csv(THS_OUTPUT_PATH, index=False)
                except Exception as e:
                    logger.warning("重写去重后的同花顺结果失败（忽略）：%s", e)
            if not existing.empty and "ts_code" in existing.columns:
                existing_codes = set(existing["ts_code"].astype(str))
                append_header = THS_OUTPUT_PATH.stat().st_size == 0
                # 为空的股票也加入重抓列表
                if "concepts_name" in existing.columns:
                    mask_empty = existing["concepts_name"].fillna("").astype(str).str.strip() == ""
                    empty_or_missing = set(existing.loc[mask_empty, "ts_code"].astype(str))
                tqdm.write(f"检测到已有同花顺数据 {len(existing_codes)} 条，其中空概念 {len(empty_or_missing)} 条，将补抓空/缺失。")
        except Exception as e:
            logger.warning("读取已有同花顺数据失败，忽略增量跳过：%s", e)
    target_codes = [c for c in codes if c not in existing_codes or c in empty_or_missing]
    if not target_codes:
        tqdm.write("所有股票均有同花顺概念数据且非空，跳过抓取。")
        return existing if not existing.empty else pd.DataFrame(columns=["ts_code", "concepts_code", "concepts_name", "stock_name"])

    session = _make_ths_session()
    failure_details: List[str] = []
    http_fail = 0
    empty_count = 0
    max_retries = len(THS_BACKOFF_DELAYS)

    def _fetch_one(code: str) -> bool:
        nonlocal http_fail, empty_count, append_header
        start = time.time()
        attempt = 0
        current_proxies = _current_ths_proxy()
        switched_after_403 = False
        while attempt <= max_retries:
            try:
                ths_limiter.wait_if_needed()
                _sleep_ths_jitter()
                url = _ths_url(code)
                resp = session.get(url, headers=_build_ths_headers(), timeout=THS_TIMEOUT, proxies=current_proxies)
                resp.raise_for_status()
                html = resp.content.decode("gbk", errors="ignore")
                concepts: List[str] = []
                table = re.search(r'gnContent">(.*?)</table>', html, re.S)
                if table:
                    rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", table.group(1), re.S)
                    for row in rows_html:
                        tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
                        if len(tds) >= 2 and tds[0].strip().isdigit():
                            name_html = tds[1]
                            text = re.sub(r"<[^>]+>", "", name_html)
                            text = text.replace("\t", "").replace("\n", "").strip()
                            if text:
                                concepts.append(text)
                names_list: List[str] = []
                seen: set[str] = set()
                for raw in concepts:
                    if not raw or raw in seen:
                        continue
                    seen.add(raw)
                    names_list.append(raw)
                if not names_list:
                    empty_count += 1
                    snippet = html[:120].replace("\n", " ")
                    logger.info("[THS] %s 返回空概念，len=%s，片段：%s", code, len(html), snippet)
                row_data = {
                    "ts_code": code,
                    "concepts_code": "",
                    "concepts_name": ",".join(names_list),
                    "stock_name": name_map.get(code, ""),
                }
                rows.append(row_data)
                # 成功即落盘，避免长跑中断丢数据
                try:
                    pd.DataFrame([row_data]).to_csv(
                        THS_OUTPUT_PATH,
                        mode="a",
                        index=False,
                        header=append_header,
                    )
                    append_header = False
                except Exception as write_err:
                    logger.warning("写入同花顺增量失败 %s: %s", code, write_err)
                return True
            except Exception as e:
                elapsed = time.time() - start
                status = getattr(getattr(e, "response", None), "status_code", None)
                text_snippet = ""
                if getattr(e, "response", None) is not None:
                    raw_text = getattr(e.response, "text", "") or ""
                    text_snippet = raw_text[:800].replace("\n", " ").replace("\r", " ")
                msg = (
                    f"{code} error ({type(e).__name__}): {e}, "
                    f"status={status}, elapsed={elapsed:.2f}s, url={url if 'url' in locals() else ''}, snippet={text_snippet}"
                )
                failure_details.append(msg)
                http_fail += 1
                logger.warning("同花顺抓取 %s 失败: %s", code, msg)

                if status == 403 and not switched_after_403 and len(THS_PROXY_OPTIONS) > 1:
                    switched_after_403 = True
                    current_proxies = _switch_ths_proxy("403 后重试")
                    continue
                # 403 且已切换代理仍失败，终止后续下载
                if status == 403 and switched_after_403:
                    tqdm.write("403 连续失败，已尝试切换代理，建议稍后再次运行。")
                    raise SystemExit(1)

                # 非 403 或未配置代理：不再重试，退出
                break
        return False

    def _run_pass(target_codes: List[str], desc: str) -> List[str]:
        failed_local: List[str] = []
        ok_count = 0
        with tqdm(total=len(target_codes), desc=desc, **_tqdm_kwargs()) as bar:
            for code in target_codes:
                ok = _fetch_one(code)
                if ok:
                    ok_count += 1
                else:
                    failed_local.append(code)
                bar.update(1)
                bar.set_postfix(ok=ok_count, fail=len(failed_local))
        return failed_local

    failed = _run_pass(target_codes, "抓取概念(同花顺)")
    retry_round = 1
    while failed and retry_round <= RETRY_ROUND_MAX:
        tqdm.write(f"同花顺第 {retry_round} 次重试，剩余失败 {len(failed)} ...")
        time.sleep(RETRY_ROUND_SLEEP)
        failed = _run_pass(failed, f"抓取概念(同花顺-重试{retry_round})")
        retry_round += 1
    session.close()
    if failed:
        tqdm.write(f"同花顺抓取未全部成功，剩余 {len(failed)}，HTTP失败 {http_fail}，空概念 {empty_count}。")
        for line in failure_details[:20]:
            tqdm.write(f"[FAIL] {line}")
    else:
        tqdm.write(f"同花顺抓取完成：全部成功，HTTP失败 {http_fail}，空概念 {empty_count}。")

    # 最终写回：去重并优先保留非空概念
    try:
        combined = existing if rows else existing.copy()
        if rows:
            combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        combined = _deduplicate_ths_output(combined)
        combined.to_csv(THS_OUTPUT_PATH, index=False)
        empty_final = combined["concepts_name"].fillna("").str.strip().eq("").sum()
        tqdm.write(f"同花顺结果已去重写回：总计 {len(combined)} 条，空概念 {empty_final}。")
    except Exception as e:
        logger.warning("写回去重后的同花顺结果失败：%s", e)

    # 返回合并后的结果（读盘保证与文件一致）
    try:
        final_df = pd.read_csv(THS_OUTPUT_PATH, dtype=str)
    except Exception:
        final_df = pd.DataFrame(rows + existing.to_dict("records") if not existing.empty else rows, dtype=str)
    return final_df


def _load_stock_list(path: Path) -> pd.DataFrame:
    """读取 stock_list.csv，缺失则返回空 DataFrame。"""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def run_em() -> pd.DataFrame:
    """抓取东财概念，返回 stock_map DataFrame。"""
    EM_DIR.mkdir(parents=True, exist_ok=True)

    # 若已有结果则跳过抓取
    if (EM_OUTPUT_PATH.exists() and (EM_DIR / "concepts.csv").exists() and (EM_DIR / "concept_members.csv").exists()):
        tqdm.write(f"检测到东财概念文件已存在，跳过抓取：{EM_DIR}")
        try:
            return pd.read_csv(EM_OUTPUT_PATH, dtype=str)
        except Exception:
            pass

    logger.debug("抓取概念列表 ...")
    try:
        concepts = fetch_concept_list()
    except Exception as e:
        tqdm.write(f"东财概念列表抓取失败：{e}")
        logger.warning("东财概念列表抓取失败，终止本轮东财抓取：%s", e)
        return pd.DataFrame()
    if concepts.empty:
        tqdm.write("概念列表为空，终止。")
        logger.warning("概念列表为空，终止。")
        return pd.DataFrame()
    tqdm.write(f"东财概念列表获取成功，共 {len(concepts)} 个；开始抓取成分…")
    concepts.to_csv(EM_DIR / "concepts.csv", index=False)
    logger.debug("概念数：%s", len(concepts))

    def _run_batch(target: List[Tuple[str, str]]) -> Tuple[List[pd.DataFrame], List[Tuple[str, str]]]:
        ok: List[pd.DataFrame] = []
        failed_local: List[Tuple[str, str]] = []
        max_workers = min(16, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(fetch_members, bk): (bk, name) for bk, name in target}
            with tqdm(total=len(future_map), desc="抓取概念成分", **_tqdm_kwargs()) as bar:
                for future in as_completed(future_map):
                    bk, name = future_map[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            if name:
                                df["concept_name"] = name
                            ok.append(df)
                            logger.debug("板块 %s %s 成分 %s 条", bk, name, len(df))
                        else:
                            logger.debug("板块 %s %s 成分为空", bk, name)
                    except Exception as e:
                        failed_local.append((bk, str(e)))
                        logger.warning("板块 %s 抓取失败: %s", bk, e)
                    bar.update(1)
                    bar.set_postfix(success=len(ok), failed=len(failed_local))
                    time.sleep(PAUSE_SEC)
        return ok, failed_local

    # 初始一轮抓取
    targets = [(row["concept_code"], row["concept_name"]) for _, row in concepts.iterrows()]
    all_members: List[pd.DataFrame] = []
    ok_batch, failed = _run_batch(targets)
    all_members.extend(ok_batch)

    # 针对失败的板块做多轮重试
    round_idx = 1
    while failed and round_idx <= EM_MEMBER_RETRY_ROUND_MAX:
        tqdm.write(f"东财成分重试第 {round_idx}/{EM_MEMBER_RETRY_ROUND_MAX} 轮，剩余失败 {len(failed)} 个板块 ...")
        time.sleep(EM_MEMBER_RETRY_SLEEP)
        retry_targets = [(bk, None) for bk, _ in failed]  # name 可为空，fetch_members 只需代码
        ok_batch, failed = _run_batch(retry_targets)
        all_members.extend(ok_batch)
        round_idx += 1

    if not all_members:
        tqdm.write("未获取到任何板块成分。")
        logger.warning("未获取到任何板块成分。")
        return pd.DataFrame()
    if failed:
        tqdm.write(f"东财成分最终仍有 {len(failed)} 个板块失败，详见日志。")

    members = pd.concat(all_members, ignore_index=True)
    members.to_csv(EM_DIR / "concept_members.csv", index=False)

    stock_map = (
        members.groupby("ts_code")
        .agg(
            concepts_code=("concept_code", lambda s: ",".join(sorted(set(s)))),
            concepts_name=("concept_name", lambda s: ",".join(sorted(set(s.dropna())))),
            stock_name=("name", "first"),
        )
        .reset_index()
    )
    stock_map.to_csv(EM_OUTPUT_PATH, index=False)

    logger.debug(
        "东财成分总计 %s 条，股票 %s；输出目录：%s",
        len(members),
        len(stock_map),
        EM_DIR,
    )
    summary = (
        f"东财抓取完成: 概念 {len(concepts)} 个，成分 {len(members)} 条，"
        f"股票 {len(stock_map)}，输出文件：{EM_OUTPUT_PATH}"
    )
    if failed:
        summary += f"（失败 {len(failed)} 个板块，详见日志）"
    tqdm.write(summary)
    return stock_map


def run_ths_all_mode() -> pd.DataFrame:
    """全量同花顺模式：抓取同花顺概念，返回 stock_map DataFrame。"""
    THS_DIR.mkdir(parents=True, exist_ok=True)
    if THS_OUTPUT_PATH.exists():
        tqdm.write(f"检测到同花顺概念文件已存在，执行补缺增量：{THS_OUTPUT_PATH}")
    stock_map = build_all_from_ths()
    if stock_map.empty:
        tqdm.write("同花顺全量抓取为空，终止。")
        return pd.DataFrame()
    tmp_path = THS_OUTPUT_PATH.with_suffix(".tmp")
    stock_map.to_csv(tmp_path, index=False)
    tmp_path.replace(THS_OUTPUT_PATH)
    summary = f"同花顺全量抓取完成：股票 {len(stock_map)}，输出文件：{THS_OUTPUT_PATH}"
    tqdm.write(summary)
    return stock_map


def main() -> None:
    global _ths_cookie
    parser = argparse.ArgumentParser(description="抓取东财/同花顺概念或获取同花顺 Cookie。")
    parser.add_argument("--get-ths-cookie", action="store_true", help="打开浏览器手动登录并提取同花顺 Cookie")
    parser.add_argument("--ths-login-url", default=THS_LOGIN_URL, help="用于登录的同花顺页面 URL")
    parser.add_argument("--driver-path", default=None, help="浏览器驱动路径（可选），默认依赖系统 PATH")
    args = parser.parse_args()

    if args.get_ths_cookie:
        new_cookie = interactive_fetch_ths_cookie(
            login_url=args.ths_login_url,
            driver_path=args.driver_path,
        )
        if new_cookie:
            os.environ[THS_COOKIE_ENV] = new_cookie
            _persist_ths_cookie(new_cookie)
            tqdm.write(f"同花顺 Cookie 已写入：{THS_COOKIE_FILE}")
        return

    sources = _load_download_sources()
    tqdm.write(f"CONCEPT_DOWNLOAD_SOURCES: {sources}")
    need_ths = "ths" in sources

    if need_ths and not _ths_cookie:
        tqdm.write("未检测到 THS_COOKIE，可能导致同花顺抓取 403。")
        choice = input("是否现在打开浏览器获取 Cookie？[y/N]: ").strip().lower()
        if choice == "y":
            new_cookie = interactive_fetch_ths_cookie(
                login_url=args.ths_login_url,
                driver_path=args.driver_path,
            )
            if new_cookie:
                _ths_cookie = new_cookie
                os.environ[THS_COOKIE_ENV] = new_cookie
                _persist_ths_cookie(new_cookie)
                tqdm.write("已更新同花顺 Cookie，继续抓取。")

    # 明确提示当前抓取流程与重试策略
    tqdm.write(
        f"抓取计划：sources={sources} | EM重试{EM_RETRY_MAX}次(指数退避) "
        f"| EM成分重试轮次={EM_MEMBER_RETRY_ROUND_MAX}, 间隔={EM_MEMBER_RETRY_SLEEP}s "
        f"| THS自动重启：间隔600s，最多3轮"
    )

    for src in sources:
        if src == "em":
            run_em()
        elif src == "ths":
            _run_ths_with_restart()
        else:
            tqdm.write(f"未知的抓取源 {src}，已跳过。")


if __name__ == "__main__":
    main()
