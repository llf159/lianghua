# def ensure_datetime_index(df):
#     df = df.sort_values('trade_date')
#     df = df.set_index('trade_date')
#     return df

import pandas as pd
import os
import re
import glob
from typing import Optional
from log_system import get_logger

# 初始化日志记录器
logger = get_logger("utils")

def ensure_datetime_index(df, file_path=None):
    for date_col in ['trade_date', 'date', '交易日期', 'datetime']:
        if date_col in df.columns:
            if df[date_col].dtype == 'int64' or df[date_col].dtype == 'float64':
                df[date_col] = df[date_col].astype(int).astype(str)
            df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
            df.index = pd.to_datetime(df.index, format='%Y%m%d', errors='coerce')
            return df.sort_index()
    # 如果没有找到合适的日期列，打印出来方便你排查
    print(f"\n数据文件{file_path if file_path else ''} 缺少日期列！表头为：{df.columns.tolist()}")
    return None


def normalize_trade_date(df: pd.DataFrame, col: str = "trade_date") -> pd.DataFrame:
    """
    规范化 trade_date 列为 YYYYMMDD 格式字符串，丢弃无法解析的日期。
    - 强制按 '%Y%m%d' 解析，避免 '0' / NaT 被转成 1970-01-01
    - 保留原 DataFrame 的其他列
    """
    if col not in df.columns:
        raise ValueError(f"缺少 {col} 列")
    # 转成字符串再解析，严格格式匹配
    td = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="coerce")
    # 丢掉解析失败的行
    mask = td.notna()
    if not mask.all():
        bad_count = (~mask).sum()
        logger.warning("丢弃无法解析的 %s 行（%d 条）", col, bad_count)
    df = df.loc[mask].copy()
    df[col] = td.dt.strftime("%Y%m%d")
    return df


def normalize_ts(ts_input: str, asset: str = "stock") -> str:
    """
    统一股票/指数代码为 Tushare 风格：'000001.SZ' / '600000.SH' / '430047.BJ'
    - 接受输入形式：'000001'、'000001.sz'、'sz000001'、'SH600000'、'600000-SH' 等
    - stock 资产：六位纯数字会按首位规则补后缀：8→BJ；{5,6,9}→SH；其余→SZ
    - 其它资产（如 index）默认仅做大小写/分隔符清洗，不强行补后缀
    """
    s = (ts_input or "").strip().upper()
    if not s:
        return s
    s = s.replace("_", ".")
    s = re.sub(r"\s+", "", s)

    # SH600000 / SZ000001 / BJ430047
    m = re.match(r"^(SH|SZ|BJ)[\.-]?(\d{6})$", s)
    if m:
        ex, code = m.group(1), m.group(2)
        return f"{code}.{ex}"

    # 600000SH / 000001.SZ
    m = re.match(r"^(\d{6})[\.-]?(SH|SZ|BJ)$", s)
    if m:
        code, ex = m.group(1), m.group(2)
        return f"{code}.{ex}"

    # 已是标准形态
    if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", s):
        return s

    # 六位纯数字：仅对股票补后缀
    if asset == "stock" and re.fullmatch(r"\d{6}", s):
        code = s
        if code.startswith("8") or code.startswith("920"):
            ex = "BJ"
        elif code[0] in {"5", "6", "9"}:
            ex = "SH"
        else:
            ex = "SZ"
        return f"{code}.{ex}"

    return s


def market_label(ts_code: str) -> str:
    """根据 ts_code 前缀粗分市场板块。"""
    s = (ts_code or "").split(".")[0]
    if s.startswith(("600","601","603","605")):
        return "沪A"
    if s.startswith(("000","001","002","003")):
        return "深A"
    if s.startswith(("300","301","302","303","304","305","306","307","308","309")):
        return "创业板"
    if s.startswith(("688","689")):
        return "科创板"
    if s.startswith((
        "430","831","832","833","834","835","836","837","838","839",
        "80","81","82","83","84","85","86","87","88","89",
        "920","921","922","923","924","925","926","927","928","929"
    )):
        return "北交所"
    return "其他"

# 策略文件读取工具
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path as _PathForStrategy
import importlib.util as _importlib_util, types as _types_for_strategy

@dataclass
class StrategySet:
    title: str
    category: str   # "ranking" | "filter" | "prediction"
    rules: List[Dict[str, Any]]
    path: str


def _import_module_from_path(_path: str, name: str = "strategy_repo_dyn") -> _types_for_strategy.ModuleType:
    spec = _importlib_util.spec_from_file_location(name, _path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载策略模块：{_path}")
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _candidate_repo_paths() -> List[_PathForStrategy]:
    here = _PathForStrategy(__file__).parent
    return [
        here / "strategies_repo.py",
        here.parent / "strategies_repo.py",
        _PathForStrategy("./strategies_repo.py"),
        _PathForStrategy("./strategies/strategies_repo.py"),
    ]


def _load_from_repo(py_path: Optional[str]) -> Optional[List[StrategySet]]:
    target = None
    if py_path:
        p = _PathForStrategy(py_path)
        if p.exists():
            target = p
    else:
        for c in _candidate_repo_paths():
            if c.exists():
                target = c
                break
    if not target:
        return None
    mod = _import_module_from_path(str(target), name=f"strategy_repo_{target.name}")
    sets: List[StrategySet] = []
    if hasattr(mod, "RANKING_RULES"):
        sets.append(StrategySet(title=str(getattr(mod, "RANKING_TITLE", "ranking")), category="ranking", rules=list(getattr(mod, "RANKING_RULES", [])), path=str(target)))
    if hasattr(mod, "FILTER_RULES"):
        sets.append(StrategySet(title=str(getattr(mod, "FILTER_TITLE", "filter")), category="filter", rules=list(getattr(mod, "FILTER_RULES", [])), path=str(target)))
    if hasattr(mod, "PREDICTION_RULES"):
        sets.append(StrategySet(title=str(getattr(mod, "PREDICTION_TITLE", "prediction")), category="prediction", rules=list(getattr(mod, "PREDICTION_RULES", [])), path=str(target)))
    return sets


def _split_config_rules_by_hard_penalty(rules: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    rank, filt = [], []
    for r in rules:
        if isinstance(r, dict) and r.get("hard_penalty", False):
            filt.append(r)
        else:
            rank.append(r)
    return {"ranking": rank, "filter": filt}


def load_strategy_sets_py(py_path: Optional[str] = None) -> List[StrategySet]:
    sets = _load_from_repo(py_path)
    if sets is not None:
        return sets
    cfg_mod = _import_module_from_path(str(_PathForStrategy(__file__).parent / "config.py"), name="config_for_strategy_fallback")
    sc_rules = list(getattr(cfg_mod, "SC_RULES", []))
    spl = _split_config_rules_by_hard_penalty(sc_rules)
    return [
        StrategySet(title="ranking@config", category="ranking", rules=spl["ranking"], path=str(cfg_mod.__file__)),
        StrategySet(title="filter@config(hard_penalty=True)", category="filter", rules=spl["filter"], path=str(cfg_mod.__file__)),
        StrategySet(title="prediction@empty", category="prediction", rules=[], path=str(cfg_mod.__file__)),
    ]


def load_rank_rules_py(py_path: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in load_strategy_sets_py(py_path):
        if s.category == "ranking":
            out.extend(s.rules)
    return out


def load_filter_rules_py(py_path: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in load_strategy_sets_py(py_path):
        if s.category == "filter":
            out.extend(s.rules)
    return out


def load_pred_rules_py(py_path: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in load_strategy_sets_py(py_path):
        if s.category == "prediction":
            out.extend(s.rules)
    return out


# ==================== 获取最新交易日相关函数 ====================

def get_latest_date_from_database() -> Optional[str]:
    """从数据库获取最新交易日期"""
    try:
        from database_manager import get_latest_trade_date
        latest = get_latest_trade_date()
        if latest:
            logger.info(f"从数据库获取最新交易日: {latest}")
            return latest
    except Exception as e:
        logger.warning(f"从数据库获取最新交易日失败: {e}")
    return None


def get_latest_date_from_daily_partition() -> Optional[str]:
    """从daily分区获取最新交易日期"""
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


def get_latest_date_from_single_dir(single_dir: Optional[str] = None) -> Optional[str]:
    """
    从single目录推断最新交易日期
    
    Args:
        single_dir: single目录路径，如果为None则尝试从config获取
        
    Returns:
        最新交易日期字符串（YYYYMMDD格式），如果无法获取则返回None
    """
    try:
        # 如果没有提供路径，尝试从config获取
        if single_dir is None:
            try:
                from config import DATA_ROOT, API_ADJ
                single_dir = os.path.join(DATA_ROOT, "stock", "single", f"single_{API_ADJ}_indicators")
            except ImportError:
                logger.debug("无法从config获取single目录路径")
                return None
        
        if not os.path.isdir(single_dir):
            return None
        
        files = glob.glob(os.path.join(single_dir, "*.parquet"))[:50]  # 取前 50 个采样
        mx = None
        for f in files:
            try:
                td = pd.read_parquet(f, columns=["trade_date"])["trade_date"]
                tmax = str(td.astype(str).max())
                if (mx is None) or (tmax > mx):
                    mx = tmax
            except Exception:
                continue
        if mx:
            logger.info(f"从single目录获取最新日期: {mx}")
            return mx
    except Exception as e:
        logger.debug(f"从single目录获取最新日期失败: {e}")
    return None
