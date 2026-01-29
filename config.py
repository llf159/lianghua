#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量化交易系统配置文件
统一管理所有模块的配置参数
"""

import os

# ================= 基础路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_NAME = "venv"  # 虚拟环境目录名（位于项目根目录下），若不存在则使用系统环境

# ================= 数据源配置 =================
# Tushare API Token（必填）
TOKEN = ""

# 数据存储根目录
DATA_ROOT = os.path.join(BASE_DIR, "stock_data")

# 数据下载配置
ASSETS = ["stock", "index"]  # 可选: ["stock"], ["index"], ["stock","index"]
START_DATE = "20250101"  # 数据下载开始日期 'YYYYMMDD'
END_DATE = "today"  # today或具体日期 'YYYYMMDD'
# 是否下载换手率因子（tor）。开启后 pro_bar 会额外调用 daily_basic，Tushare 限频以 daily_basic 的 200/min 为准
DOWNLOAD_TOR = True

# 指数白名单
INDEX_WHITELIST = [
    "000001.SH", "399001.SZ", "399300.SZ", "399905.SZ", 
    "399006.SZ", "000016.SH", "000852.SH"
]

# 复权方式
API_ADJ = "qfq"  # 可选: "qfq" | "hfq" | "raw"

# 概念显示黑名单（列表中的概念名称不展示）
CONCEPT_BLACKLIST = ["百元股", "创业板综", "融资融券", "QFII重仓", "并购重组概念", "2025中报扭亏", "参股保险", "2025三季报预增", 
                     "注册制次新股", "国家大基金持股", "同花顺果指数", "科创次新股", "2025中报预增", "中字头股票", "回购增持再贷款概念",
                     "新股与次新股"]
# 概念读取数据源：下载阶段固定同时抓取东财 + 同花顺，读取时可选择
CONCEPT_READ_SOURCE = "ths"  # em=仅东财；ths=仅同花顺；mix=东财优先，不足用同花顺补
# 概念抓取下载源：控制爬虫阶段抓取哪些数据源，默认同时抓取东财和同花顺
# 可选值示例：["em", "ths"]（默认）；["ths"] 仅抓取同花顺；["em"] 仅抓取东财；"all"/"both" 等同默认
CONCEPT_DOWNLOAD_SOURCES = ["ths"]
# 概念榜小样本收缩强度（置信调整），值越大对覆盖数少的概念压制越强，常用 5~10
CONCEPT_SHRINK_ALPHA = 5

# 同花顺抓取代理配置（供 scrape_concepts 使用；端口<=0 表示仅直连）
THS_PROXY_HOST = os.getenv("THS_PROXY_HOST", "127.0.0.1")
try:
    THS_PROXY_PORT = int(os.getenv("THS_PROXY_PORT", "12334"))
except Exception:
    THS_PROXY_PORT = 12334

# ================= 限频控制配置 =================
# API调用频率限制：你的权限额度
CALLS_PER_MIN = 500
# 安全限频：留出安全边距，避免触发限频
SAFE_CALLS_PER_MIN = 490
# 令牌桶容量（设置为覆盖线程数，允许更多突发）
RATE_BUCKET_CAPACITY = 12
# 令牌补充速率（次/秒）
RATE_BUCKET_REFILL_RATE = 8.0
# 最小等待时间（秒）
RATE_BUCKET_MIN_WAIT = 0.02
# 接近限频阈值时的额外延迟（秒）
RATE_BUCKET_EXTRA_DELAY = 0.25
# 触发额外延迟的调用次数阈值
RATE_BUCKET_EXTRA_DELAY_THRESHOLD = 495

# ================= 线程配置 =================
# 增量下载线程数
STOCK_INC_THREADS = 12
# 快速初始化线程数
FAST_INIT_THREADS = 16
# 增量重算工作线程数
INC_RECALC_WORKERS = 32

# ================= 数据库配置 =================
# 统一数据库存储配置
UNIFIED_DB_PATH = "stock_data.db"

# DuckDB连接配置（由数据库连接配置管理器统一管理）
# 注意：DuckDB要求连接到同一个数据库文件的所有连接必须使用相同的配置参数
# 因此，所有连接（只读和读写）都使用相同的配置参数
DUCKDB_THREADS = 16  # DuckDB使用的线程数（所有连接统一使用此值）
DUCKDB_MEMORY_LIMIT = "18GB"  # DuckDB内存限制（所有连接统一使用此值）
DUCKDB_TEMP_DIR = os.path.join(DATA_ROOT, "duckdb_tmp")  # 临时文件目录
DUCKDB_CLEAR_DAILY_BEFORE = False  # 是否在写入前清理每日数据
# 全表内存缓存开关（针对小体量数据库，默认开启；关闭后始终走数据库查询）
FULL_STOCK_CACHE_ENABLED = True
# 全表缓存阈值配置（优先级：FULL_STOCK_CACHE_MAX_MB > FULL_STOCK_CACHE_RATIO）
# 设置 FULL_STOCK_CACHE_DISABLE=True 可彻底禁用
FULL_STOCK_CACHE_MAX_MB = None  # 显式MB上限（None或<=0 表示使用占比阈值）
FULL_STOCK_CACHE_RATIO = 0.2    # 占用物理内存比例上限（0.01~0.8）
FULL_STOCK_CACHE_DISABLE = False

# 数据库连接配置（由数据库连接配置管理器统一应用）
DB_QUERY_TIMEOUT = 30  # 查询超时时间（秒），读写连接为2倍
DB_ENABLE_INDEXES = True  # 是否启用索引
DB_BATCH_SIZE = 1000  # 批处理大小

# 快速初始化配置
FAST_INIT_STOCK_DIR = os.path.join(DATA_ROOT, "fast_init_symbol")

# 日志配置
# 落盘最低等级：DEBUG/INFO/WARNING/ERROR/CRITICAL，低于该级别的文件不会创建
LOG_FILE_LEVEL = "INFO"

# ================= 评分系统配置 =================
# 基础配置
SC_REF_DATE = "today"  # 参考日：'today' 或 'YYYYMMDD'
SC_LOOKBACK_D = 60     # 打分窗口（日线）
SC_PRESCREEN_LOOKBACK_D = 180  # 初选窗口（多用于周/月线）

# 评分参数
SC_BASE_SCORE = 50  # 基础分数
SC_MIN_SCORE = 0  # 最低分数
SC_TIE_BREAK = "kdj_j_asc"  # 并列打破：使用 KDJ 的 J 值（越小越靠前）

# 并行与读取优化
import multiprocessing
SC_MAX_WORKERS = multiprocessing.cpu_count() or 4  # 默认与 CPU 核心数一致，UI 可覆盖
SC_READ_TAIL_DAYS = None       # 若不为 None，则强制只读最近 N 天数据

# 执行器选择（实验特性）：是否在安全条件下使用进程池
SC_USE_PROCESS_POOL = True

# 记录与注入开关（性能测试用）
SC_ENABLE_RULE_DETAILS = True          # 规则明细与批量缓冲
SC_ENABLE_CUSTOM_TAGS = True           # 自定义标签注入
SC_ENABLE_VERBOSE_SCORE_LOG = True     # 评分过程中的细粒度调试日志
SC_ENABLE_BATCH_XSEC = False           # 启用横截面批量排名（实验）
SC_DYNAMIC_RESAMPLE = True             # 仅当规则用到 W/M 时才触发重采样

# 输出目录
SC_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "score")
SC_CACHE_DIR = os.path.join(BASE_DIR, "cache", "scorelists")

# 个股详情存储配置
SC_DETAIL_STORAGE = "database"     # 存储方式：'json' | 'database' | 'both'
SC_DETAIL_DB_TYPE = "duckdb"       # 数据库类型：'sqlite' | 'duckdb'
SC_DETAIL_DB_PATH = "details/details.db"   # 数据库文件路径（相对于SC_OUTPUT_DIR）
SC_USE_DB_STORAGE = True  # 是否使用数据库存储
SC_DB_FALLBACK_TO_JSON = True  # 数据库存储失败时是否回退到JSON

# 特别关注榜配置
SC_ATTENTION_SOURCE = "top"         # 统计来源：'top' | 'white' | 'black'
SC_ATTENTION_WINDOW_D = 20          # 统计窗口：最近 N 个"交易日"
SC_ATTENTION_MIN_HITS = 2           # 至少上榜次数
SC_ATTENTION_TOP_K = 200            # 输出前多少名

# 指数对比配置
SC_BENCH_CODES = []  # 基准指数清单 - 临时禁用
SC_BENCH_WINDOW = 20                # 特征滚动窗口（天）
SC_BENCH_FILL = "ffill"             # 基准对齐方式：'ffill' 前向填充 或 'drop' 只保留共同交易日
SC_BENCH_FEATURES = ["rs", "exret", "beta", "corr"]  # 输出哪些特征

# 其他配置
SC_UNIVERSE = "all"                 # 打分范围：all / white / black / attention 或 直接给一个 ts_code 列表
SC_TRACKING_TOP_N = 200             # 排名跟踪：只跟踪前多少名（默认200）

# ================= 投资组合配置 =================
# 账本配置
PF_LEDGER_NAME = "default"  # 账本名称

# 费率配置（以 BP 为单位，1bp = 0.01%）
PF_FEE_BPS_BUY = 15     # 买入费率
PF_FEE_BPS_SELL = 15    # 卖出费率
PF_MIN_FEE = 0.0        # 最低费用（元）

# 资金配置
PF_INIT_CASH = 1_000_000.0  # 初始资金（元）
PF_INIT_AVAILABLE = PF_INIT_CASH  # 初始可用资金（元）

# 成交价模式
PF_TRADE_PRICE_MODE = "next_open"  # 可选: "next_open" | "close"
