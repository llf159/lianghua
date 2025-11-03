#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量化交易系统配置文件
统一管理所有模块的配置参数
"""

import os

# ================= 基础路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= 数据源配置 =================
# Tushare API Token（必填）
TOKEN = ""

# 数据存储根目录
DATA_ROOT = os.path.join(BASE_DIR, "stock_data")

# 数据下载配置
ASSETS = ["stock", "index"]  # 可选: ["stock"], ["index"], ["stock","index"]
START_DATE = "20250101"  # 数据下载开始日期 'YYYYMMDD'
END_DATE = "today"  # today或具体日期 'YYYYMMDD'

# 指数白名单
INDEX_WHITELIST = [
    "000001.SH", "399001.SZ", "399300.SZ", "399905.SZ", 
    "399006.SZ", "000016.SH", "000852.SH"
]

# 复权方式
API_ADJ = "qfq"  # 可选: "qfq" | "hfq" | "raw"

# ================= 限频控制配置 =================
# API调用频率限制：你的权限额度
CALLS_PER_MIN = 500
# 安全限频：留出安全边距，避免触发限频
SAFE_CALLS_PER_MIN = 490
# 令牌桶容量（设置为较小的值，避免初始爆发）
RATE_BUCKET_CAPACITY = 8
# 令牌补充速率（次/秒）
RATE_BUCKET_REFILL_RATE = 8.0
# 最小等待时间（秒）
RATE_BUCKET_MIN_WAIT = 0.05
# 接近限频阈值时的额外延迟（秒）
RATE_BUCKET_EXTRA_DELAY = 0.5
# 触发额外延迟的调用次数阈值
RATE_BUCKET_EXTRA_DELAY_THRESHOLD = 480

# 重试配置
RETRY_TIMES = 5
RETRY_DELAY_SEQUENCE = [10, 10, 5]  # 固定序列；超过长度后都用最后一个值(5)
RETRY_JITTER_RANGE = (-0.5, 0.5)    # 每次等待加的随机抖动秒数范围
RETRY_LOG_LEVEL = "INFO"            # 等待日志级别：INFO / DEBUG

# ================= 线程配置 =================
# 增量下载线程数
STOCK_INC_THREADS = 12
# 快速初始化线程数
FAST_INIT_THREADS = 16
# 失败重试线程数
FAILED_RETRY_THREADS = 10
# 增量重算工作线程数
INC_RECALC_WORKERS = 32

# ================= 数据库配置 =================
# 数据存储模式选择
DATA_STORAGE_MODE = "duckdb"  # 可选: "duckdb" | "parquet" | "auto"
# 当设置为"auto"时，优先使用DuckDB，如果DuckDB不可用则使用Parquet
# 当设置为"duckdb"时，强制使用DuckDB数据库
# 当设置为"parquet"时，强制使用Parquet文件存储

# 统一数据库存储配置
USE_UNIFIED_DB_STORAGE = True
UNIFIED_DB_TYPE = "duckdb"  # 推荐使用DuckDB，性能更好
UNIFIED_DB_PATH = "stock_data.db"
KEEP_PARQUET_FILES = True  # 是否保留原有Parquet文件（用于迁移和备份）

# DuckDB连接配置（由数据库连接配置管理器统一管理）
DUCKDB_THREADS = 16  # DuckDB使用的线程数（只读连接使用完整线程数，读写连接使用一半）
DUCKDB_MEMORY_LIMIT = "18GB"  # DuckDB内存限制
DUCKDB_TEMP_DIR = os.path.join(DATA_ROOT, "duckdb_tmp")  # 临时文件目录
DUCKDB_CLEAR_DAILY_BEFORE = False  # 是否在写入前清理每日数据

# 数据库连接配置（由数据库连接配置管理器统一应用）
DB_QUERY_TIMEOUT = 30  # 查询超时时间（秒），读写连接为2倍
DB_ENABLE_INDEXES = True  # 是否启用索引
DB_BATCH_SIZE = 1000  # 批处理大小

# 评分系统数据库查询缓存配置
SC_DB_CACHE_TTL = 900    # 缓存有效期（秒），尽量覆盖一整轮评分时间（默认15分钟）
SC_DB_CACHE_MAX = 128    # 最大缓存条目数

# ================= 数据处理配置 =================
# 流式处理配置
STREAM_FLUSH_DATE_BATCH = 80      # 缓冲多少个不同 trade_date 就刷盘一次
STREAM_FLUSH_STOCK_BATCH = 200    # 处理多少只股票后强制刷盘
STREAM_LOG_EVERY = 300            # 每处理多少只股票打印一次进度日志

# 增量处理配置
INC_STREAM_COMPUTE_INDICATORS = True     # 边下载边计算指标
INC_STREAM_UPDATE_FAST_CACHE = True      # 边下载边更新 fast_init 缓存
INC_STREAM_MERGE_IND_SUBSET = True       # 边下载边合并指标到 daily 分区

# 增量重算优化
INC_IND_ALL_INMEM = True                 # 一次性内存微批：DuckDB 一把拉 + 内存重算
INC_INMEM_CUTOFF_BACK_DAYS = 365         # 从指标分区回看多少天来估每股 last_date
INC_INMEM_PADDING_DAYS = 5               # warm-up 下界再多回看几天
INC_INMEM_CHUNK_TS = 800                 # 非常多股票时的分片规模
INC_SKIP_OLD_READ = True                 # 跳过旧文件 warm-read
INC_ENABLE_PARALLEL_COMPUTE = True       # 是否启用并行计算指标（仅计算，数据库写入仍为串行）

# 文件完整性检查
CHECK_SKIP_MIN_MAX = True                 # 是否启用跳过前检查
CHECK_SKIP_READ_COLUMNS = ["trade_date"]  # 读取的列，尽量最少减少 IO
CHECK_SKIP_ALLOW_LAG_DAYS = 0             # 允许已有文件的最大日期距离 end_date 的"滞后"天数
SKIP_CHECK_START_ENABLED = False          # 是否启用开始日期检查

# 失败重试配置
FAILED_RETRY_ONCE = True          # 第一次下载后自动对失败股票再跑一轮
FAILED_RETRY_WAIT = 5             # 下载结束到补抓之间的等待秒

# ================= 输出配置 =================
# 单股成品输出控制
WRITE_SYMBOL_PLAIN = True            # 是否输出「不带指标」的单股文件
WRITE_SYMBOL_INDICATORS = True       # 是否输出「带指标」的单股文件
SYMBOL_PRODUCT_FORMATS = {
    "plain": ["parquet", "csv"],      # 同时导出 Parquet + CSV
    "ind":   ["parquet", "csv"]       # 同时导出 Parquet + CSV
}
SYMBOL_PRODUCT_INDICATORS = "all"        # 需要计算哪些指标，如果需要全部则 "all"
SYMBOL_PRODUCT_WARMUP_DAYS = 120        # 增量重算指标的 warm-up 天数
SYMBOL_PRODUCT_OUT = None                # None → 自动写到 <base>/stock/by_symbol_<adj>

# 快速初始化配置
FAST_INIT_STOCK_DIR = os.path.join(DATA_ROOT, "fast_init_symbol")
CLEAR_CACHE_AFTER_FAST_INIT = True       # 快速初始化完成后清除缓存文件

# 日志配置
LOG_LEVEL = "INFO"  # 已关闭DEBUG级别日志

# ================= 评分系统配置 =================
# 基础配置
SC_DO_TRACKING = False
SC_DO_SURGE = False
SC_REF_DATE = "today"  # 参考日：'today' 或 'YYYYMMDD'
SC_LOOKBACK_D = 60     # 打分窗口（日线）
SC_PRESCREEN_LOOKBACK_D = 180  # 初选窗口（多用于周/月线）

# 评分参数
SC_BASE_SCORE = 50  # 基础分数
SC_MIN_SCORE = 0  # 最低分数
SC_TOP_K = 100  # 输出前K名
SC_TIE_BREAK = "kdj_j_asc"  # 并列打破：使用 KDJ 的 J 值（越小越靠前）

# 并行与读取优化
# SC_MAX_WORKERS 默认值：min(2*CPU, 16)，可按需在 UI 中覆盖
import multiprocessing
_SC_DEFAULT_WORKERS = min((multiprocessing.cpu_count() or 4) * 2, 16)
SC_MAX_WORKERS = _SC_DEFAULT_WORKERS  # 默认 min(2*CPU, 16)，UI 可覆盖
SC_READ_TAIL_DAYS = None       # 若不为 None，则强制只读最近 N 天数据

# 执行器选择（实验特性）：是否在安全条件下使用进程池
SC_USE_PROCESS_POOL = True

# 记录与注入开关（性能测试用）
SC_ENABLE_RULE_DETAILS = True          # 规则明细与批量缓冲
SC_ENABLE_CUSTOM_TAGS = True           # 自定义标签注入
SC_ENABLE_VERBOSE_SCORE_LOG = True     # 评分过程中的细粒度调试日志
SC_ENABLE_VECTOR_BOOL = False          # 启用向量化布尔求值（实验）
SC_ENABLE_BATCH_XSEC = False           # 启用横截面批量排名（实验）
SC_DYNAMIC_RESAMPLE = True             # 仅当规则用到 W/M 时才触发重采样

# 输出目录
SC_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "score")
SC_CACHE_DIR = os.path.join(BASE_DIR, "cache", "scorelists")

# 个股详情存储配置
SC_DETAIL_STORAGE = "database"     # 存储方式：'json' | 'database' | 'both'
SC_DETAIL_DB_TYPE = "duckdb"     # 数据库类型：'sqlite' | 'duckdb' | 'postgres'
SC_DETAIL_DB_PATH = "details/details.db"   # 数据库文件路径（相对于SC_OUTPUT_DIR，postgres模式下忽略）
SC_USE_DB_STORAGE = True  # 是否使用数据库存储
SC_DB_FALLBACK_TO_JSON = True  # 数据库存储失败时是否回退到JSON

# PostgreSQL 配置
SC_PG_DSN = "postgresql://postgres:password@localhost:5432/stock_data"  # 通用PostgreSQL连接字符串
SC_DETAIL_DB_DSN = "postgresql://postgres:password@localhost:5432/stock_details"  # 专用详情数据库连接字符串

# 明细写入策略
SC_DETAIL_WRITE_MODE = "json_then_import"  # 写入策略: "immediate" | "queued" | "json_then_import"
# SQLite busy_timeout 毫秒（写锁冲突时的等待时间）
SC_SQLITE_BUSY_TIMEOUT_MS = 60000

# 名单配置
SC_WRITE_WHITELIST = True   # 写白名单 cache/…/whitelist.csv
SC_WRITE_BLACKLIST = True   # 写黑名单 cache/…/blacklist.csv

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
SC_HIDE_FORMULA = True              # details JSON 不写 when；UI 可选择显示
SC_TOPK_ROWS = 30                   # UI：Top-K 显示行数（仅用于前端展示）

# ================= 回测配置 =================
# 策略参数
HOLD_DAYS = 2                       # 买入持有天数
STRATEGY_START_DATE = "20220601"  # 回测开始日期 'YYYYMMDD'
STRATEGY_END_DATE = "20250801"  # 回测结束日期 'YYYYMMDD'
MAX_HOLD_DAYS = 60                  # -1 表示不限制持有天数，其他正整数表示最多持有几天

# 交易模式
BUY_MODE = "open"                   # 可选: "open" | "close" | "signal_open"
SELL_MODE = "other"                 # 可选: "open" | "close" | "strategy" | "other"
FALLBACK_SELL_MODE = "open"         # 超过持有天数后强制卖出的模式

# 数据目录（回测用）
DATA_DIR = "E://gupiao-hfq"  # 回测数据目录路径

# TDX规则文件路径
TDX_BUY_PATH = "./buy_rules.txt"  # 买入规则文件路径
TDX_SELL_PATH = "./sell_rules.txt"  # 卖出规则文件路径
TDX_SELL_OPEN_PATH = "./sell_open_rules.txt"  # 开盘卖出规则文件路径
TDX_SELL_CLOSE_PATH = "./sell_close_rules.txt"  # 收盘卖出规则文件路径

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
