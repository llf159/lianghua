import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# ================= backtest配置区 =====================
# True 使用 Parquet，False 使用旧 CSV
USE_PARQUET = True   
# USE_PARQUET = False

PARQUET_BASE = r"E:\stock_data"   # 与 download_new.py 的 DATA_ROOT 保持一致
PARQUET_ADJ = "qfq"          # 可选: "daily" | "raw" | "qfq" | "hfq"
PARQUET_USE_INDICATORS = True     # True: *_indicators 分区；False: 非 indicators 分区

# 配置参数
HOLD_DAYS = 2  # 买入持有天数
STRATEGY_START_DATE = "20250101"
STRATEGY_END_DATE = "20250801"
# 数据目录路径（可绝对路径或相对路径）
DATA_DIR = "E://gupiao-hfq"
TDX_BUY_PATH = "./buy_rules.txt"
TDX_SELL_PATH = "./sell_rules.txt"
TDX_SELL_OPEN_PATH = "./sell_open_rules.txt"
TDX_SELL_CLOSE_PATH = "./sell_close_rules.txt"

# DATA_DIR = os.path.join(BASE_DIR, "test") 
#open为次日开盘价买入，close为当日收盘价买入,signal_open为信号当天开盘买入；open涨停不买入
# BUY_MODE = "open"
# BUY_MODE = "close"
BUY_MODE = "signal_open"

#open为买入后n日开盘价卖出，close为收盘价卖出，strategy为策略卖出
SELL_MODE = "strategy"
# SELL_MODE = "open"
# SELL_MODE = "close"

MAX_HOLD_DAYS = -1  # -1 表示不限制持有天数，其他正整数表示最多持有几天

FALLBACK_SELL_MODE = "open"#超过持有天数后强制卖出的模式

# ================= download配置区 =====================
TOKEN = ""  # <-- 必填
# DATA_ROOT = "./data"             # 下载数据目录(可改为绝对路径)
DATA_ROOT = r"E:\stock_data"
ASSETS = ["stock", "index"]      # 可选: ["stock"], ["index"], ["stock","index"]
START_DATE = "20050101"
END_DATE = "today"               # 或具体日期 'YYYYMMDD'
INDEX_WHITELIST = [
    "000001.SH","399001.SZ","399300.SZ","399905.SZ","399006.SZ","000016.SH","000852.SH"
]

# 通用限频：你的权限额度
CALLS_PER_MIN = 470
RETRY_TIMES = 5
PARQUET_ENGINE = "pyarrow"
LOG_LEVEL = "INFO"
STOCK_INC_THREADS = 40         # 增量下载线程数

# -------- FAST INIT(按股票多线程全历史回补)开关 --------
FAST_INIT_MODE = True                     # 首次全历史快速抓取
FAST_INIT_THREADS = 50                    # 并发线程数
FAST_INIT_STOCK_DIR = os.path.join(DATA_ROOT, "fast_init_symbol")
API_ADJ = "qfq"                           # qfq/hfq/raw
# 若 FAST_INIT_MODE=True，可通过设置 API_ADJ 控制接口返回的复权方式：

# -------- 单股成品输出控制---------
WRITE_SYMBOL_PLAIN = True            # 是否输出「不带指标」的单股文件
WRITE_SYMBOL_INDICATORS = True       # 是否输出「带指标」的单股文件
# 为两类输出分别指定格式；可选: "parquet", "csv"
SYMBOL_PRODUCT_FORMATS = {
    "plain": ["parquet","csv"],      # 同时导出 Parquet + CSV
    "ind":   ["parquet","csv"]       # 同时导出 Parquet + CSV
}

SYMBOL_PRODUCT_INDICATORS = "all"        # 需要计算哪些指标，如果需要全部则 "all"
SYMBOL_PRODUCT_WARMUP_DAYS = 90          # 增量重算指标的 warm-up 天数
SYMBOL_PRODUCT_OUT = None                # None → 自动写到 <base>/stock/by_symbol_<adj>

# ===== 重试策略配置 (固定序列 + 抖动) =====
RETRY_DELAY_SEQUENCE = [10, 10, 5]   # 固定序列；超过长度后都用最后一个值(5)
RETRY_JITTER_RANGE = (-0.5, 0.5)     # 每次等待加的随机抖动秒数范围 (可调为 (-1,1))
RETRY_LOG_LEVEL = "INFO"             # 等待日志级别：INFO / DEBUG
# ==========================================

# ==== Streaming Merge 配置 ================
STREAM_FLUSH_DATE_BATCH = 80      # 缓冲多少个不同 trade_date 就刷盘一次
STREAM_FLUSH_STOCK_BATCH = 200    # 处理多少只股票后强制刷盘(避免长时间不落盘)
STREAM_LOG_EVERY = 300            # 每处理多少只股票打印一次进度日志
FAILED_RETRY_ONCE = True          # 第一次下载后自动对失败股票再跑一轮
FAILED_RETRY_THREADS = 8          # 失败补抓的线程数(可低一些)
FAILED_RETRY_WAIT = 5             # 下载结束到补抓之间的等待秒(缓冲限频)
# ==========================================

# ====== Skip 文件完整性快速检查参数 ======
CHECK_SKIP_MIN_MAX = True                 # 是否启用跳过前检查
CHECK_SKIP_READ_COLUMNS = ["trade_date"]  # 读取的列，尽量最少减少 IO
CHECK_SKIP_ALLOW_LAG_DAYS = 0           # 允许已有文件的最大日期距离 end_date 的“滞后”天数 (0=必须等于 end_date)
SKIP_CHECK_START_ENABLED = False          # 是否启用开始日期检查(如果不需要可以关闭，减少接口调用)
# ==========================================

# ==== DuckDB 分批归并配置 =================
DUCKDB_BATCH_SIZE = 300          # 每批处理的“单股票文件”数量(内存紧 → 降到 150/100)
DUCKDB_THREADS = 10              # DuckDB 并行线程 (2~8 之间；太大内存峰值上升)
DUCKDB_MEMORY_LIMIT = "18GB"      # 给 DuckDB 的内存上限(小机器可设 "4GB")
DUCKDB_TEMP_DIR = "duckdb_tmp"   # Spill 目录(磁盘剩余空间要够)
DUCKDB_CLEAR_DAILY_BEFORE = False # 首次构建或要完全重建设 True，会清空 daily 目录
DUCKDB_COLUMNS = "*"             # 列裁剪：可改成 "ts_code,trade_date,open,high,low,close,vol,amount"
DUCKDB_ENABLE_COMPACT_AFTER = True       # True,False,"if_needed"
COMPACT_MAX_FILES_PER_DATE = 12          # 超过 12 个 part 的日期执行压实
COMPACT_TMP_DIR = "compact_tmp"          # 若 compact 函数里需要临时目录(当前版本没用到)
DUCK_MERGE_DAY_LAG = 5          # parquet 最大日期距离 duck 表 > 5 天才触发合并
DUCK_MERGE_MIN_ROWS = 1_000_000 # 或过去 5 天新增行数 > 100 万行才触发
# ==========================================