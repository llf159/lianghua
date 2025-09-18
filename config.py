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
STRATEGY_START_DATE = "20220601"
STRATEGY_END_DATE = "20250801"
# 数据目录路径（可绝对路径或相对路径）
DATA_DIR = "E://gupiao-hfq"
TDX_BUY_PATH = "./buy_rules.txt"
TDX_SELL_PATH = "./sell_rules.txt"
TDX_SELL_OPEN_PATH = "./sell_open_rules.txt"
TDX_SELL_CLOSE_PATH = "./sell_close_rules.txt"

# DATA_DIR = os.path.join(BASE_DIR, "test") 
#open为次日开盘价买入，close为当日收盘价买入,signal_open为信号当天开盘买入；open涨停不买入
BUY_MODE = "open"
# BUY_MODE = "close"
# BUY_MODE = "signal_open"

#open为买入后n日开盘价卖出，close为收盘价卖出，strategy为策略卖出，other为统计MAX_HOLD_DAYS内最大涨幅
SELL_MODE = "other"
MAX_HOLD_DAYS = 60  # -1 表示不限制持有天数，其他正整数表示最多持有几天

FALLBACK_SELL_MODE = "open"#超过持有天数后强制卖出的模式

# ================= download配置区 =====================
TOKEN = ""  # <-- 必填
# DATA_ROOT = "./data"             # 下载数据目录(可改为绝对路径)
DATA_ROOT = r"E:\stock_data"
ASSETS = ["stock", "index"]      # 可选: ["stock"], ["index"], ["stock","index"]
START_DATE = "20220101"
END_DATE = "today"               # 或具体日期 'YYYYMMDD'
INDEX_WHITELIST = [
    "000001.SH","399001.SZ","399300.SZ","399905.SZ","399006.SZ","000016.SH","000852.SH"
]

# 通用限频：你的权限额度
CALLS_PER_MIN = 470
RETRY_TIMES = 5
PARQUET_ENGINE = "pyarrow"
LOG_LEVEL = "INFO"
STOCK_INC_THREADS = 8         # 增量下载线程数

# -------- FAST INIT(按股票多线程全历史回补) --------
FAST_INIT_THREADS = 8                    # 并发线程数
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
SYMBOL_PRODUCT_WARMUP_DAYS = 120          # 增量重算指标的 warm-up 天数
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
CHECK_SKIP_ALLOW_LAG_DAYS = 0             # 允许已有文件的最大日期距离 end_date 的“滞后”天数 (0=必须等于 end_date)
SKIP_CHECK_START_ENABLED = False          # 是否启用开始日期检查(如果不需要可以关闭，减少接口调用)
# ==========================================

# ==== DuckDB 分批归并配置 =================
DUCKDB_BATCH_SIZE = 300          # 每批处理的“单股票文件”数量(内存紧 → 降到 150/100)
DUCKDB_THREADS = 16              # DuckDB 并行线程 (2~8 之间；太大内存峰值上升)
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

# === 增量重算(指标) 的 I/O 优化 ===
INC_IND_ALL_INMEM = True                 # 一次性内存微批：DuckDB 一把拉 + 内存重算
INC_INMEM_CUTOFF_BACK_DAYS = 365         # 从指标分区回看多少天来估每股 last_date（避免全库扫描）
INC_INMEM_PADDING_DAYS = 5               # warm-up 下界再多回看几天，跨节假日更稳
INC_INMEM_CHUNK_TS = 800                 # （可选）非常多股票时的分片规模
INC_SKIP_OLD_READ = True                 # _WRITE_SYMBOL_INDICATORS 跳过旧文件 warm-read（由上游预热）
INC_RECALC_WORKERS = 32                  # None=自动(≈2×CPU)，也可设定具体整数

# ===================== Scoring 系统 =====================
SC_DO_TRACKING = False
SC_DO_SURGE = False
# 参考日：'today' 或 'YYYYMMDD'
SC_REF_DATE = "today"
# 打分窗口（日线）、初选窗口（多用于周/月线）
SC_LOOKBACK_D = 60
SC_PRESCREEN_LOOKBACK_D = 180
# 基础分与下限
SC_BASE_SCORE = 50
SC_MIN_SCORE = 0
# 结果数量、Tie-break（并列打破）：使用 KDJ 的 J 值（越小越靠前）
SC_TOP_K = 100
SC_TIE_BREAK = "kdj_j_asc"
# 并行与读取优化
SC_MAX_WORKERS = None          # None 表示 CPU-1
SC_READ_TAIL_DAYS = None       # 若不为 None，则强制只读最近 N 天数据
# 输出目录与缓存目录
SC_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "score")
SC_CACHE_DIR  = os.path.join(BASE_DIR, "cache", "scorelists")
# —— 名单开关（可选写入）——
SC_WRITE_WHITELIST = True   # 写白名单 cache/…/whitelist.csv
SC_WRITE_BLACKLIST = True   # 写黑名单 cache/…/blacklist.csv
# —— 特别关注榜（周期上榜次数统计）——
SC_ATTENTION_SOURCE    = "top"         # 统计来源：'top' | 'white' | 'black'
SC_ATTENTION_WINDOW_D  = 20            # 统计窗口：最近 N 个“交易日”
SC_ATTENTION_MIN_HITS  = 2             # 至少上榜次数
SC_ATTENTION_TOP_K     = 200           # 输出前多少名
SC_ATTENTION_BACKFILL_ENABLE = True    # 是否需要滚动补算
# ====== Scoring：指数对比（Benchmark） ======
SC_BENCH_CODES   = ["399300.SZ", "399001.SZ"]      # 基准指数清单；可多只，比如 ["000001.SH","399300.SZ"]
SC_BENCH_WINDOW  = 20                 # 特征滚动窗口（天）
SC_BENCH_FILL    = "ffill"            # 基准对齐方式：'ffill' 前向填充 或 'drop' 只保留共同交易日
SC_BENCH_FEATURES = ["rs","exret","beta","corr"]  # 输出哪些特征：相对强弱/超额/β/相关

# —— 打分范围：all / white / black / attention 或 直接给一个 ts_code 列表
SC_UNIVERSE = "all"

SC_HIDE_FORMULA = True   # details JSON 不写 when；UI 可选择显示（从 config 读取）

# —— UI：Top-K 显示行数（仅用于前端展示，不影响计算）——
SC_TOPK_ROWS = 30

# ===================== Portfolio 模拟持仓（全局配置） =====================
# 账本（可按需扩展）
PF_LEDGER_NAME = "default"
# 费率：以 BP 为单位（1bp = 0.01%）
PF_FEE_BPS_BUY  = 15     # 买入费率
PF_FEE_BPS_SELL = 15     # 卖出费率
PF_MIN_FEE      = 0.0    # 最低费用（元）
# 资金（初始总额与可用）
PF_INIT_CASH    = 1_000_000.0
PF_INIT_AVAILABLE = PF_INIT_CASH
# 成交价模式：'next_open' 或 'close'
PF_TRADE_PRICE_MODE = "next_open"

# ========== 规则样例（你可随意增删改；支持 TDX 表达式 + scope/clauses） ==========
SC_RULES = [
#     # 1) 任意一根满足：放量长阳 或 60日新高
#     {
#         "name": "D_放量长阳_或_60日新高",
#         "timeframe": "D",
#         "window": 60,
#         "when": "((SAFE_DIV(C - O, O) >= 0.03) AND (V > 1.8 * MA(V, 20))) OR (C >= HHV(H, 60)))",
#         "scope": "ANY",
#         "points": +6,
#         "explain": "出现放量长阳或创60日新高"
#     },
#     # 2) 连续条件：连续3天收盘高于 MA20
#     {
#         "name": "D_连阳收盘高于MA20",
#         "timeframe": "D",
#         "window": 20,
#         "when": "C>MA(C,20)",
#         "scope": "CONSEC>=3",
#         "points": +4,
#         "explain": "连续3天收盘站上MA20"
#     },
#     # 3) 跨周期组合：日线放量突破 + 周线均线多头（两个子句都命中才加分）
#     {
#         "name": "D突破+W多头",
#         "clauses": [
#             {"timeframe":"D","window":40,"when":"C>HHV(H,40) AND V>1.5*MA(V,20)","scope":"ANY"},
#             {"timeframe":"W","window":20,"when":"MA(C,5)>MA(C,10)","scope":"LAST"}
#         ],
#         "points": +8,
#         "explain": "日线放量突破且周线均线多头排列"
#     },
    # {
    #     "name": "当日机会",
    #     "timeframe": "D",
    #     "window": 2,
    #     "when": "TAG_HITS('opportunity') > 3",
    #     "scope": "LAST",
    #     "points": +15,
    #     "explain": "b1plus"
    # },
    # {
    #     "name": "相对强于深证",
    #     "timeframe": "D",
    #     "window": 20,
    #     "when": "RS_399001_SZ_3 > 1.02",   # 20日RS>1.02（强于基准≈2%）
    #     "scope": "ANY",
    #     "points": +4,
    #     "explain": "20日跑赢深证"
    # },
    # {
    #     "name": "当日振幅≥5%",
    #     "timeframe": "D",
    #     "window": 10,
    #     "when": "SAFE_DIV(H - L, REF(C,1)) >= 0.05 AND SAFE_DIV(ABS(C - REF(C,1)), REF(C,1)) <= 0.02",
    #     "scope": "EACH",
    #     "points": -5,
    #     "explain": "大波动"
    # },
    # {
    #     "name": "健康缩量",
    #     "timeframe": "D",
    #     "window": 60,
    #     "when": "(COUNT( (CROSS(C, HHV(H, 60)) AND V <= 1.5 * MA(V, 20)), 5 ) >= 1) AND (TS_PCT(V, 20) <= 0.35)",
    #     "scope": "ANY",
    #     "points": +5,
    #     "explain": "健康缩量",
    #     "show_reason": False
    # },
    # {
    #     "name": "3/4 阴量线",
    #     "timeframe": "D",
    #     "window": 20,
    #     "when": "REF(TS_PCT(C,20),1) > 0.9 AND (C < O) AND (C < REF(C, 1)) AND (SAFE_DIV(V, REF(V, 1)) >= 0.6) AND (SAFE_DIV(V, REF(V, 1)) <= 0.8)",
    #     "scope": "ANY",
    #     "points": -15,
    #     "explain": "3/4 阴量线",
    # },
]

# 初选（硬淘汰）样例：命中任一即淘汰，并写入 blacklist.csv
# 你可以把 “ST/上市天数< N/停牌”等标的过滤，也写成这里的规则。
SC_PRESCREEN_RULES = [
#     # a) 周线下行并放量：12周内至少3次
#     {
#         "name": "W_下行放量_硬淘汰",
#         "timeframe": "W",
#         "window": 12,
#         "when": " (C<REF(C,1)) AND (V>1.5*MA(V,10)) ",
#         "scope": "COUNT>=3",
#         "hard_penalty": True,
#         "reason": "周线下行并放量(12周内≥3次)"
#     },
#     # b) 月线破位（跌破半年均线）
#     {
#         "name": "M_跌破半年均线_硬淘汰",
#         "timeframe": "M",
#         "window": 12,
#         "when": " C<MA(C,6) ",
#         "scope": "LAST",
#         "hard_penalty": True,
#         "reason": "月线跌破半年均线"
#     },
]
