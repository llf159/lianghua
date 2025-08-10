import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# True 使用 Parquet，False 使用旧 CSV
USE_PARQUET = True   
# USE_PARQUET = False

PARQUET_BASE = r"E:\stock_data"   # 与 download_new.py 的 DATA_ROOT 保持一致
PARQUET_ADJ = "daily_qfq_indicators"         # parquet 复权方式
# {"daily","qfq","hfq","daily_indicators","daily_qfq_indicators","daily_hfq_indicators"}

# 配置参数
HOLD_DAYS = 2  # 买入持有天数
START_DATE = "20250101"
END_DATE = "20250801"
# 数据目录路径（可绝对路径或相对路径）
DATA_DIR = "E://gupiao-hfq"
TDX_BUY_PATH = "./buy_rule.txt"
TDX_SELL_PATH = "./sell_rule.txt"
TDX_SELL_OPEN_PATH = "./sell_open_rule.txt"
TDX_SELL_CLOSE_PATH = "./sell_close_rule.txt"

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
