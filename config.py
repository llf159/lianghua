import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# 配置参数
HOLD_DAYS = 2  # 买入持有天数

START_DATE = "2005-01-01"
END_DATE = "2025-06-01"

# 数据目录路径（可绝对路径或相对路径）
DATA_DIR = "E://gupiao-hfq"
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
