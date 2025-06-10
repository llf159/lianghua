# 配置参数
HOLD_DAYS = 2  # 买入持有天数

START_DATE = "2010-01-01"
END_DATE = "2025-06-01"

# 数据目录路径（可绝对路径或相对路径）
# DATA_DIR = "E://股票//主板"
DATA_DIR = "test"

#open为次日开盘价买入，close为当日收盘价买入,single_open；open涨停不买入
BUY_MODE = "open"

#open为买入后n日开盘价卖出，close为收盘价卖出，strategy为策略卖出
SELL_MODE = "strategy"

MAX_HOLD_DAYS = 3  # -1 表示不限制持有天数，其他正整数表示最多持有几天

FALLBACK_SELL_MODE = "open"#超过持有天数后强制卖出的模式