# 配置参数
HOLD_DAYS = 2  # 买入持有天数
START_DATE = "2010-01-01"
END_DATE = "2025-06-01"
# 数据目录路径（可绝对路径或相对路径）
# DATA_DIR = "data"
DATA_DIR = "E://gupiao"
#open为次日开盘价买入，close为当日收盘价买入；open涨停不买入
BUY_MODE = "open"
#open为买入后n日开盘价卖出，close为收盘价卖出
SELL_MODE = "open"