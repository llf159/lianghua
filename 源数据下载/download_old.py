import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime
from tqdm import tqdm

# ======== 用户配置 ========
TOKEN = ''#请自行到 https://tushare.pro 获取token
SAVE_DIR = r'E:\gupiao'
START_DATE = '20050101'
END_DATE = datetime.today().strftime('%Y%m%d')
MAX_PER_MIN = 50
FAILED_LOG = os.path.join(SAVE_DIR, 'failed_log.txt')
# ==========================

# 初始化
ts.set_token(TOKEN)
pro = ts.pro_api()
os.makedirs(SAVE_DIR, exist_ok=True)

# 获取A股列表
print("正在获取 A 股列表...")
stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
stocks = stocks[stocks['ts_code'].str.startswith(('60', '000', '001'))]
print(f"获取到 {len(stocks)} 支 A 股")

# 已完成的股票（文件名为 ts_code_名称.csv）
completed = set([f.split('_')[0] for f in os.listdir(SAVE_DIR) if f.endswith('.csv')])
failed = []

# 下载过程
count = 0
start_time = time.time()

print("\n开始下载历史行情数据...\n")
for i, row in tqdm(stocks.iterrows(), total=len(stocks), ncols=100):
    ts_code = row['ts_code']
    name = row['name']
    filename = f"{ts_code}_{name}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    if ts_code in completed:
        continue

    try:
        df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
        if df.empty:
            failed.append(ts_code)
        else:
            df.sort_values('trade_date', inplace=True)
            df.to_csv(filepath, index=False)
    except Exception as e:
        failed.append(ts_code)

    count += 1
    if count % MAX_PER_MIN == 0:
        elapsed = time.time() - start_time
        if elapsed < 60:
            time.sleep(60 - elapsed)
        start_time = time.time()

# 写入失败的代码
if failed:
    with open(FAILED_LOG, 'w') as f:
        for code in failed:
            f.write(code + '\n')
    print(f"\n共 {len(failed)} 支股票下载失败，已记录在：{FAILED_LOG}")
else:
    print("\n所有股票均成功下载！")

print(f"\n下载完成！数据已保存在：{SAVE_DIR}")
