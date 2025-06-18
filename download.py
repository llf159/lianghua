import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime
from tqdm import tqdm
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')  # 让 print 不乱码
os.environ["PYTHONIOENCODING"] = "utf-8"

# ======== 用户配置 ========
TOKEN = ''#请自行到 https://tushare.pro 获取token
SAVE_DIR = r'E:\\股票'
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
print(f"获取到 {len(stocks)} 支 A 股")

# 下载过程
failed = []
count = 0
start_time = time.time()

print("\n 开始下载历史行情数据...\n")
for i, row in tqdm(stocks.iterrows(), total=len(stocks), ncols=100):
    ts_code = row['ts_code']
    name = row['name']
    filename = f"{ts_code}_{name}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    try:
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            last_date = existing_df['trade_date'].max()
            update_start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y%m%d')
            if update_start > END_DATE:
                continue  # 已是最新
            df = pro.daily(ts_code=ts_code, start_date=update_start, end_date=END_DATE)
            if not df.empty:
                df.sort_values('trade_date', inplace=True)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.drop_duplicates(subset='trade_date', keep='last', inplace=True)
                updated_df.sort_values('trade_date', inplace=True)
                updated_df.to_csv(filepath, index=False)
        else:
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
