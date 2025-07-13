import tushare as ts
import pandas as pd
import os
from datetime import datetime, timedelta
import time

TOKEN = '你的token'
SAVE_DIR = r'E:\gupiao'
MAX_PER_MIN = 50

ts.set_token(TOKEN)
pro = ts.pro_api()
os.makedirs(SAVE_DIR, exist_ok=True)

stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
stocks = stocks[stocks['ts_code'].str.startswith(('60', '000', '001'))]

count = 0
start_time = time.time()

for _, row in stocks.iterrows():
    ts_code, name = row['ts_code'], row['name']
    filename = f"{ts_code}_{name}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    if os.path.exists(filepath):
        df_old = pd.read_csv(filepath)
        last_date = df_old['trade_date'].max()
        start_date = (datetime.strptime(last_date, "%Y%m%d") + timedelta(days=1)).strftime('%Y%m%d')
    else:
        start_date = '20050101'
        df_old = pd.DataFrame()

    end_date = datetime.today().strftime('%Y%m%d')
    if start_date > end_date:
        continue

    try:
        df_new = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if not df_new.empty:
            df_new.sort_values('trade_date', inplace=True)
            df = pd.concat([df_old, df_new], ignore_index=True)
            df.drop_duplicates(subset='trade_date', keep='last', inplace=True)
            df.to_csv(filepath, index=False)
    except Exception as e:
        print(f"更新失败：{ts_code} - {e}")

    count += 1
    if count % MAX_PER_MIN == 0:
        elapsed = time.time() - start_time
        if elapsed < 60:
            time.sleep(60 - elapsed)
        start_time = time.time()
