import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ========== 用户配置 ==========
TOKEN = ''
API_NAME = "pro_bar"
START_DATE = '20050101'
END_DATE = datetime.today().strftime('%Y%m%d')
MAX_WORKERS = 20  # 并发线程数
SAVE_DIR = r'E:\gupiao'
# ==============================

ts.set_token(TOKEN)
pro = ts.pro_api()
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 缓存目录设置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(CACHE_DIR, exist_ok=True)

def load_or_fetch(name, fetch_func):
    path = os.path.join(CACHE_DIR, f'{name}.csv')
    if os.path.exists(path):
        ans = input(f"[提示] 检测到已有缓存 {name}.csv，是否更新？(y/n)：").strip().lower()
        if ans != 'y':
            print(f"[缓存] 使用本地文件：{path}")
            return pd.read_csv(path, dtype=str, encoding='utf-8')
        else:
            print(f"[更新] 正在重新获取 {name} 数据...")
    else:
        print(f"[首次] 获取 {name} 数据...")

    df = fetch_func()
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"[缓存] 已保存至：{path}")
    return df

# ========== 获取股票列表 ==========
stocks = load_or_fetch(
    "stock_list",
    lambda: pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
)
print(f"[提示] 共获取到 {len(stocks)} 支股票")

# ========== 下载函数 ==========
def download_one_stock(row, retries=1, retry_delay=60):
    ts_code = row['ts_code']
    name = row['name']
    filename = f"{ts_code}_{API_NAME}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    if os.path.exists(filepath):
        return (ts_code, '已完成')
    
    err_msg = "未知错误"

    for attempt in range(1, retries + 1):
        try:
            df = ts.pro_bar(
                ts_code=ts_code,
                start_date=START_DATE,
                end_date=END_DATE,
                adj='qfq',
                freq='D',
                asset='E'
            )
            if df is None or df.empty:
                return (ts_code, '空数据')

            df.sort_values('trade_date', inplace=True)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            return (ts_code, '成功')

        except Exception as e:
            err_msg = str(e)
            if '每分钟最多访问该接口' in err_msg or '超过访问频次' in err_msg:
                print(f"[限频] {ts_code} 第{attempt}次失败：触发限频，暂停 {retry_delay} 秒...")
                time.sleep(retry_delay)
            else:
                print(f"[错误] {ts_code} 第{attempt}次失败：{err_msg}")
                time.sleep(3)

    return (ts_code, f'失败: {err_msg}')

# ========== 多线程执行 ==========
results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_one_stock, row): row['ts_code'] for _, row in stocks.iterrows()}

    pbar = tqdm(
        total=len(futures),
        desc="下载进度",
        ncols=120,
        unit="支",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for future in as_completed(futures):
        result = future.result()
        ts_code, status = result
        pbar.set_description(f"下载中：{ts_code} ({status})")
        pbar.update(1)
        results.append(result)

    pbar.close()

# ========== 统计结果 ==========
success, skipped, failed, empty = [], [], [], []
for ts_code, status in results:
    if status == '成功':
        success.append(ts_code)
    elif status == '已完成':
        skipped.append(ts_code)
    elif status == '空数据':
        empty.append(ts_code)
    else:
        failed.append(f"{ts_code} - {status}")

print(f"\n下载完成：成功 {len(success)}，已跳过 {len(skipped)}，空数据 {len(empty)}，失败 {len(failed)}")

# ========== 记录失败 ==========
if failed:
    failed_log = os.path.join(SAVE_DIR, f'failed_{API_NAME}.txt')
    with open(failed_log, 'w', encoding='utf-8') as f:
        for item in failed:
            f.write(item + '\n')
    print(f"失败记录已保存至：{failed_log}")
