import pandas as pd
import sys
from pathlib import Path

def parquet_to_csv(parquet_path: str, csv_path: str = None):
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {parquet_path}")

    # 默认输出路径
    if csv_path is None:
        csv_path = parquet_path.with_suffix(".csv")

    print(f"读取 {parquet_path} ...")
    df = pd.read_parquet(parquet_path)

    print(f"写入 {csv_path} ...")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("完成 ✅")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python parquet_to_csv.py input.parquet [output.csv]")
        sys.exit(1)

    parquet_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    parquet_to_csv(parquet_file, csv_file)
