import pandas as pd
import argparse
import os
from rich.console import Console
from rich.table import Table

def print_parquet_head(file_path: str, n: int = 10):
    if not os.path.isfile(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return

    try:
        df = pd.read_parquet(file_path)
        if df.empty:
            print("[信息] 文件为空")
            return

        df = df.head(n)

        console = Console()
        table = Table(title=f"{file_path} (前 {n} 行)")

        for col in df.columns:
            table.add_column(col, overflow="fold")

        for _, row in df.iterrows():
            table.add_row(*[str(x) for x in row.tolist()])

        console.print(table)

    except Exception as e:
        print(f"[错误] 读取失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parquet 文件查看器")
    parser.add_argument("file", help="parquet 文件路径")
    parser.add_argument("-n", "--num", type=int, default=10, help="显示前几行 (默认 10)")
    args = parser.parse_args()

    print_parquet_head(args.file, args.num)
