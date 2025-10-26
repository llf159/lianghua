# Details 数据库读取工具

这个工具包提供了读取和查询股票评分详情数据库的功能。

## 文件说明

- `read_details_db.py` - 主要的数据库读取类 `DetailsDBReader`
- `example_read_details.py` - 使用示例脚本
- `README_details_db.md` - 本说明文档

## 主要功能

### DetailsDBReader 类

提供以下主要方法：

1. **get_table_info()** - 获取数据库表信息和统计
2. **query_by_stock(ts_code, limit)** - 根据股票代码查询详情
3. **query_by_date(ref_date, limit)** - 根据日期查询详情
4. **query_top_stocks(ref_date, top_k)** - 查询指定日期的Top-K股票
5. **query_score_range(ref_date, min_score, max_score)** - 查询指定分数范围的股票
6. **query_recent_dates(days)** - 查询最近的N个交易日
7. **get_stock_summary(ts_code)** - 获取股票的历史评分摘要
8. **export_to_csv(output_file, ref_date, limit)** - 导出数据到CSV文件

## 使用方法

### 1. 命令行使用

```bash
# 显示数据库信息
python read_details_db.py --info

# 查询指定股票
python read_details_db.py --stock 000001.SZ

# 查询指定日期的Top-20股票
python read_details_db.py --date 20241201 --top 20

# 查询指定分数范围的股票
python read_details_db.py --date 20241201 --score-min 60 --score-max 80

# 查询最近5个交易日
python read_details_db.py --recent 5

# 获取股票历史摘要
python read_details_db.py --summary 000001.SZ

# 导出数据到CSV
python read_details_db.py --export output.csv --date 20241201 --limit 100
```

### 2. 编程使用

```python
from read_details_db import DetailsDBReader

# 初始化读取器
reader = DetailsDBReader()

# 获取数据库信息
info = reader.get_table_info()
print(f"总记录数: {info['total_records']}")

# 查询Top-10股票
top_stocks = reader.query_top_stocks("20241201", 10)
print(top_stocks[['ts_code', 'score', 'rank']])

# 查询特定股票
stock_data = reader.query_by_stock("000001.SZ", 5)
print(stock_data[['ref_date', 'score', 'rank']])

# 导出数据
reader.export_to_csv("export.csv", "20241201", 100)
```

### 3. 运行示例

```bash
# 运行完整示例
python example_read_details.py
```

## 数据库结构

数据库表 `stock_details` 包含以下主要字段：

- `ts_code` - 股票代码
- `ref_date` - 参考日期
- `score` - 评分
- `rank` - 排名
- `total` - 总股票数
- `tiebreak` - 并列打破值
- `highlights` - 亮点（JSON格式）
- `drawbacks` - 缺点（JSON格式）
- `opportunities` - 机会（JSON格式）
- `rules` - 规则详情（JSON格式）

## 注意事项

1. 确保数据库文件存在且可访问
2. 查询大量数据时注意设置合适的 `limit` 参数
3. 导出CSV文件时会自动添加UTF-8 BOM，确保中文正确显示
4. 所有查询都支持超时设置，默认30秒

## 错误处理

- 数据库文件不存在时会抛出 `FileNotFoundError`
- 查询超时会抛出超时异常
- 所有方法都包含异常处理，会返回空结果而不是崩溃
