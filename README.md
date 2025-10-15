# 量化交易系统

一个功能完整的量化交易系统，集成了数据下载、技术指标计算、股票评分、策略回测、组合管理、预测模拟等核心功能。

## 🚀 核心功能

### 📊 数据管理
- **Tushare数据下载**：支持股票、指数历史数据全量/增量下载
- **多格式存储**：Parquet + CSV双格式存储，支持DuckDB加速查询
- **数据完整性**：自动检查数据缺口，支持断点续传
- **指标计算**：内置KDJ、RSI、BBI、量比等20+技术指标

### 🎯 智能评分系统
- **多维度评分**：基于技术指标、趋势分析、量价关系等综合评分
- **规则引擎**：支持通达信风格表达式，灵活配置评分规则
- **实时排名**：全市场股票实时评分排名，支持Top-K筛选
- **黑白名单**：自动生成初选白名单和淘汰黑名单

### 📈 策略回测
- **多策略支持**：排名策略、筛选策略、预测策略、持仓检查策略
- **回测引擎**：支持多种买卖模式，可配置手续费和滑点
- **性能分析**：详细的回测报告，包含收益率、最大回撤、夏普比率等
- **组合管理**：支持多组合并行管理，实时持仓跟踪

### 🔮 预测模拟
- **明日模拟**：基于历史数据模拟次日开盘表现
- **场景分析**：支持多种市场场景的预测分析
- **缓存机制**：智能缓存预测结果，提高计算效率
- **风险评估**：预测结果包含置信度和风险提示

### 💾 数据存储
- **数据库支持**：SQLite/DuckDB双数据库支持
- **JSON存储**：结构化JSON存储个股详情
- **自动回退**：数据库失败时自动回退到JSON存储
- **数据迁移**：支持历史数据迁移和格式转换

### 🖥️ 用户界面
- **Web界面**：基于Streamlit的现代化Web界面
- **多标签页**：排名、详情、持仓、预测、规则编辑等
- **实时更新**：支持实时数据更新和进度显示
- **数据可视化**：丰富的图表和统计展示

## 📋 系统要求

### 环境配置
- **Python版本**：推荐 Python 3.10+（避免使用 Python 3.14）
- **操作系统**：Windows 10/11、macOS、Linux
- **内存**：建议 8GB+（大数据量处理需要更多内存）
- **存储**：建议 50GB+ 可用空间

### 依赖包
```bash
# 基础依赖
pip install pandas numpy pyarrow duckdb tushare tqdm streamlit plotly matplotlib xlsxwriter tabulate openpyxl

# 可选依赖（用于特定功能）
pip install gradio  # 用于Parquet浏览界面
```

## ⚙️ 快速开始

### 1. 环境配置

#### 1.1 安装依赖
```bash
# 克隆项目
git clone <repository-url>
cd 量化

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -U pip
pip install pandas numpy pyarrow duckdb tushare tqdm streamlit plotly matplotlib xlsxwriter tabulate openpyxl
```

#### 1.2 Windows一键脚本（可选）
- `setup_win.bat`：一键安装常用包
- `app_score.bat`：一键启动评分界面

### 2. 配置Tushare Token

#### 2.1 获取Token
1. 访问 [Tushare官网](https://tushare.pro) 注册并登录
2. 进入"个人中心 / 账号 / Token"复制个人Token
3. 购买200元积分（推荐）或使用免费数据

#### 2.2 配置Token
**方式A：环境变量**
```bash
# Windows PowerShell
$env:TUSHARE_TOKEN="你的token"

# Windows CMD
set TUSHARE_TOKEN=你的token

# macOS/Linux
export TUSHARE_TOKEN="你的token"
```

**方式B：配置文件（推荐）**
编辑 `config.py`：
```python
TOKEN = "你的token"
```

### 3. 数据下载

#### 3.1 首次全量下载
```bash
python download.py
# 按提示选择：是否第一次下载？→ y
```

#### 3.2 日常增量更新
```bash
python download.py
# 按提示选择：是否第一次下载？→ n
```

### 4. 启动系统

#### 4.1 启动评分界面
```bash
# 方式1：直接启动
streamlit run score_ui.py

# 方式2：Windows一键启动
app_score.bat
```

#### 4.2 启动Parquet浏览界面
```bash
streamlit run app_pv.py
```

## 📁 项目结构

```
量化/
├── 核心模块/
│   ├── config.py              # 配置文件
│   ├── download.py            # 数据下载模块
│   ├── scoring_core.py        # 评分核心引擎
│   ├── predict_core.py        # 预测模拟模块
│   ├── stats_core.py          # 统计分析模块
│   ├── detail_db.py           # 数据库存储模块
│   └── indicators.py          # 技术指标模块
├── 界面模块/
│   ├── score_ui.py            # 主评分界面
│   ├── app_pv.py              # Parquet浏览界面
│   └── parquet_viewer.py      # 数据查看工具
├── 策略模块/
│   ├── strategies_repo.py     # 策略仓库
│   ├── tdx_compat.py          # 通达信兼容层
│   └── backtest_core.py       # 回测核心
├── 工具模块/
│   ├── utils.py               # 工具函数
│   ├── tools/                 # 工具脚本目录
│   └── 手册/                  # 使用手册
├── 数据目录/
│   ├── cache/                 # 缓存目录
│   ├── output/                # 输出目录
│   └── log/                   # 日志目录
└── 配置文件/
    ├── setup_win.bat          # Windows安装脚本
    └── app_score.bat          # Windows启动脚本
```

## 🔧 配置说明

### 数据配置
```python
# config.py 关键配置
PARQUET_BASE = r"E:\stock_data"        # 数据存储根目录
PARQUET_ADJ = "qfq"                    # 复权方式：qfq/hfq/raw/daily
PARQUET_USE_INDICATORS = True          # 是否使用带指标的分区
```

### 评分配置
```python
# 评分系统配置
SC_REF_DATE = "today"                  # 参考日期
SC_LOOKBACK_D = 60                     # 评分窗口（天）
SC_TOP_K = 100                         # Top-K数量
SC_BASE_SCORE = 50                     # 基础分数
SC_MIN_SCORE = 0                       # 最低分数
```

### 数据库配置
```python
# 数据库存储配置
SC_DETAIL_STORAGE = "database"         # 存储方式：database/json/both
SC_DETAIL_DB_TYPE = "sqlite"           # 数据库类型：sqlite/duckdb
SC_USE_DB_STORAGE = True               # 是否启用数据库存储
SC_DB_FALLBACK_TO_JSON = True          # 数据库失败时回退到JSON
```

## 📊 使用指南

### 1. 数据管理

#### 查看数据
```python
from parquet_viewer import read_by_symbol, read_range

# 读取单股数据
df = read_by_symbol("E:/stock_data", adj="qfq", ts_code="600519.SH", with_indicators=True)

# 读取时间区间数据
df = read_range("E:/stock_data", asset="stock", adj="qfq", 
                ts_code="600519.SH", start="20240101", end="20240630")
```

#### 数据完整性检查
```bash
# 启动Parquet浏览界面
streamlit run app_pv.py
```

### 2. 评分系统

#### 运行评分
```python
from scoring_core import run_for_date

# 运行当日评分
result_path = run_for_date()

# 运行指定日期评分
result_path = run_for_date("20250115")
```

#### 自定义评分规则
编辑 `strategies_repo.py` 或通过Web界面添加规则：
```python
{
    'name': '自定义规则',
    'timeframe': 'D',
    'window': 20,
    'when': 'C > MA(C, 20)',
    'scope': 'LAST',
    'points': 5,
    'explain': '收盘价高于20日均线'
}
```

### 3. 预测模拟

#### 运行预测
```python
from predict_core import run_prediction, PredictionInput

# 创建预测输入
inp = PredictionInput(
    ref_date="20250115",
    codes=["600519.SH", "000001.SZ"],
    scenarios=["bull", "bear", "neutral"]
)

# 运行预测
result = run_prediction(inp)
```

### 4. 组合管理

#### 创建组合
通过Web界面的"组合模拟/持仓"标签页创建和管理投资组合。

#### 持仓跟踪
系统自动跟踪组合表现，提供详细的持仓分析和风险提示。

## 🛠️ 高级功能

### 1. 自定义指标

在 `indicators.py` 中添加自定义指标：
```python
REGISTRY["my_indicator"] = IndMeta(
    name="my_indicator",
    out={"my_col": 3},
    tdx="MY_COL := EMA(CLOSE, 10) - EMA(CLOSE, 30);",
    py_func=lambda df, **kw: my_indicator_func(df, **kw),
    kwargs={"fast": 10, "slow": 30},
    tags=["product"]
)
```

### 2. 策略回测

#### 配置回测参数
```python
# config.py
STRATEGY_START_DATE = "20220601"
STRATEGY_END_DATE = "20250801"
HOLD_DAYS = 2
BUY_MODE = "open"  # open/close/signal_open
SELL_MODE = "other"  # open/close/strategy/other
```

#### 运行回测
```python
from backtest_core import run_backtest
result = run_backtest()
```

### 3. 数据迁移

#### 迁移到数据库
```bash
python tools/migrate_details_to_db.py
```

#### 格式转换
```bash
python tools/parquet_to_csv.py
```

## 📈 性能优化

### 1. 内存优化
- 调整 `DUCKDB_MEMORY_LIMIT` 控制DuckDB内存使用
- 使用 `SC_READ_TAIL_DAYS` 限制读取数据量
- 启用 `INC_STREAM_COMPUTE_INDICATORS` 流式计算指标

### 2. 并发优化
- 调整 `SC_MAX_WORKERS` 控制并行度
- 使用 `FAST_INIT_THREADS` 优化初始下载
- 启用 `STREAM_FLUSH_*` 配置流式处理

### 3. 存储优化
- 使用Parquet格式提高压缩率
- 启用 `DUCKDB_ENABLE_COMPACT_AFTER` 自动压实
- 定期清理缓存目录

## 🔍 故障排查

### 常见问题

#### 1. Tushare Pro未配置
```
错误：Tushare Pro 未配置
解决：检查 config.TOKEN 或环境变量 TUSHARE_TOKEN
```

#### 2. 内存不足
```
错误：DuckDB内存不足
解决：降低 DUCKDB_MEMORY_LIMIT 或 DUCKDB_THREADS
```

#### 3. 数据读取失败
```
错误：读取不到数据
解决：检查 PARQUET_BASE 与 DATA_ROOT 是否对齐
```

#### 4. 依赖包缺失
```bash
# 安装缺失的包
pip install pyarrow duckdb tushare streamlit
```

### 日志查看
- 评分日志：`log/score.log`
- 下载日志：`log/fast_init.log`
- 系统日志：控制台输出

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 开发指南

### 1. 添加新功能
- 在对应模块中添加功能函数
- 更新 `config.py` 添加配置项
- 在Web界面中添加对应标签页

### 2. 自定义策略
- 在 `strategies_repo.py` 中添加策略规则
- 使用通达信风格表达式编写条件
- 通过Web界面测试和调试

### 3. 扩展指标
- 在 `indicators.py` 中注册新指标
- 提供TDX脚本和Python实现
- 设置合适的预热天数

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 📞 支持

如有问题或建议，请：
1. 查看本文档的故障排查部分
2. 检查项目的 Issues 页面
3. 提交新的 Issue 描述问题

---

**注意**：本系统仅供学习和研究使用，不构成投资建议。投资有风险，入市需谨慎。