# 量化交易系统

一个功能完整的Python量化交易系统，集成了数据下载、技术指标计算、股票评分、策略回测、预测模拟等核心功能。

## ✨ 核心特性

- 📊 **数据管理**：Tushare数据下载、DuckDB/SQLite存储、自动增量更新
- 🎯 **智能评分**：多维度股票评分、规则引擎、实时排名、黑白名单
- 📈 **策略回测**：多策略支持、回测引擎、性能分析、组合管理
- 🔮 **预测模拟**：明日K线模拟、场景分析、指标反推、缓存优化
- 🛠️ **规则引擎**：通达信风格表达式、多周期支持、标签系统
- 🖥️ **Web界面**：Streamlit现代化界面、实时更新、数据可视化

## 📋 目录

- [快速开始](#快速开始)
- [系统要求](#系统要求)
- [核心功能](#核心功能)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [使用指南](#使用指南)
- [高级功能](#高级功能)
- [性能优化](#性能优化)
- [故障排查](#故障排查)
- [开发指南](#开发指南)

## 🚀 快速开始

### 1. 环境准备

#### 系统要求
- **Python版本**：3.10+（建议3.10-3.13，避免3.14）
- **操作系统**：Windows 10/11、macOS、Linux
- **内存**：建议 8GB+（大数据量处理需要更多）
- **存储**：建议 2GB+ 可用空间

#### Windows一键脚本
- `tools/setup_win.bat`：一键安装常用包
- `ui_run.bat`：一键启动评分界面

### 2. 配置Tushare Token

#### 获取Token
1. 访问 [Tushare官网](https://tushare.pro) 注册并登录
2. 进入"个人中心 / 账号 / Token"复制个人Token
3. 购买200元积分（推荐）或使用免费数据

#### 配置Token
**方式A：环境变量（推荐）**
```bash
# Windows PowerShell
$env:TUSHARE_TOKEN="你的token"

# Windows CMD
set TUSHARE_TOKEN=你的token

# macOS/Linux
export TUSHARE_TOKEN="你的token"
```

**方式B：配置文件**
编辑 `config.py`：
```python
TOKEN = "你的token"
```

### 3. 数据下载

#### 首次全量下载
```bash
python download.py
# 按提示选择：是否第一次下载？→ y
```

#### 日常增量更新
```bash
python download.py
# 按提示选择：是否第一次下载？→ n
```

### 4. 启动系统

#### 启动评分界面
```bash
# 方式1：直接启动
streamlit run score_ui.py

# 方式2：Windows一键启动
ui_run.bat
```

#### 启动数据浏览界面
```bash
streamlit run app_pv.py
```

## 📊 核心功能

### 数据管理

#### Tushare数据下载
- 支持股票、指数历史数据下载
- 全量/增量下载模式
- 自动检查数据缺口，支持断点续传
- 流式处理，边下载边计算指标
- 多线程并发下载优化

#### 数据存储
- **DuckDB/SQLite**：高性能列式存储，支持复杂查询
- **Parquet/CSV**：标准格式存储，便于数据交换
- **自动回退**：数据库失败时自动回退到JSON存储
- **数据迁移**：支持历史数据迁移和格式转换

#### 技术指标
内置20+技术指标，支持自定义扩展：
- 价格指标：MA、EMA、SMA、BBI等
- 动量指标：KDJ、RSI、MACD、DIFF等
- 量价指标：量比(VR)、Z评分等
- 趋势指标：双均线、多空线等

### 智能评分系统

#### 评分规则引擎
- **通达信风格表达式**：兼容TDX语法，易于编写
- **多周期支持**：日线(D)、周线(W)、月线(M)
- **命中口径**：LAST、ANY、ALL、COUNT、CONSEC等
- **前置门槛**：支持gate条件，实现复杂规则
- **多子句组合**：跨周期AND组合

#### 评分输出
- **全市场排名**：全市场股票实时评分排名
- **Top-K筛选**：自动生成Top-K股票列表
- **黑白名单**：自动生成初选白名单和淘汰黑名单
- **个股详情**：详细的评分理由和指标数据
- **特别关注榜**：基于历史榜单的股票追踪

### 策略回测

#### 回测引擎
- **多策略支持**：排名策略、筛选策略、预测策略、持仓检查策略
- **买卖模式**：支持多种买卖模式，可配置手续费和滑点
- **持仓管理**：支持止损、止盈、加仓等操作
- **性能分析**：详细的回测报告，包含收益率、最大回撤、夏普比率等

### 预测模拟

#### 明日模拟
- **K线生成**：支持多种价格模式和成交量模式
- **场景分析**：支持牛市、熊市、中性等多种市场场景
- **指标反推**：反推到指定指标值时的价格
- **缓存机制**：智能缓存预测结果，提高计算效率

#### 模拟模式
- `close_pct`：收盘涨跌
- `open_pct`：开盘涨跌
- `gap_then_close_pct`：跳空+收盘涨跌
- `flat`：平盘
- `limit_up/limit_down`：涨停/跌停

### 用户界面

#### Web界面功能
- **股票排名**：实时查看全市场股票评分排名
- **个股详情**：查看个股详细评分理由和指标数据
- **持仓管理**：管理投资组合，跟踪持仓表现
- **预测模拟**：模拟明日K线，分析不同场景
- **规则编辑**：可视化编辑评分规则，实时测试
- **数据浏览**：浏览和查询历史数据

## 📁 项目结构

```
lianghua/
├── 核心模块/
│   ├── config.py              # 配置文件
│   ├── download.py            # 数据下载模块
│   ├── database_manager.py    # 数据库管理模块
│   ├── scoring_core.py        # 评分核心引擎
│   ├── predict_core.py        # 预测模拟模块
│   ├── stats_core.py          # 统计分析模块
│   ├── indicators.py          # 技术指标模块
│   ├── strategies_repo.py     # 策略仓库
│   └── tdx_compat.py          # 通达信兼容层
├── 界面模块/
│   ├── score_ui.py            # 主评分界面
│   └── app_pv.py              # 数据浏览界面
├── 工具模块/
│   ├── utils.py               # 工具函数
│   └── tools/                 # 工具脚本目录
├── 手册/                      # 使用手册
│   ├── README.md              # 手册索引（本文件）
│   ├── 快速入门.md            # 快速入门指南
│   ├── 规则编辑方法.md        # 规则编辑手册
│   └── 支持的表达式与计算方法.md  # 表达式语法参考
├── 数据目录/
│   ├── stock_data/            # 股票数据（DuckDB/SQLite）
│   ├── cache/                 # 缓存目录
│   ├── output/                # 输出目录
│   └── log/                   # 日志目录
└── 配置文件/
    ├── ui_run.bat             # Windows启动脚本
    └── ...
```

## ⚙️ 配置说明

### 基础配置

编辑 `config.py` 进行配置：

#### 数据源配置
```python
# Tushare API Token（必填）
TOKEN = "你的token"

# 数据存储根目录
DATA_ROOT = "stock_data"

# 数据下载配置
ASSETS = ["stock", "index"]  # 可选: ["stock"], ["index"], ["stock","index"]
START_DATE = "20250101"
END_DATE = "today"  # today或具体日期 'YYYYMMDD'

# 复权方式
API_ADJ = "qfq"  # 可选: "qfq" | "hfq" | "raw"
```

#### 数据库配置
```python
# 数据存储模式
DATA_STORAGE_MODE = "duckdb"  # 可选: "duckdb" | "parquet" | "auto"
UNIFIED_DB_TYPE = "duckdb"    # 推荐使用DuckDB，性能更好
UNIFIED_DB_PATH = "stock_data.db"

# DuckDB配置
DUCKDB_THREADS = 16
DUCKDB_MEMORY_LIMIT = "18GB"
DUCKDB_BATCH_SIZE = 300
```

#### 评分系统配置
```python
# 基础配置
SC_REF_DATE = "today"        # 参考日期：'today' 或 'YYYYMMDD'
SC_LOOKBACK_D = 60           # 评分窗口（天）
SC_PRESCREEN_LOOKBACK_D = 180 # 初选窗口（天）

# 评分参数
SC_BASE_SCORE = 50           # 基础分数
SC_MIN_SCORE = 0             # 最低分数
SC_TOP_K = 100               # Top-K数量
SC_TIE_BREAK = "kdj_j_asc"   # 并列打破规则

# 并行配置
SC_MAX_WORKERS = 16          # 并行工作线程数（默认：min(2*CPU, 16)）
SC_READ_TAIL_DAYS = None    # 若不为None，则强制只读最近N天数据

# 存储配置
SC_DETAIL_STORAGE = "database"  # 存储方式：'json' | 'database' | 'both'
SC_DETAIL_DB_TYPE = "duckdb"    # 数据库类型：'sqlite' | 'duckdb' | 'postgres'
SC_USE_DB_STORAGE = True
SC_DB_FALLBACK_TO_JSON = True
```

#### 线程配置
```python
# 增量下载线程数
STOCK_INC_THREADS = 12

# 快速初始化线程数
FAST_INIT_THREADS = 16

# 失败重试线程数
FAILED_RETRY_THREADS = 10
```

## 📖 使用指南

### 数据管理

#### 查看数据
```python
from database_manager import query_stock_data, get_trade_dates

# 查询股票数据
df = query_stock_data(
    ts_code="600519.SH",
    start_date="20240101",
    end_date="20241231",
    columns=["trade_date", "open", "high", "low", "close", "vol"]
)

# 获取交易日列表
dates = get_trade_dates(start_date="20240101", end_date="20241231")
```

#### 数据完整性检查
```bash
# 启动数据浏览界面
streamlit run app_pv.py
```

### 评分系统

#### 运行评分
```python
from scoring_core import run_for_date

# 运行当日评分
result_path = run_for_date()

# 运行指定日期评分
result_path = run_for_date("20250115")
```

#### 查看评分结果
评分结果保存在 `output/score/` 目录：
- `score_all_YYYYMMDD.csv`：全市场评分排名
- `score_top_YYYYMMDD.csv`：Top-K评分排名
- `details/`：个股详情数据库
- `cache/scorelists/YYYYMMDD/`：黑白名单缓存

#### 自定义评分规则
编辑 `strategies_repo.py` 或通过Web界面添加规则：

```python
# 在 strategies_repo.py 的 RANKING_RULES 中添加
{
    'name': '放量突破',
    'timeframe': 'D',
    'window': 20,
    'when': 'C > HHV(H,20) AND V > 1.5*MA(V,20)',
    'scope': 'LAST',
    'points': 15,
    'explain': '放量突破20日高点'
}
```

### 预测模拟

#### 运行预测
```python
from predict_core import run_prediction, PredictionInput

# 创建预测输入
inp = PredictionInput(
    ref_date="20250115",
    codes=["600519.SH", "000001.SZ"],
    scenarios=[
        {
            "mode": "gap_then_close_pct",
            "gap_pct": -0.5,
            "pct": 2.0,
            "hl_mode": "atr_like",
            "atr_mult": 1.2,
            "vol_mode": "mult",
            "vol_arg": 1.5
        }
    ]
)

# 运行预测
result = run_prediction(inp)
```

### Web界面使用

#### 股票排名
1. 启动界面：`streamlit run score_ui.py`
2. 选择日期：在"股票排名"标签页选择评分日期
3. 查看排名：查看全市场股票评分排名和Top-K列表
4. 筛选股票：可按分数、代码等条件筛选

#### 个股详情
1. 点击股票代码：在排名列表中点击股票代码
2. 查看详情：查看个股详细评分理由和指标数据
3. 查看历史：查看个股历史评分趋势

#### 规则编辑
1. 进入"规则编辑"标签页
2. 添加/编辑规则：使用可视化编辑器编辑规则
3. 实时测试：输入股票代码实时测试规则效果
4. 保存规则：将规则保存到 `strategies_repo.py`

#### 持仓管理
1. 进入"组合模拟/持仓"标签页
2. 添加持仓：手动添加或从排名中选择
3. 查看表现：查看持仓股票的表现和评分
4. 检查策略：使用持仓检查策略判断买卖时机

## 🛠️ 高级功能

### 自定义指标

在 `indicators.py` 中注册新指标：

```python
from indicators import IndMeta, REGISTRY

def my_indicator_func(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """自定义指标计算函数"""
    fast = kwargs.get('fast', 10)
    slow = kwargs.get('slow', 30)
    result = pd.DataFrame()
    result['my_col'] = df['close'].ewm(span=fast).mean() - df['close'].ewm(span=slow).mean()
    return result

# 注册指标
REGISTRY["my_indicator"] = IndMeta(
    name="my_indicator",
    out={"my_col": 3},  # 输出列名和预热天数
    tdx="MY_COL := EMA(CLOSE, 10) - EMA(CLOSE, 30);",  # TDX脚本
    py_func=my_indicator_func,
    kwargs={"fast": 10, "slow": 30},
    tags=["product"]
)
```

### 表达式语法

支持通达信风格表达式，详见[支持的表达式与计算方法.md](手册/支持的表达式与计算方法.md)

#### 基础变量
- `C/Close`：收盘价
- `O/Open`：开盘价
- `H/High`：最高价
- `L/Low`：最低价
- `V/Vol`：成交量

#### 常用函数
- `MA(C, 20)`：20日移动平均
- `EMA(C, 12)`：12日指数移动平均
- `HHV(H, 20)`：20日内最高价
- `LLV(L, 20)`：20日内最低价
- `CROSS(MA(C,5), MA(C,20))`：金叉
- `REF(C, 1)`：前一日收盘价

#### 逻辑运算符
- `AND` / `&`：逻辑与
- `OR` / `|`：逻辑或
- `NOT` / `~`：逻辑非

### 规则编辑

详见[规则编辑方法.md](手册/规则编辑方法.md)

#### 最小模板
```json
{
  "when": "C > MA(C, 20)",
  "scope": "LAST",
  "points": 5,
  "explain": "收盘价高于20日均线"
}
```

#### 复杂规则
```json
{
  "name": "放量突破确认",
  "when": "C > HHV(H,20)",
  "gate": "V > 2*MA(V,20)",
  "scope": "LAST",
  "points": 20,
  "explain": "放量突破20日高点"
}
```

## 📈 性能优化

### 内存优化
- **调整DuckDB内存限制**：`DUCKDB_MEMORY_LIMIT = "18GB"`
- **限制读取数据量**：`SC_READ_TAIL_DAYS = 60`（只读最近60天）
- **启用流式计算**：`INC_STREAM_COMPUTE_INDICATORS = True`

### 并发优化
- **调整并行度**：`SC_MAX_WORKERS = 16`（根据CPU核心数调整）
- **优化线程数**：`STOCK_INC_THREADS = 12`、`FAST_INIT_THREADS = 16`
- **启用进程池**：`SC_USE_PROCESS_POOL = True`（安全条件下）

### 存储优化
- **使用DuckDB**：列式存储，查询性能更好
- **启用自动压实**：`DUCKDB_ENABLE_COMPACT_AFTER = True`
- **定期清理缓存**：清理 `cache/` 目录

### 数据库优化
- **连接池配置**：`DB_POOL_SIZE = 16`（与SC_MAX_WORKERS对齐）
- **批量写入**：`DUCKDB_BATCH_SIZE = 300`
- **启用索引**：`DB_ENABLE_INDEXES = True`

## 🔍 故障排查

### 常见问题

#### 1. Tushare Pro未配置
```
错误：Tushare Pro 未配置
解决：
1. 检查 config.TOKEN 是否已设置
2. 检查环境变量 TUSHARE_TOKEN 是否已设置
3. 确认Token是否有效
```

#### 2. 内存不足
```
错误：DuckDB内存不足
解决：
1. 降低 DUCKDB_MEMORY_LIMIT（如"8GB"）
2. 降低 SC_MAX_WORKERS（如8）
3. 设置 SC_READ_TAIL_DAYS = 60（限制读取数据量）
```

#### 3. 数据读取失败
```
错误：读取不到数据
解决：
1. 检查 DATA_ROOT 路径是否正确
2. 检查数据库文件是否存在
3. 运行数据完整性检查：streamlit run app_pv.py
```

#### 4. 依赖包缺失
```bash
# 安装缺失的包
pip install pyarrow duckdb tushare streamlit plotly pandas numpy scipy
```

#### 5. 数据库连接失败
```
错误：数据库连接失败
解决：
1. 检查数据库文件权限
2. 检查磁盘空间
3. 运行修复工具：python tools/fix_database_size.py
```

### 日志查看

日志文件位于 `log/` 目录：
- `scoring_core_info.log`：评分系统日志
- `predict_core_info.log`：预测系统日志
- `database_manager_info.log`：数据库管理日志
- `download.log`：数据下载日志

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 或在 config.py 中设置
LOG_LEVEL = "DEBUG"
```

## 📚 开发指南

### 代码结构

- **核心模块**：`scoring_core.py`、`predict_core.py`、`database_manager.py`
- **策略模块**：`strategies_repo.py`、`tdx_compat.py`
- **指标模块**：`indicators.py`
- **界面模块**：`score_ui.py`、`app_pv.py`

### 扩展功能

1. **添加新指标**：在 `indicators.py` 中注册
2. **添加新策略**：在 `strategies_repo.py` 中添加规则
3. **扩展表达式**：在 `tdx_compat.py` 中添加函数支持
4. **添加界面功能**：在 `score_ui.py` 中添加新标签页

