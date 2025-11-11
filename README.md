# 量化交易系统

一个功能完整的Python量化交易系统，集成了数据下载、技术指标计算、股票评分、策略回测、预测模拟等核心功能。

## 核心特性

- **数据管理**：Tushare数据下载、DuckDB/SQLite存储、自动增量更新
- **智能评分**：多维度股票评分、规则引擎、实时排名、黑白名单
- **策略回测**：多策略支持、回测引擎、性能分析、组合管理
- **预测模拟**：明日K线模拟、场景分析、指标反推、缓存优化
- **规则引擎**：通达信风格表达式、多周期支持、标签系统
- **Web界面**：Streamlit现代化界面、实时更新、数据可视化

## 快速开始

### 1. 环境要求

- Python 3.10+（建议3.10-3.13）
- Windows 10/11、macOS、Linux
- 建议 8GB+ 内存

### 2. 安装依赖

```bash
# Windows一键安装
tools/setup_win.bat

# 或手动安装
pip install pyarrow duckdb tushare streamlit plotly pandas numpy scipy
```

### 3. 配置Tushare Token

获取Token：访问 [Tushare官网](https://tushare.pro) 注册并获取个人Token

配置方式（二选一）：
- **环境变量**（推荐）：
  ```bash
  # Windows PowerShell
  $env:TUSHARE_TOKEN="你的token"
  
  # macOS/Linux
  export TUSHARE_TOKEN="你的token"
  ```
- **配置文件**：编辑 `config.py`，设置 `TOKEN = "你的token"`

### 4. 数据下载

```bash
# 首次全量下载
python download.py
# 按提示选择：是否第一次下载？→ y

# 日常增量更新
python download.py
# 按提示选择：是否第一次下载？→ n
```

### 5. 启动系统

```bash
# 启动评分界面
streamlit run score_ui.py

# 或Windows一键启动
ui_run.bat

# 启动数据浏览界面
streamlit run app_pv.py
```

## 项目结构

```
lianghua/
├── 核心模块/
│   ├── config.py              # 配置文件
│   ├── download.py            # 数据下载
│   ├── scoring_core.py        # 评分引擎
│   ├── predict_core.py        # 预测模拟
│   ├── strategies_repo.py     # 策略仓库
│   └── ...
├── 界面模块/
│   ├── score_ui.py            # 主评分界面
│   └── app_pv.py              # 数据浏览界面
├── 手册/                      # 使用手册
│   ├── 规则编辑方法.md        # 规则编辑手册
│   └── 支持的表达式与计算方法.md  # 表达式语法参考
└── ...
```

## 主要功能

### 数据管理
- Tushare数据下载（全量/增量）
- DuckDB/SQLite存储
- 20+技术指标计算

### 智能评分
- 通达信风格表达式
- 多周期支持（日线/周线/月线）
- 实时排名和黑白名单生成
- 可视化规则编辑器

### 策略回测
- 多策略支持（排名/筛选/预测/持仓）
- 性能分析和组合管理

### 预测模拟
- 明日K线模拟
- 多种价格和成交量模式
- 场景分析

## 使用手册

详细使用说明请参考：

- [规则编辑方法](手册/规则编辑方法.md) - 规则配置和表达式语法
- [支持的表达式与计算方法](手册/支持的表达式与计算方法.md) - 函数和变量参考

## 配置说明

主要配置在 `config.py` 文件中：

- **数据源**：Tushare Token、数据存储路径
- **评分系统**：评分窗口、Top-K数量、并行配置
- **数据库**：存储模式、内存限制、线程数

## 许可证

本项目仅供学习和研究使用。
