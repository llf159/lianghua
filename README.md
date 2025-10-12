# 上手手册

---

## 1) 环境配置（优先）

### 1.1 Python 版本
- 推荐 **Python 3.10+**。  
- 注意不要使用 **Python 3.14**。

### 1.2 安装依赖
**基础依赖（必须）**：
```bash
pip install -U pip
pip install pandas numpy pyarrow duckdb tushare tqdm streamlit plotly matplotlib xlsxwriter tabulate openpyxl
```
### 1.3 Windows 一键脚本（可选）
- `setup_win.bat` 会一次性安装常用包（若缺 `duckdb/tushare/streamlit`，请再手动执行上面的基础依赖安装）。  
- `app_score.bat` 一键启动评分界面（等同 `streamlit run score_ui.py`）。

> 提示：双击 `.bat` 之前，**先**在当前目录打开 `cmd` 或 PowerShell 并激活虚拟环境（见 1.1）。

---

## 2) 获取并配置 Tushare Token（重点）

### 2.1 如何获取 Token
1. 访问 [Tushare 官网](https://tushare.pro)，注册并登录 **Tushare Pro**；  

2. 进入“个人中心 / 账号 / Token”，复制你的 **个人 Token**。

3. 本程序使用了付费数据，请通过官网付费通道购买200元积分；

   或使用免费不复权数据，需要修改download接口调用和限频器参数

> 建议把 Token 视为**私密凭证**，不要提交到公开仓库。

### 2.2 在本项目里配置 Token 的两种方式

**方式 A：环境变量  

- Windows PowerShell：
  ```powershell
  $env:TUSHARE_TOKEN="你的token"
  ```
- Windows CMD：
  ```cmd
  set TUSHARE_TOKEN=你的token
  ```
- macOS / Linux（bash/zsh）：
  ```bash
  export TUSHARE_TOKEN="你的token"
  ```

**方式 B：写入 `config.py`**  (简单)

- 打开项目根目录 `config.py`，设置：
  ```python
  TOKEN = "你的token"
  ```

> 运行时，程序**优先读取 `config.TOKEN`**，若为空则回退到环境变量 `TUSHARE_TOKEN`。

---

## 3) 数据目录与对齐关系（很重要）

- 下载任务使用 `download.py` 的根目录：`DATA_ROOT`  
- 评分/可视化使用 `config.py` 的：
  - `PARQUET_BASE`（**需与 `DATA_ROOT` 对齐**）
  - `PARQUET_ADJ`：`"daily" | "raw" | "qfq" | "hfq"`
  - `PARQUET_USE_INDICATORS`：`True` 使用 `*_indicators` 分区

> 简言之：**哪里下载，哪里可视化**。把 `PARQUET_BASE` 指到 `DATA_ROOT` 所在目录即可。

---

## 4) 首次全量 & 日常增量

### 4.1 首次全量（第一次执行，历史全拉）
```bash
python download.py
# 按提示选择：是否第一次下载？→ y
```
**会完成：**  
- 股票/指数全历史拉取（按 `trade_date=YYYYMMDD` 分区到 Parquet）；  
- 生成/更新 **单股成品**（可选带指标列）；  
- 生成元信息、必要的索引/合并。

**关键参数（在 `download.py` 或其配置段中）：**
- `ASSETS=["stock","index"]`（可先只拉 `"stock"`）
- `API_ADJ="qfq"`（初次全量的复权口径）
- 并发：`FAST_INIT_THREADS`、`STOCK_INC_THREADS`
- DuckDB 资源：`DUCKDB_THREADS`、`DUCKDB_MEMORY_LIMIT`、`DUCKDB_ENABLE_COMPACT_AFTER`

### 4.2 日常增量（每天或按需补最新交易日）
```bash
python download.py
# 按提示选择：是否第一次下载？→ n
```
**会完成：**  
- 拉取新增交易日；  
- 识别受影响股票，**只对增量区间**重算单股成品（含指标）；  
- 必要时做分区合并/压实。

> 常见问题：若提示 “Tushare Pro 未配置”，请检查 2.2 的 Token 设置；若读取 Parquet 报错，确保已安装 `pyarrow`。

---

## 5) 查看数据（Parquet 浏览/读取）

### 5.1 图形界面（*Parquet 浏览 App*）
仓库若包含 app_pv.py，可直接双击运行：

**界面能力（典型）：**选择股票、时间区间、是否使用带指标分区、显示 Schema/样例行、导出片段等。

### 5.2 编程方式（`parquet_viewer.py`）
```python
from parquet_viewer import read_by_symbol, read_range, asset_root, allowed_stock_adjs

BASE = r"E:/stock_data"  # 与 DATA_ROOT / PARQUET_BASE 一致

# 单股成品（带指标）
df1 = read_by_symbol(BASE, adj="qfq", ts_code="600519.SH", with_indicators=True)

# 按区间读取（日线分区，优先 duckdb，回退 pandas/pyarrow）
df2 = read_range(BASE, asset="stock", adj="qfq",
                 ts_code="600519.SH", start="20240101", end="20240630",
                 columns=["ts_code","trade_date","open","high","low","close","vol"])
```

**常用函数速览**
- `allowed_stock_adjs(base)`：列出已有复权口径/分区（如 `daily_qfq_indicators`）  
- `normalize_stock_adj(base, adj_kind, with_indicators)`：从 `qfq/raw/hfq/daily` + 是否带指标 → 实际目录名  
- `list_symbols(root)` / `list_trade_dates(root)`：扫描已有股票与日期分区  
- `scan_with_duckdb(...)` / `scan_with_pandas(...)`：两种后端（自动择优）

---

## 6) 指标系统（`indicators.py`）——介绍与配置（加量）

### 6.1 已内置指标（节选）
| 名称 | 输出列（小数位） | 说明 / 等价表达 |
|---|---|---|
| `kdj` | `j`(2) | 经典 KDJ 的 J 值（`RSV→K→D→J`） |
| `volume_ratio` | `vr`(4) | 量比，`V / MA(V, 20)` |
| `bbi` | `bbi`(2) | 多均线均值：MA(3,6,12,24) 平均 |
| `rsi` | `rsi`(2) | 相对强弱指标，默认 `N=14` |

> 每个指标在 `REGISTRY` 中以 `IndMeta` 描述：输出列、小数位、TDX 脚本与 Python 兜底函数等。TDX 失败时自动回退 Python。

### 6.2 选择要计算的指标
在增量/全量产品化过程中，可通过配置控制：
- `SYMBOL_PRODUCT_INDICATORS = "all"`（默认）→ 计算全部已注册指标；  
- 或设为**逗号分隔**列表：`"kdj,rsi,bbi,duokong_short,duokong_long"`；  
- `SYMBOL_PRODUCT_WARMUP_DAYS = 120~200`：增量重算时为保证指标稳定所需的 **warm‑up** 天数。

### 6.3 添加自定义指标（进阶）
在 `indicators.py` 中向 `REGISTRY` 添加：
```python
REGISTRY["my_ind"] = IndMeta(
    name="my_ind",
    out={"my_col": 3},
    tdx="MY_COL := EMA(CLOSE, 10) - EMA(CLOSE, 30);",  # 可选
    py_func=lambda df, **kw: my_ind(df, **kw),         # 可选
    kwargs={"fast": 10, "slow": 30},
    tags=["product"]
)
```
- **最少**需要：`name` 与 `out`（输出列名与小数位）。  
- 二选一或都写：`tdx`（脚本）/`py_func`（Python 实现）。  
- 新增后，确保 `SYMBOL_PRODUCT_INDICATORS` 包含它（或置为 `"all"`）。

---

## 7) 评分界面（`score_ui.py`）与一键脚本（Windows）

### 7.1 快速启动（Windows）
- 双击 `app_score.bat`（内容：`streamlit run score_ui.py`）。
- 首次使用前，如果缺包：先双击 `setup_win.bat` 安装通用依赖，随后**务必**补装：
  ```bash
  pip install duckdb tushare streamlit pyarrow
  ```

### 7.2 常见设置（在 `config.py`）
- `USE_PARQUET=True`；`PARQUET_BASE` 指向数据目录；`PARQUET_ADJ`/`PARQUET_USE_INDICATORS` 与你下载的口径一致；  
- 回测窗口：`STRATEGY_START_DATE/STRATEGY_END_DATE`；  
- 其它评分选项按页面表单为准。

> 规则编写与表达式口径，请参考《策略测试器_字段与表达式参考.md》。

---

## 8) 使用方法（把脚本放进流程里）

**Windows 新人 10 分钟上手：**
1. 安装 Python 3.10+，在项目目录执行：  
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. 双击 `setup_win.bat`（或手动执行 §1.2 的 `pip install ...`）。  
3. 配置 Token（§2.2），并把 `PARQUET_BASE` 指到一个**可写**目录（如 `D:\stock_data`）。  
4. **首次全量**：`python download.py` → 回答 `y`。  
5. **启动评分 UI**：双击 `app_score.bat`（或 `streamlit run score_ui.py`）。  
6. **Parquet 浏览 App**（如仓库包含）：`streamlit run parquet_viewer_app.py`。  
7. **日常增量**：`python download.py` → 回答 `n`。

---

## 9) 故障排查（FAQ）

- “Tushare Pro 未配置” → 检查 `config.TOKEN` 或 `TUSHARE_TOKEN`。  
- “需要 pyarrow/duckdb” → `pip install pyarrow duckdb`。  
- DuckDB 内存不足 → 调低 `DUCKDB_THREADS`，或设 `DUCKDB_MEMORY_LIMIT="4GB"`。  
- 读取不到数据 → 确认 `PARQUET_BASE` 与 `DATA_ROOT` 对齐；复权口径与是否带指标一致。  
- Windows 中文路径/盘符问题 → 尽量使用英文路径（例如 `D:\stock_data`）。  
- 代理/网络 → 默认不走代理；若必须代理，请在启动前设置系统级代理环境变量。
