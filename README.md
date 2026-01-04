# 量化选股系统

基于 Python + Streamlit 的量化研究与交易工具链，覆盖数据抓取、指标计算、评分/回测、预测模拟、策略管理与可视化界面。规则语法兼容通达信，并提供标签体系、指标兜底、规则编辑与语法检查工具。

## 核心能力

- **数据层**：Tushare 获取 A 股多周期数据（D/W/M/60MIN），DuckDB/SQLite 存储，支持全量/增量更新与缓存。  
- **指标层**：内置 KDJ、量比（vr/vr_prev）、BBI、RSI、MACD DIFF 等；TDX 计算失败自动回退 Python，warmup 行遮挡，统一小数位控制。  
- **规则引擎**：TDX 风格表达式，多周期、多窗口；scope 支持 COUNT/CONSEC/RECENT/DIST/NEAR/ANY_n/ALL_n；门槛分层（gate/trigger/require），多子句 AND（clauses）；标签联动（ANY_TAG/TAG_HITS）。  
- **策略类型**：RANKING、FILTER、OPPORTUNITY、POSITION、PREDICTION；评分/黑白名单、买点、持仓信号和模拟场景。  
- **预测/模拟**：明日 K 线生成（跳空/涨跌停/ATR 振幅/成交量模式）；DIFF 反推价格；warmup 天数控制。  
- **界面**：`score_ui.py`（评分、策略测试、规则编辑器、语法检查器）、`app_pv.py`（数据浏览）；Windows 一键 `ui_run.bat`。  
- **工具链**：规则编辑器可视化生成 JSON；策略语法检查器校验字段/表达式/列缺失/废弃字段。

## 环境与安装

- Python 3.10+（建议 3.10–3.13），Windows/macOS/Linux，建议 8GB+ 内存。  
- 依赖安装：

```bash
# Windows 一键
tools/setup_win.bat

# 其他平台或手动
pip install pyarrow duckdb tushare streamlit plotly pandas numpy scipy
```

## 准备工作

1) **配置 Tushare Token（必填）**：在 [tushare.pro](https://tushare.pro) 获取。  
   - 环境变量（推荐）：`$env:TUSHARE_TOKEN="你的token"` 或 `export TUSHARE_TOKEN="你的token"`  
   - 或 `config.py`：`TOKEN = "你的token"`

2) **初始化数据**（首次/增量均自动判断，无需手动输入“是否首次下载”）：

```bash
# 自动判断是否为首次下载，按需全量或增量
python download.py
```

- 程序会读取状态文件/数据库自动判断首次下载与否并决定日期范围，不再需要手动回答“是否首次下载”。  
- 如需全交互模式（自填日期、资产类型、限频等），可运行 `python download.py --interactive`；非交互模式默认无提问。

### 下载行为与参数（结合 download.py 逻辑）

- **Token 解析顺序**：命令行传参 → 环境变量 `TUSHARE_TOKEN/TS_TOKEN` → `config.TOKEN` → 交互式输入（仅 tty）。占位符（your_token/空字符串）视为未配置。  
- **首次/增量自动判断**：依据状态文件+数据库最新交易日，自动选择全量或从最新日的下一日增量；若最新日缺失部分股票，会自动补齐。  
- **结束日期智能处理**：`END_DATE="today"` 时，若今天交易日且 <15:00 取前一交易日，>=15:00 取当日，否则取最近交易日。  
- **限频**：默认启用自适应令牌桶（速率试探/回退），`--no-adaptive-rate-limit` 可关闭。  
- **warmup**：默认开启指标 warmup 计算，前 warmup 行会置空；`--no-warmup` 可关闭。  
- **资产选择**：默认下载股票，`--assets stock index` 可同时拉取指数；交互模式提供选项。  
- **概念数据**：若缺少概念文件，非交互模式默认抓取，交互模式会询问。  
- **断点与重试**：下载失败会列出失败代码并自动重试一次（按失败列表逐股重试）。  
- **常用 CLI 参数**：`--start/--end/--adj(qfq|hfq|raw)/--threads/--token/--assets/--interactive`，详见 `python download.py --help`。

3) **启动界面**：

```bash
# 评分/策略测试/导出
streamlit run score_ui.py
# Windows 一键
ui_run.bat
# 数据浏览
streamlit run app_pv.py
```

## 常用工作流

- **新增/修改策略**：在 `score_ui.py` → “规则编辑辅助工具”可视化配置 → 生成 JSON → 写入 `strategies_repo.py` 对应列表 → 运行语法检查器。  
- **调试单条规则**：评分界面“策略测试器（单条规则）”输入规则，查看命中/解释/所需列。  
- **批量校验**：语法检查器扫描 `strategies_repo.py`，检查必填字段、类型、scope/dist_points 格式、表达式合法性、缺失列、废弃字段。  
- **数据维护**：定期增量 `download.py`；新增指标时在 `indicators.py` 注册并设置 warmup/精度；必要时重算指标列。  
- **买点/持仓**：`strategies_repo.py` 的 `OPPORTUNITY_POLICIES` / `POSITION_POLICIES`；可勾选“基于明日虚拟日检查”复用模拟场景。  
- **模拟场景**：在 PREDICTION 策略中配置 `scenario`（价格/高低点/成交量模式、warmup、个股 override），用于回测或买点验证。

## 目录导览

- `config.py`：Token、数据存储、评分窗口/Top-K、线程/缓存等。  
- `download.py`：数据下载与增量更新入口。  
- `scoring_core.py`：评分引擎、scope 统计、dist_points 打分、门槛与 clauses 合并、标签联动。  
- `predict_core.py`：场景模拟、指标反推、缓存。  
- `strategies_repo.py`：RANKING/FILTER/OPPORTUNITY/POSITION/PREDICTION 列表。  
- `indicators.py`：指标注册表（TDX/Python 兜底、warmup、精度）。  
- `score_ui.py` / `app_pv.py`：Streamlit 前端。  
- `手册/规则编辑方法.md`：规则/策略配置指南。  
- `手册/支持的表达式与计算方法.md`：表达式、函数、指标、标签、解析流程详解。

## 规则/表达式要点

- 变量大小写不敏感；价格别名 O/H/L/C/V；DataFrame 数值列自动注入上下文。  
- 逻辑 `AND/OR/NOT` → `&/|/~`，`=` 自动转 `==`；比较子句自动加括号以适配向量化。  
- scope：`LAST/ANY/ALL/EACH/COUNT>=k/CONSEC>=m/ANY_n/ALL_n/RECENT/DIST/NEAR`（距离类需 `dist_points`）。  
- 门槛：`gate/trigger/require` 等价，默认 `scope="LAST"`，可写字符串/子规则/子句数组。  
- 字段：使用 `score_windows`，`window` 已废弃；标签函数 `ANY_TAG/TAG_HITS/ANY_TAG_AT_LEAST` 支持自定义标签。

## 示例命令

- 运行下载：`python download.py`（自动判断全量/增量，无需输入是否首次）  
- 启动评分界面：`streamlit run score_ui.py`  
- 启动数据浏览：`streamlit run app_pv.py`

## 许可证

仅供学习与研究使用。
