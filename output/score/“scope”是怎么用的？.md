**“scope”是怎么用的？**
 scope 决定**在窗口内**如何把一串布尔条件（由 `when` 表达式逐K计算得到）**折叠**成最终“命中/不命中”。内置几种写法：

- `LAST`：只看**窗口最后一根**（参考日对应那根）是否为 True；

- `ANY`：窗口里**任意一根**为 True 即命中；

- `ALL`：窗口里**全部**为 True 才命中；

- EACH：逐K计分

- `COUNT>=k`：窗口里 True 的**个数至少 k**；

- `CONSEC>=m`：窗口里存在**至少 m 根连续**为 True 的段。

- `ANY_3`：外层 window 内，**存在某个长度=3 的连续子窗**，在**这3天里至少一天**满足 `when`。
   （注意：当外层 window ≥ n 时，`ANY_n` 与普通 `ANY` 在语义上通常等价；只是当 window<n 时会变为 False。）

  `ALL_3`：外层 window 内，**存在某个长度=3 的连续子窗**，在**这3天里每天都**满足 `when`。（≈ `CONSEC>=3` 的语义）

```python
SC_RULES += [
    {
        "name": "相对强于沪深300",
        "timeframe": "D",
        "window": SC_BENCH_WINDOW,
        "when": "RS_399300_SZ_20 > 1.02",   # 20日RS>1.02（强于基准≈2%）
        "scope": "LAST",
        "points": +4,
        "explain": "20日跑赢沪深300"
    },
    {
        "name": "超额动量累积",
        "timeframe": "D",
        "window": SC_BENCH_WINDOW,
        "when": "EXRET_SUM_399300_SZ_20 > 0.05",  # 20日累计超额 > 5%
        "scope": "LAST",
        "points": +3,
        "explain": "超额收益动量"
    },
    {
        "name": "低相关+高β过滤",
        "timeframe": "D",
        "window": SC_BENCH_WINDOW,
        "when": "(CORR_399300_SZ_20 < 0.4) AND (BETA_399300_SZ_20 > 1.05)",
        "scope": "LAST",
        "points": +2,
        "explain": "风格独立且进攻性更强"
    },
]
```

- `RECENT` 规则（举例：当日 +3，近 1~3 天 +2，近 4~5 天 +1）：

```
{
  "name": "放量突破（按远近加权）",
  "timeframe": "D",
  "window": 60,
  "scope": "RECENT",
  "when": "S:=(C>MA(C,5)) & (V>1.5*MA(V,5)); S",
  "dist_points": [
    [0, 0, 3],
    [1, 3, 2],
    [4, 5, 1]
  ],
  "explain": "近端强信号"
}
```

**直接用**

- 打开 `socre_app.py`（Streamlit 工作台）→ 切到“普通选股”页签
- 填表达式（TDX 风格），选择 `D/W/M` + 窗口、`scope`，点“运行选股”
- 命中就会：
  - 出现在表格中
  - 被写入当日白名单（可选把未命中写黑名单）
  - 结果另存 `output/score/select/` 便于复盘

**和黑白名单的联动**

- 名单会落在：`SC_CACHE_DIR/<ref_date>/whitelist.csv|blacklist.csv`（你评分引擎本身也使用这同一路径）。
- “特别关注榜”可选择 `source='white'|'black'|'top'` 聚合次数，与你当前统计函数保持一致（不需改动）。

**表达式写法 & 范围（scope）**

- **示例**：
  
  - 最近 N 根里**至少**有 k 根满足：`COUNT>=k`（如 `COUNT>=3`）
  - 最近 N 根里出现**连续 m 根**满足：`CONSEC>=m`
  - 仅看**最后一根**是否满足：`LAST`
  - 任意一根满足：`ANY`；全部满足：`ALL`
  
- 例子：`timeframe=W, window=12, scope=COUNT>=5, when=MA(C,5)>MA(C,10)`

  
