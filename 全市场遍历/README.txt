量化策略回测工具 v2 - 使用说明
============================

【运行方法】
------------
方式一：回测指定文件
    python main.py --file data/000001.SZ_平安银行.csv

方式二：回测 data/ 目录下所有CSV文件
    python main.py

【配置修改】
------------
编辑 config.py 中的参数：
- HOLD_DAYS: 持有天数
- START_DATE / END_DATE: 回测的起止时间

【策略修改】
------------
编辑 strategy.py 中的 buy_signal 函数，返回一个布尔Series。
你可以使用 indicators.py 中提供的常用技术指标。

【结果输出】
------------
控制台打印每个文件的回测结果，同时保存 summary 到 results/summary.txt。
