# GSM8K / MMLU-Pro 已批准执行协议

| 数据集 | train | val | 作者 test | train 来源 | test 来源 | 角色路径 |
|---|---:|---:|---:|---|---|---|
| GSM8K | 24 | 120 | 300 | 官方 train7473 固定 seed42 抽样 | 作者 GSM8K300 | MathSolver / FinalRefer |
| MMLU-Pro | 14 | 56 | 280 | 官方 validation70，每类 1 题 | 作者 MMLU-Pro280 | AnalyzeAgent / FinalRefer |

每个数据集训练 labels 独立产生；val 只用于一次冻结 checkpoint 的比较与诊断，test 只在源代码、训练和策略锁定后运行。适配阶段最多占用两张 27B 主 GPU 并行；适配结束后清理主服务，再串行使用额外 9B 合并 GPU 跑原 MOC baseline。全流程不启动第三张 GPU。

适配方法继承 v2.1：MiniLM 消息 sketch，最多 4 跳候选，空集/单条/有直接来源边的二元组，1024 token 通信预算，cost penalty 0.02，compact22 加 ridge10，bootstrap32 仅用于训练诊断。每题两个预先规划、结果无关的接收角色 episode；下游最终答案 correctness 作为标签，排除题只报告不替补。

冻结的比较组：
1. AQuA checkpoint 的 frozen-transfer（只在 val，用来观察跨任务行为）；
2. 本任务 train 后冻结的 adapted v2.1（val 与作者 test）；
3. 原 MOC（作者 test，必要时只补缺失条件）。

所有结果按 question_id、arm、density 保存；invalid 终态也提交并跳过。报告每密度配对准确率、bootstrap(seed42) 区间、McNemar、全流程 main+merger token，并分列训练/编码/失败/未知费用。
