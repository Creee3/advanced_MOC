# 跨数据集实验：GSM8K 与 MMLU-Pro

## 已批准范围

用户于 2026-10-15 明确批准按以下配置执行：

- GSM8K：24 train / 120 val / 作者 test 300；
- MMLU-Pro：14 train / 56 val / 作者 test 280；
- 两个数据集最多占用两张 GPU，先各用一张 27B 主卡并行完成训练、拟合、val 与适配方法 test；两批适配阶段结束并释放主卡后，再串行运行各自原 MOC baseline（27B 主模型 + 9B 合并器）。
- 每 10 分钟检查一次，只管理这两批；旧 AQuA、HumanEval、MBPP 和已完成 v1/v2 任务不恢复。

## 研究问题

在作者固定 DAG、角色、提示词、最终汇总和解码设置下，少量本任务标签训练的 v2.1 是否能在作者 test 上维持或提升答案质量，同时减少全流程通信 token。GSM8K 使用原 MathSolver/FinalRefer 路径；MMLU-Pro 使用原 AnalyzeAgent/FinalRefer 路径，Wiki Searcher 不纳入本轮。

## 数据隔离

GSM8K 的 24/120 从官方 train7473 按 seed42 的固定哈希抽取；MMLU-Pro 没有官方 train，本轮从官方 validation70 按 14 类分层抽取每类 1 题作 train、每类其余 4 题作 val。作者仓库原始 GSM8K300 与 MMLU-Pro280 保留为 test。三 split 规范化题文交集为 0；答案只在本地 gold 文件，绝不进入云端 prompt。

## 执行约束

每批 8 小时绝对截止、20,000 新调用、16,000,000 模型 token 上限；等待时间计入截止，恢复不重置预算。先完成 CPU 预检和唯一快照，再上传并启动协调器。终端 invalid 结果保留，重复运行只从精确兼容缓存续未提交条件。适配 checkpoint 在 val/test 前冻结，不根据 test 调参。
