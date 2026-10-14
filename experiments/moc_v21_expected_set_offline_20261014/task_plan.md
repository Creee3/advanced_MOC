# v2.1 离线修订

目标：修正 v2 的效用估计/选择目标错位，复用已有训练材料实现可检验的整组选择。本轮不启动 GPU，不重跑基准，不跑 test；保留冻结 v2 原件。

1. 审计停止区间、bootstrap 截距及状态覆盖 — complete。
2. 冻结仅用 train35 的离线协议 — complete（219aa66，先于数值比较）。
3. 实现正确中心化/题级加权的期望收益模型与一次性证据组选择 — complete（bfc2204）。
4. 运行问题隔离 OOF、必要行为测试及源码审查 — complete；修正schema后的唯一固定复核仍全部fail。
5. 记录通过/失败，提交可复现代码与下一步边界 — complete。

本轮工程/离线评价完成，研究质量成本目标未实现；没有晋级模型、运行新cal/val/GPU/test或重跑基准。19项新增、12项v2、6项v1检查通过，offline_audit通过；原件及修复前结果保留。

反思：本轮离线结果出来后立即收口，不反复搜索超参。通过代理指标也不等于全图质量已提高。

## 已知限制与错误
- 原 v2 的单次结果残差区间 radius>1，不是期望效用置信区间；所有上界净收益为正，停止不可能。
- 题级 bootstrap 使用未中心化的设计矩阵并固定均值截距，存在实现偏差。
- 训练只有 selected_count=0/1，不能支持无限逐条追加。
- 子代理在预注册前复核了旧 checkpoint 的 cal12 均值决策；此检查仅为已有版本的探索性审计，不能用于新模型选择或作为独立验证。
- 首次OOF后发现旧collect的singleton继承episode pair.path，connected-path特征与production不同。原结果已归档attempt00_stored_features；更正协议78faecb先提交，只修改派生数据的path/flag，固定原方案一次复核，无新增调用/参数搜索。
