# v2.1：期望收益整组选择，离线协议

## 范围
在 frozen v2 的 47 个有效 train 题中，仍用原 35 个 fit 题开发；12 个 calibration 题只允许在训练内验收通过后做一次最终诊断。已看过的 val60 不进入拟合、特征筛选、超参选择或离线评价，不运行 test/LLM/GPU，不新增题或调用，不重跑论文基准。

本协议是基于 v2 失败诊断的探索性方法修订，不能当作独立确认实验。旧 v2 的 cal12 也已被查看，最终诊断不具有首次盲验性质。

## 实质修订
1. 训练对象从条件逐步追加收益改为初始空通信状态下的整组 downstream correctness delta。只用 left=empty 的已观测转换；同一 episode 的 empty/A/B/AB 是相关潜在结果，不当作独立实验。
2. 每个题等权、题内每个 episode 等权、episode 内动作等权；标准化只用训练折。每次 bootstrap 正确加权中心化 X/y 并恢复不惩罚的截距。帮助/损害两头用于描述，signed mean=help-harm。
3. 一次性枚举最多四跳的 singleton 和最多两条消息的连接路径组，与空集一起比较总期望净收益 mean - 0.02 * communication_tokens / 1024。不按收益/token 排序，不用单次结果预测上界驱动探索，不递归进入无训练支持的 selected_count>=2 状态。输出仍保留作者文本/顺序，最终汇总不改。
4. Bootstrap 分布只作模型稳定性诊断，不声称为期望收益的有效置信区间。元数据无覆盖的候选明确弃权，不以宽区间强迫读取；零收益空集合法，不人为设置必须读/必须组的配额。

## 锁定的比较
以原 fit question_id 排序、seed42 划分固定5折，任何题的两个 episode 与所有标签同折。只比较下列三项，不做超参网格：
- frozen_v2_mean：在同一 initial-state 数据上重新拟合原 v2 公式，保留其原截距实现，用均值净效用决策（旧实现参照，非论文新基准）。
- centered_full：修正中心化/题级加权的两头 ridge，原77特征；regularization=10、bootstrap=32、seed42。
- centered_compact：同修正模型；固定22特征：13项 GROUP_EXTRA_NAMES；member_mean_base_similarity、member_mean_ladies_mass、member_mean_same_role；member_mean_receiver_solver/receiver_analyst/receiver_programmer/receiver_inspector；member_max_base_similarity、member_max_ladies_mass。无新增特征或 val 驱动阈值。

修正模型的主预测为全fit确定性加权岭回归的两头，clip至[0,1]并在和>1时归一化，然后help-harm。32个题级bootstrap复本仅报告稳定性，不改变主均值决策。总拟合权重归一至当前行数，保留ridge=10的量纲。

无需把 offline_null（恒零预测/空集）或 fold_train_mean 当成论文基准；它们是学习器正确性的无信号参照。

## 度量与验收
主要：题级等权 OOF gain MSE，与恒零和每折训练先验比较。另报 help/harm Brier。
次要：每个完整已观测 episode 仅在 empty/A/B/AB 的实际可行动作中作决策，统计实际 downstream delta、净收益、通信输入 token、empty/singleton/bundle 行为及题级 bootstrap 区间。绝不把该 restricted replay 当成完整 candidate-pool 或 Graph.arun 结果；也不把消息输入 token 当全流程 token。

训练内候选只有同时优于两个无信号 MSE 参照、观测题级平均净收益>0，才允许保留为待全图验证候选。符合时选 MSE 最低者；否则保留实现供审查，但禁止因数学修复而直接扩跑 GPU。行为防塌缩统计只报告，不用人为配额让它通过。

训练内通过后，选定模型先冻结/hash，再且仅再做一次 cal12 诊断，不根据 cal 结果重选/调参；若未通过则不读取新 cal/val 结果、不假造效用标签。本轮无论结果如何都记录并停止搜索。

## 可重现与成本
记录输入两个 questions.jsonl 的 SHA256、固定折、源码哈希、原 train/cal 边界、模型配置。仅 CPU 数值拟合，不调用主 LLM；新增模型 token=0。冻结 v1/v2 原数据/源码/报告保持不变。
