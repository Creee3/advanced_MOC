# Findings

v2 的 radius=1.22035 来自单次 delta 残差，与模型参数/期望收益不确定性不同。宽上界与 U/token 排序共同导致近全读且组读取为零。

新方法先与可观测训练动作对齐：只在空集条件下预测整组收益，一次性做有限集合决策。此改造修正数学/支持范围，尚无改善准确率或全流程 token 的证据。

旧 cal12 只读审计：旧 checkpoint 用 mean-cost 在24个已观测 episode 选择 empty15/singleton9/bundle0，实际 delta 总和0；这能停止强制全读，但不能证明路由信号可靠。该探索性检查不用于新模型排序。

独立审查确认旧collector给singleton保留pair.path造成训练/生产feature不一致，176初始fit动作中106条受影响。只修正派生数据的path/flag，首次结果/source精确归档；原label、费用、其他76特征/v2原件不变。

修复后35题5fold：zero MSE0.304762/fold prior0.303869，完整模型0.370540/compact22为0.350914，净收益均为负，组动作均0。三项固定比较全fail，不启GPU或新cal/val。当前语义相似度/角色/拓扑特征与单次最终delta监督未支持可靠证据价值学习；不能通过阈值或组配额包装出进展。数据有限、候选选择偏倚、固定材料下游proxy及噪声限制仍在，不能推断任何更好表示均无效。
