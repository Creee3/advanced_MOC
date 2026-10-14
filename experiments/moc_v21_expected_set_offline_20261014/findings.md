# Findings

v2 的 radius=1.22035 来自单次 delta 残差，与模型参数/期望收益不确定性不同。宽上界与 U/token 排序共同导致近全读且组读取为零。

新方法先与可观测训练动作对齐：只在空集条件下预测整组收益，一次性做有限集合决策。此改造修正数学/支持范围，尚无改善准确率或全流程 token 的证据。

旧 cal12 只读审计：旧 checkpoint 用 mean-cost 在24个已观测 episode 选择 empty15/singleton9/bundle0，实际 delta 总和0；这能停止强制全读，但不能证明路由信号可靠。该探索性检查不用于新模型排序。
