# MOC 原文基准线与代码委托

原文：`多智能体通信_指定会议论文_2023-2026/09_弱网与语义通信/ICML2026_MOC_Guan_et_al.pdf`。
作者源码：`coding_base/MOC-official`，commit `9c67c92507570704a7df73e452552a3f49e83897`。

| 原文位置 | 比较组 | 本扩展如何委托 |
|---|---|---|
| Table 1 主表 | Single LLM | 直接调用 `experiments/run_direct.py::process_single_question`，一次同模型调用，作者 direct constraint 与答案提取 |
| Table 1 主表 | Vanilla MAS | 作者 `Graph`，同7节点、角色、DAG、模型、最终聚合，`use_neighbor_summary=False`，读直接前驱 |
| Table 1 主表 | MOC | 作者 `Graph`，`use_neighbor_summary=True`，默认 K=2，保留 ISM/五候选整合 |
| Figure 3(b) 消融 | MOC w/o Φ | 保留作者多阶收集与未整合序列化，仅替换 Φ 为原文拼接 |
| Figure 3(a) 消融 | K=1、2、3 | 改作者 `neighbor_hops`；原文已测 K=3，并非原论文只测到2跳 |

`adaptive_sampling` 是本轮候选方法，不伪装成原文 baseline。过去的无通信多代理、Prefix、25%/50%抽取仍是历史分析，当前主比较不使用它们替代 Single LLM / Vanilla MAS / MOC。G-Designer+MOC 是作者额外通用性实验，不替换主表基准线。

## 配置核对

- 主表同7代理，Random DAG 密度0.3/0.5/0.7/1.0；直接委托 `run_experiment.py::get_kwargs` 的随机拓扑、链骨架、其余前向边采样。节点编号与输出源标签由作者产生。
- 默认 MOC K=2、r=1、epsilon=0.1、kappa=45%、五个9B候选；主模型 Gemma-2-27B-Instruct（或原文其他模型）。代码 CLI epsilon 默认0.01，与论文0.1不同，本 recipe 显式传0.1。
- 原 `--use_neighbor_summary` 的 `store_true/default=True` 无法从 CLI 关闭。Vanilla 通过同一原 Graph 的布尔参数关闭，未自己写一个新 MAS。
- 作者 AQuA role 是全局 cycle。对同一轮比较，先由作者分配一次，再快照角色，用 `node_kwargs` 原接口复用，避免顺序创建不同 arm 时角色轮换造成混淆。
- `PaperRecipe.use_cot=False` 是当前代码默认，不声称是原文全部模型实验的已确认 CoT 开关；正式协议必须对齐原文/作者命令，并在所有组一致记录。
- 默认 num_rounds=1；本候选只支持 MathSolver（AQuA/GSM8K/SVAMP）和 FinalRefer，暂不支持 CodeWriting / Wiki工具路径、多轮 temporal 边。原基准线仍按作者实现可用，本扩展不会为这些任务伪造替代实现。

## 可复现修正与未解决点

`AQUAPromptSet.postprocess_answer` 被作者标记为 `@staticmethod` 却签名是 `(self,answer)`。直接脚本以一个实参调用会抛 TypeError，随后把样本记为错误。本扩展只在发现这个精确签名时补参数绑定，并仍调用原函数体；提示词、模型调用、解析规则均不变。该修正必须在实验记录中披露，不能把错误基准线的低准确率算成方法收益。

作者普通消息顺序是“最短 hop 从远到近、同 hop 按 BFS 遍历”。候选保留其子序列，不把它误称为严格全局拓扑序。FinalRefer 依旧读取全部普通节点回复，没有把最终聚合器也偷换成预算剪枝。

本轮仅源码接入与离线结构检查，不宣称复现 Table 1 数字。需要真正跑模型时，必须记录 main/merger 实际 temperature、上下文/输出限长和部署；`load_author_api`/本 recipe 不在本轮启动服务，也不代替后端 usage 计数。
