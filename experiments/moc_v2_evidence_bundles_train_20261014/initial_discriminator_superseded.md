# SUPERSEDED, not executed: initial train-only discriminator

用户后续要求回到train-val-test完整流程。本草案没有启动推理，不作为当前协议或推进门槛；当前协议见protocol.md。

用户授权启动 v2；先实施此前建议的小型训练集鉴别。冻结 v1，不重新跑全部基准。

## Questions and candidate units

从原 v1 已完整有效的 fit 题组（原 index<36）中按原顺序取前 12 题，使用其已保存的 Vanilla 上游输出、工具后缀、角色、实际 DAG 与 MiniLM sketch。不补替代题，不读旧 val/test 金标。根据预先定义的角色轮换和输入长度选择一个接收方/证据组；每种角色覆盖两密度。优先能在 1024 通信 token 中放下的相邻祖先二元组，其次单条，按最远 hop 和 SHA256 做输入条件选择。具体选择在 pilot_manifest.json 中冻结。Inspector 只有一个祖先，所以四个 Inspector 单元为二元对照；其余八个单元为四条件。

四条件为空、A、B、A+B。单条也是合法动作，不要求所有远端证据都带中间正文。相邻路径只是候选生成规则；没有等 token 的无关二元组对照，本鉴别不能宣称路径式方案优于任意成组方案。

## Intervention and downstream continuation

接收方沿用原 MathSolver._async_execute，空 spatial/temporal，按原作者未合并消息布局传入所选祖先。空集为真正无正文，没有 fallback。其后沿实际 DAG 顺序重算接收方所有后代，用原 Vanilla 一跳空间输入继续。其他节点输出完全冻结。最终原 FinalRefer 按原节点顺序汇总全部输出。记录局部解析结果和最终解析结果；监督是固定材料、Vanilla continuation 的最终效用，尚非部署 v2 的全图效用。

上游材料在原 Vanilla 图中生成，因此含原通信历史。这是局部上下文干预，不是从无通信运行开始的全局因果估计。

## Exact reuse and costs

模型 digest、完整有序 messages、全部显式 options（seed42、temperature0、8192 ctx、2000 output、num_gpu999）、stream=false 完全相同才命中调用缓存。同请求历史输出矛盾则拒绝复用，不挑有利答案。缓存记录与历史 trace SHA256 和 event_index 绑定。历史已完成代码执行结果按请求键复用；新工具执行仍限 5 秒。

没有新的 Single/Vanilla/MOC/w/o Phi 全图基准。每个变体报告两个账本：新支出的实际 prompt+completion 与假设重算完整干预图的逻辑模型 token。后者含未改变上游节点原费用及命中缓存的调用费用，不能把缓存节省当成通信算法节省。

每个条件的原始请求/响应先保存，完整单元通过后提交结果。输出截断、上下文超限、工具超时或无效响应使对应整个题单元排除，没有 replacement，费用保留。8 个 quartets 和 4 个 dyads 不能视为独立的40个题。

## Prespecified measurements and gate

- 四条件的局部/最终正确数、从空集到单条/二元组的帮助与损害。
- 2x2 interaction = y_AB - y_A - y_B + y_empty。重点计数联合才变正确（前三者均错、AB对）和联合才变错误；二元正确性非线性，不能据 interaction 自动推出普遍语义协同。
- a→ab、b→ab、empty→ab/单条的局部与最终 delta 是否不同；同题多个转换作为一个题组。
- 分角色、密度、hop、实际调用、缓存复用、完整逻辑费用及本次新增费用。

进入扩展 v2 标签采集/正式选择器训练的门槛：至少3个完整独立题组出现最终联合才正确，或至少3个独立题组出现局部非正但最终正的增量效用；同时记录相反方向，不能隐去联合损害。门槛是探索筛选，非统计证实或准确率保证。若未过，结束本批并讨论最小下一步，不无限扩样本、不用 val 调阈值。

代码中可以离线验证未训练的帮助/损害两头、问题级 bootstrap 和组最大残差校准；未经门槛与独立训练/calibration 协议，不能部署它或声称 v2 有精度收益。少量 calibration 题不足时区间必须保留宽界，不得制造置信停止。

## Resources and end state

现有 LXC 和模型，至多1 main GPU，私有11448端口。GPU util<=15%且free>=21000MiB连续两次，CUDA分配和短GPU生成核验全显存驻留。禁止CPU回退、驱动/模型安装、其他任务操作。最长90分钟、至多256新增调用、250000新增模型token；预算/STOP/故障清理自有服务及子进程，保留raw与费用。没有新监控、val或test。

推理前冻结 config、manifest、所有执行源及输入 SHA256 并提交 Git。最终下载/核对原始产物，报告实际有效题数、排除与统计限制。
