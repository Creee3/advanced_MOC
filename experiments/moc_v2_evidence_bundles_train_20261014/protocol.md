# v2: downstream group utility, train → calibration → frozen val

用户明确要求v2回到train-val-test主流程。最初12题独立机制鉴别未执行，不再作为推进门槛；机制指标作为本轮训练数据的附带分析。v1保持冻结，已有匹配基准直接复用。

## Data boundary

沿用原v1的47道完整有效官方train题、旧index32输出上限排除记录不补题。原index<36为35道fit，36:48为12道calibration。仅按已冻结问题索引和内容元数据选材料；训练/calibration不同题组，禁止按val挑超参。

每题使用已保存Vanilla上游输出、工具后缀、角色、实际DAG和MiniLM sketch。按两密度轮换四种接收角色，每题2个不同角色，共94个训练单元。按输入token预算1024挑相邻祖先二元组，优先更远hop、最后用SHA256作输入条件tie-break。仅一个祖先的Inspector保留单条。实际71个四条件单元、23个二条件单元；training_manifest.json列出所有选择。

四条件为空、A、B、A+B；单条为空、A。选择单条合法，不要求远端证据携带所有中间正文。路径只是候选生成规则，没有等token无关二元组控制，不能宣称优于任意成组方案。

## Supervision and learner

接收方沿用作者MathSolver._async_execute，在原未合并消息布局下读取所选组；空集真正无正文，没有fallback。随后按实际DAG重算受影响后代，用原Vanilla一跳输入继续。未受影响输出冻结；原FinalRefer按原节点顺序汇总全部输出。

每四条件单元构造empty→A/B/AB与A/B→AB共5个转换；单条1个。目标为最终解析正确性的变化{-1,0,+1}，同时记录接收方局部变化。它是固定材料、Vanilla continuation的下游代理效用，尚非部署v2时的全图效用。上游材料含原Vanilla通信历史，不能称全局无通信因果实验。

77维特征为成员v1接收状态特征均值/最大值、组内相似度、hop跨度、路径、角色、已选状态与成本，不含gold或输出对错。两头ridge分别拟合帮助(delta>0)与损害(delta<0)，regularization固定10，问题级bootstrap32次、seed42，不在val调参。期望代理量为help-harm。

以12个独立calibration题组的最大绝对残差构造alpha=.1的代理区间；不足有限顺序统计量则保持[-1,1]，角色/hop/组大小/已选数量未覆盖也保持宽界。没有部署全图或自适应多步选择的覆盖/非劣保证。只有全部可行候选的代理上界减成本非正才停止；否则选正下界候选或预算内最乐观候选。宽界可能花满预算，必须报告，不能强制通信后称解决信息价值学习。

每完整题提交一个原子JSON训练单元。截断、工具超时、无完整比较题不训练、不补替代，raw和费用保留。拟合/校准最低30/8完整题组，不足则停止不运行val。学习退化或校准不佳照实报告，不暗换策略。

## Cache and accounting

模型digest、有序messages、全部显式options（seed42、temperature0、8192ctx、2000output、num_gpu999）与stream=false完全相同才命中缓存。历史键绑定trace SHA256/event_index；同请求有不同历史输出/工具结果则全部拒绝复用，不择对的答案。新结果固定下来复用，原始响应先保存再检查完整性。

报告每条件完整图逻辑模型token（含冻结上游原费用和缓存费用）与本批实际新增模型token，缓存节省不能当通信算法节省。训练、排除、废弃、warmup费用另计。历史同请求输出不总相同，单次干预标签有生成噪声，不能把delta当确定因果真值。

## Validation

模型、config、训练提交及执行源哈希在首个val前冻结。用原val60、原7角色、DAG .3/.7、解码、主模型和工具控制，仅运行v2的120条件，保持完整Graph.arun与所有输出FinalRefer。Single/Vanilla/MOC/w/o Phi从原540结果读取，不新造/重跑；兼容旧调用也可缓存。旧val已用于v1诊断，本轮是开发验证，不声称未污染独立确认。

分密度比较准确率和完整推理token，问题级paired bootstrap；不完整输出共同排除同题的敏感性与主结果分开。报告全停、hop/组读取、乐观探索/置信停止、训练费用与角色覆盖；同题两密度不能当120独立题。成功不预设，不根据本轮val重训/改阈值。

最终方法固定后再制定test评估，本批不跑test，历史test暴露记录不抹除。

## Resources and termination

仅已分配LXC、现有Gemma2-27B/MiniLM/cuda_v13，最多2main GPU、私有11448/11449。util<=15%且free>=21000MiB连续两次；真实CUDA分配、短GPU生成和全显存驻留核验。禁止CPU回退、驱动/新模型安装、其他任务操作。

新v2批最多3h/2400新增调用/250万新增模型token；共享锁管理预留成本与实际费用。STOP、预算/墙钟或故障清理自有服务/子进程并保留终态。不新建监控/无限重启，不复活已结束v1。

推理前冻结config、manifest、source/data哈希并提交Git；训练结束冻结checkpoint再val。结束下载和逐文件SHA核对、审计及配对分析，缺失与费用均保留。
