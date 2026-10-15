# 原MOC的MMLU-Pro val补测

用户在2026-10-15明确批准：同一val56、密度0.3/0.7、原MOC共112条件；临时新增两张GPU独立运行，现有两批不打断，重叠时本聊天合计最多四张。原两批的预算与截止不重置。

复制父实验的原作者AnalyzeAgent/FinalRefer、七角色、DAG/seed42、Gemma2:27b/9b Q4_0、MiniLM、原MOC K2/ISM/五候选、temperature0/.1、context8192、main output2000、num_gpu999和受限执行。父实验val输入逐字一致；gold仅本地评分，不上传。新目录只执行val/moc，绝不训练或运行test；父实验代码、配置、策略与现有结果均不改动。

独立预算：7200秒绝对截止（含等待），6000新增调用、4000000模型token；actual+unknown保守上界+pending共同计入。两个worker共享新27B服务11451，9B服务11452；只允许GPU0与GPU4对应UUID，连续两次10秒util<=15%且main free>=21000MiB、merger>=10000MiB，再真实CUDA分配、短生成、digest和size_vram>=size核验。健康不重复launch。恢复前唯一归档、费用结算审计与自有资源清理核验；invalid保留并跳过，相同根因最多两次，预算/STOP优先。

新增运行管理仅将supervise限定为val/moc、覆盖112终态和指定GPU，并把合并器trace的stage标签改为val；runner、原模型请求和算法代码不改。空历史缓存明确归档；所有调用和费用保留，不复跑已完成对照。输出完整后下载raw/trace/results/SHA，原作者解析器在本地val gold评分，与已完成两组按题配对比较，并报告主模型+合并器全流程逻辑token、实际新增/失败/未知成本。首次训练成本在父实验单列。
