# 2026-10-15 用户批准40 train + author test

已核查现有两批：GSM原MOC205/600，适配600终态保留；Pro适配560已清理等待原序MOC。旧两批健康、锁匹配，无恢复或改截止。

新两批尚未启动模型调用；当前无符合主GPU门槛卡。读取旧协调器config.json、runtime/train_policy.py及advanced_MOC/tests均不存在，已确认配置在各实验目录、训练逻辑在runner.py，不依赖这些误路径。

按最新指令替代先前114/48训练和val草案；不新增验证或迁移比较组。

CPU preflight: MMLU 6 Graph conditions plus4 math namespace/4 isolation/timeout passed. SVAMP failed before model calls: author registry has no svamp prompt key; author run_experiment explicitly uses prompt_domain=gsm8k. Original no-call bundle archived preflight_attempt1_LOCAL_ONLY; repair only native prompt-domain mapping, no algorithm/author source changes. Test gold stays local.

Both native preflights passed (6 Graph conditions each;4 namespace/4 denial/timeout). Original regression first attempt had13 missing fixture-path errors;8 admission tests passed. Corrected only copied test AUTHOR/ASSETS paths and supplied unchanged ladies fixture;8 admission and53 original CPU regressions now passed. One local ladies fixture lookup mistakenly included advanced_MOC prefix, corrected to workspace root. All production source/data locks verified locally and44 uploaded files per batch verified on cloud. No test gold uploaded.

2026-10-16 00:05 dispatch首个准入tick，两次10秒检查仍无符合main卡门槛资源，waiting_for_gpu；两批run_start不存在、模型调用与费用均0。00:06只读核查两批冻结源码/数据锁全匹配，test0；未启动不应将空进程列表all([])当成真实存活，已修正新只读helper摘要。moc十分钟监控工具已成功更新ACTIVE，保留旧三批并新增两批等卡/首次启动/分阶段监控。dispatch是有界tick，不是常驻进程，退出或actual_cmdline=null正常。一次本地plan读取默认GBK报错，改显式UTF-8后继续，未影响云端锁。
