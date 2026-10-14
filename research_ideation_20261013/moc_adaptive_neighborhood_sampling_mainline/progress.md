# Progress

2026-10-14 14:02:55：v1 train+val收口，47有效train/1排除、376对子，540val条件生成完成、自有资源全清理；1条稀疏MOC输出限额例外明确披露。稀疏v1 19/60对MOC24/60，推理少10.88%；稠密28/60对28/60，少67.02%，但普通消息全停，不能作为有效多阶选择证据。训练657,528token未摊回，校准未优于恒零，角色覆盖缺Programming Expert。完整raw归档/hash和费用一致审计完成，现阶段不扩跑、不调val、不启动test。详见experiments/moc_stateful_sampling_val_20261014/完成结果与首次分析_20261014.md。

2026-10-14 10:30北京时间：用户授权v1实现+云端GPU+val。stateful.py实现受控配对局部效用ridge学习、已选状态残差更新、read/expand/stop；22/22离线结构检查通过。沿用原val60清单，另用官方train48；训练36/校准12，val前冻结，不运行test。云端GPU0/1主27B、GPU2摘要9B全部真实GPU生成、全显存检查通过，已进入training_labels，后续自动进入540val条件。首次CPU启动故障0实验调用已清理并保留；旧任务不恢复。v1性能仍待验证。

2026-10-14晨间收口：首轮真实方法试验210/210完成，01:55:07北京时间释放所有自有资源。完整210条件调用、token、GPU与cleanup审计通过；结果与机制限制已写首次结果分析_20261014.md。四跳采样尚未实际筛掉大量消息，稠密图省总token主要来自取消9B合并。保留方法主线，暂不原样扩样本；具体下一版待与用户讨论。本轮没有新计算。

2026-10-14 01:05北京时间：用户新授权“可以去跑一下实验”，启动 `experiments/moc_adaptive_sampling_20261014`，与旧六组任务分离。官方dev30、密度0.3/0.7，Single LLM+Vanilla MAS+MOC+4hop候选，共210条件，2main GPU(0/3)+1merger GPU(5)，≤6小时，开发集结束即释放。代码/数据/预算1024/解码固定，gold留本地。此前“本轮不启动GPU”记录为其各自历史时点，不再代表当前授权。

2026-10-14 会话收尾：用ara-research-manager建立本主线下ara记录，区分用户确认方向与AI实现规则；2项决策、1项离线检查事件、3条有范围限制的claim，4份YAML解析通过。未改用户级memories，未新增自动化或后台任务。

2026-10-14 本轮代码迁移完成：固定GraphSAGE/FastGCN/AS-GCN/LADIES官方源码与PDF；以LADIES层条件采样为主内核，补接收方语义状态、互补/token准入、3–4跳、空回退防护、成本账本和hidden/KV显式pool接口。`coding_base/moc_adaptive_sampling`独立接入原Graph.arun/MathSolver；原文三基准线使用作者路径。16/16本地检查通过；27份资产SHA256核验，45份作者src未改，Git clean。0真实模型调用、0GPU实验、0云端任务恢复。prototype参数与增量效用代理未校准；KV真实后端、CodeWriting/Wiki、多轮未接入，不声称完整方法已验证。已保存源码映射、baseline定义、verification.json与offline_trace.json。

2026-10-14 00:03（北京时间）：用户要求收口旧六组全图实验。通过STOP正常停止并核验自有进程退出、11436/11437关闭；dev180/180、test541/1524结果保留。全部runs及冻结关键记录已下载为stopped_snapshot_20261014.tar.gz，SHA256与远端一致。旧六组不自动续跑，当前继续方法设计，未启动新GPU实验。

2026-10-13 23:59左右（北京时间）：用户确认dev30/val60/test254划分。只读核查云端：历史六组dev180/180完成，test已完成raw254、independent50共254，no_communication当时25/254，总计533/1524；原全图监督器、worker与11436服务仍在运行。成本基线烟测实际结果6/6（status completed字段停在5，以results和最终日志为准），自有进程退出、11440/11441关闭；早期hop pilot128/128完成。未新启动或改动实验。

2026-10-13：用户确认把本次“3–4 跳 + 接收方状态条件 + GNN 邻域采样迁移 + hidden/KV sketch 边界”作为当前大主线。

2026-10-13：完成本地盘点。保留 MOC 全图、固定回放、成本基线、自适应 hop pilot 和跨域文献原始资产；只计划清理研究/实验目录下的 Python 缓存。

2026-10-13：本阶段明确不启动实验。下一步只做方法定义、原始论文核对和数据流插入点设计。
