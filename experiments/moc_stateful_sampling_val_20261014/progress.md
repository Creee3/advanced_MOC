# Progress

2026-10-14: restored source, experiment and research state; read shared-server constraints, rechecked existing models/runtime and inspected current GPU availability. Implementation and controlled training-label protocol started. No new GPU process launched yet.

2026-10-14: v1 implemented in stateful.py with 32 fixed sketch/selected-state/topology/cost features, paired local-utility ridge learner, residual recomputation, read/expand/stop and serialized-token ceiling. 22/22 local source-path and state/budget/stop checks passed. Existing dev30 and original val60 recovered exactly from 2026-10-11 manifest; official train97,467 records downloaded locally, selected48 excluding official dev/test content. Frozen protocol and isolated three-GPU supervisor prepared; source/baseline checkout preserved.

10:16北京时间首次启动：Ollama自动发现CUDA12/13各30s超时后落到CPU，warmup显存检查拒绝该运行，0train/val调用。监督器完成端口/子进程cleanup；启动证据保留。独立GPU0/3/7 cuCtxCreate/cuMemAlloc/free全部成功；显式加载现有cuda_v13 libggml-cuda后llama-server成功识别GPU。为有界重试指定既有v13库、预分配探针和warmup num_gpu999，不安装/换后端，也不更改实验算法/数据参数。

10:26:42北京时间重试启动；10:28:18 main0真实GPU warmup OK，size==size_vram==17,121,820,016bytes。Ollama discovery仍误报CPU但显式CUDA库和num_gpu999实际将模型/KV/计算放在GPU，显存及日志核验成立。10:29:13在任何worker启动前同步所有实验请求num_gpu999，避免从warmup999切auto触发reload；该项是全部组共享的部署控制，不是选择器超参数。val前还会冻结执行代码的SHA256。

10:33北京时间：两路training worker各完成1/24题，共16配对标签；48题训练阶段正在运行，val0/540正常。三个服务warmup均GPU全驻留，0当前错误。已下载live_snapshot并核验与远端SHA256一致：5e5a05feda9372f24e22c807b242420174d91ac96c976ffff183cfe7f0d80113。保留完整启动证据与首批标签，startup_verified.json记录本轮已核验状态。后续训练/val/结束审计仍未完成，性能结论pending。

最新下载快照：两路training worker各完成2/24，共4/48题完成，累计37条标签（含正在执行题的已完成对子）。此快照实际数量以startup_verified.json和remote_snapshot为准；上段16条为更早只读观察。

2026-10-14 10:41北京时间只读状态核查：training_labels，两路完成6/24与7/24，共13/48题；累计111条标签（含当前题已完成对子），val0/540尚未开始。训练trace错误计数0，两路/api/ps当前size==size_vram，GPU0/1利用率93%/95%、各约18.3GiB。GPU2已按keepalive卸载9B，训练阶段无摘要调用，属于正常闲置。没有新启动或调整任何实验参数。

11:34北京时间用户状态查询：发现10:56监督器因worker0 output-length断言失败停止，35/48完整有效题，280有效标签；原active labels290条，10条来自失败/未完成题，不能作为训练数据。自有端口/进程/子进程清理确认，GPU占用0。失败快照已下载、hash与远端一致a0606f3934aeff2b3e8ebe2c2cff74071943ba876f51eb8ed238ebf3f23622b9。恢复修复为题目级标签提交和输出上限题整题排除，不改变2000cap/模型/算法/val；保留原48题不另补题，实际有效数另报。3项新增完整性检查通过，续跑剩余原选题前归档全部旧rawtrace，已有GPU时间从总6h扣除。

11:39:03北京时间已续启动监督器1502607。旧raw、运行配置、服务日志完整保存在attempts/training_failure_02；active保留35有效题/280对子，index32题整题排除（18次调用/20,949tokens），被中断的下一题15次调用/9,596tokens另记废弃成本并待重跑。恢复只需处理原剩余12题，无替代样本；原35题不重跑。实际GPU耗时1758.878s已从21600s扣除，剩余19841s。恢复报告training_recovery.json与最新配置/部署hash已同步本地。

用户明确授权每10分钟检查云端并遇断点自动续跑。已通过Codex app创建当前聊天heartbeat，automationId=moc，名称MOC云端实验检查与断点续跑，ACTIVE/10min；并非服务器新cron。约束：只管理本批train/val，健康不重复启动；恢复保留断点/raw/费用且不修改冻结算法/数据/参数，总GPU6h不重置，用户STOP/暂停不复活，同根因两次失败后通知。完成下载/审计/cleanup后暂停此检查。初次create缺destination=thread报schema错误，修正后创建成功。

2026-10-14 11:55–11:56北京时间首次heartbeat：实际训练47有效/1排除完成，376对子（fit35题280对子，calibration12题96对子），已拟合ridge并冻结policy_lock进入val_smoke_running。最初检查11/540条件，两个真实worker和监督器存活、trace持续更新，无trace错误；未重启或调参。已下载新快照并匹配远端hash76364d5b9eaf482225f8ce3e34f63ae96a3584b5fea9aa27dc13ef6c5a1c8c39，checkpoint/config/val-input哈希核验一致，val未用于训练。

2026-10-14 12:07北京时间第二次heartbeat：val_running，49/540条件，分片25/270与24/270；18条件部署smoke已通过并复用结果。监督器1502607与正式worker1513739/1513740实际存活，trace时间持续更新，0错误，主GPU全驻留；GPU0利用率96%，GPU1瞬时0%为正常调用间隙，未据单次利用率重启。正常推进，不重复launch、不改policy/config、不读取val gold，也不重复发送进度通知。

2026-10-14 12:15北京时间第三次heartbeat：val_running，94/540条件，分片41/270与53/270；监督器和两个worker实际存活，trace更新到数秒内，0错误，主GPU利用率54%/96%。相较上次增加45条件，未发生断点，无需恢复。没有修改/重启任务或读取val gold，正常检查保持安静。

2026-10-14 12:25北京时间第四次heartbeat：val_running，status130/540，分片61/270与69/270；worker0日志已到62，属于状态刷新间隙。监督器与两个worker实际存活，trace持续更新，0错误，主GPU利用率92%/95%。相较上次至少增加36条件，运行健康，无断点，不重启、不调参、不读取val gold，保持安静。

2026-10-14 12:35北京时间第五次heartbeat：val_running，168/540，分片80/270与88/270；真实监督器/worker全部存活，trace更新在数秒内，无错误，主GPU利用率96%/94%。较上次增加38条件，未发生中断；不重复启动、修改冻结配置或读取val gold，正常检查保持安静。

2026-10-14 12:45北京时间第六次heartbeat：val_running，209/540，分片95/270与114/270；真实监督器及两个worker存活，trace持续更新，0错误。主GPU0利用率92%、摘要GPU2为55%；GPU1瞬时空闲对应等待摘要调用，未误判卡住。较上次增加41条件，运行正常，无重启/调参/val评分，保持安静。

2026-10-14 12:55北京时间第七次heartbeat：val_running，254/540，分片122/270与132/270；真实监督器和worker全部存活，trace更新在数秒内，无错误，主GPU利用率92%/94%。较上次增加45条件，健康推进，无断点，不重启、不修改冻结设置、不读取val gold，保持安静。

2026-10-14 13:05北京时间第八次heartbeat：val_running，306/540，分片145/270与161/270；真实监督器/worker存活，trace更新到当前数秒内，0错误，主GPU利用率95%/92%。较上次增加52条件，正常推进，未重启或改动冻结参数、未读取val gold，保持安静。

2026-10-14 13:15北京时间第九次heartbeat：val_running，status345/540，分片166/270与179/270；worker0日志已到167，为刷新间隙。真实监督器/worker存活、trace持续更新，0错误，主GPU利用率均96%。较上次至少增加39条件，无断点；未重复启动、修改冻结配置或读取val gold，正常检查保持安静。

2026-10-14 13:25北京时间第十次heartbeat：val_running，397/540，分片191/270与206/270；监督器/worker真实存活，trace在数秒内更新，0错误，自有GPU0/1利用率92%/94%。较上次增加52条件，无断点；GPU5/6现有其他任务占用，不操作它们。未重启或改动冻结设置、未读取val gold，正常检查保持安静。

2026-10-14 13:32北京时间用户状态查询：val_running，431/540（79.8%），分片203/270与228/270，真实监督器/worker存活，无错误，主GPU利用率92%/93%。训练已完成且冻结；剩余109条件，不重启、不调参。按最近速度粗估仍需20–30分钟，非保证完成时间；结束后还有下载/一致性审计与本地评分。

2026-10-14 13:35北京时间第十一次heartbeat：val_running，450/540，分片213/270与237/270；真实监督器及worker存活、trace持续更新，0错误，自有主GPU利用率95%/94%。比13:25增53条件、比13:32增19，无断点；正常检查不重启/调参/读取val gold，保持安静。

2026-10-14 13:45北京时间第十二次heartbeat：val_running，status498/540，分片235/270与263/270，日志已分别236与264，为刷新间隙。真实监督器/worker存活、trace持续更新，无错误，GPU0/1利用率95%/93%。比上次至少增48条件，剩42条件；不重启、不改变冻结配置、不读取val gold，正常检查保持安静。

2026-10-14 13:55北京时间第十三次heartbeat：val_running，status525/540；shard1已270/270并正常退出，shard0为255/270（日志已256），监督器及shard0实际存活且GPU0利用率96%，0错误。GPU1模型已按keepalive卸载为正常空闲，不重启已完成shard1；等待剩余原条件完成，不调参/读val gold，保持安静。

2026-10-14 14:02:55北京时间全540条件完成，cleanup确认自有进程/子进程/端口全部释放，实时ps/ss再次核验。完整raw160文件已下载，归档SHA256869146b31a36282a3241fb5bb9deb1ace578e3da012e547219bd4f9d2c587f37与远端一致，逐文件哈希全通过。初次audit因1条MOC .3主模型length2000拒绝；保留并明确标记输出完整性例外，没有删除/重跑或更改冻结参数。审计报告区分记录费用一致通过与539/540未截断，另做所有稀疏组共同剔除同题的59题敏感性，不按对错筛选。已执行本地分析：.3 v1 19/60对MOC24/60，推理token-10.88%；.7 28/60对28/60、-67.02%，但360非根节点全停普通消息。训练657,528token未摊回；两密度配对区间不支持非劣。完成结果与首次分析_20261014.md已保存，本批终止、不追加GPU/test。

收尾：automationId=moc已确认为PAUSED并通过app update保持暂停，避免对已结束任务重复轮询。ARA会话003记录用户授权与AI实现/恢复/分析的不同来源，结论C05/C06与原raw绑定；用户级memories未修改。

2026-10-14 14:27北京时间按用户要求再次只读SSH核验：540/540 complete；6个记录的监督器/worker/服务PID均不存在，/proc扫描无当前项目实验或私有Ollama服务，11448/11449/11450无监听。项目目录仅留2个VS Code交互终端，无实验进程。nvidia-smi仍显示共享GPU占用，但其进程不可在本LXC识别，不能据此归属本批或清理。检查原始证据保存cloud_residual_check_20261014.json；没有启动/停止任何进程。

用户明确要求复用已完成基准、不反复重跑：同题、同模型、同图/角色/解码及兼容执行协议的后续诊断直接复用已有raw基准，只补新方法或缺失条件。已查看val60作为诊断数据，不重新声称独立验证；若未来独立评估换题或条件变化导致旧基准不适用，明确匹配缺口再制定最小补跑范围，不能将旧题基准与新题方法结果直接配对。
